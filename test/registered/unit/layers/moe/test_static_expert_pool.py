"""Units for the CUDA-graph-safe static expert pool (Phase 2, spec §3.1)."""

import unittest
import torch

from sglang.srt.layers.moe.expert_cache import ModelSpec, HardwareSpec
from sglang.srt.layers.moe.expert_cache.static_pool import StaticExpertPool
from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.test_utils import CustomTestCase

register_cuda_ci(est_time=30, suite="base-b-test-1-gpu-small")


def _model():
    m = ModelSpec()
    m.hidden_size = 64
    m.intermediate_size = 128
    return m


@unittest.skipUnless(torch.cuda.is_available(), "CUDA required")
class TestStaticExpertPool(CustomTestCase):
    def test_allocated_shapes_and_static_addresses(self):
        pool = StaticExpertPool(_model(), HardwareSpec(), None, 32)

        self.assertEqual(tuple(pool.pool_w13().shape), (32, 64, 256))
        self.assertEqual(tuple(pool.pool_w2().shape), (32, 128, 64))

        addr_before = pool.pool_w13().data_ptr()
        pool.copy_in(
            3,
            torch.zeros(64, 256, dtype=torch.float8_e4m3fn),
            torch.zeros(128, 64, dtype=torch.float8_e4m3fn),
        )
        pool.wait_all()
        # Address must not move: the whole point of the static pool.
        self.assertEqual(pool.pool_w13().data_ptr(), addr_before)

    def test_copy_in_roundtrip(self):
        pool = StaticExpertPool(_model(), HardwareSpec(), None, 32)
        # Source tensors must already be pool dtype (fp8) to honor no-alloc.
        w13 = torch.randn(64, 256, dtype=torch.float16).to(torch.float8_e4m3fn)
        w2 = torch.randn(128, 64, dtype=torch.float16).to(torch.float8_e4m3fn)
        pool.copy_in(7, w13, w2)
        pool.wait_all()
        got13 = pool.pool_w13()[7].float().cpu()
        got2 = pool.pool_w2()[7].float().cpu()
        self.assertTrue(torch.equal(got13, w13.float().cpu()))
        self.assertTrue(torch.equal(got2, w2.float().cpu()))

    def test_record_step_flags_pending_copies(self):
        pool = StaticExpertPool(_model(), HardwareSpec(), None, 32)
        # No copies issued yet -> not dirty.
        self.assertFalse(pool.record_step())
        pool.copy_in(
            1,
            torch.zeros(64, 256, dtype=torch.float8_e4m3fn),
            torch.zeros(128, 64, dtype=torch.float8_e4m3fn),
        )
        # A copy was issued this "step".
        self.assertTrue(pool.record_step())
        # Consumed the flag.
        self.assertFalse(pool.record_step())

    def test_precreated_events(self):
        pool = StaticExpertPool(_model(), HardwareSpec(), None, 32)
        ev = pool.event_of(0)
        self.assertIsInstance(ev, torch.cuda.Event)
        # Same object reused (pre-created once, not per call).
        self.assertIs(pool.event_of(0), ev)


@unittest.skipUnless(torch.cuda.is_available(), "CUDA required")
class TestCudaTransferBackend(CustomTestCase):
    def test_load_writes_into_pool_slot(self):
        from sglang.srt.layers.moe.expert_cache import (
            CudaTransferBackend,
            ExpertCache,
            ExpertKey,
            RouterChoice,
            make_policy,
        )

        m = _model()
        num_slots = 8
        pool = StaticExpertPool(m, HardwareSpec(), None, num_slots)
        backend = CudaTransferBackend(h2d_bw=25e9)
        backend.set_pool(pool)

        # Pinned fp8 expert store (matches pool dtype; copy_in asserts it).
        source = {}
        for e in range(4):
            base13 = torch.randn(64, 256)
            base2 = torch.randn(128, 64)
            source[ExpertKey(0, e)] = (
                base13.to(torch.float8_e4m3fn).pin_memory(),
                base2.to(torch.float8_e4m3fn).pin_memory(),
            )
        backend.set_expert_source(lambda k: source[k])

        cache = ExpertCache(m, HardwareSpec(), make_policy("lru", num_slots), backend)
        slot = cache.acquire(RouterChoice(ExpertKey(0, 1), 0.9))
        pool.wait_all()

        # The expert landed in slot.node.index, mapped 1:1 to the pool.
        self.assertIsNotNone(slot)
        got13 = pool.pool_w13()[slot.node.index].cpu()
        got2 = pool.pool_w2()[slot.node.index].cpu()
        self.assertTrue(torch.equal(got13, source[ExpertKey(0, 1)][0]))
        self.assertTrue(torch.equal(got2, source[ExpertKey(0, 1)][1]))


@unittest.skipUnless(torch.cuda.is_available(), "CUDA required")
class TestEagerDecodeHarness(CustomTestCase):
    def test_hit_rate_matches_simulator_under_capacity_pressure(self):
        import random

        from sglang.srt.layers.moe.expert_cache import (
            CudaTransferBackend, ExpertCache, ExpertKey, HardwareSpec,
            ModelSpec, RouterChoice, SimBackend, StaticPoolOrchestrator,
            make_policy,
        )

        m = ModelSpec()
        m.num_layers = 1
        m.num_experts = 256
        m.top_k = 6
        m.shared_experts = 0
        m.hidden_size = 64
        m.intermediate_size = 128

        num_slots = 128  # capacity-bound: cache < key universe
        hw = HardwareSpec()
        hw.vram_bytes = 6 * 1024**3
        hw.h2d_bw = 25e9

        # --- reference hit rate from the Phase-1 simulator ---
        from sglang.srt.layers.moe.expert_cache import TraceGenerator, run_simulation
        gen = TraceGenerator(m, seed=7)
        trace = gen.generate(prefill_tokens=16, decode_tokens=100)
        ref = run_simulation(
            m, hw, "logitgds", trace, warm_tokens=4,
            prefetch_enabled=True, slot_cap_override=num_slots,
        )
        ref_hit = ref.decode_hit_rate
        self.assertGreater(ref_hit, 0.5)

        # --- real CUDA path through the pool ---
        # fp32 pool: the harness only exercises hit-rate semantics + copies,
        # not quantized weights (that is Task 7's job). fp8 would force the
        # store to pre-quantize every expert, adding noise to a hit-rate test.
        pool = StaticExpertPool(m, hw, None, num_slots, dtype=torch.float32)
        backend = CudaTransferBackend(h2d_bw=hw.h2d_bw)
        backend.set_pool(pool)
        cache = ExpertCache(m, hw, make_policy("logitgds", num_slots), backend)
        orch = StaticPoolOrchestrator(cache, pool, num_experts=m.num_experts,
                                      num_layers=m.num_layers)

        # pinned CPU expert store: key -> (w13_pinned, w2_pinned)
        store = {}
        rng = random.Random(1)
        for e in range(m.num_experts):
            w13 = torch.randn(64, 256, dtype=torch.float32)
            w2 = torch.randn(128, 64, dtype=torch.float32)
            store[ExpertKey(0, e)] = (
                w13.pin_memory(), w2.pin_memory(),
            )
        backend.set_expert_source(lambda k: store[k])

        # deterministic decode: emit the same expert selections the simulator
        # saw for decode tokens (reuse trace[·][0] router choices). Hits are
        # counted BEFORE the step's acquire (= demand hit at step start, the
        # simulator's definition), and the NEXT token's choices are prefetched
        # to match the reference's prefetch_enabled=True.
        warm_tokens = 4
        decode_trace = [tok[0] for tok in trace[warm_tokens:]]
        hits = 0
        total = 0
        for i, tok in enumerate(decode_trace):
            expert_ids = [c.key.expert for c in tok]
            logits = [c.logit for c in tok]
            hits += sum(1 for e in expert_ids
                        if cache.is_resident(ExpertKey(0, e)))
            total += len(expert_ids)
            next_ids = ([c.key.expert for c in decode_trace[i + 1]]
                        if i + 1 < len(decode_trace) else [])
            orch.on_router_output(0, expert_ids, logits, next_ids=next_ids)
            orch.step_commit()
        pool.wait_all()

        hit_rate = hits / total
        # Eager CUDA path must reproduce the simulator within a tolerance.
        self.assertGreaterEqual(hit_rate, ref_hit - 0.05)


@unittest.skipUnless(torch.cuda.is_available(), "CUDA required")
class TestCapturedDecodeHarness(CustomTestCase):
    def test_graph_replay_tracks_updated_slot_contents(self):
        from sglang.srt.layers.moe.expert_cache import (
            CudaTransferBackend, ExpertCache, ExpertKey, HardwareSpec,
            ModelSpec, RouterChoice, StaticPoolOrchestrator, make_policy,
        )

        m = ModelSpec()
        m.num_layers = 1
        m.num_experts = 64
        m.top_k = 4
        m.shared_experts = 0
        m.hidden_size = 32
        m.intermediate_size = 16   # so h@w13 -> gate/up -> mid@w2 -> h composes

        num_slots = 16
        hw = HardwareSpec()
        hw.vram_bytes = 6 * 1024**3
        hw.h2d_bw = 25e9

        pool = StaticExpertPool(m, hw, None, num_slots, dtype=torch.float32)
        backend = CudaTransferBackend(h2d_bw=hw.h2d_bw)
        backend.set_pool(pool)
        cache = ExpertCache(m, hw, make_policy("logitgds", num_slots), backend)
        orch = StaticPoolOrchestrator(cache, pool, m.num_experts, m.num_layers)

        store = {}
        for e in range(m.num_experts):
            store[ExpertKey(0, e)] = (
                torch.randn(32, 32, dtype=torch.float32).pin_memory(),
                torch.randn(16, 32, dtype=torch.float32).pin_memory(),
            )
        backend.set_expert_source(lambda k: store[k])

        # ---- Phase A once, before capture warmup ----
        first_ids = [1, 2, 3, 4]
        orch.on_router_output(0, first_ids, [0.9] * 4, next_ids=[])
        orch.step_commit()
        pool.wait_all()

        # ---- static capture inputs (pre-allocated, no alloc in capture) ----
        activations = torch.randn(1, 32, device="cuda")
        topk_ids_buf = torch.zeros(1, 4, dtype=torch.int64, device="cuda")
        slot_map_buf = orch.slot_map_tensor().to(dtype=torch.int64, device="cuda")
        out_buf = torch.zeros(1, 4, 32, device="cuda")

        def moe_ffn(w13, w2):
            """Shared FFN: weights already gathered per routed expert."""
            up = torch.matmul(activations, w13)        # [1, n, 2I]
            gate, act = up.chunk(2, dim=-1)            # [1, n, I], [1, n, I]
            mid = gate * torch.gelu(act)               # [1, n, I]
            return torch.matmul(mid, w2)               # [1, n, H]

        def fake_decode_forward():
            # Captured: wait on per-step event, map topk ids -> slot ids via the
            # static slot-map buffer, gather pool weights, run the FFN.
            torch.cuda.current_stream().wait_event(pool.step_event())
            ids = topk_ids_buf.view(-1)                     # [n]
            slot_ids = slot_map_buf.index_select(0, ids)    # [n] (no alloc)
            out_buf.copy_(moe_ffn(pool.pool_w13()[slot_ids],
                                  pool.pool_w2()[slot_ids]))

        # warmup on a side stream (allocations go to the graph pool — legal)
        g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g, stream=torch.cuda.Stream()):
            fake_decode_forward()

        def eager_reference(ids):
            # Authoritative oracle: read the CPU store by expert id, NOT the
            # pool. If a pool slot ever held stale/mispaired bytes this diverges
            # from the graph's out_buf, so a wrong outcome cannot silently pass.
            ws = [store[ExpertKey(0, e)] for e in ids]
            w13 = torch.stack([w[0].to("cuda") for w in ws])  # [n, H, 2I]
            w2 = torch.stack([w[1].to("cuda") for w in ws])   # [n, I, H]
            return moe_ffn(w13, w2)

        # Route MORE distinct experts than num_slots (32 > 16) so a slot is
        # evicted and re-copied with a DIFFERENT expert mid-test; the graph must
        # read the updated bytes the reference expects. Before each replay we
        # ensure residency (updating the slot map + issuing any demand copies
        # into the same static addresses) and re-point the buffers.
        for step in range(8):
            ids = [step * 4 + k for k in range(4)]
            orch.on_router_output(0, ids, [0.9] * 4, next_ids=[])
            # Spec §6: a committed step never maps an expert to -1.
            for e in ids:
                self.assertNotEqual(orch.slot_id_of(0, e), -1,
                                    f"step {step}: expert {e} not resident")
            orch.step_commit()
            pool.wait_all()
            topk_ids_buf.copy_(torch.tensor([ids], dtype=torch.int64, device="cuda"))
            slot_map_buf.copy_(orch.slot_map_tensor().to(dtype=torch.int64, device="cuda"))
            g.replay()
            torch.cuda.synchronize()
            self.assertTrue(torch.allclose(out_buf, eager_reference(ids),
                                           atol=1e-4), f"step {step} mismatch")
