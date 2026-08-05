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
        pool = StaticExpertPool(_model(), HardwareSpec(), None, num_slots=32)

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
        pool = StaticExpertPool(_model(), HardwareSpec(), None, num_slots=32)
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
        pool = StaticExpertPool(_model(), HardwareSpec(), None, num_slots=32)
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
        pool = StaticExpertPool(_model(), HardwareSpec(), None, num_slots=32)
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
        pool = StaticExpertPool(m, HardwareSpec(), None, num_slots=num_slots)
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
        pool = StaticExpertPool(m, hw, None, num_slots=num_slots, dtype=torch.float32)
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
