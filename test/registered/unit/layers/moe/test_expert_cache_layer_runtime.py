"""CPU units for the per-layer runtime: streaming experts into real weight
tensors via the orchestrator's acquire path (Phase 3, Task 4)."""

import unittest

import torch

from sglang.srt.layers.moe.expert_cache import (
    ExpertCache,
    ExpertHostStore,
    ExpertKey,
    HardwareSpec,
    ModelSpec,
    SimBackend,
    make_policy,
)
from sglang.srt.layers.moe.expert_cache.layer_runtime import LayerRuntime
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=10, suite="base-a-test-cpu")

NUM_SLOTS = 4
NUM_EXPERTS = 8


def _gate_pool_rows_match(case, pool, slot, w13, w2, w13_scale_inv=None,
                          w2_scale_inv=None):
    """Residency byte-gate extended to scales: every pool row owned by `slot`
    must be BYTE-equal to the host-store entry — weight rows always, scale
    rows whenever the entry carries scales. Pool scale-param presence must
    mirror the entry's (a pool with params can never hold a scale-less expert
    and vice versa without breaking the quantization contract)."""
    case.assertTrue(
        torch.equal(pool.w13[slot], w13),
        f"pool w13 row != store bytes (slot {slot})",
    )
    case.assertTrue(
        torch.equal(pool.w2[slot], w2),
        f"pool w2 row != store bytes (slot {slot})",
    )
    case.assertEqual(pool.w13_scale is None, w13_scale_inv is None,
                     "pool/entry w13 scale-presence mismatch")
    case.assertEqual(pool.w2_scale is None, w2_scale_inv is None,
                     "pool/entry w2 scale-presence mismatch")
    if w13_scale_inv is not None:
        case.assertTrue(
            torch.equal(pool.w13_scale[slot], w13_scale_inv),
            f"pool w13 scale row != store bytes (slot {slot})",
        )
    if w2_scale_inv is not None:
        case.assertTrue(
            torch.equal(pool.w2_scale[slot], w2_scale_inv),
            f"pool w2 scale row != store bytes (slot {slot})",
        )


def _build(layer_id=0, num_slots=NUM_SLOTS):
    torch.manual_seed(0)
    m = ModelSpec()
    m.num_layers = 1
    m.num_experts = NUM_EXPERTS
    m.top_k = 1
    m.shared_experts = 0
    cache = ExpertCache(m, HardwareSpec(), make_policy("lru", num_slots), SimBackend(1e12))

    store = ExpertHostStore(num_layers=1, num_experts=NUM_EXPERTS)
    stored = {}
    for e in range(NUM_EXPERTS):
        w13 = torch.randn(4, 6)
        w2 = torch.randn(6, 4)
        store.put(ExpertKey(0, e), w13, w2)
        stored[e] = (w13, w2)

    # Slot-major device weight tensors for a FusedMoE layer.
    w13_weight = torch.full((num_slots, 4, 6), float("nan"))
    w2_weight = torch.full((num_slots, 6, 4), float("nan"))

    rt = LayerRuntime(
        layer_id=layer_id,
        cache=cache,
        store=store,
        w13_weight=w13_weight,
        w2_weight=w2_weight,
        num_experts=NUM_EXPERTS,
    )
    return rt, cache, stored


def _build_fp8(num_slots=NUM_SLOTS):
    """Like _build but every expert is an fp8 block: e4m3fn payload + a [1, 1]
    fp32 block scale, wired into a RealWeightPool WITH matching scale params.
    Pool rows start as NaN canaries so any uncopied row fails torch.equal."""
    torch.manual_seed(0)
    m = ModelSpec()
    m.num_layers = 1
    m.num_experts = NUM_EXPERTS
    m.top_k = 1
    m.shared_experts = 0
    cache = ExpertCache(m, HardwareSpec(), make_policy("lru", num_slots), SimBackend(1e12))

    store = ExpertHostStore(num_layers=1, num_experts=NUM_EXPERTS)
    stored = {}
    for e in range(NUM_EXPERTS):
        w13 = torch.randn(4, 6).to(torch.float8_e4m3fn)
        w2 = torch.randn(6, 4).to(torch.float8_e4m3fn)
        s13 = torch.rand(1, 1, dtype=torch.float32)
        s2 = torch.rand(1, 1, dtype=torch.float32)
        store.put(ExpertKey(0, e), w13, w2, w13_scale_inv=s13, w2_scale_inv=s2)
        # Gate against what the STORE holds, not the pre-put locals.
        stored[e] = store.get(ExpertKey(0, e))

    fp8 = torch.float8_e4m3fn
    w13_weight = torch.full((num_slots, 4, 6), float("nan"), dtype=fp8)
    w2_weight = torch.full((num_slots, 6, 4), float("nan"), dtype=fp8)
    w13_scale = torch.full((num_slots, 1, 1), float("nan"), dtype=torch.float32)
    w2_scale = torch.full((num_slots, 1, 1), float("nan"), dtype=torch.float32)

    rt = LayerRuntime(
        layer_id=0,
        cache=cache,
        store=store,
        w13_weight=w13_weight,
        w2_weight=w2_weight,
        num_experts=NUM_EXPERTS,
        w13_scale_param=w13_scale,
        w2_scale_param=w2_scale,
    )
    return rt, cache, stored


class TestLayerRuntime(CustomTestCase):
    def test_roundtrip_copies_stored_weights_into_mapped_slot(self):
        rt, cache, stored = _build()

        rt.ensure_resident([3])

        slot = rt.orch.slot_id_of(0, 3)
        self.assertGreaterEqual(slot, 0)
        self.assertTrue(torch.equal(rt.pool.w13[slot], stored[3][0]))
        self.assertTrue(torch.equal(rt.pool.w2[slot], stored[3][1]))
        self.assertFalse(torch.isnan(rt.pool.w13[slot]).any())
        self.assertFalse(torch.isnan(rt.pool.w2[slot]).any())

    def test_second_call_is_hit_no_reload(self):
        rt, cache, _stored = _build()
        rt.ensure_resident([3])
        loads_after_first = cache.stats.loads
        slot_after_first = rt.orch.slot_id_of(0, 3)

        rt.ensure_resident([3])

        # Resident expert must be served from cache: no new load, same slot.
        self.assertEqual(cache.stats.loads, loads_after_first)
        self.assertEqual(rt.orch.slot_id_of(0, 3), slot_after_first)
        self.assertGreater(cache.stats.hits, 0)

    def test_slot_map_mirror_matches_slot_id_of(self):
        rt, _cache, _stored = _build()
        rt.ensure_resident([3, 5])

        mirror = rt.slot_map_device(torch.device("cpu"))

        self.assertIsNotNone(rt.device_map)
        for e in (3, 5):
            self.assertEqual(int(mirror[e]), rt.orch.slot_id_of(0, e))
            self.assertGreaterEqual(int(mirror[e]), 0)
        # Untouched experts stay unmapped.
        self.assertEqual(int(mirror[0]), -1)

    def test_wait_all_fences_copies_issued_since_record_step(self):
        rt, _cache, stored = _build()
        rt.ensure_resident([3])
        slot = rt.orch.slot_id_of(0, 3)
        pool = rt.pool
        syncs_after_ensure = pool._sync_count
        self.assertGreaterEqual(syncs_after_ensure, 1)

        # A fresh copy issued after the orchestrator's record_step must still
        # be fenced by wait_all (record_step clears copy-batch state, NOT the
        # inflight-transfer fence).
        pool.copy_in(slot, stored[3][0], stored[3][1])
        self.assertEqual(pool._inflight, 1)
        pool.record_step()
        pool.wait_all()

        self.assertEqual(pool._sync_count, syncs_after_ensure + 1)
        self.assertEqual(pool._inflight, 0)

        # And a no-op wait_all must not fence.
        pool.wait_all()
        self.assertEqual(pool._sync_count, syncs_after_ensure + 1)

    def test_wait_all_noop_without_inflight_is_not_a_fence(self):
        rt, _cache, _stored = _build()
        pool = rt.pool
        pool.wait_all()
        self.assertEqual(pool._sync_count, 0)

    def test_telemetry_miss_then_hit_stall_accounting(self):
        rt, _cache, _stored = _build()

        # First call: demand miss -> a load happens, stall window recorded.
        rt.ensure_resident([3])
        tel = rt.telemetry()
        self.assertEqual(
            set(tel),
            {"hits", "misses", "loads", "stall_ms_total", "stall_ms_last"},
        )
        self.assertGreaterEqual(tel["misses"], 1)
        self.assertGreaterEqual(tel["loads"], 1)
        # Miss path: stall_ms_last is wall time around real copies, so it
        # must be strictly positive; total subsumes it.
        self.assertGreater(tel["stall_ms_last"], 0)
        self.assertGreaterEqual(tel["stall_ms_total"], tel["stall_ms_last"])
        loads_after_first = tel["loads"]
        total_after_first = tel["stall_ms_total"]

        # Second call: all-hit -> no new load, no stall.
        rt.ensure_resident([3])
        tel = rt.telemetry()
        self.assertEqual(tel["loads"], loads_after_first)
        self.assertEqual(tel["stall_ms_last"], 0)
        self.assertEqual(tel["stall_ms_total"], total_after_first)
        self.assertGreaterEqual(tel["hits"], 1)

    def test_eviction_reload_pool_rows_byte_match_store(self):
        # Two slots over eight experts forces real evictions. After each
        # residency call, every currently mapped expert must own a slot whose
        # pool rows are BYTE-equal to the host-store weights — the
        # pool-row<->store gate applied post-residency (CPU, stream=None).
        rt, _cache, stored = _build(num_slots=2)

        rt.ensure_resident([0, 1])
        rt.ensure_resident([2, 3])  # evicts 0 and 1

        for e in (2, 3):
            slot = rt.orch.slot_id_of(0, e)
            self.assertGreaterEqual(slot, 0)
            w13, w2 = stored[e]
            _gate_pool_rows_match(self, rt.pool, slot, w13, w2)

        # Reload after eviction: re-routing expert 0 must land its exact
        # store bytes in a pool row again.
        rt.ensure_resident([0])
        slot0 = rt.orch.slot_id_of(0, 0)
        self.assertGreaterEqual(slot0, 0)
        w13_0, w2_0 = stored[0]
        _gate_pool_rows_match(self, rt.pool, slot0, w13_0, w2_0)


    def test_fp8_store_roundtrips_payload_and_scales_into_pool(self):
        # fp8 store entry streamed through ensure_resident into a pool WITH
        # scale params: weight row AND scale row must equal the store bytes.
        rt, _cache, stored = _build_fp8()

        rt.ensure_resident([3])

        slot = rt.orch.slot_id_of(0, 3)
        self.assertGreaterEqual(slot, 0)
        entry = stored[3]
        _gate_pool_rows_match(
            self, rt.pool, slot,
            entry.w13, entry.w2, entry.w13_scale_inv, entry.w2_scale_inv,
        )

    def test_copy_in_enforces_scale_contract(self):
        rt, _cache, stored = _build_fp8()
        pool = rt.pool
        entry = stored[3]

        # Mismatched scale SHAPE raises (pool scale params are [1, 1]).
        with self.assertRaises(AssertionError):
            pool.copy_in(
                0, entry.w13, entry.w2,
                torch.zeros(2, 1, dtype=torch.float32), entry.w2_scale_inv,
            )
        # Mismatched scale DTYPE raises.
        with self.assertRaises(AssertionError):
            pool.copy_in(
                0, entry.w13, entry.w2,
                entry.w13_scale_inv.to(torch.float64),
                entry.w2_scale_inv.to(torch.float64),
            )
        # Pool HAS scale params -> omitting them raises.
        with self.assertRaises(AssertionError):
            pool.copy_in(0, entry.w13, entry.w2)

    def test_bf16_scaleless_pool_unchanged_rejects_stray_scales(self):
        # bf16 store into a scale-less pool: unchanged roundtrip behavior.
        rt, _cache, stored = _build()
        rt.ensure_resident([3])
        slot = rt.orch.slot_id_of(0, 3)
        w13, w2 = stored[3]
        _gate_pool_rows_match(self, rt.pool, slot, w13, w2)

        # Scale-less pool -> stray scales in copy_in raise.
        stray = torch.rand(1, 1, dtype=torch.float32)
        with self.assertRaises(AssertionError):
            rt.pool.copy_in(slot, w13, w2, stray, stray)


def _build_shared(num_layers):
    """One shared cache + store spanning `num_layers` model layers."""
    torch.manual_seed(0)
    m = ModelSpec()
    m.num_layers = num_layers
    m.num_experts = NUM_EXPERTS
    m.top_k = 1
    m.shared_experts = 0
    cache = ExpertCache(m, HardwareSpec(), make_policy("lru", NUM_SLOTS), SimBackend(1e12))

    store = ExpertHostStore(num_layers=num_layers, num_experts=NUM_EXPERTS)
    stored = {}
    for layer in range(num_layers):
        for e in range(NUM_EXPERTS):
            w13 = torch.randn(4, 6)
            w2 = torch.randn(6, 4)
            store.put(ExpertKey(layer, e), w13, w2)
            stored[(layer, e)] = (w13, w2)
    return cache, store, stored


class TestLayerRuntimeSharedCache(CustomTestCase):
    def test_layers_route_distinct_keys_no_collision(self):
        cache, store, stored = _build_shared(num_layers=2)
        runtimes = []
        for layer in range(2):
            w13_weight = torch.full((NUM_SLOTS, 4, 6), float("nan"))
            w2_weight = torch.full((NUM_SLOTS, 6, 4), float("nan"))
            runtimes.append(
                LayerRuntime(
                    layer_id=layer,
                    cache=cache,
                    store=store,
                    w13_weight=w13_weight,
                    w2_weight=w2_weight,
                    num_experts=NUM_EXPERTS,
                    num_layers=2,
                )
            )
        rt0, rt1 = runtimes

        rt0.ensure_resident([5])
        rt1.ensure_resident([5])

        # Same expert id on two layers: distinct cache keys -> two demand
        # misses (a key collision would serve the second as a hit).
        self.assertEqual(cache.stats.misses, 2)
        s0 = rt0.orch.slot_id_of(rt0.layer_id, 5)
        s1 = rt1.orch.slot_id_of(rt1.layer_id, 5)
        self.assertGreaterEqual(s0, 0)
        self.assertGreaterEqual(s1, 0)
        # Both resident simultaneously => distinct slots.
        self.assertNotEqual(s0, s1)
        # Each slot row holds its OWN layer's weights.
        self.assertTrue(torch.equal(rt0.pool.w13[s0], stored[(0, 5)][0]))
        self.assertTrue(torch.equal(rt0.pool.w2[s0], stored[(0, 5)][1]))
        self.assertTrue(torch.equal(rt1.pool.w13[s1], stored[(1, 5)][0]))
        self.assertTrue(torch.equal(rt1.pool.w2[s1], stored[(1, 5)][1]))

        # Re-routing each layer hits its own key.
        misses_before = cache.stats.misses
        rt0.ensure_resident([5])
        rt1.ensure_resident([5])
        self.assertEqual(cache.stats.misses, misses_before)


if __name__ == "__main__":
    unittest.main()
