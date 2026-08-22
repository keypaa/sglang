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


def _build(layer_id=0):
    torch.manual_seed(0)
    m = ModelSpec()
    m.num_layers = 1
    m.num_experts = NUM_EXPERTS
    m.top_k = 1
    m.shared_experts = 0
    cache = ExpertCache(m, HardwareSpec(), make_policy("lru", NUM_SLOTS), SimBackend(1e12))

    store = ExpertHostStore(num_layers=1, num_experts=NUM_EXPERTS)
    stored = {}
    for e in range(NUM_EXPERTS):
        w13 = torch.randn(4, 6)
        w2 = torch.randn(6, 4)
        store.put(ExpertKey(0, e), w13, w2)
        stored[e] = (w13, w2)

    # Slot-major device weight tensors for a FusedMoE layer.
    w13_weight = torch.full((NUM_SLOTS, 4, 6), float("nan"))
    w2_weight = torch.full((NUM_SLOTS, 6, 4), float("nan"))

    rt = LayerRuntime(
        layer_id=layer_id,
        cache=cache,
        store=store,
        w13_weight=w13_weight,
        w2_weight=w2_weight,
        num_experts=NUM_EXPERTS,
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


if __name__ == "__main__":
    unittest.main()
