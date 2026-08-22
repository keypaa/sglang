"""CPU units for the Phase-A orchestrator (spec §3.3)."""

import unittest
import torch

from sglang.srt.layers.moe.expert_cache import (
    ExpertCache, ExpertKey, HardwareSpec, ModelSpec, RouterChoice,
    SimBackend, make_policy,
)
from sglang.srt.layers.moe.expert_cache.orchestrator import StaticPoolOrchestrator
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=10, suite="base-a-test-cpu")


class FakePool:
    """CPU stand-in for StaticExpertPool exposing the orchestrator's surface."""

    def __init__(self, num_slots):
        self.num_slots = num_slots
        self.copied = []          # [(slot_id, w13, w2)]
        self.pending = 0

    def copy_in(self, slot_id, w13, w2):
        self.copied.append((slot_id, w13, w2))
        self.pending += 1

    def record_step(self) -> bool:
        if self.pending == 0:
            return False
        self.pending = 0
        return True


def _cache(num_slots=8, policy="lru"):
    m = ModelSpec()
    m.num_layers = 2
    m.num_experts = 16
    m.top_k = 1
    m.shared_experts = 0
    return ExpertCache(m, HardwareSpec(), make_policy(policy, num_slots), SimBackend(1e12))


class TestOrchestrator(CustomTestCase):
    def test_slot_map_maps_routed_experts_to_real_slots(self):
        pool = FakePool(8)
        cache = _cache(8)
        orch = StaticPoolOrchestrator(cache, pool, num_experts=16, num_layers=2)

        orch.on_router_output(layer=0, expert_ids=[3], logit=[0.9], next_ids=[5])
        orch.step_commit()

        # Demand miss -> load into slot 0 (the only free slot at first).
        self.assertGreaterEqual(orch.slot_id_of(0, 3), 0)
        self.assertTrue(orch.slot_map_tensor()[3].item() >= 0)
        self.assertEqual(orch.slot_map_tensor()[7].item(), -1)  # untouched

    def test_hit_does_not_reload(self):
        pool = FakePool(8)
        cache = _cache(8)
        orch = StaticPoolOrchestrator(cache, pool, num_experts=16, num_layers=2)
        orch.on_router_output(0, [3], [0.9], [])
        orch.step_commit()
        hits_after_first = cache.stats.hits
        misses_after_first = cache.stats.misses
        slot_after_first = orch.slot_id_of(0, 3)
        orch.on_router_output(0, [3], [0.9], [])
        orch.step_commit()
        # A hit must not reload: one more demand hit, no new miss/load.
        self.assertEqual(cache.stats.hits, hits_after_first + 1)
        self.assertEqual(cache.stats.misses, misses_after_first)
        self.assertEqual(orch.slot_id_of(0, 3), slot_after_first)

    def test_step_commit_flags_only_when_copies_pending(self):
        pool = FakePool(8)
        cache = _cache(8)
        orch = StaticPoolOrchestrator(cache, pool, num_experts=16, num_layers=2)
        # Hit only -> no pending copies -> the pool flag stays clear.
        orch.on_router_output(0, [3], [0.9], [])
        orch.step_commit()
        self.assertEqual(pool.pending, 0)
        # First routing was a demand miss -> one load, no additional miss.
        self.assertEqual(cache.stats.misses, 1)
        self.assertEqual(cache.stats.loads, 1)
        # Second routing of the same expert is a hit (no reload), still clean.
        hits_before = cache.stats.hits
        orch.on_router_output(0, [3], [0.9], [])
        orch.step_commit()
        self.assertEqual(cache.stats.hits, hits_before + 1)
        self.assertEqual(cache.stats.misses, 1)
        self.assertEqual(pool.pending, 0)

    def test_rejected_one_shot_is_upgraded_to_resident(self):
        # A low-logit one-shot may be admission-rejected (transient, index < 0);
        # the orchestrator must upgrade it via acquire_resident so the committed
        # step never maps -1 (spec §6). Needs LogitGDS (LRU never rejects).
        # The hot set must build up value first (repeated high-logit routing),
        # otherwise admission accepts a fresh low-logit key and the upgrade path
        # is never exercised.
        pool = FakePool(8)
        cache = _cache(8, policy="logitgds")
        orch = StaticPoolOrchestrator(cache, pool, num_experts=16, num_layers=2)
        for _ in range(5):
            for e in range(8):
                orch.on_router_output(0, [e], [0.9], [])
                orch.step_commit()
        orch.on_router_output(0, [10], [0.01], [])
        orch.step_commit()
        self.assertGreaterEqual(orch.slot_id_of(0, 10), 0)
        self.assertEqual(orch.slot_map_tensor()[10].item(), orch.slot_id_of(0, 10))

    def test_out_of_range_expert_raises(self):
        pool = FakePool(8)
        cache = _cache(8)
        orch = StaticPoolOrchestrator(cache, pool, num_experts=16, num_layers=2)
        with self.assertRaises(IndexError):
            orch.on_router_output(0, [99], [0.9], [])

    def test_negative_expert_id_raises(self):
        # -1 is the slot map's unmapped sentinel; routing it must not wrap to
        # the last column via negative indexing.
        pool = FakePool(8)
        cache = _cache(8)
        orch = StaticPoolOrchestrator(cache, pool, num_experts=16, num_layers=2)
        with self.assertRaises(IndexError):
            orch.on_router_output(0, [-1], [0.9], [])

    def test_slot_map_is_keyed_by_layer_not_shared(self):
        # Two layers route the SAME expert id; each layer's map entry must be
        # independent (the graph reads layer L's own map).
        pool = FakePool(8)
        cache = _cache(8)
        orch = StaticPoolOrchestrator(cache, pool, num_experts=16, num_layers=2)

        orch.on_router_output(0, [3], [0.9], [])
        orch.step_commit()
        layer0_slot = orch.slot_id_of(0, 3)
        self.assertGreaterEqual(layer0_slot, 0)

        orch.on_router_output(1, [3], [0.9], [])
        orch.step_commit()
        layer1_slot = orch.slot_id_of(1, 3)
        self.assertGreaterEqual(layer1_slot, 0)

        # Layer-0's mapping for expert 3 is untouched by layer 1's routing.
        self.assertEqual(orch.slot_id_of(0, 3), layer0_slot)
        # slot_map_tensor() reflects the ACTIVE layer (layer 1 here).
        self.assertEqual(orch.slot_map_tensor()[3].item(), layer1_slot)
