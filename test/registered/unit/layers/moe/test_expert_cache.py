"""Unit tests for the MoE expert cache: slot lifecycle, eviction policies,
and prefill-thrash / admission-filter behavior.

These pin the Python port against the C++ reference simulator
(MoE-LRU/tests/test_cache.cpp). Reproducing the C++ numbers:

    test_prefill_thrash_survivors  ->  LRU: 0/12, TinyLFU: 12/12, LogitGDS: 12/12
    test_admission_filter          ->  8/8 hot-set survivors

The cache is pure Python (no torch, no GPU), so it runs on CPU CI.
"""

import unittest

from sglang.srt.layers.moe.expert_cache import (
    ExpertCache,
    ExpertKey,
    HardwareSpec,
    ModelSpec,
    RouterChoice,
    SimBackend,
    SlotState,
    make_policy,
)
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


def small_model(num_experts=16, num_layers=2):
    m = ModelSpec()
    m.num_layers = num_layers
    m.num_experts = num_experts
    m.top_k = 2
    m.shared_experts = 1
    m.bytes_on_disk = m.num_layers * m.num_experts * 1024
    return m


class TestSlotLifecycle(CustomTestCase):
    def test_acquire_release_refcounts(self):
        cache = ExpertCache(
            small_model(),
            HardwareSpec(),
            make_policy("lru", 8),
            SimBackend(1e12),
        )
        key = ExpertKey(0, 3)
        first = cache.acquire(RouterChoice(key, 0.5))
        self.assertEqual(first.state, SlotState.READY)
        # Re-acquire while held must bump the refcount, not double-load.
        second = cache.acquire(RouterChoice(key, 0.5))
        self.assertIs(second, first)
        self.assertEqual(cache.stats.hits, 1)
        self.assertEqual(cache.stats.misses, 1)
        self.assertEqual(cache.stats.loads, 1)
        cache.release(first)
        cache.release(second)

    def test_eviction_waits_for_release(self):
        cache = ExpertCache(
            small_model(),
            HardwareSpec(),
            make_policy("lru", 2),
            SimBackend(1e12),
        )
        s0 = cache.acquire(RouterChoice(ExpertKey(0, 0), 0.5))
        s1 = cache.acquire(RouterChoice(ExpertKey(0, 1), 0.5))
        # Capacity is full and both slots are held; a third expert must not
        # evict either, it is served transiently instead.
        transient = cache.acquire(RouterChoice(ExpertKey(0, 2), 0.5))
        self.assertFalse(cache.is_resident(ExpertKey(0, 2)))
        self.assertTrue(cache.is_resident(ExpertKey(0, 0)))
        self.assertTrue(cache.is_resident(ExpertKey(0, 1)))
        cache.release(transient)
        # After both held slots are released, a new expert can evict the LRU.
        cache.release(s0)
        cache.release(s1)
        cache.acquire(RouterChoice(ExpertKey(0, 3), 0.5))
        self.assertTrue(cache.is_resident(ExpertKey(0, 1)))
        self.assertFalse(cache.is_resident(ExpertKey(0, 0)))

    def test_out_of_range_logit_is_tolerated(self):
        # The C++ contract documents logit in [0, 1] but does not enforce it;
        # the port must degrade gracefully rather than crash.
        cache = ExpertCache(
            small_model(),
            HardwareSpec(),
            make_policy("logitgds", 4),
            SimBackend(1e12),
        )
        for logit in (1.5, -0.1, 0.0, 1.0):
            s = cache.acquire(RouterChoice(ExpertKey(0, 0), logit))
            cache.release(s)
        self.assertTrue(cache.is_resident(ExpertKey(0, 0)))


class TestLRUPolicy(CustomTestCase):
    def test_recency_ordering(self):
        # Mirrors C++ test_lru_recency: with capacity 8, acquiring a 9th
        # distinct expert evicts the least recently used one.
        cache = ExpertCache(
            small_model(),
            HardwareSpec(),
            make_policy("lru", 8),
            SimBackend(1e12),
        )
        held = [cache.acquire(RouterChoice(ExpertKey(0, i), 0.5)) for i in range(8)]
        for s in held:
            cache.release(s)
        s0 = cache.acquire(RouterChoice(ExpertKey(0, 0), 0.5))
        cache.acquire(RouterChoice(ExpertKey(0, 99), 0.5))
        self.assertTrue(cache.is_resident(ExpertKey(0, 0)))
        self.assertFalse(cache.is_resident(ExpertKey(0, 1)))
        cache.release(s0)


class TestPrefillThrash(CustomTestCase):
    """A long prefill floods the cache with one-shot experts; only a
    frequency-aware policy may keep the warm-set experts resident."""

    NUM_EXPERTS = 64
    CAPACITY = 12
    WARM_SET = 6  # per layer; x NUM_LAYERS = 12 total warm keys
    NUM_LAYERS = 2

    def _run(self, policy_name):
        cache = ExpertCache(
            small_model(self.NUM_EXPERTS),
            HardwareSpec(),
            make_policy(policy_name, self.CAPACITY),
            SimBackend(1e12),
        )
        L = cache.model.num_layers
        hot = [ExpertKey(l, e) for l in range(L) for e in range(self.WARM_SET)]
        for _ in range(6):
            for k in hot:
                s = cache.acquire(RouterChoice(k, 0.8))
                cache.release(s)
        for t in range(200):
            for l in range(L):
                e = self.WARM_SET + (t * 7 + l) % (self.NUM_EXPERTS - self.WARM_SET)
                s = cache.acquire(RouterChoice(ExpertKey(l, e), 0.1))
                cache.release(s)
        return sum(1 for k in hot if cache.is_resident(k))

    def test_lru_loses_warm_set(self):
        self.assertEqual(self._run("lru"), 0)

    def test_tinylfu_retains_warm_set(self):
        self.assertEqual(self._run("tinylfu"), self.WARM_SET * self.NUM_LAYERS)

    def test_logitgds_retains_warm_set(self):
        self.assertEqual(self._run("logitgds"), self.WARM_SET * self.NUM_LAYERS)


class TestAdmissionFilter(CustomTestCase):
    """Low-confidence one-shot experts must not flush a high-confidence hot
    set; this is what distinguishes LogitGDS from plain recency caches."""

    NUM_EXPERTS = 1024
    CAPACITY = 8

    def _run(self, policy_name):
        cache = ExpertCache(
            small_model(self.NUM_EXPERTS),
            HardwareSpec(),
            make_policy(policy_name, self.CAPACITY),
            SimBackend(1e12),
        )
        for _ in range(8):
            for e in range(self.CAPACITY):
                s = cache.acquire(RouterChoice(ExpertKey(0, e), 0.9))
                cache.release(s)
        for t in range(1000):
            s = cache.acquire(RouterChoice(ExpertKey(1, 100 + t), 0.05))
            cache.release(s)
        return sum(1 for e in range(self.CAPACITY) if cache.is_resident(ExpertKey(0, e)))

    def test_lru_flushes_hot_set(self):
        self.assertLess(self._run("lru"), self.CAPACITY)

    def test_logitgds_keeps_hot_set(self):
        self.assertEqual(self._run("logitgds"), self.CAPACITY)


class TestModelSpecShapes(CustomTestCase):
    def test_expert_shapes(self):
        m = ModelSpec()
        m.hidden_size = 3840
        m.intermediate_size = 1536
        w13_shape, w2_shape = m.expert_shapes()
        self.assertEqual(w13_shape, (3840, 3072))
        self.assertEqual(w2_shape, (1536, 3840))


if __name__ == "__main__":
    unittest.main()
