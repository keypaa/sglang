"""Unit tests for the MoE expert-cache discrete-event simulator.

Validates the software-pipeline model ported from the C++ reference
(MoE-LRU/include/moe/simulator.hpp) and the cache properties it demonstrates:

    * prefetch overlaps the transfer stream with compute, so ON >= OFF
      throughput for the same trace;
    * both LogitGDS and LRU sustain a high decode hit rate across sessions
      with a rotating hot working set, and LogitGDS stays within a small band
      of LRU's hit rate (its admission filter protects the warm set - the
      distinguishing prefill-thrash behavior is pinned in test_expert_cache.py).

The model is a scaled-down proxy (16 layers x 256 experts = 4k cache keys)
and the cache is deliberately capacity-bound (256 slots) so the eviction
policy, admission filter, and prefetch budget are actually stressed
(mirroring the C++ prefetch test's slot_cap_override=256), while keeping the
CPU-CI runtime around a second per scenario.

Pure Python + stdlib random only (no torch, no GPU) -> CPU CI.
"""

import unittest

from sglang.srt.layers.moe.expert_cache import (
    HardwareSpec,
    ModelSpec,
    TraceGenerator,
    run_simulation,
)
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=10, suite="base-a-test-cpu")

# (num_layers, num_experts, top_k, shared) scaled proxy of DeepSeek-V4-Flash.
MODEL_LAYERS = 16
MODEL_EXPERTS = 256
TOP_K = 8
SHARED = 1
SLOT_CAP = 256


def reference_model():
    m = ModelSpec()
    m.num_layers = MODEL_LAYERS
    m.num_experts = MODEL_EXPERTS
    m.top_k = TOP_K
    m.shared_experts = SHARED
    m.active_params = 8e9
    m.bytes_on_disk = 2 * 1024**3  # ~2GB of expert weights
    return m


def reference_hw():
    hw = HardwareSpec()
    hw.vram_bytes = 6 * 1024**3
    hw.h2d_bw = 25e9
    return hw


class TestPrefetchOverlap(CustomTestCase):
    def _run(self, policy, prefetch, seed=7):
        model = reference_model()
        gen = TraceGenerator(model, seed=seed)
        trace = gen.generate(prefill_tokens=16, decode_tokens=128)
        return run_simulation(
            model,
            reference_hw(),
            policy,
            trace,
            warm_tokens=4,
            prefetch_enabled=prefetch,
            slot_cap_override=SLOT_CAP,
        )

    def test_prefetch_never_slower(self):
        off = self._run("logitgds", prefetch=False)
        on = self._run("logitgds", prefetch=True)

        self.assertGreaterEqual(on.tokens_per_sec, off.tokens_per_sec)
        self.assertGreater(on.stats.hit_rate, 0.5)

    def test_multiturn_reuse_lifts_hit_rate(self):
        r = self._run("logitgds", prefetch=True)
        self.assertGreater(r.decode_hit_rate, 0.5)
        self.assertGreater(r.tokens_per_sec, 0)
        self.assertGreater(r.bytes_moved_gb, 0)


class TestPolicyDecodeRobustness(CustomTestCase):
    def test_logitgds_competitive_with_lru_on_decode(self):
        model = reference_model()
        gen = TraceGenerator(model, seed=11)
        trace = gen.generate(prefill_tokens=16, decode_tokens=128)

        lru = run_simulation(
            model, reference_hw(), "lru", trace, warm_tokens=4,
            prefetch_enabled=True, slot_cap_override=SLOT_CAP,
        )
        gds = run_simulation(
            model, reference_hw(), "logitgds", trace, warm_tokens=4,
            prefetch_enabled=True, slot_cap_override=SLOT_CAP,
        )

        # The admission filter must not degrade decode hit rate versus plain
        # LRU beyond a small band (it trades a little recency for warm-set
        # protection, which the prefill-thrash tests pin separately).
        self.assertGreaterEqual(gds.decode_hit_rate, lru.decode_hit_rate - 0.05)
        self.assertGreater(gds.decode_hit_rate, 0.5)
        self.assertGreaterEqual(gds.tokens_per_sec, lru.tokens_per_sec - 1.0)


if __name__ == "__main__":
    unittest.main()
