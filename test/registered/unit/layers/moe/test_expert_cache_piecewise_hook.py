"""CPU units for the tc_piecewise prepare hook: the split custom op that
streams routed experts into residency and returns slot-remapped ids on every
decode-step replay (Phase 3, M3, Task 2)."""

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
from sglang.srt.layers.moe.expert_cache.piecewise_hook import (
    expert_cache_prepare_impl,
    register_expert_cache_runtime,
    unregister_expert_cache_runtime,
)
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=10, suite="base-a-test-cpu")

NUM_SLOTS = 4
NUM_EXPERTS = 8


class _RecordingPool:
    def __init__(self, log):
        self._log = log

    def wait_all(self):
        self._log.append("wait_all")


class _RecordingRuntime:
    """Mirrors ONLY the runtime surface the prepare hook touches."""

    def __init__(self, device_map):
        self.log = []
        self.pool = _RecordingPool(self.log)
        self._device_map = device_map

    def ensure_resident(self, expert_ids):
        self.log.append(("ensure_resident", list(expert_ids)))

    def slot_map_device(self, device):
        self.log.append(("slot_map_device", device))
        return self._device_map


def _build_real_runtime():
    torch.manual_seed(0)
    m = ModelSpec()
    m.num_layers = 1
    m.num_experts = NUM_EXPERTS
    m.top_k = 1
    m.shared_experts = 0
    cache = ExpertCache(m, HardwareSpec(), make_policy("lru", NUM_SLOTS), SimBackend(1e12))

    store = ExpertHostStore(num_layers=1, num_experts=NUM_EXPERTS)
    for e in range(NUM_EXPERTS):
        store.put(ExpertKey(0, e), torch.randn(4, 6), torch.randn(6, 4))

    w13_weight = torch.full((NUM_SLOTS, 4, 6), float("nan"))
    w2_weight = torch.full((NUM_SLOTS, 6, 4), float("nan"))

    return LayerRuntime(
        layer_id=0,
        cache=cache,
        store=store,
        w13_weight=w13_weight,
        w2_weight=w2_weight,
        num_experts=NUM_EXPERTS,
    )


class TestPrepareHookCallSequence(CustomTestCase):
    def test_records_ensure_then_fence_then_map_refresh(self):
        # Slot layout chosen so the remap result is distinguishable per id.
        device_map = torch.tensor([7, -1, 5, -1, 2, -1, 0, -1], dtype=torch.int32)
        rt = _RecordingRuntime(device_map)
        register_expert_cache_runtime(11, rt)
        self.addCleanup(unregister_expert_cache_runtime, 11)

        ids = torch.tensor([[2, 0], [2, 2]], dtype=torch.int64)
        out = expert_cache_prepare_impl(ids, 11)

        # Body order per spec §2.2: host-read -> residency -> fence ->
        # device-map refresh -> remap. The 2D ids arrive FLAT (residency
        # consumes Sequence[int]; duplicates are the runtime's business).
        self.assertEqual(rt.log[0], ("ensure_resident", [2, 0, 2, 2]))
        self.assertEqual(rt.log[1], "wait_all")
        self.assertEqual(rt.log[2], ("slot_map_device", torch.device("cpu")))
        self.assertEqual(len(rt.log), 3)

        self.assertEqual(out.tolist(), [[5, 7], [5, 5]])
        self.assertEqual(out.shape, ids.shape)

    def test_missing_runtime_raises(self):
        # No half-wired window: replaying a layer whose runtime was never
        # registered must fail loudly instead of gathering raw expert ids
        # against pool-sized weights.
        ids = torch.tensor([[3, 5]], dtype=torch.int64)
        with self.assertRaises(KeyError):
            expert_cache_prepare_impl(ids, 999)


class TestPrepareHookEndToEnd(CustomTestCase):
    def test_routes_and_remaps_via_real_layer_runtime(self):
        rt = _build_real_runtime()
        register_expert_cache_runtime(21, rt)
        self.addCleanup(unregister_expert_cache_runtime, 21)

        ids = torch.tensor([[3, 5], [3, 1]], dtype=torch.int64)
        out = expert_cache_prepare_impl(ids, 21)

        flat_expected = [
            int(rt.orch.slot_id_of(0, int(e))) for e in ids.reshape(-1)
        ]
        self.assertEqual(out.tolist(), [flat_expected[:2], flat_expected[2:]])
        for slot in flat_expected:
            self.assertGreaterEqual(slot, 0)
            self.assertLess(slot, NUM_SLOTS)

        # The device mirror the graph-side gather reads is refreshed too.
        self.assertIsNotNone(rt.device_map)
        self.assertEqual(int(rt.device_map[3]), flat_expected[0])
        self.assertEqual(int(rt.device_map[5]), flat_expected[1])
        # Experts never routed stay unmapped in the refreshed map.
        self.assertEqual(int(rt.device_map[2]), -1)

        # A second replay (all-hit) keeps the same stable mapping.
        out_again = expert_cache_prepare_impl(ids, 21)
        self.assertEqual(out_again.tolist(), out.tolist())

    def test_output_preserves_input_dtype_shape_and_device(self):
        rt = _build_real_runtime()
        register_expert_cache_runtime(22, rt)
        self.addCleanup(unregister_expert_cache_runtime, 22)

        ids = torch.tensor([[6, 0], [4, 7]], dtype=torch.int32)
        out = expert_cache_prepare_impl(ids, 22)

        self.assertEqual(out.dtype, torch.int32)
        self.assertEqual(out.shape, ids.shape)
        self.assertEqual(out.device, ids.device)


class TestRegistryHelpers(CustomTestCase):
    def test_register_makes_lookup_work_unregister_breaks_it(self):
        rt = _RecordingRuntime(torch.full((NUM_EXPERTS,), -1, dtype=torch.int32))
        ids = torch.tensor([[0]], dtype=torch.int64)

        with self.assertRaises(KeyError):
            expert_cache_prepare_impl(ids, 31)

        register_expert_cache_runtime(31, rt)
        self.addCleanup(unregister_expert_cache_runtime, 31)
        expert_cache_prepare_impl(ids, 31)

        unregister_expert_cache_runtime(31)
        with self.assertRaises(KeyError):
            expert_cache_prepare_impl(ids, 31)

    def test_unregister_absent_layer_is_a_noop(self):
        unregister_expert_cache_runtime(4096)


class TestSplitOpRegistration(CustomTestCase):
    def test_op_is_registered_in_sglang_namespace(self):
        # Registration happens eagerly at import; the CUDA-gated dispatch
        # itself cannot run on this CPU-only machine, so only existence is
        # asserted here (replay-through-dispatch is validated on GPU).
        self.assertTrue(hasattr(torch.ops.sglang, "expert_cache_prepare"))


if __name__ == "__main__":
    unittest.main()
