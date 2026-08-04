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
