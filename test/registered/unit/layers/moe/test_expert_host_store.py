"""CPU units for the pinned expert host store."""

import unittest

import torch

from sglang.srt.layers.moe.expert_cache import ExpertKey, ExpertHostStore
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


class TestExpertHostStore(CustomTestCase):
    def test_put_get_roundtrip_pins_memory(self):
        store = ExpertHostStore(num_layers=2, num_experts=4)
        w13 = torch.randn(8, 16)
        w2 = torch.randn(4, 8)
        store.put(ExpertKey(0, 1), w13, w2)
        entry = store.get(ExpertKey(0, 1))
        self.assertTrue(torch.equal(entry.w13, w13))
        self.assertTrue(torch.equal(entry.w2, w2))
        if torch.cuda.is_available():
            self.assertTrue(entry.w13.is_pinned())

    def test_duplicate_put_raises(self):
        store = ExpertHostStore(1, 2)
        e = (torch.randn(2, 2), torch.randn(2, 2))
        store.put(ExpertKey(0, 0), *e)
        with self.assertRaises(RuntimeError):
            store.put(ExpertKey(0, 0), *e)

    def test_missing_key_raises(self):
        store = ExpertHostStore(1, 2)
        with self.assertRaises(KeyError):
            store.get(ExpertKey(0, 7))

    def test_bytes_accounting(self):
        store = ExpertHostStore(1, 2)
        store.put(ExpertKey(0, 0), torch.zeros(8, 4), torch.zeros(4, 8))
        self.assertEqual(store.total_bytes, 8 * 4 * 4 + 4 * 8 * 4)


if __name__ == "__main__":
    unittest.main()
