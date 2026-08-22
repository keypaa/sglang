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

    def test_put_get_with_scales_roundtrip(self):
        store = ExpertHostStore(2, 4)
        w13 = torch.randn(8, 16)
        w2 = torch.randn(4, 8)
        w13_scale_inv = torch.randn(2, 1, 16)
        w2_scale_inv = torch.randn(2, 8)
        store.put(
            ExpertKey(0, 1),
            w13,
            w2,
            w13_scale_inv=w13_scale_inv,
            w2_scale_inv=w2_scale_inv,
        )
        entry = store.get(ExpertKey(0, 1))
        self.assertTrue(torch.equal(entry.w13, w13))
        self.assertTrue(torch.equal(entry.w2, w2))
        self.assertIsNotNone(entry.w13_scale_inv)
        self.assertIsNotNone(entry.w2_scale_inv)
        self.assertTrue(torch.equal(entry.w13_scale_inv, w13_scale_inv))
        self.assertTrue(torch.equal(entry.w2_scale_inv, w2_scale_inv))

    def test_total_bytes_includes_scale_bytes(self):
        store = ExpertHostStore(1, 2)
        w13 = torch.zeros(8, 4)
        w2 = torch.zeros(4, 8)
        w13_scale_inv = torch.zeros(1, 4)
        w2_scale_inv = torch.zeros(1, 8)
        store.put(
            ExpertKey(0, 0),
            w13,
            w2,
            w13_scale_inv=w13_scale_inv,
            w2_scale_inv=w2_scale_inv,
        )
        expected = (
            w13.numel() * w13.element_size()
            + w2.numel() * w2.element_size()
            + w13_scale_inv.numel() * w13_scale_inv.element_size()
            + w2_scale_inv.numel() * w2_scale_inv.element_size()
        )
        self.assertEqual(store.total_bytes, expected)

    def test_two_arg_put_leaves_scales_none(self):
        store = ExpertHostStore(1, 2)
        store.put(ExpertKey(0, 0), torch.randn(2, 2), torch.randn(2, 2))
        entry = store.get(ExpertKey(0, 0))
        self.assertIsNone(entry.w13_scale_inv)
        self.assertIsNone(entry.w2_scale_inv)


class TestCudaBackendSourceCompat(CustomTestCase):
    """CudaTransferBackend.load accepts both source shapes (regression)."""

    class _FakeSlot:
        class node:
            index = 3

    class _FakePool:
        def __init__(self):
            self.copied = None

        def copy_in(self, slot_id, w13, w2):
            self.copied = (slot_id, w13, w2)

        def wait_all(self):
            pass

    def _backend(self, pool):
        from sglang.srt.layers.moe.expert_cache import CudaTransferBackend

        b = CudaTransferBackend(h2d_bw=1e12)
        b.set_pool(pool)
        return b

    def test_plain_tuple_source_still_works(self):
        pool = self._FakePool()
        b = self._backend(pool)
        b.set_expert_source(lambda k: (torch.zeros(2, 2), torch.zeros(2, 2)))
        b.load(ExpertKey(0, 0), 16, self._FakeSlot())
        self.assertEqual(pool.copied[0], 3)

    def test_host_store_entry_source_unpacks(self):
        store = ExpertHostStore(1, 2)
        w13 = torch.zeros(2, 3)
        w2 = torch.zeros(3, 2)
        store.put(ExpertKey(0, 0), w13, w2)
        pool = self._FakePool()
        b = self._backend(pool)
        b.set_expert_source(store.get)
        b.load(ExpertKey(0, 0), 24, self._FakeSlot())
        self.assertTrue(torch.equal(pool.copied[1], w13))
        self.assertTrue(torch.equal(pool.copied[2], w2))

    def test_entry_with_scales_raises_clear_error(self):
        store = ExpertHostStore(1, 2)
        s = torch.ones(1, 1)
        store.put(ExpertKey(0, 0), torch.zeros(2, 3), torch.zeros(3, 2), s, s)
        b = self._backend(self._FakePool())
        b.set_expert_source(store.get)
        with self.assertRaises(NotImplementedError):
            b.load(ExpertKey(0, 0), 24, self._FakeSlot())


if __name__ == "__main__":
    unittest.main()
