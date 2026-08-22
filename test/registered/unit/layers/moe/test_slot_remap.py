"""CPU units for the topk->slot id remap."""

import torch

from sglang.srt.layers.moe.expert_cache import remap_topk_ids
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


class TestSlotRemap(CustomTestCase):
    def test_remaps_each_id_independently(self):
        # experts 0..7 -> slots 5,3,1,0,7,6,4,2
        m = torch.tensor([5, 3, 1, 0, 7, 6, 4, 2], dtype=torch.int32)
        ids = torch.tensor([[0, 1], [2, 3]], dtype=torch.int64)
        out = remap_topk_ids(ids, m)
        self.assertEqual(out.tolist(), [[5, 3], [1, 0]])
        self.assertEqual(out.dtype, torch.int64)

    def test_preserves_int32(self):
        m = torch.arange(10, dtype=torch.int32)
        ids = torch.tensor([[4, 9]], dtype=torch.int32)
        self.assertEqual(remap_topk_ids(ids, m).dtype, torch.int32)

    def test_layer_maps_are_independent_inputs(self):
        # caller picks which layer's map to pass; helper is stateless
        m0 = torch.tensor([9, 8], dtype=torch.int32)
        m1 = torch.tensor([1, 2], dtype=torch.int32)
        ids = torch.tensor([[0, 1]], dtype=torch.int64)
        self.assertEqual(remap_topk_ids(ids, m0).tolist(), [[9, 8]])
        self.assertEqual(remap_topk_ids(ids, m1).tolist(), [[1, 2]])
