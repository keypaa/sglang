"""topk expert-id -> pool-slot-id remap (spec §4).

The MoE kernel gathers weight rows by "expert id"; passing it slot ids makes
it read exactly the streamed rows. Pure gather, no allocation beyond output.
"""
from __future__ import annotations

import torch


def remap_topk_ids(topk_ids: torch.Tensor, device_slot_map: torch.Tensor) -> torch.Tensor:
    flat = topk_ids.reshape(-1).long()
    out = torch.index_select(device_slot_map, 0, flat)
    return out.reshape(topk_ids.shape).to(topk_ids.dtype)
