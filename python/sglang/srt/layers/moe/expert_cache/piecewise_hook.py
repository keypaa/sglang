"""tc_piecewise prepare hook (spec §2.2).

Under tc_piecewise, the region registered as a split custom op re-executes as
Python on every decode-step replay. This op is that split point for
expert-cached MoE layers: given a layer's router ``topk_ids`` it streams the
routed experts into residency, fences pending H2D copies, refreshes the device
slot map, and returns slot ids for the captured MoE graph to gather pool rows
by. Deliberately synchronous and demand-driven (spec §2.4: no prefetch).
"""
from __future__ import annotations

from typing import Dict

import torch

from sglang.srt.layers.moe.expert_cache.layer_runtime import LayerRuntime
from sglang.srt.layers.moe.expert_cache.slot_remap import remap_topk_ids
from sglang.srt.utils.custom_op import register_custom_op

# Per-layer runtimes, populated once at model wiring time and read on every
# replay. A missing entry must raise: forwarding a pool-sized layer without
# its runtime would silently gather wrong weight rows.
_EXPERT_CACHE_RUNTIMES: Dict[int, LayerRuntime] = {}


def register_expert_cache_runtime(layer_id: int, runtime: LayerRuntime) -> None:
    _EXPERT_CACHE_RUNTIMES[layer_id] = runtime


def unregister_expert_cache_runtime(layer_id: int) -> None:
    _EXPERT_CACHE_RUNTIMES.pop(layer_id, None)


def expert_cache_prepare_impl(
    topk_ids: torch.Tensor, layer_id: int
) -> torch.Tensor:
    try:
        runtime = _EXPERT_CACHE_RUNTIMES[layer_id]
    except KeyError:
        raise KeyError(
            f"no expert-cache runtime is wired for layer {layer_id}; "
            "expert weights were not intercepted into the host store"
        ) from None
    runtime.ensure_resident(topk_ids.reshape(-1).tolist())
    runtime.pool.wait_all()
    return remap_topk_ids(topk_ids, runtime.slot_map_device(topk_ids.device))


@register_custom_op(out_shape="topk_ids")
def expert_cache_prepare(topk_ids: torch.Tensor, layer_id: int) -> torch.Tensor:
    return expert_cache_prepare_impl(topk_ids, layer_id)
