from __future__ import annotations

from typing import List, Sequence

import torch

from sglang.srt.layers.moe.expert_cache.expert_cache import ExpertCache, Slot
from sglang.srt.layers.moe.expert_cache.types import ExpertKey, RouterChoice


class StaticPoolOrchestrator:
    """Host-side Phase-A adapter (spec §3.3).

    Turns router output into cache acquires + pool copies and exposes the
    static slot map the captured graph reads via gather. All host-side; safe
    to call every decode step between graph replays.

    Refcount discipline: every expert acquired this step is released on
    step_commit (mirrors simulator.py:212's per-token acquire/release), so a
    busy slot never leaks and the cache never deadlocks into the all-busy
    acquire_resident RuntimeError.
    """

    def __init__(
        self,
        cache: ExpertCache,
        pool,
        num_experts: int,
        num_layers: int,
    ):
        self._cache = cache
        self._pool = pool
        self._num_experts = num_experts
        self._num_layers = num_layers
        self._slot_map = torch.full((num_experts,), -1, dtype=torch.int32)
        self._held: List[Slot] = []

    def on_router_output(
        self,
        layer: int,
        expert_ids: Sequence[int],
        logit: Sequence[float],
        next_ids: Sequence[int],
    ) -> None:
        if layer >= self._num_layers or any(
            e >= self._num_experts for e in expert_ids
        ) or any(e >= self._num_experts for e in next_ids):
            raise IndexError("expert id out of range")

        self._held = []
        for eid, conf in zip(expert_ids, logit):
            choice = RouterChoice(ExpertKey(layer, eid), conf)
            # Normal path honors admission (matches the simulator's hit rate);
            # only fall back to the demand-guarantee when admission served a
            # transient (-1) slot that a committed step cannot tolerate.
            slot = self._cache.acquire(choice)
            if slot.node.index < 0:
                self._cache.release(slot)
                slot = self._cache.acquire_resident(choice)
            self._slot_map[eid] = slot.node.index
            self._held.append(slot)

        if next_ids:
            # Prefetch next layer's candidates; at the last layer, wrap around
            # to the next token's layer-0 experts (matches simulator.py:198-206).
            nxt_layer = layer + 1 if layer + 1 < self._num_layers else 0
            self._cache.prefetch(
                [RouterChoice(ExpertKey(nxt_layer, int(e)), 0.5) for e in next_ids],
                budget=self._cache.prefetch_budget(),
            )

    def step_commit(self) -> None:
        self._pool.record_step()
        for s in self._held:
            self._cache.release(s)
        self._held = []

    def slot_map_tensor(self) -> torch.Tensor:
        return self._slot_map

    def slot_id_of(self, layer: int, expert_id: int) -> int:
        return int(self._slot_map[expert_id])
