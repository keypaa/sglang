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

    Refcount discipline: every expert acquired by on_router_output is released
    on step_commit. Commit rhythm is PER LAYER: call on_router_output(layer)
    then step_commit() once per layer, after that layer's on_router_output and
    before replaying that layer's graph (mirrors simulator.py:186-211's
    per-layer acquire/release). on_router_output replaces the held batch, so a
    second call without an intervening step_commit pins the previous batch's
    refcounts. With this rhythm a busy slot never leaks and the cache never
    deadlocks into the all-busy acquire_resident RuntimeError.
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
        # The cache's resident set and the pool's slot count must match: the
        # backend writes slot.node.index (0..capacity-1) into pool.copy_in,
        # which asserts 0 <= slot_id < num_slots. A mismatch silently diverges.
        assert cache.capacity == pool.num_slots, (
            f"cache capacity {cache.capacity} != pool.num_slots {pool.num_slots}"
        )
        # Keyed by (layer, expert): each layer owns its own expert id space, so
        # two layers routing the same id must not clobber one slot-map entry.
        # Pinned when CUDA is available so copy_slot_map_into can H2D the map
        # with non_blocking=True — no per-step staging allocation (the .to()
        # refill pattern measured as the dominant all-hit host cost).
        # device="cpu" is EXPLICIT: an ambient torch.device("cuda") context
        # would otherwise inject device=cuda into this constructor and pinning
        # a CUDA tensor raises ("Only dense CPU tensors can be pinned").
        self._slot_map = torch.full(
            (num_layers, num_experts),
            -1,
            dtype=torch.int32,
            device="cpu",
            pin_memory=torch.cuda.is_available(),
        )
        # The layer whose map slot_map_tensor() exposes (the graph reads one
        # layer's static map per Phase-A instance).
        self._active_layer = 0
        self._held: List[Slot] = []

    def on_router_output(
        self,
        layer: int,
        expert_ids: Sequence[int],
        logit: Sequence[float],
        next_ids: Sequence[int],
    ) -> None:
        if layer >= self._num_layers or any(
            e < 0 or e >= self._num_experts for e in expert_ids
        ) or any(e < 0 or e >= self._num_experts for e in next_ids):
            # Negative ids must be rejected too: -1 is the slot map's unmapped
            # sentinel, and _slot_map[layer, -1] would silently wrap columns.
            raise IndexError("expert id out of range")

        # Defensive: release a stale batch instead of silently leaking its
        # refcounts (the documented rhythm is on_router_output -> step_commit
        # per layer; a second call without the commit must not wedge slots).
        for s in self._held:
            self._cache.release(s)
        self._held = []
        self._active_layer = layer
        for eid, conf in zip(expert_ids, logit):
            choice = RouterChoice(ExpertKey(layer, eid), conf)
            # Normal path honors admission (matches the simulator's hit rate);
            # only fall back to the demand-guarantee when admission served a
            # transient (-1) slot that a committed step cannot tolerate.
            slot = self._cache.acquire(choice)
            if slot.node.index < 0:
                self._cache.release(slot)
                slot = self._cache.acquire_resident(choice)
            self._slot_map[layer, eid] = slot.node.index
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
        # Flat view of the active layer's expert->slot map. The captured graph
        # gathers by expert id on this single layer's map (one map per
        # Phase-A graph instance).
        return self._slot_map[self._active_layer]

    def copy_slot_map_into(self, dst: torch.Tensor) -> None:
        """H2D the active layer's map into a device tensor, no allocation.

        The map is pinned (when CUDA is available), so this is a single
        non_blocking device copy — replaces the per-step
        `slot_map_tensor().to(dtype, device)` staging alloc.
        """
        dst.copy_(self._slot_map[self._active_layer], non_blocking=True)

    def slot_id_of(self, layer: int, expert_id: int) -> int:
        if layer < 0 or layer >= self._num_layers:
            raise IndexError("layer out of range")
        return int(self._slot_map[layer, expert_id])
