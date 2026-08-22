"""Per-layer expert-cache runtime (Phase 3).

Streams experts from the pinned host store into arbitrary FusedMoE weight
tensors. The real FusedMoE weights are plain ``torch.Tensor`` params, not a
``StaticExpertPool``, so :class:`RealWeightPool` adapts them to the pool
surface the orchestrator/backend contract expects (the Phase-2 FakePool:
``copy_in`` / ``record_step`` / ``num_slots`` / ``wait_all``).
"""
from __future__ import annotations

from typing import Optional, Sequence

import torch

from sglang.srt.layers.moe.expert_cache.expert_cache import ExpertCache
from sglang.srt.layers.moe.expert_cache.host_store import ExpertHostStore
from sglang.srt.layers.moe.expert_cache.orchestrator import StaticPoolOrchestrator
from sglang.srt.layers.moe.expert_cache.types import CacheStats, ExpertKey


class RealWeightPool:
    """Adapts FusedMoE weight params to the pool surface copy_in/record_step."""

    def __init__(
        self,
        w13_weight: torch.Tensor,
        w2_weight: torch.Tensor,
        transfer_stream: Optional["torch.cuda.Stream"] = None,
    ):
        assert w13_weight.dim() == 3 and w2_weight.dim() == 3
        self.w13 = w13_weight
        self.w2 = w2_weight
        self.num_slots = w13_weight.shape[0]
        self._stream = transfer_stream or (
            torch.cuda.Stream() if torch.cuda.is_available() else None
        )
        self._pending = 0

    def copy_in(self, slot_id: int, w13: torch.Tensor, w2: torch.Tensor) -> None:
        assert 0 <= slot_id < self.num_slots
        assert self.w13[slot_id].shape == w13.shape and self.w13.dtype == w13.dtype
        assert self.w2[slot_id].shape == w2.shape and self.w2.dtype == w2.dtype
        if self._stream is not None:
            with torch.cuda.stream(self._stream):
                self.w13[slot_id].copy_(w13, non_blocking=True)
                self.w2[slot_id].copy_(w2, non_blocking=True)
        else:
            self.w13[slot_id].copy_(w13)
            self.w2[slot_id].copy_(w2)
        self._pending += 1

    def record_step(self) -> bool:
        if self._pending == 0:
            return False
        self._pending = 0
        return True

    def wait_all(self) -> None:
        if self._pending == 0:
            return
        if self._stream is not None:
            self._stream.synchronize()


class LayerRuntime:
    """Residency manager for ONE layer's FusedMoE weight tensors.

    Routes requested experts through the shared ExpertCache via a per-layer
    StaticPoolOrchestrator (num_layers keeps the slot map per-layer) and
    streams any non-resident expert's weights from the host store into the
    mapped slot row of the real weight params.
    """

    def __init__(
        self,
        layer_id: int,
        cache: ExpertCache,
        store: ExpertHostStore,
        w13_weight: torch.Tensor,
        w2_weight: torch.Tensor,
        num_experts: int,
        num_layers: int = 1,
    ):
        self.layer_id = layer_id
        self.pool = RealWeightPool(w13_weight, w2_weight)
        self.cache = cache          # shared across layers
        self.store = store
        self.orch = StaticPoolOrchestrator(cache, self.pool, num_experts, num_layers)
        self.device_map = None      # lazily created device mirror (CUDA)
        # A real discrete-GPU backend can copy straight into this pool during
        # the cache's own load path; then ensure_resident must not re-copy.
        self._backend_copies = False
        backend = getattr(cache, "_backend", None)
        if hasattr(backend, "set_pool") and hasattr(backend, "set_expert_source"):
            backend.set_pool(self.pool)
            backend.set_expert_source(store.get)
            self._backend_copies = True

    @property
    def stats(self) -> CacheStats:
        return self.cache.stats

    def ensure_resident(self, expert_ids: Sequence[int]) -> None:
        # Per-expert on_router_output -> step_commit rhythm (the documented
        # commit discipline): each copy lands before the next acquire, so a
        # mid-batch eviction can never leave a stale map pointing at a slot
        # whose contents belong to a different expert.
        for eid in dict.fromkeys(int(e) for e in expert_ids):
            key = ExpertKey(self.layer_id, eid)
            resident_before = self.cache.is_resident(key)
            self.orch.on_router_output(0, [eid], [0.9], next_ids=[])
            self.orch.step_commit()
            if not resident_before and not self._backend_copies:
                slot = self.orch.slot_id_of(0, eid)
                assert slot >= 0, f"no resident slot for expert {eid}"
                entry = self.store.get(key)
                self.pool.copy_in(slot, entry.w13, entry.w2)
        self.pool.wait_all()

    def slot_map_device(self, device) -> torch.Tensor:
        if self.device_map is None:
            self.device_map = torch.full((self.orch._num_experts,), -1,
                                         dtype=torch.int32, device=device)
        self.orch.copy_slot_map_into(self.device_map)
        return self.device_map
