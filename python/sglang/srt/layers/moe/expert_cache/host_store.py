"""Pinned host-RAM store for all MoE expert weights (spec §5).

Holds every layer's expert blocks so device-side pools can be streamed
arbitrarily. Pinned memory is mandatory: pageable H2D measured ~3x slower
(benchmark/moe_expert_cache_bench.py).
"""
from __future__ import annotations

from typing import Dict, NamedTuple

import torch

from .types import ExpertKey


class HostStoreEntry(NamedTuple):
    w13: torch.Tensor
    w2: torch.Tensor


class ExpertHostStore:
    def __init__(self, num_layers: int, num_experts: int):
        self._num_layers = num_layers
        self._num_experts = num_experts
        self._entries: Dict[ExpertKey, HostStoreEntry] = {}
        self._total_bytes = 0

    def put(self, key: ExpertKey, w13: torch.Tensor, w2: torch.Tensor) -> None:
        if key in self._entries:
            raise RuntimeError(f"duplicate expert weight put: {key}")
        if not key.valid() or key.layer >= self._num_layers or key.expert >= self._num_experts:
            raise KeyError(f"key out of range for store: {key}")
        entry = HostStoreEntry(w13=self._pin(w13), w2=self._pin(w2))
        self._entries[key] = entry
        self._total_bytes += entry.w13.numel() * entry.w13.element_size()
        self._total_bytes += entry.w2.numel() * entry.w2.element_size()

    @staticmethod
    def _pin(t: torch.Tensor) -> torch.Tensor:
        t = t.detach().contiguous().cpu()
        return t.pin_memory() if torch.cuda.is_available() else t

    def get(self, key: ExpertKey) -> HostStoreEntry:
        return self._entries[key]  # raises KeyError on missing

    def contains(self, key: ExpertKey) -> bool:
        return key in self._entries

    def expert_bytes(self, key: ExpertKey) -> int:
        e = self._entries[key]
        return (e.w13.numel() + e.w2.numel()) * e.w13.element_size()

    @property
    def total_bytes(self) -> int:
        return self._total_bytes
