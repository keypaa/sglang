from __future__ import annotations

from dataclasses import dataclass
from enum import IntEnum
from typing import NamedTuple


class ExpertKey(NamedTuple):
    """A cache key identifying one expert weight block: (layer, expert index).

    Mirrors ``moe::ExpertKey`` in the C++ simulator (include/moe/types.hpp).
    """

    layer: int = -1
    expert: int = -1

    def valid(self) -> bool:
        return self.layer >= 0 and self.expert >= 0

    def hash(self) -> int:
        # layer in high bits, expert in low bits -> unique for the model.
        return (self.layer << 32) | (self.expert & 0xFFFFFFFF)

    def id(self, experts_per_layer: int) -> int:
        return self.layer * experts_per_layer + self.expert

    @staticmethod
    def from_id(expert_id: int, experts_per_layer: int) -> ExpertKey:
        return ExpertKey(expert_id // experts_per_layer, expert_id % experts_per_layer)


@dataclass
class RouterChoice:
    """A router decision for one expert: which expert, and how confident the
    router was (softmax probability / logit). This is the signal the prefetcher
    uses. Mirrors ``moe::RouterChoice``."""

    key: ExpertKey
    logit: float = 0.0  # [0,1] after softmax, or raw logit normalized


class SlotState(IntEnum):
    EMPTY = 0
    LOADING = 1
    READY = 2
    EVICTING = 3
    TRANSIENT = 4


@dataclass
class CacheStats:
    hits: int = 0
    misses: int = 0
    transient_served: int = 0
    prefetches_issued: int = 0
    prefetch_used: int = 0  # prefetched entries later hit
    evictions: int = 0
    loads: int = 0
    bytes_moved: int = 0
    admission_rejected: int = 0

    @property
    def hit_rate(self) -> float:
        total = self.hits + self.misses
        return self.hits / total if total else 0.0


@dataclass
class ModelSpec:
    """DeepSeek-V4-Flash-0731 model parameters. All values are runtime-tunable;
    nothing below is hard-coded into the cache logic. Mirrors ``moe::ModelSpec``."""

    num_layers: int = 43  # layers
    num_experts: int = 256  # routed experts per layer
    top_k: int = 6  # routed experts selected per token
    shared_experts: int = 1  # always-active experts per layer
    total_params: int = 284_000_000_000
    active_params: int = 13_000_000_000
    bytes_on_disk: int = 167_000_000_000

    def expert_bytes(self) -> int:
        """One expert weight block size (uniform within a layer). ~15.2 MB."""
        return self.bytes_on_disk // (self.num_layers * self.num_experts)

    def total_expert_keys(self) -> int:
        """Total cache keys across the whole model (routed + shared experts)."""
        return self.num_layers * (self.num_experts + self.shared_experts)

    def active_bytes_per_layer(self) -> int:
        """Bytes activated per token per layer: top_k routed + shared."""
        return self.expert_bytes() * (self.top_k + self.shared_experts)


@dataclass
class HardwareSpec:
    """Hardware / bandwidth parameters. Bandwidth units are bytes/second.
    Mirrors ``moe::HardwareSpec``."""

    vram_bw: float = 936e9  # GB/s : RTX 3090 GDDR6X
    h2d_bw: float = 25e9  # GB/s : PCIe 4.0 x16 usable host->device
    host_bw: float = 80e9  # GB/s : DDR5-6000 dual channel
    nvme_bw: float = 7e9  # GB/s : PCIe 4.0 NVMe cold store
    vram_bytes: int = 24 * 1024 * 1024 * 1024  # per-device budget
    host_bytes: int = 192 * 1024 * 1024 * 1024  # system RAM
    nvme_bytes: int = 0  # 0 => model fits in RAM
    unified_memory: bool = False  # DGX Spark / Apple silicon style tiering

    def cache_bytes(self) -> int:
        """What the cache (L1) may occupy: after dense weights + KV + scratch.
        Default: 45% of VRAM for the expert cache on discrete cards."""
        budget = int(self.vram_bytes * (0.30 if self.unified_memory else 0.45))
        return budget - (budget % 4096)
