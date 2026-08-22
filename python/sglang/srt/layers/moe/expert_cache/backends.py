from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Callable, Optional

if TYPE_CHECKING:
    import torch

    from .expert_cache import Slot
    from .types import ExpertKey

# ---------------------------------------------------------------------------
# TransferBackend abstracts the memory fabric between the host expert store and
# the GPU-fast tier. The cache layer is identical for every backend; only the
# mechanism for "make this expert's weights available in GPU-fast memory"
# differs (mirrors include/moe/backend.hpp).
#
#   Discrete  : pinned host RAM -> VRAM slot via cudaMemcpyAsync.
#   Sim       : CPU timing model used by the offline simulator. Cost is a
#               function of bytes and the configured H2D bandwidth; it reports
#               the time the transfer would take on the target hardware.
#
# The unit of transfer is one expert block (expert_bytes from ModelSpec).
# ---------------------------------------------------------------------------


class TransferBackend(ABC):
    @property
    @abstractmethod
    def name(self) -> str:
        ...

    # Ask the backend to make `nbytes` of weights for `key` available in
    # GPU-fast memory at the fixed slot address. Returns the *elapsed time* the
    # transfer occupies the fabric (0 for unified-residency, >0 for sim).
    # `on_done` is invoked (async semantics) once the weights are resident.
    @abstractmethod
    def load(
        self,
        key: ExpertKey,
        nbytes: int,
        slot: Slot,
        on_done: Optional[Callable[[], None]] = None,
    ) -> float:
        ...

    # Invalidate/forget a slot's residency.
    @abstractmethod
    def evict(self, slot: Slot) -> None:
        ...

    # Bandwidth (bytes/s) used for cost modeling.
    @abstractmethod
    def effective_bw(self) -> float:
        ...

    @abstractmethod
    def reset(self) -> None:
        ...

    @abstractmethod
    def total_bytes_moved(self) -> int:
        ...


# ---------------------------------------------------------------------------
# CPU/sim backend: models transfer cost but does not touch hardware.
# ---------------------------------------------------------------------------


class SimBackend(TransferBackend):
    def __init__(self, h2d_bw: float):
        self._bw = h2d_bw
        self._moved = 0

    @property
    def name(self) -> str:
        return "sim"

    def load(
        self,
        key: ExpertKey,
        nbytes: int,
        slot: Slot,
        on_done: Optional[Callable[[], None]] = None,
    ) -> float:
        self._moved += nbytes
        # Simulator clocks are in milliseconds.
        t = nbytes / self._bw * 1000.0
        if on_done is not None:
            on_done()
        return t

    def evict(self, slot: Slot) -> None:
        pass

    def effective_bw(self) -> float:
        return self._bw

    def reset(self) -> None:
        self._moved = 0

    def total_bytes_moved(self) -> int:
        return self._moved


# ---------------------------------------------------------------------------
# Real discrete-GPU backend: pinned host RAM -> VRAM slot via async copies on a
# dedicated transfer stream. Guards the torch/CUDA imports so the rest of the
# package (and its CPU-only unit tests) never depend on a CUDA runtime.
# ---------------------------------------------------------------------------


class CudaTransferBackend(TransferBackend):
    """Real discrete-GPU backend: pinned host RAM -> STATIC pool slot."""

    @property
    def name(self) -> str:
        return "cuda"

    def __init__(
        self,
        h2d_bw: float,
        device: Optional[torch.device] = None,
        transfer_stream: Optional[torch.cuda.Stream] = None,
    ):
        import torch

        self._torch = torch
        self._bw = h2d_bw
        self._device = device if device is not None else torch.device("cuda")
        self._stream = transfer_stream
        self._expert_source = None
        self._pool = None
        self._moved = 0

    def set_pool(self, pool) -> None:
        self._pool = pool

    def set_expert_source(self, source) -> None:
        """source(key) -> (w13_cpu_pinned, w2_cpu_pinned)."""
        self._expert_source = source

    def load(self, key, nbytes, slot, on_done=None) -> float:
        assert self._expert_source is not None, "set_expert_source() first"
        assert self._pool is not None, "set_pool() first"
        if slot.node.index < 0:
            # Transient/scratch slot: nothing to copy into the static pool. The
            # orchestrator upgrades transients to real slots (acquire_resident)
            # before committing a step, so scratch loads are never read by the
            # graph. Return zero time so the simulator clock does not stall.
            if on_done is not None:
                on_done()
            return 0.0
        src = self._expert_source(key)
        # Expert sources may be plain (w13, w2) tuples (Phase-2 harnesses) or
        # HostStoreEntry namedtuples carrying optional fp8 scale blocks.
        if hasattr(src, "w13"):
            w13, w2 = src.w13, src.w2
            w13_scale, w2_scale = src.w13_scale_inv, src.w2_scale_inv
        else:
            w13, w2 = src
            w13_scale = w2_scale = None
        if w13_scale is not None or w2_scale is not None:
            # Scale blocks only make sense for pools that accept them
            # (RealWeightPool); the Phase-2 StaticExpertPool has none.
            raise NotImplementedError(
                "expert source carries scale blocks but this pool does not "
                "accept them"
            )
        # slot.node.index == pool slot id (0..capacity-1); no new field.
        self._pool.copy_in(slot.node.index, w13, w2)
        self._moved += nbytes
        if on_done is not None:
            on_done()
        return nbytes / self._bw * 1000.0

    def wait_ready(self, slot) -> None:
        # Async copy is on the transfer stream; the graph's baked step-event
        # wait covers reads. For eager callers, block on the pool.
        if self._pool is not None:
            self._pool.wait_all()

    def evict(self, slot) -> None:
        pass  # pool owns storage; addresses are stable.

    def effective_bw(self) -> float:
        return self._bw

    def reset(self) -> None:
        self._moved = 0

    def total_bytes_moved(self) -> int:
        return self._moved
