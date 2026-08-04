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
        self._expert_source: Optional[Callable[[ExpertKey], torch.Tensor]] = None
        self._moved = 0

    @property
    def name(self) -> str:
        return "cuda"

    # The pinned host expert store. `source(key)` must return a pinned CPU
    # tensor holding that expert's weights.
    def set_expert_source(self, source: Callable[[ExpertKey], torch.Tensor]) -> None:
        self._expert_source = source

    def load(
        self,
        key: ExpertKey,
        nbytes: int,
        slot: Slot,
        on_done: Optional[Callable[[], None]] = None,
    ) -> float:
        assert self._expert_source is not None, "set_expert_source() first"
        torch = self._torch
        src = self._expert_source(key)
        dst = slot.addr
        if dst is None or dst.shape != src.shape or dst.dtype != src.dtype:
            dst = torch.empty(src.shape, device=self._device, dtype=src.dtype)
            slot.addr = dst
        stream = self._stream if self._stream is not None else torch.cuda.current_stream()
        with torch.cuda.stream(stream):
            dst.copy_(src, non_blocking=True)
        slot.event = torch.cuda.Event()
        slot.event.record(stream)
        self._moved += nbytes
        if on_done is not None:
            on_done()
        # Estimated fabric time for the virtual-clock bookkeeping.
        return nbytes / self._bw * 1000.0

    # Block the current stream until the slot's async copy completes.
    def wait_ready(self, slot: Slot) -> None:
        if slot.event is not None:
            self._torch.cuda.current_stream().wait_event(slot.event)

    def evict(self, slot: Slot) -> None:
        slot.event = None

    def effective_bw(self) -> float:
        return self._bw

    def reset(self) -> None:
        self._moved = 0

    def total_bytes_moved(self) -> int:
        return self._moved
