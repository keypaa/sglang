from __future__ import annotations

from typing import Optional

import torch

from sglang.srt.layers.moe.expert_cache.types import HardwareSpec, ModelSpec


class StaticExpertPool:
    """Fixed-address VRAM store for resident MoE experts (spec §3.1).

    Pre-allocates twin ML-shaped pools and all CUDA events before any graph
    capture. Slot addresses never move; only the bytes change via copy_in on the
    transfer stream. The captured CUDA graph reads pool[slot_map[topk_ids]].

    Hard rule: NO allocation and NO Event() creation after __init__.
    """

    def __init__(
        self,
        model: ModelSpec,
        hw: HardwareSpec,
        backend: Optional["CudaTransferBackend"],  # noqa: F821
        num_slots: int,
        /,
        *,
        dtype: torch.dtype = torch.float8_e4m3fn,
        transfer_stream: Optional[torch.cuda.Stream] = None,
    ):
        assert torch.cuda.is_available(), "StaticExpertPool requires CUDA"
        self._model = model
        self._hw = hw
        self._num_slots = num_slots
        self._dtype = dtype
        w13_shape, w2_shape = model.expert_shapes()
        # Static twin pools: allocated exactly once, at fixed addresses.
        self._w13 = torch.empty((num_slots, *w13_shape), device="cuda", dtype=dtype)
        self._w2 = torch.empty((num_slots, *w2_shape), device="cuda", dtype=dtype)
        # Pre-created event: one reusable per-step barrier (decision-log #2:
        # a single step event, re-recorded each step — NOT per-slot events).
        self._step_event = torch.cuda.Event()
        # Transfer stream leased once; never created inside capture.
        self._stream = (
            transfer_stream if transfer_stream is not None else torch.cuda.Stream()
        )
        self._pending_copies = 0
        if backend is not None:
            backend.set_pool(self)

    @property
    def num_slots(self) -> int:
        return self._num_slots

    def pool_w13(self) -> torch.Tensor:
        return self._w13

    def pool_w2(self) -> torch.Tensor:
        return self._w2

    def copy_in(self, slot_id: int, w13: torch.Tensor, w2: torch.Tensor) -> None:
        """Async H2D of one expert on the transfer stream. No allocation."""
        assert 0 <= slot_id < self._num_slots
        # Hard rule: no allocation in copy_in. Callers must hand tensors already
        # in the pool dtype and shape (the pinned expert store provides them).
        assert w13.shape == self._w13[slot_id].shape and w13.dtype == self._dtype
        assert w2.shape == self._w2[slot_id].shape and w2.dtype == self._dtype
        with torch.cuda.stream(self._stream):
            self._w13[slot_id].copy_(w13, non_blocking=True)
            self._w2[slot_id].copy_(w2, non_blocking=True)
        self._pending_copies += 1

    def record_step(self) -> bool:
        """Record step_event after all demand copies of this step. Returns True
        if any copy was issued. Re-recorded each Phase A; the captured graph's
        baked wait waits on the latest record."""
        if self._pending_copies == 0:
            return False
        with torch.cuda.stream(self._stream):
            self._step_event.record(self._stream)
        self._pending_copies = 0
        return True

    def wait_all(self) -> None:
        """Block the host until all in-flight copies complete (tests/shutdown).

        Only the transfer stream needs syncing: every pool write goes through
        it, so once it is drained no later device work anywhere can race a
        pool read. The previous extra torch.cuda.synchronize() doubled the
        host stall for nothing (measured: dominates all-hit step latency).
        """
        self._stream.synchronize()

    def step_event(self) -> torch.cuda.Event:
        return self._step_event

    def reset(self) -> None:
        """Zero pools and clear pending state; call once before capture warmup."""
        self._w13.zero_()
        self._w2.zero_()
        self._pending_copies = 0
