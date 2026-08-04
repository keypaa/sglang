from __future__ import annotations

from typing import List, Optional, Set

from .backends import TransferBackend
from .freq_sketch import FreqSketch
from .policies import EvictionPolicy, KeyNode
from .types import CacheStats, ExpertKey, HardwareSpec, ModelSpec, RouterChoice, SlotState

# ---------------------------------------------------------------------------
# ExpertCache - the L1 GPU-fast memory cache.
#
# Owns a fixed array of VRAM slots (fixed-address, allocated once), a hash
# index keyed by (layer, expert), an eviction policy, and a transfer backend.
# Exposes the demand-fetch and prefetch lifecycles described in DESIGN.md. The
# simulator drives it in virtual time; on real hardware the same API is called
# from the engine's scheduler with the CUDA backend.
#
# This is a faithful Python port of include/moe/expert_cache.hpp.
# ---------------------------------------------------------------------------


class Slot:
    __slots__ = (
        "node",
        "state",
        "refcount",
        "addr",
        "ready_tick",
        "load_start",
        "event",
    )

    def __init__(self, index: int):
        self.node = KeyNode(index)  # policy tracks by node identity
        self.state = SlotState.EMPTY
        self.refcount = 0
        self.addr = None  # fixed device address / GPU tensor (real backend)
        self.ready_tick = 0.0  # virtual time the weights are resident
        self.load_start = 0.0
        self.event = None  # CUDA event set by the CUDA backend


class ExpertCache:
    kMinPrefetchFreq = 2.0  # demonstrated reuse
    kHighConfidenceLogit = 0.6

    def __init__(
        self,
        model: ModelSpec,
        hw: HardwareSpec,
        policy: EvictionPolicy,
        backend: TransferBackend,
        num_scratch_slots: int = 4,
    ):
        self._model = model
        self._hw = hw
        self._policy = policy
        self._backend = backend

        capacity = policy.capacity()
        self._slots: List[Slot] = [Slot(i) for i in range(capacity)]
        self._scratch: List[Slot] = [
            Slot(-1 - i) for i in range(num_scratch_slots)
        ]
        self._scratch_next = 0
        self._free: List[Slot] = list(self._slots)

        self._index: dict[ExpertKey, Slot] = {}
        self._shared_sketch = FreqSketch()

        self._stats = CacheStats()
        self._transfer_busy_until = 0.0
        self._now = 0.0

    # ---- demand fetch ------------------------------------------------------
    # The scheduler calls this for every expert the router selected. Returns
    # the slot whose weights must be resident before compute starts; the caller
    # waits until slot.ready_tick (sim) or slot.event (CUDA) before launching
    # the FFN kernel.
    def acquire(self, choice: RouterChoice) -> Slot:
        k = choice.key
        self._shared_sketch.record(k)
        s = self._index.get(k)
        if s is not None:
            if s.state == SlotState.READY:
                self._policy.on_access(s.node, choice.logit)
                s.refcount += 1
                self._stats.hits += 1
                return s
            # LOADING (prefetch already issued): attach as a waiter.
            s.refcount += 1
            self._stats.hits += 1  # counted as a soft hit: no new fabric work
            self._stats.prefetch_used += 1
            return s

        self._stats.misses += 1
        if self._policy.should_admit(k, choice.logit):
            s = self._load_into_cache(k, choice.logit, False)
            if s is not None:
                s.refcount += 1
                return s
            # No evictable slot (all candidates busy this layer): serve
            # transiently - weights load, but nothing resident is displaced.
            self._stats.admission_rejected += 1
            return self._load_transient(k)
        # Admission rejected: serve transiently, never polluting the cache.
        self._stats.admission_rejected += 1
        return self._load_transient(k)

    # ---- release -----------------------------------------------------------
    def release(self, s: Slot) -> None:
        if s.refcount > 0:
            s.refcount -= 1

    # ---- prefetch ----------------------------------------------------------
    # Router output for layer L+1 arrives while layer L computes. Rank the
    # candidates by predicted reuse, respect budget + admission, and issue
    # asynchronous loads that overlap the compute of layer L.
    def prefetch(self, choices: List[RouterChoice], budget: int) -> None:
        # dedup by key + drop already-resident (must re-check per candidate:
        # earlier candidates in this same batch may load the same key).
        cand: List[RouterChoice] = []
        for c in choices:
            if c.key in self._index:
                continue  # resident
            if any(d.key == c.key for d in cand):
                continue
            cand.append(c)
        cand.sort(key=lambda c: self._prefetch_rank(c), reverse=True)

        used = 0
        for c in cand:
            if used >= budget:
                break
            if c.key in self._index:
                continue  # became resident
            self._shared_sketch.record(c.key)
            # Reuse gate (DESIGN section 4.2): only prefetch experts that are
            # likely to be re-requested. A fresh one-shot (sketch<2) with weak
            # router confidence would waste PCIe bandwidth and cache slots.
            reuse = self._shared_sketch.estimate(c.key)
            if reuse < self.kMinPrefetchFreq and c.logit < self.kHighConfidenceLogit:
                continue
            if not self._policy.should_admit(c.key, c.logit):
                continue
            s = self._load_into_cache(c.key, c.logit, True)
            if s is None:
                continue
            self._stats.prefetches_issued += 1
            used += 1

    # ---- misc ---------------------------------------------------------------
    def is_resident(self, k: ExpertKey) -> bool:
        s = self._index.get(k)
        return s is not None and s.state == SlotState.READY

    def prefetch_budget(self, beta: float = 0.5) -> int:
        free = len(self._free)
        hit = self._stats.hit_rate
        dyn = int(beta * (1.0 - hit) * self._policy.capacity())
        return free + dyn

    def on_phase_change(self) -> None:
        self._policy.on_phase_change()

    # Advance the cache's virtual clock (used by the simulator so loads issued
    # later start at the correct time).
    def advance_time(self, t: float) -> None:
        self._now = max(self._now, t)

    @property
    def stats(self) -> CacheStats:
        return self._stats

    @property
    def model(self) -> ModelSpec:
        return self._model

    @property
    def hw(self) -> HardwareSpec:
        return self._hw

    # Virtual time at which the transfer stream becomes free (serialization).
    def transfer_stream_free(self) -> float:
        return self._transfer_busy_until

    # Bytes actually moved across the fabric (from the backend's accounting).
    def bytes_moved(self) -> int:
        return self._backend.total_bytes_moved()

    # Predicted near-term reuse of a key (shared sketch, policy-independent).
    def predicted_reuse(self, k: ExpertKey) -> int:
        return self._shared_sketch.estimate(k)

    # Block (CUDA) until the slot's async load completes.
    def wait_ready(self, s: Slot) -> None:
        self._backend.wait_ready(s)

    # ---- private -------------------------------------------------------------

    def _load_into_cache(self, k: ExpertKey, logit: float, is_prefetch: bool) -> Optional[Slot]:
        # Defensive: never double-load a key that is already resident.
        already = self._index.get(k)
        if already is not None and already.state != SlotState.EMPTY:
            return already

        # Only evict when the policy is at capacity; otherwise a free slot is
        # cheaper (no reuse of a live entry). Collect in-flight (busy) slots so
        # the policy never offers one for eviction.
        busy: Set[KeyNode] = set()
        for s in self._slots:
            if s.refcount > 0 or s.state == SlotState.LOADING:
                busy.add(s.node)
        victim = (
            self._policy.victim(busy)
            if self._policy.size() >= self._policy.capacity()
            else None
        )
        slot: Optional[Slot] = None

        if victim is not None:
            vs = self._index.get(victim.key)
            if vs is not None:
                # Never evict an in-flight load or an in-use compute slot.
                if vs.state == SlotState.READY and vs.refcount == 0:
                    slot = vs
                    del self._index[victim.key]
                    # Tell the policy the old occupant is gone (node.key still
                    # holds the old key here).
                    self._policy.on_evict(vs.node)
                    self._backend.evict(vs)
                    self._stats.evictions += 1
                else:
                    victim = None  # busy: fall through to free slot / transient
        if slot is None:
            if not self._free:
                return None  # no capacity: caller falls back
            slot = self._free.pop()

        slot.node.key = k
        slot.node.freq = 0
        slot.node.flags = 0
        slot.refcount = 0
        slot.state = SlotState.LOADING

        # Transfer stream: serialized, overlapped with compute by the simulator.
        start = max(self._now, self._transfer_busy_until)
        duration = self._backend.load(k, self._model.expert_bytes(), slot, None)
        end = start + duration
        self._transfer_busy_until = end
        slot.load_start = start
        slot.ready_tick = end
        slot.state = SlotState.READY  # transfer complete; slot is resident
        self._now = max(self._now, end)

        self._index[k] = slot

        if is_prefetch:
            self._policy.on_prefetch(slot.node, logit)
        else:
            self._policy.insert(slot.node)
        self._stats.loads += 1
        return slot

    def _load_transient(self, k: ExpertKey) -> Slot:
        s = self._scratch[self._scratch_next % len(self._scratch)]
        self._scratch_next += 1
        s.node.key = k
        s.state = SlotState.TRANSIENT
        start = max(self._now, self._transfer_busy_until)
        duration = self._backend.load(k, self._model.expert_bytes(), s, None)
        self._transfer_busy_until = start + duration
        s.load_start = start
        s.ready_tick = self._transfer_busy_until
        self._now = max(self._now, self._transfer_busy_until)
        s.refcount += 1
        self._stats.transient_served += 1
        return s

    def _prefetch_rank(self, c: RouterChoice) -> float:
        reuse = self._shared_sketch.estimate(c.key)
        return reuse + 2.0 * c.logit
