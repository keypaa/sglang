from __future__ import annotations

from abc import ABC, abstractmethod
from collections import OrderedDict
from typing import List, Optional, Set, Tuple

from .freq_sketch import FreqSketch
from .types import ExpertKey


class KeyNode:
    """The per-slot bookkeeping node tracked by eviction policies.

    In the C++ simulator the cache owns a fixed array of ``KeyNode`` objects
    (one per VRAM slot) and the policy tracks them by pointer. Here a node is
    owned by a ``Slot`` and the policy tracks it by object identity. ``index``
    is the owning slot's index, used as a deterministic tie-breaker so heap
    orderings are stable across runs (mirrors pointer comparison in C++).
    """

    __slots__ = ("key", "freq", "flags", "index")

    def __init__(self, index: int = -1):
        self.key: Optional[ExpertKey] = None
        self.freq: int = 0  # per-entry saturating frequency (LogitGDS)
        self.flags: int = 0  # policy-specific
        self.index: int = index


# ---------------------------------------------------------------------------
# EvictionPolicy interface (mirrors include/moe/policy.hpp).
# ---------------------------------------------------------------------------


class EvictionPolicy(ABC):
    @property
    @abstractmethod
    def name(self) -> str:
        ...

    @abstractmethod
    def on_access(self, node: KeyNode, logit: float) -> None:
        ...

    @abstractmethod
    def on_prefetch(self, node: KeyNode, logit: float) -> None:
        ...

    @abstractmethod
    def on_evict(self, node: KeyNode) -> None:
        ...

    # The cache assigned a slot to a new resident key (node->key already set).
    @abstractmethod
    def insert(self, node: KeyNode) -> None:
        ...

    @abstractmethod
    def should_admit(self, key: ExpertKey, logit: float) -> bool:
        ...

    # Next resident node to evict, or None if none is eligible. Nodes in
    # `unavailable` are in-flight (refcount > 0) and must be skipped.
    @abstractmethod
    def victim(self, unavailable: Set[KeyNode]) -> Optional[KeyNode]:
        ...

    # Global signal: a long cold-access stretch (prefill) or session switch.
    @abstractmethod
    def on_phase_change(self) -> None:
        ...

    @abstractmethod
    def clear(self) -> None:
        ...

    @abstractmethod
    def size(self) -> int:
        ...

    @abstractmethod
    def capacity(self) -> int:
        ...


# ---------------------------------------------------------------------------
# Plain LRU. Baseline. Known to fail under prefill: a long prompt touches
# every expert once and wipes the hot set before decode even starts.
# Mirrors include/moe/policy_lru.hpp.
# ---------------------------------------------------------------------------


class LRUPolicy(EvictionPolicy):
    def __init__(self, capacity: int):
        self._capacity = capacity
        self._index: dict[ExpertKey, KeyNode] = {}
        # OrderedDict preserves insertion order: oldest (LRU) first, MRU last.
        self._order: OrderedDict[KeyNode, None] = OrderedDict()

    @property
    def name(self) -> str:
        return "LRU"

    def on_access(self, node: KeyNode, logit: float) -> None:
        self._order.move_to_end(node, last=True)

    # Prefetched entries are tracked but placed at the LRU end: they never
    # displace real recent usage until actually accessed.
    def on_prefetch(self, node: KeyNode, logit: float) -> None:
        if node in self._order:
            return
        self._index[node.key] = node
        self._order[node] = None  # new entries land at the LRU (oldest) end

    def on_evict(self, node: KeyNode) -> None:
        self._order.pop(node, None)
        self._index.pop(node.key, None)

    def should_admit(self, key: ExpertKey, logit: float) -> bool:
        return True

    def victim(self, unavailable: Set[KeyNode]) -> Optional[KeyNode]:
        # Walk from the LRU end (oldest) skipping in-flight nodes.
        for node in self._order.keys():
            if node not in unavailable:
                return node
        return None

    def on_phase_change(self) -> None:
        pass

    def clear(self) -> None:
        self._index.clear()
        self._order.clear()

    def size(self) -> int:
        return len(self._order)

    def capacity(self) -> int:
        return self._capacity

    def insert(self, node: KeyNode) -> None:
        if node in self._order:
            return
        self._index[node.key] = node
        self._order[node] = None
        self._order.move_to_end(node, last=True)  # MRU end


# ---------------------------------------------------------------------------
# W-TinyLFU (the Caffeine/Guava cache design). Mirrors
# include/moe/policy_tinylfu.hpp.
#
#   window  : tiny LRU (default 1% of capacity, min 8) that absorbs bursts and
#             short-term recency.
#   main    : larger LRU holding the long-lived, frequently-used set.
#   sketch  : 4-bit Count-Min frequency estimate over ALL observed accesses.
#
# The crux is the *admission filter*. A candidate is admitted only if
# sketch[candidate] >= sketch[main_tail]. A one-shot prefill expert has
# frequency ~1 and loses against any hot expert - so prefill garbage can never
# displace the hot set. Frequency is aged by periodic decay, so an old
# session's hot set doesn't pin the cache forever.
# ---------------------------------------------------------------------------


class TinyLFUPolicy(EvictionPolicy):
    kInWindow = 1
    kInMain = 2

    def __init__(self, capacity: int, window_fraction: float = 0.01):
        self._capacity = capacity
        self._window_cap = max(8, int(capacity * window_fraction))
        self._sketch = FreqSketch()
        self._window: OrderedDict[KeyNode, None] = OrderedDict()
        self._main: OrderedDict[KeyNode, None] = OrderedDict()

    @property
    def name(self) -> str:
        return "TinyLFU"

    def on_access(self, node: KeyNode, logit: float) -> None:
        self._sketch.record(node.key)
        if node.flags & self.kInMain:
            self._main.move_to_end(node, last=True)
        elif node.flags & self.kInWindow:
            self._window.move_to_end(node, last=True)

    def on_prefetch(self, node: KeyNode, logit: float) -> None:
        self._sketch.record(node.key)
        # prefetched entries land in the window (probationary) so they age out
        # quickly if never used.
        if node.flags & (self.kInWindow | self.kInMain):
            return
        node.flags |= self.kInWindow
        self._window[node] = None
        self._filter_window()

    def on_evict(self, node: KeyNode) -> None:
        self._window.pop(node, None)
        self._main.pop(node, None)
        node.flags = 0

    def should_admit(self, key: ExpertKey, logit: float) -> bool:
        self._sketch.record(key)
        total = len(self._window) + len(self._main)
        if total < self._capacity:
            return True  # room to grow: admit
        if not self._main:
            return True
        # Full: only admit if the candidate is at least as frequent as the
        # current main tail.
        main_tail = next(reversed(self._main))
        return self._sketch.estimate(key) >= self._sketch.estimate(main_tail.key)

    def victim(self, unavailable: Set[KeyNode]) -> Optional[KeyNode]:
        # Evict from main tail (oldest) first; if main is empty, the window.
        for node in self._main.keys():
            if node not in unavailable:
                return node
        for node in self._window.keys():
            if node not in unavailable:
                return node
        return None

    def on_phase_change(self) -> None:
        self._sketch.decay(1)

    def clear(self) -> None:
        self._sketch.reset()
        self._window.clear()
        self._main.clear()

    def size(self) -> int:
        return len(self._window) + len(self._main)

    def capacity(self) -> int:
        return self._capacity

    def insert(self, node: KeyNode) -> None:
        if node.flags & (self.kInWindow | self.kInMain):
            return
        node.flags |= self.kInWindow
        self._window[node] = None
        self._filter_window()

    # Move window-tail into main when the window exceeds its cap. Admission
    # (candidate freq >= victim freq) was ALREADY applied by should_admit
    # before the cache assigned a slot, so this promotion is unconditional.
    def _filter_window(self) -> None:
        while len(self._window) > self._window_cap:
            cand, _ = self._window.popitem(last=False)
            cand.flags = self.kInMain
            self._main[cand] = None


# ---------------------------------------------------------------------------
# LogitGDS - router-confidence x Greedy-Dual aging. This is the recommended
# production policy because it exploits the ONE signal plain caches lack: the
# router's logits, which tell us how likely an expert is to be needed again
# right now. Mirrors include/moe/policy_logit_gds.hpp.
#
#   score(n) = w_f * freq(n) + w_l * logit_ema(n)
#   H(n)     = score(n) + L            (Greedy-Dual: global aging value L)
#
# * freq(n) is the per-entry saturating counter, also mirrored in the shared
#   Count-Min sketch so that admission can compare candidates vs victims.
# * logit_ema(n) is the exponentially-decayed router confidence.
# * Victim = node with minimal H (lazy-deleted min-heap).
# * Admission: admit if the candidate's score beats the current victim's, or if
#   the router confidence alone exceeds a threshold.
#
# L monotonically increases on eviction, so long-resident low-value entries are
# progressively cheapened - this is what prevents LFU-style staleness.
# ---------------------------------------------------------------------------


class LogitGDSPolicy(EvictionPolicy):
    def __init__(self, capacity: int, logit_admit_threshold: float = 0.35):
        self._capacity = capacity
        self._admit_threshold = logit_admit_threshold

        self._sketch = FreqSketch()
        self._index: dict[ExpertKey, KeyNode] = {}
        self._logit_ema: dict[KeyNode, float] = {}
        # Indexed min-heap keyed on INTRINSIC value. GDS's aging value L is
        # global, so min(value + L) == min(value): the heap order never goes
        # stale when L advances, and there is exactly one live entry per
        # resident node. Entries are (value, node) tuples: the value is the
        # stored H at push time, but the heap orders by intrinsic value.
        self._heap: List[Tuple[float, KeyNode]] = []
        self._pos: dict[KeyNode, int] = {}
        self._L: float = 0.0

        self._w_f = 1.0
        self._w_l = 2.0
        self._alpha = 0.7
        self._slack = 0.5

    @property
    def name(self) -> str:
        return "LogitGDS"

    # ---- bookkeeping ------------------------------------------------------

    @staticmethod
    def _sat_add(v: int) -> int:
        return 15 if v >= 15 else v + 1

    def _value(self, node: KeyNode) -> float:
        l = self._logit_ema.get(node, 0.0)
        return self._w_f * float(node.freq) + self._w_l * l

    # GDS key at push time. Stored H = value + L ages entries that are not
    # re-accessed as L grows, exactly like classic Greedy-Dual.
    def _score(self, node: KeyNode) -> float:
        return self._value(node) + self._L

    def on_access(self, node: KeyNode, logit: float) -> None:
        self._sketch.record(node.key)
        node.freq = self._sat_add(node.freq)
        ema = self._alpha * self._logit_ema.get(node, 0.0) + (
            1.0 - self._alpha
        ) * max(0.0, logit)
        self._logit_ema[node] = ema
        self._push(node)

    def on_prefetch(self, node: KeyNode, logit: float) -> None:
        # Prefetched-but-unused entries must not dominate eviction; give them a
        # suppressed score so they age out quickly if never touched.
        self._sketch.record(node.key)
        if node.key not in self._index:
            self._index[node.key] = node
        self._logit_ema[node] = max(
            self._logit_ema.get(node, 0.0), min(logit, 0.5 * 15.0)
        )
        node.freq = 0
        self._push(node)

    def on_evict(self, node: KeyNode) -> None:
        self._index.pop(node.key, None)
        self._logit_ema.pop(node, None)
        self._heap_erase(node)
        # advance the global aging value by the victim's contribution
        self._L += self._value(node)
        node.freq = 0

    # ---- admission --------------------------------------------------------

    def should_admit(self, key: ExpertKey, logit: float) -> bool:
        self._sketch.record(key)
        if len(self._index) < self._capacity:
            return True
        if not self._heap:
            return True
        # Compare INTRINSIC values (not H = value + L): Greedy-Dual's aging L is
        # common to every entry and grows without bound, so H-only comparison
        # would reject every candidate once L is large.
        cand = self._w_f * self._sketch.estimate(key) + self._w_l * logit
        vic = self._value(self._heap[0][1])
        return cand >= vic - self._slack or logit >= self._admit_threshold

    # ---- eviction ---------------------------------------------------------

    def victim(self, unavailable: Set[KeyNode]) -> Optional[KeyNode]:
        return self._live_victim(unavailable)

    # Return the current min-value resident node not in `unavailable`, or None
    # if none. Peek-only: the winner stays in the heap; the cache erases it via
    # on_evict when the eviction actually happens (so a fall-through to a free
    # slot loses nothing). Busy nodes are temporarily parked aside and restored.
    def _live_victim(self, unavailable: Set[KeyNode]) -> Optional[KeyNode]:
        aside: List[KeyNode] = []
        winner: Optional[KeyNode] = None
        while self._heap:
            n = self._heap[0][1]
            self._heap_erase(n)  # pop the min out of the heap
            if n in unavailable:
                aside.append(n)
                continue
            winner = n
            break
        for a in aside:
            self._push(a)
        if winner is not None:
            self._push(winner)  # restore; on_evict removes it if evicted
        return winner

    # ---- misc -------------------------------------------------------------

    def on_phase_change(self) -> None:
        self._sketch.decay(1)
        # halve per-entry frequencies (stale sessions) and re-push.
        for node in list(self._index.values()):
            node.freq = node.freq >> 1
            self._push(node)

    def clear(self) -> None:
        self._index.clear()
        self._logit_ema.clear()
        self._heap.clear()
        self._pos.clear()
        self._L = 0.0

    def size(self) -> int:
        return len(self._index)

    def capacity(self) -> int:
        return self._capacity

    def insert(self, node: KeyNode) -> None:
        if node.key in self._index:
            return
        self._index[node.key] = node
        node.freq = 1
        self._logit_ema[node] = 0.0
        self._push(node)

    # Exposed for the prefetcher's ranking: predicted near-term reuse.
    def predicted_reuse(self, key: ExpertKey) -> float:
        return self._w_f * self._sketch.estimate(key)

    # ---- indexed min-heap --------------------------------------------------

    @staticmethod
    def _parent(i: int) -> int:
        return (i - 1) // 2

    @staticmethod
    def _left(i: int) -> int:
        return 2 * i + 1

    @staticmethod
    def _right(i: int) -> int:
        return 2 * i + 2

    def _heap_less(self, a: int, b: int) -> bool:
        va, na = self._heap[a][0], self._heap[a][1]
        vb, nb = self._heap[b][0], self._heap[b][1]
        return va < vb or (va == vb and na.index < nb.index)

    def _heap_swap(self, a: int, b: int) -> None:
        self._heap[a], self._heap[b] = self._heap[b], self._heap[a]
        self._pos[self._heap[a][1]] = a
        self._pos[self._heap[b][1]] = b

    def _sift_up(self, i: int) -> None:
        while i > 0 and self._heap_less(i, self._parent(i)):
            self._heap_swap(i, self._parent(i))
            i = self._parent(i)

    def _sift_down(self, i: int) -> None:
        n = len(self._heap)
        while True:
            m, l, r = i, self._left(i), self._right(i)
            if l < n and self._heap_less(l, m):
                m = l
            if r < n and self._heap_less(r, m):
                m = r
            if m == i:
                return
            self._heap_swap(i, m)
            i = m

    # Insert or update the node's single heap entry with its current H.
    def _push(self, node: KeyNode) -> None:
        v = self._score(node)
        if node not in self._pos:
            self._pos[node] = len(self._heap)
            self._heap.append((v, node))
            self._sift_up(len(self._heap) - 1)
        else:
            i = self._pos[node]
            old = self._heap[i][0]
            self._heap[i] = (v, node)
            if v < old:
                self._sift_up(i)
            else:
                self._sift_down(i)

    # Remove node from the heap (no-op if absent). O(log n).
    def _heap_erase(self, node: KeyNode) -> None:
        if node not in self._pos:
            return
        i = self._pos.pop(node)
        last = len(self._heap) - 1
        if i == last:
            self._heap.pop()
            return
        self._heap[i] = self._heap[last]
        self._pos[self._heap[i][1]] = i
        self._heap.pop()
        self._sift_up(i)
        self._sift_down(i)


# ---------------------------------------------------------------------------
# Policy factory (mirrors src/policy.cpp make_policy).
# ---------------------------------------------------------------------------


def make_policy(
    name: str, capacity_entries: int, logit_admit_threshold: float = 0.35
) -> Optional[EvictionPolicy]:
    n = name.lower()
    if n == "lru":
        return LRUPolicy(capacity_entries)
    if n in ("tinylfu", "lfu"):
        return TinyLFUPolicy(capacity_entries)
    if n in ("logitgds", "gds", "confidence", "logit"):
        return LogitGDSPolicy(capacity_entries, logit_admit_threshold)
    raise ValueError(f"unknown eviction policy {name!r} (lru | tinylfu | logitgds)")
