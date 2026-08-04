from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .types import ExpertKey

# ---------------------------------------------------------------------------
# Count-Min frequency sketch (mirrors include/moe/freq_sketch.hpp).
#
# Used for:
#   * the W-TinyLFU admission filter (is the candidate hotter than the victim?)
#   * LogitGDS long-term frequency term
#   * prefetcher ranking (predicted future reuse)
#
# 4-bit counters keep memory tiny: 4 hashes x 2^16 buckets x 0.5 B = 128 KB
# for the whole model. Counters saturate; a global "aging" sweep divides every
# counter by 2 to forget stale sessions (multi-turn switching).
# ---------------------------------------------------------------------------


class FreqSketch:
    kRows = 4
    kBits = 16  # 65536 buckets / row
    kBuckets = 1 << kBits
    kMaxCount = 15  # 4-bit saturate

    def __init__(self, seed: int = 0x9E3779B97F4A7C15):
        self._seed = seed
        self._table = bytearray(self.kRows * self.kBuckets)

    # Four independent 64-bit hashes over (layer, expert). Cheap: two mixes of
    # a splitmix-based hash with different seeds.
    def _hashes(self, key: ExpertKey):
        v = (key.layer << 32) | (key.expert & 0xFFFFFFFF)
        for r in range(self.kRows):
            h = (v ^ (self._seed + r * 0x9E3779B97F4A7C15)) & 0xFFFFFFFFFFFFFFFF
            h ^= h >> 30
            h = (h * 0xBF58476D1CE4E5B9) & 0xFFFFFFFFFFFFFFFF
            h ^= h >> 27
            h = (h * 0x94D049BB133111EB) & 0xFFFFFFFFFFFFFFFF
            h ^= h >> 31
            yield h & (self.kBuckets - 1)

    def record(self, key: ExpertKey, amount: int = 1) -> None:
        for r, b in enumerate(self._hashes(key)):
            idx = r * self.kBuckets + b
            v = self._table[idx] + amount
            self._table[idx] = self.kMaxCount if v > self.kMaxCount else v

    # Estimated frequency; min over rows (standard Count-Min underestimate-free
    # upper-bound: we use the min of the per-row counters).
    def estimate(self, key: ExpertKey) -> int:
        m = self.kMaxCount
        for r, b in enumerate(self._hashes(key)):
            c = self._table[r * self.kBuckets + b]
            if c < m:
                m = c
        return m

    # Decay all counters (called periodically or after a session switch).
    def decay(self, shift: int = 1) -> None:
        for i in range(len(self._table)):
            self._table[i] >>= shift

    def reset(self) -> None:
        self._table = bytearray(self.kRows * self.kBuckets)

    def bytes(self) -> int:
        return len(self._table)
