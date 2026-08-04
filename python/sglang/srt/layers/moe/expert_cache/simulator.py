from __future__ import annotations

import math
import random
from dataclasses import dataclass, field
from typing import List

from .backends import SimBackend
from .expert_cache import ExpertCache, Slot
from .policies import make_policy
from .types import CacheStats, ExpertKey, HardwareSpec, ModelSpec, RouterChoice

# ---------------------------------------------------------------------------
# TraceGenerator models the three regimes of MoE serving that stress caches
# (mirrors include/moe/simulator.hpp):
#   1. prefill    - a long prompt touching a wide, near-uniform spread of
#                   experts (designed to defeat plain LRU).
#   2. decode     - Zipf-skewed selection: ~17% of experts carry ~80% of
#                   requests.
#   3. multi-turn - burstiness: within a session, recently-used experts are
#                   re-requested across consecutive tokens (temporal locality).
# ---------------------------------------------------------------------------


class _Zipf:
    def __init__(self, n: int, s: float, rng: random.Random):
        self._rng = rng
        self._cdf: List[float] = [0.0]
        h = 0.0
        for r in range(1, n + 1):
            h += 1.0 / math.pow(float(r), s)
            self._cdf.append(h)
        for i in range(len(self._cdf)):
            self._cdf[i] /= h

    def sample(self) -> int:
        u = self._rng.random()
        # lower_bound(cdf, u) - 1
        lo, hi = 0, len(self._cdf)
        while lo < hi:
            mid = (lo + hi) // 2
            if self._cdf[mid] < u:
                lo = mid + 1
            else:
                hi = mid
        return lo - 1


class TraceGenerator:
    def __init__(self, model: ModelSpec, seed: int = 1234):
        self._model = model
        self._rng = random.Random(seed)

    def generate(self, prefill_tokens: int, decode_tokens: int) -> List[List[List[RouterChoice]]]:
        """Returns a trace: list of tokens; each token has `num_layers`
        layer-lists; each layer-list has top_k + shared choices."""
        L = self._model.num_layers
        E = self._model.num_experts

        zipfs = [_Zipf(E, 1.05, self._rng) for _ in range(L)]

        trace: List[List[List[RouterChoice]]] = []

        # --- prefill: near-uniform, low reuse ------------------------------
        for _ in range(prefill_tokens):
            tok: List[List[RouterChoice]] = []
            for l in range(L):
                seen = set()
                layer_choices: List[RouterChoice] = []
                for _ in range(self._model.top_k + self._model.shared_experts):
                    e = self._rng.randrange(E)
                    if e in seen:
                        e = (e + 1) % E
                    seen.add(e)
                    layer_choices.append(
                        RouterChoice(ExpertKey(l, e), 0.05 + 0.20 * self._rng.random())
                    )
                tok.append(layer_choices)
            trace.append(tok)

        # --- decode: concentrated session working set + Zipf tail ----------
        kSessionHot = 120  # hot experts per layer
        kSessionLen = 400  # tokens per session
        session_hot: List[List[int]] = [[] for _ in range(L)]
        session_age = [0] * L
        recent: List[List[int]] = [[] for _ in range(L)]

        def refresh_session(l: int) -> None:
            session_hot[l] = [zipfs[l].sample() for _ in range(kSessionHot)]
            session_age[l] = 0

        for l in range(L):
            refresh_session(l)

        for _ in range(decode_tokens):
            tok: List[List[RouterChoice]] = []
            for l in range(L):
                seen = set()
                layer_choices: List[RouterChoice] = []
                for _ in range(self._model.top_k + self._model.shared_experts):
                    if self._rng.random() < 0.70:
                        e = session_hot[l][self._rng.randrange(len(session_hot[l]))]
                    elif self._rng.random() < 0.55 and recent[l]:
                        e = recent[l][self._rng.randrange(len(recent[l]))]
                    else:
                        e = zipfs[l].sample()
                    if e in seen:
                        e = zipfs[l].sample()
                    seen.add(e)
                    recent[l].append(e)
                    if len(recent[l]) > 64:
                        recent[l].pop(0)
                    layer_choices.append(
                        RouterChoice(ExpertKey(l, e), self._confidence_of(e, E))
                    )
                tok.append(layer_choices)
                session_age[l] += 1
                if session_age[l] > kSessionLen:
                    refresh_session(l)
            trace.append(tok)
        return trace

    def _confidence_of(self, expert: int, total: int) -> float:
        base = 1.0 - math.log10(expert + 1) / math.log10(total)
        base = max(0.15, min(0.95, base))
        return base * (0.7 + 0.6 * self._rng.random())


# ---------------------------------------------------------------------------
# Discrete-event simulator. Models a software-pipelined decoder:
#   - the transfer stream (S_transfer) serializes expert loads;
#   - the compute stream (S_compute) runs attention+FFN per layer;
#   - layer L's prefetch for layer L+1 overlaps layer L's compute.
# Returns aggregate timing + cache stats. Mirrors include/moe/simulator.hpp.
# ---------------------------------------------------------------------------


@dataclass
class SimResult:
    stats: CacheStats = field(default_factory=CacheStats)
    total_time_ms: float = 0.0
    tokens_per_sec: float = 0.0
    bytes_moved_gb: float = 0.0
    decode_hit_rate: float = 0.0
    decode_hits: int = 0
    decode_misses: int = 0


def run_simulation(
    model: ModelSpec,
    hw: HardwareSpec,
    policy_name: str,
    trace: List[List[List[RouterChoice]]],
    warm_tokens: int,
    prefetch_enabled: bool = True,
    slot_cap_override: int = 0,
) -> SimResult:
    capacity = (
        slot_cap_override
        if slot_cap_override
        else hw.cache_bytes() // model.expert_bytes()
    )
    cache = ExpertCache(
        model,
        hw,
        make_policy(policy_name, capacity),
        SimBackend(hw.h2d_bw),
    )

    L = model.num_layers
    active_bytes_per_layer = (model.top_k + model.shared_experts) * model.expert_bytes()
    compute_per_layer_ms = (
        active_bytes_per_layer / hw.vram_bw * 1000.0
        + model.active_params / model.num_layers / hw.vram_bw * 1000.0
    )

    now = 0.0
    snap_hits, snap_misses = 0, 0

    for t in range(len(trace)):
        if t == warm_tokens:
            snap_hits = cache.stats.hits
            snap_misses = cache.stats.misses
        tok = trace[t]
        for layer in range(L):
            cache.advance_time(now)

            # 1. demand-acquire this layer's experts (misses load first)
            max_ready = now
            slots: List[Slot] = []
            for c in tok[layer]:
                s = cache.acquire(c)
                slots.append(s)
                max_ready = max(max_ready, s.ready_tick)

            # 2. prefetch the next layer's choices to overlap this layer's
            #    compute (software pipeline across layers / tokens)
            if prefetch_enabled:
                if layer + 1 < L:
                    next_choices = tok[layer + 1]
                elif t + 1 < len(trace):
                    next_choices = trace[t + 1][0]
                else:
                    next_choices = []
                if next_choices:
                    cache.prefetch(next_choices, cache.prefetch_budget())

            # 3. compute overlaps the transfer stream
            compute_start = max(now, max_ready)
            compute_end = compute_start + compute_per_layer_ms
            for s in slots:
                cache.release(s)
            cache.advance_time(compute_end)
            now = compute_end

    r = SimResult()
    r.stats = cache.stats
    r.total_time_ms = now
    r.tokens_per_sec = len(trace) / (now / 1000.0)
    r.bytes_moved_gb = cache.bytes_moved() / 1e9

    r.decode_hits = r.stats.hits - snap_hits
    r.decode_misses = r.stats.misses - snap_misses
    total = r.decode_hits + r.decode_misses
    r.decode_hit_rate = r.decode_hits / total if total else 0.0
    return r
