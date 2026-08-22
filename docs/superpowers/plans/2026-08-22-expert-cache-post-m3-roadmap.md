# Expert-Cache Post-M3 Roadmap: Milestones 4A-6

**Date:** 2026-08-22
**Status:** Roadmap draft — each milestone gets its own brainstorm → spec → plan cycle before execution
**Context:** Assumes Phase 3 M3 (CUDA-graph-safe tc_piecewise integration) has landed.

Order rationale: 4A unblocks real models → 4B measures whether stalls matter → 5 makes it fast if they do → 6 proves it at production scale. Do NOT start 4A mid-M3 (touches the same deepseek_v2.py load-path regions as the M3 hook).

---

## Milestone 4A: Split-HF-checkpoint interception

**Objective:** Load stock DeepSeek HF checkpoints (`gate_proj` / `up_proj` /
`down_proj.weight` + per-projection `.weight_scale_inv`) under the expert cache.
Today only fused `w13_weight`/`w2_weight` names are intercepted; stock repos fail
loudly.

**Design shape (to validate at its own brainstorm):**
- Extend `_EXPERT_WEIGHT_NAME_RE` / kind map with split kinds (`gate`, `up`,
  `down`, plus each `*_scale_inv`). The interceptor's pending dict accumulates
  per-expert parts; `_finish_expert_cache_wiring` fuses them into store entries:
  - w13 payload = `cat([gate_q, up_q], dim=0)`; w13 scales =
    `cat([gate_s, up_s], dim=0)` (row-block seam must land exactly at
    `ceil(I/128)` boundaries).
  - down payload/scale map to w2 entries unchanged.
- **Critical correctness constraint:** the fused stacking order and scale-row
  seams MUST match how `expert_params_mapping` shards gate/up into w13 for the
  dense path (shard ids w1/w3). Verify against `fused_moe_triton/layer.py`
  weight_loader before writing code; add a unit test asserting our fused bytes ==
  what the dense loader produces for the same checkpoint tensors.
- Pairing validation extends: all six tensors per expert (3 payloads + 3 scales)
  or none.

**Acceptance:** tiny-model parity gates pass feeding SPLIT names through B's
interception while A receives the same split names through its normal loader;
byte gate compares store vs dense-loader-fused reference.

**Open questions:** mixed checkpoints (some layers fused-format)? ignored_layers
interaction? fp8-per-tensor split checkpoints stay rejected?

---

## Milestone 4B: Throughput benchmark

**Objective:** Measure decode tok/s: (a) cache-on + tc_piecewise graphs,
(b) cache-off + tc_piecewise, (c) eager M1-style cache, (d) dense-if-it-fit
upper reference at tiny scale. On Modal L4 with realistic batch sizes (1, 8, 32)
and trace-driven routing.

**Design shape:**
- Reuse `benchmark/moe_expert_cache_bench.py` patterns + the M3 engine harness;
  report tok/s, per-layer stall totals (telemetry), hit rates, PCIe utilization.
- Sweep `--moe-cache-slots` ∈ {16, 64, 128} at fixed VRAM budget.

**Acceptance:** a numbers table committed under `docs/` + a decision recorded:
"stalls dominate → do M5" or "hit-rate hides them → M5 optional".

**Open questions:** batch>1 changes miss concurrency (topk across batch shares
experts — misses amortize?).

---

## Milestone 5: Prefetch pipelining (conditional on 4B)

**Objective:** Hide demand-miss H2D stalls behind compute. Physics say copies
fully pipeline when not barriered (measured 8.7 ms floor vs 23.7 ms barriered).

**Design shape:**
- In the segment gap: issue layer L+1 candidate copies WITHOUT waiting; fence
  only before the MoE piece reads (the existing replay-time barrier moves to
  just-in-time per layer).
- Hint sources, in increasing ambition: (i) next token's layer-L+1 topk from the
  PREVIOUS step's captured logits buffers (router stability ≈ measured 90%
  overlap), (ii) shared-expert-free heuristic (route frequency sketch already
  exists — FreqSketch), (iii) full simulator-guided admission.
- Graph-legality: copies stay outside capture; only the fence ordering changes.

**Acceptance:** parity gates still pass; stall_ms_total drops ≥50% at equal
output quality on the 4B harness.

---

## Milestone 6: Real-checkpoint smoke at production scale

**Objective:** Run a real DeepSeek-class checkpoint end-to-end.

**Requirements:** host with ≥192 GB system RAM (pinned store) + 24 GB GPU;
checkpoint download (~167 GB); engine launch with cache flags.

**Validation strategy (dense reference cannot run anywhere):**
- Internal consistency: eager-cache vs tc_piecewise-cache outputs identical;
- Quality heuristics: greedy continuations coherent on curated prompts;
- Telemetry: hit rates in a sane band vs the Zipf-trace predictions; stall
  profile matches characterization math;
- Memory audit: peak VRAM ≤ pool + dense weights + KV as designed.

**Acceptance:** recorded transcript + telemetry snapshot committed under docs/;
known-issue list written.

**Open questions:** which exact public checkpoint (V3/R1 vs V4-Flash availability);
pinned-RAM allocation limits (vm.overcommit, hugetlbfs?) on the chosen host.
