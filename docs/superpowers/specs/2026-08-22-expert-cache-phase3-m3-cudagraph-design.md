# Phase 3 Milestone 3: CUDA-Graph-Safe Expert Cache (tc_piecewise)

**Date:** 2026-08-22
**Status:** Approved for implementation planning
**Builds on:** Phase 3 M1/M2 (pool-as-weights integration; eager-only until now)

---

## 1. Context and Problem

M1/M2 require `--disable-cuda-graph`: under full-graph capture the router runs
inside the replayed graph, so a host-side orchestrator never sees routed ids
before the kernel reads pool rows — and our Phase-2 finding shows an in-graph
residency wait is capture-illegal (`cudaErrorStreamCaptureIsolation`).

Exploration found that upstream already ships **segmented capture** with an
eager boundary exactly where we need one:

- `TcPiecewiseCudaGraphBackend` registers
  `sglang.moe_forward_piecewise_cuda_graph_impl` as a split op at the MoE
  forward (torch.compile piecewise; deepseek_v2.py is already piecewise-aware).
- `BreakableCudaGraphBackend` documents the same per-replay eager-gap property
  ("metadata recomputed at replay outside captured segments").

**Load-bearing assumption (to be probed empirically before building, Task 0):**
the split-op region re-executes as Python on every decode-step replay with the
step's real `topk_ids`. If false, fall back to extending the breakable backend
(appendix A).

## 2. Design

### 2.1 Components

| Piece | Location | Role |
|---|---|---|
| Guard update | `validate_moe_expert_cache` + model-init guards | Cache-on requires segmented decode graphs: either `--disable-cuda-graph` (eager, M1/M2 behavior) or decode backend = `tc_piecewise`. Full/breakable remain rejected. |
| Prepare hook | new registered split-op in `expert_cache/` (`expert_cache_prepare`) called from `DeepseekV2MoE`'s tc_piecewise branch | Runs in the segment gap each replay: host-read routed ids → `ensure_resident` → fence → device slot-map refresh → returns slot-remapped ids. |
| Wiring | `DeepseekV2MoE.forward` tc_piecewise branch | Routes topk output through the prepare-op when cache enabled. |

The MoE piece stays captured and reads pool rows via remapped ids. No weight
bytes move inside any graph.

### 2.2 Decode data flow (graphs ON)

```
[layer L attn piece: captured]
→ segment gap (Python, every replay):
    hook: ids = topk_ids.tolist()
          ensure_resident(ids); wait fence
          refresh device slot map; slot_ids = map[ids]
→ [layer L MoE piece: captured, gathers pool rows by slot_ids]
```

Prefetch across layers/tokens is out of scope (demand misses only at gaps);
follow-up optimization.

### 2.3 Error handling

- Cache-on with decode backend `full` or `breakable`: startup error naming
  `tc_piecewise` or `--disable-cuda-graph`.
- Probe failure (Task 0): milestone pivots to appendix-A design before any
  product code lands.
- Hook raises if runtime missing (no half-wired window, carried from M1).

## 3. Testing

- **Task 0 probe (GPU, go/no-go):** instrument the piecewise MoE boundary with
  a replay counter + wall timing across N engine decode steps; assert >1
  executions and record per-gap cost.
- **Unit (CPU):** guard matrix additions; hook logic (ids → residency → remap)
  with FakePool.
- **Milestone gate (GPU):** tiny-model end-to-end `Engine` test — cache-off vs
  cache-on with tc_piecewise decode graphs, identical prompts, output-token
  equality; bf16 and fp8 variants. Manual `torch.cuda.graph` unit emulation is
  rejected: it never re-runs Python at replay, so it cannot exercise the hook
  honestly.

## 4. Out of scope

- Cross-layer/token prefetch pipelining.
- Breakable-backend support (appendix A documents the fallback design).
- TP/EP, NextN, split-name checkpoints (unchanged from M1/M2 constraints).

## Appendix A: fallback design (breakable backend)

If the probe disproves per-replay execution of tc_piecewise split ops: extend
`BreakableCudaGraphBackend` to treat the MoE forward as a break boundary the
same way attention boundaries work (register boundary, recompute inputs at
replay), then invoke the same prepare-hook in that gap. More machinery, no
torch.compile dependency; testing identical (§3 gate).

## Decisions Log

1. Target tc_piecewise only (user pick); breakable documented as fallback.
2. Probe-first (user pick): Task 0 go/no-go gates the whole design.
3. Done = bf16+fp8 parity with graphs enabled + telemetry (user pick);
   throughput measurement deferred to a later benchmark pass.
