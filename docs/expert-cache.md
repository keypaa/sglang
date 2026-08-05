# MoE Expert-Cache (DeepSeek-V4-0731)

A locality-aware on-GPU cache for MoE expert weights. It keeps a small,
fixed-resident subset of experts hot in VRAM and streams the rest in/out of
host memory **off the critical path**, so a decode step never waits on a
router-selected expert's weights being fetched. Built for consumer GPUs
(24/48 GB) where the full expert set cannot stay resident.

The design, decisions, and the task-by-task build record live in:

- Design spec: `docs/superpowers/specs/2026-08-02-expert-cache-phase2-design.md`
- Execution plan: `docs/superpowers/plans/2026-08-02-expert-cache-phase2.md`

This file is the "what this is, and where it stands" snapshot.

## Why

DeepSeek-V4-style MoE has hundreds of experts per layer across 43 layers; the
full expert set (~167 GB on-disk) does not fit in VRAM. Per token only `top_k` +
shared experts are active, but *which* ones is data-dependent at decode time.
The cache selects which experts to keep resident (a fixed-size, static pool)
and issues async host->device copies for the rest, overlapped with compute.

## Architecture

Package root: `python/sglang/srt/layers/moe/expert_cache/`.

| Component | File | Role |
|---|---|---|
| `ExpertCache` | `expert_cache.py` | L1 decision logic: fixed-slot array, hash index keyed by `ExpertKey(layer, expert)`, refcounting, demand-fetch, prefetch, eviction. |
| `EvictionPolicy` (LRU / LogitGDS / TinyLFU) | `policies.py` | Decides admission + eviction. **LogitGDS** = router-confidence × greedy-dual aging; default. |
| `StaticExpertPool` | `static_pool.py` | Fixed-address VRAM store, two pre-allocated tensors (`w13`, `w2`). **No allocation / no `Event()` creation after `__init__`** — capture-safe. |
| `CudaTransferBackend` | `backends.py` | Async host->device copies of one expert block into a pool slot on a dedicated transfer stream. `SimBackend` is the CPU timing twin. |
| `StaticPoolOrchestrator` | `orchestrator.py` | Host-side Phase-A adapter: turns router output into cache acquires + pool copies, and maintains the **static slot map** the CUDA graph reads. |
| simulator | `simulator.py` | Discrete-event simulator (`TraceGenerator`, `run_simulation`) used to validate the Python port against the C++ `MoE-LRU` reference. |

Data flow (per decode step, per layer):

1. Router produces `top_k` `(expert_id, logit)` choices.
2. `StaticPoolOrchestrator.on_router_output(layer, ids, logits, next_ids)`:
   - `ExpertCache.acquire` each choice; a fresh miss either admits into a real
     slot or returns a transient (`index < 0`).
   - Transformer: if a routed expert landed on a transient slot, the
     orchestrator upgrades it via **`ExpertCache.acquire_resident`** — the
     demand-guarantee that a committed step **never maps an expert to -1**.
   - The chosen slot id is written into the orchestrator's
     `(layer, expert_id)` slot map.
   - `next_ids` are prefetched (async, overlapped) for the following layer.
3. `step_commit()`: records the step's copy-event on the pool; releases the
   held refcounts.
4. The captured graph waits on the step event, indexes the slot map by the
   top-k ids, gathers pool weights, and runs the FFN.

## What changed since Phase 1

Phase 1 (`f7498f7e9`) delivered the pure-Python cache + simulator and its 12
tests. Phase 2 (HEAD since `16f4bf596`) added the GPU-fast, capture-safe tier:

- `StaticExpertPool` — static twin pools, pre-allocated events, step-event
  barrier; the load path (`copy_in`, `record_step`, `wait_all`, `reset`)
  performs no allocation.
- `CudaTransferBackend` — now writes into the static pool (via `set_pool`);
  `slot.node.index` is the pool slot id; transient (negative-index) loads
  no-op.
- `StaticPoolOrchestrator` + additive **`ExpertCache.acquire_resident`** —
  the demand-guarantee so a committed step never commits a transient slot.
- Slot map keyed by `(layer, expert)` so two layers routing the same id do not
  clobber each other's entry.
- Two decode harnesses (both CUDA-gated): an **eager** harness that
  reproduces the simulator's decode hit rate through the real pool, and a
  **captured** `torch.cuda.graph` harness proving the graph reads freshly
  updated pool weights.
- Orchestrator-constructor invariant: `cache capacity == pool.num_slots`.

## Verification status

CPU tests (no CUDA available locally): **21 passed / 7 skipped** — run with

```bash
PYTHONPATH=<shim> python -m pytest \
  test/registered/unit/layers/moe/test_expert_cache.py \
  test/registered/unit/layers/moe/test_expert_cache_simulator.py \
  test/registered/unit/layers/moe/test_expert_cache_orchestrator.py \
  test/registered/unit/layers/moe/test_static_expert_pool.py -q
```

Verified on CPU:

- all orchestrator units (hit/miss bookkeeping, transient→resident upgrade,
  `-1` never committed, cross-layer independence, bounds checks);
- eager harness hit rate `0.7708` ≥ `0.7565` reference (sim `0.8065` − tol).

**Pending GPU run** (must verify on a CUDA host / CI before trusting):

- the 4 CUDA-gated test classes (skips locally): `StaticExpertPool` copy
  units, the eager harness, and especially the captured-graph harness
  (replay-stream legality, fp8 copy, gather/GEMM numerics).

Known environment caveat (unrelated): 4 sibling tests under
`test/registered/unit/layers/moe/` (`test_aiter_runner.py`, etc.) fail to
**collect** on this machine because they import `sglang.srt.layers.moe.moe_runner`,
which requires the full package + `pybase64`; they are not part of the
expert-cache suite.

## How the pieces were validated

Per-task commits, review findings, and the whole-branch review are captured in
the SDD ledger `.superpowers/sdd/2026-08-02-expert-cache-phase2/progress.md`
alongside the per-task briefs and review diffs. Two cross-layer defects that a
single-layer test suite alone could not surface — the slot-map aliasing across
layers and the missing capacity invariant — were found in the final whole-branch
review and fixed thereafter.