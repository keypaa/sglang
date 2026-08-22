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

## Phase 3 milestone 1: wired into DeepSeek MoE (pool-as-weights)

Phase 3 turns the library into a serving path: `DeepseekV2MoE`
(`models/deepseek_v2.py`) rebuilds its routed-experts layer sized to
`moe_cache_slots` when the cache is on, so **a pool slot row IS one expert** —
streaming weights into a slot row is indistinguishable from having loaded the
model. At `load_weights` time the model intercepts routed-expert fused
tensors before param loading and diverts them into a pinned host store
(`ExpertHostStore`); pool rows start uninitialized and are filled on demand.
Each cache-enabled layer gets a `LayerRuntime` whose `RealWeightPool` adapts
the real FusedMoE weight params to the pool surface (`copy_in` /
`record_step` / `wait_all`); the eager forward remaps raw `topk_ids` through
the static slot map before the MoE kernel runs.

Flags:

| Flag | Default | Meaning |
|---|---|---|
| `--enable-moe-expert-cache` | off | Stream MoE experts from pinned host RAM into a fixed VRAM pool (DeepSeek MoE). |
| `--moe-cache-slots N` | 128 | Pool slots (= resident experts) per layer; clamped to the routed-expert count. |

Constraints (enforced by `validate_moe_expert_cache` + model-init guards,
fail fast at startup):

- **Eager only**: requires `--disable-cuda-graph` (the router runs inside the
  captured decode graph; graph-safe design is future work).
- **TP=1 / EP=1** only.
- **bf16, fused-name checkpoints** of the standard format only (unquantized
  MoE, no FlashInfer-TRTLLM runner backend; fused shared experts off; no hash
  layers).
- **Big-RAM host**: every routed expert's full weight set must fit pinned in
  system RAM (~167 GB for DeepSeek-V4-class models).

Parity gate: `test_deepseek_moe_expert_cache_parity.py` builds a tiny DeepSeek
MoE twice (dense vs pool-sized + streaming, attention bypassed byte-identical)
and asserts equal logits — **passes on Modal L4** (32 passed total).

Telemetry: each layer's `LayerRuntime.telemetry()` returns

- `hits`, `misses`, `loads` — from the shared `ExpertCache` stats;
- `stall_ms_total`, `stall_ms_last` — wall time inside `ensure_resident`
  while copies were pending; an all-hit call resets `stall_ms_last` to 0.

## Verification status

**GPU gate: 32 passed / 0 failed** on a Modal L4 (sm_89,
`lmsysorg/sglang:latest` image), including the tiny-model parity gate
above.

Local suite (no CUDA needed): **41 passed / 9 skipped** across Phase-2
units plus the Phase-3 server-args, host-store, slot-remap, layer-runtime,
wiring, and parity tests — the 7 pool-copy CUDA units, the wiring smoke,
and the parity gate skip locally and execute only on the GPU gate.

CPU-only re-run (no CUDA needed):

```bash
PYTHONPATH=python python -m pytest \
  test/registered/unit/layers/moe/test_expert_cache.py \
  test/registered/unit/layers/moe/test_expert_cache_simulator.py \
  test/registered/unit/layers/moe/test_expert_cache_orchestrator.py \
  test/registered/unit/layers/moe/test_static_expert_pool.py -q
```

GPU harness: `/tmp/opencode/modal_gpu_verify.py`
(`python -m modal run /tmp/opencode/modal_gpu_verify.py`) — official
sglang Docker image + the repo mounted via `PYTHONPATH`; hand-picked
PyPI deps do **not** work (pip's `sgl_kernel` wheel ships sm100 binaries
only and lacks `libnvrtc.so.12`).

Design findings from the GPU runs (worth knowing before extending this):

- A captured graph may **not** wait on `pool.step_event()`
  (`cudaErrorStreamCaptureIsolation`, persists under
  `capture_error_mode="relaxed"`): a captured stream cannot depend on
  uncaptured transfer-stream work. Copy→read ordering is therefore
  enforced at replay time (`pool.wait_all()` before replay), not by a
  baked in-graph wait. Spec §5's original "graph waits on step event"
  wording is superseded.
- cuBLAS must be warmed up on a side stream before capture —
  `cublasCreate` inside a capture fails.

Known environment caveat (unrelated): 4 sibling tests under
`test/registered/unit/layers/moe/` (`test_aiter_runner.py`, etc.) fail to
collect without the full package deps; they are not part of the
expert-cache suite.

## Performance characterization (Modal L4, PCIe Gen4)

Measured with DeepSeek-V4-Flash shapes (H=3840, I=1536, top_k=6, fp8 expert
block = 17.7 MB) by `benchmark/moe_expert_cache_bench.py`:

| Measurement | Result |
|---|---|
| H2D, pinned, one expert block | **1.45 ms** (~12.2 GB/s achieved) |
| H2D, pageable | 4.12 ms (~4.3 GB/s) — pin your memory |
| top-6 burst (pinned) | 8.7 ms |
| Decode GEMMs, one layer, top-6 | 0.88 ms (launch-bound at batch 1) |
| Copies + GEMMs, barrier every step | 23.7 ms/iter |
| Same, pipelined (drain once) | **8.7 ms/iter = the PCIe floor; compute fully hidden** |
| Orchestrated captured step, all-hit | 3.38 ms (host/harness-bound) |
| Orchestrated step with demand misses | 8.1 ms (~0.42 ms stall/miss, ~11 loads/step) |

Simulator sweep (Phase-1 sim, Zipf trace seed 7, per layer): hit rate climbs
3% → 51% → 83% → **90%** at 8 → 32 → 64 → 128 slots and saturates there;
slots beyond 128 buy nothing because the residual misses are *cold*
(first-touch), not capacity misses.

Phase-3 implications:

1. **Plan around ~12 GB/s**, not the assumed 25 GB/s (`HardwareSpec.h2d_bw`
   should be re-calibrated); a real expert fetch is ~1.45 ms.
2. **Pinned memory is mandatory** (pageable costs ~3x).
3. **Never barrier mid-step**: pipelining copies behind compute is the entire
   ballgame (8.7 vs 23.7 ms). The §5 replay-time-ordering amendment aligns
   with this.
4. **Cache sweet spot ≈ half the expert population per layer** (128 of 256
   slots ≈ 2.3 GB fp8); more VRAM is better spent on KV.
5. **Host overhead was the next bottleneck — largely fixed**: the hot path
   now syncs the transfer stream only when copies are pending, uses a pinned
   slot map with a single non_blocking H2D refill (`copy_slot_map_into`),
   and no longer records an unconsumed event per copy. Demand-miss steps
   dropped 12.1 → 8.1 ms (stall/miss 761 → 422 µs). The residual all-hit
   step cost is harness-side (per-step `torch.tensor(..., device="cuda")`
   + full-device synchronize), not library-side.
6. Back-of-envelope throughput ceiling on this trace: cold misses/token
   (~26 across 43 layers) × 1.45 ms ≈ 38 ms/token of unavoidable transfer,
   partially overlappable — i.e. streaming helps most when locality is high;
   when it isn't, PCIe is the hard floor.

## How the pieces were validated

Per-task commits, review findings, and the whole-branch review are captured in
the SDD ledger `.superpowers/sdd/2026-08-02-expert-cache-phase2/progress.md`
alongside the per-task briefs and review diffs. Two cross-layer defects that a
single-layer test suite alone could not surface — the slot-map aliasing across
layers and the missing capacity invariant — were found in the final whole-branch
review and fixed thereafter.