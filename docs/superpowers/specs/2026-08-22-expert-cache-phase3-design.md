# Phase 3 Design (Milestone 1): MoE Expert Cache — Real-Model Integration

**Date:** 2026-08-22
**Status:** Approved for implementation planning
**Builds on:** Phase 1 (validated cache/simulator), Phase 2 (GPU-trusted static pool, orchestrator, harnesses; performance-characterized)

---

## 1. Context and Problem

Phases 1–2 delivered a GPU-verified expert-cache building block: a static,
CUDA-safe pool, a host-side orchestrator, and harnesses proving the captured
and eager decode patterns. Nothing in SGLang's model path uses it yet.

The blocker this phase removes: DeepSeek-V2/V3-family models keep all 256
expert weights per layer on device (`FusedMoE.w13_weight/w2_weight`,
~2.9 GB fp8 per layer, ~127 GB across 43 layers) — impossible on the 24/48 GB
consumer GPUs this project targets. The expert weights must live in pinned
host RAM and stream into a small fixed device pool that the MoE kernel reads
directly.

### The core constraint discovered in exploration

Any integration that materializes full-size expert weight tensors is dead on
arrival: there is no VRAM to gather into. Therefore **the pool must be the
weight tensor**: the layer's `FusedMoE` is constructed with
`num_experts = num_slots`, and routed experts are addressed through a slot-id
remap rather than by their checkpoint ids.

### Milestone scoping (user-approved)

- **Eager-first:** milestone 1 requires `--disable-cuda-graph`. The router
  runs inside the captured decode graph in normal serving, so a host-side
  orchestrator cannot see routed ids before replay; that problem gets its own
  design round in a later milestone.
- **TP1 / EP1 only.**
- **Success = numerical parity + telemetry**, not throughput.

## 2. Approach: Pool-as-Weights

Construct the layer's experts as `FusedMoE(num_experts=num_slots)`. Its
`w13_weight` / `w2_weight` tensors (plus fp8 scale tensors) ARE the static
pool: fixed addresses, ~2.3 GB at 128 slots. Rejected alternatives:
pool-backed custom Parameters inside full-size FusedMoE (deep quant-internals
surgery), and D2D gather into full-size weights (requires the impossible
tensor).

## 3. Architecture

| Component | Location | Role |
|---|---|---|
| Server flags | `server_args.py` | `--enable-moe-expert-cache`, `--moe-cache-slots N` (default 128) |
| `ExpertHostStore` | new module in `python/sglang/srt/layers/moe/expert_cache/` | Pinned host RAM store for ALL layers' experts, keyed `(layer, expert)` → (w13, w2, scales). ~167 GB for DeepSeek-V4-Flash: big-RAM host is a documented requirement |
| Per-layer wiring | `DeepseekV2MoE.__init__` | Flag on → `self.experts = FusedMoE(num_experts=num_slots, ...)`; one `StaticPoolOrchestrator` per layer over the shared store/backend; `num_fused_shared_experts` forced to 0 |
| Slot-id remap | helper in `expert_cache/` | Device gather `slot_ids = slot_map[topk_ids]` |

Startup guards (fail fast): cache requires `--disable-cuda-graph`; TP=1 and
EP=1; fused shared experts disabled; `slots <= n_routed_experts` (clamp +
warning otherwise); host RAM shortfall reported with actual numbers at store
build time.

## 4. Decode-Step Data Flow (eager)

```
hidden → gate (256-expert logits, unchanged) → topk (unchanged)
  → NEW: host reads topk_ids (.tolist(); legal in eager mode)
      orch.on_router_output(layer, ids, logits, next_layer_hints)
      step_commit(); pool.wait_all()   # zero-cost when no copies pending
  → NEW: slot_ids = slot_map[topk_ids]  # tiny device gather
  → self.experts(hidden, TopKOutput(topk_weights, slot_ids, logits))
```

- Router/topk semantics untouched (real expert ids end to end up to the remap).
- Kernel gathers pool rows exactly as it would gather full-size weight rows.
- Prefetch of layer L+1 candidates issues during layer L compute — matches the
  measured pipelining regime (8.7 ms PCIe floor hides <1 ms compute).
- Correctness invariant (Phase-2 §6 carried forward): every id in the remapped
  `topk_ids` addresses a pool row holding that expert's exact checkpoint bytes
  (payload + scales) before the kernel reads it.

## 5. Weight Loading

In `DeepseekV2ForCausalLM.load_weights` (deepseek_v2.py:3112), when enabled:
expert tensors (`mlp.experts.{e}.w13_weight`, `w2_weight`, scale invs) are
intercepted and written to the `ExpertHostStore` (pinned) instead of device
params. FusedMoE pool params are allocated but never checkpoint-initialized;
rows start invalid and are overwritten by `copy_in` before any read.

Layout caveat: copies must target the post-`process_weights_after_loading`
layout (fp8 block scales may be transposed/reshaped); store entries capture
tensors after the loader's transform step so `copy_in` is layout-exact.

## 6. Error Handling

| Condition | Behavior |
|---|---|
| Cache + CUDA graph enabled | startup error → `--disable-cuda-graph` |
| Cache + TP>1 or EP>1 | startup error |
| `--moe-cache-slots` > routed experts | clamp + warning |
| Host RAM insufficient | store-build error with actual shortfall |
| `acquire_resident` all-busy RuntimeError | propagate (per-layer commit rhythm makes it unreachable; if hit, real bug) |

## 7. Testing Strategy

1. **Unit (CPU):** slot-id remap correctness incl. layer independence;
   startup guards; host-store put/get round-trip.
2. **Parity (GPU) — the milestone gate:** a tiny synthetic MoE config (dense
   fits in VRAM) served twice from identical weights: uncached vs cached with
   slots ≪ E; same prompts; outputs must match within tolerance. This is the
   only honest parity check: no GPU can run dense DeepSeek-V4-Flash.
3. **Telemetry:** cached tiny-model run shows nonzero hits, recorded stall
   times, per-layer counters. Mechanism: each layer's existing Phase-2
   `CacheStats` plus a wall-clock stall accumulator on the orchestrator,
   surfaced via periodic scheduler logging (no new metrics framework in
   milestone 1).
4. **Regression:** Phase-2 suite stays green; Modal verify script extended
   with the new CPU units.

## 8. Out of Scope (later milestones)

- CUDA-graph-compatible routing (graph break or router pre-pass).
- TP/EP sharding of the pool.
- Quantization beyond fp8 block-wise.
- Throughput optimization as an explicit goal (parity first).

## 9. Decisions Log

1. Eager-first milestone (user): correctness before the graph problem.
2. Pool-as-weights (user, after exploration killed alternatives): FusedMoE
   sized to the pool; no full-size expert tensors ever exist on device.
3. Parity via tiny synthetic model: dense reference is otherwise unrunnable.
4. Phase-2's replay-time ordering amendment carries over unchanged: eager has
   no capture constraints, but the same wait-only-when-pending discipline
   applies.
