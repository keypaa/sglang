# Phase 2 Design: Static MoE Expert Pool (CUDA-Graph-Safe)

**Date:** 2026-08-02
**Status:** Implemented
**Builds on:** Phase 1 (pure-Python `expert_cache` package, validated against the C++ MoE-LRU simulator)

**Phase 2 implemented** 2026-08-05: `StaticExpertPool` (§3.1), `CudaTransferBackend` writes into the pool (§3.2), `StaticPoolOrchestrator` with demand-guarantee `acquire_resident` (§3.3, §6), plus the 4-test matrix (§7): CPU orchestration unit tests, CUDA copy unit tests, the eager decode harness, and the captured CUDA-graph decode harness. Test commits: `b2c2a1fb0` … `7b1b272ee`. CUDA-gated tests run pending a CUDA host/CI.

---

## 1. Context and Problem

DeepSeek-V4-Flash (`num_layers=43`, `num_experts=256` routed/layer, `top_k=6`,
`shared_experts=1`, mixed FP4/FP8 quantized weights, ~15.2 MB/expert) does not
fit in the 24 GB / 48 GB GDDR of consumer GPUs. Phase 1 proved—in a pure-Python
simulator faithful to the C++ baseline—that an LRU / W-TinyLFU / **LogitGDS**
expert cache keeps the ~12-expert hot set resident through prefill thrash and
rejects low-confidence one-shots, sustaining a 96.7-96.8% decode hit rate on the
C++ reference prefetch test (and 87-95% on shorter capacity-bound traces; see
note below).

Phase 2 ports that cache to **real CUDA memory** with a VRAM *static pool* that
is safe to read from inside a captured CUDA graph.

### The core tension

SGLang captures the decode forward pass as a single CUDA graph. CUDA graph
capture demands:
- **static memory** (fixed addresses, no `torch.empty` during capture),
- **no event creation / stream sync** during capture,
- the graph's kernels are replayed from baked-in addresses.

But an expert cache is inherently *dynamic*: which experts are needed is only
known when the **router runs**, and the router runs *inside* the graph.

**Resolution:** keep the **slot addresses static**, decouple *what* the graph
reads from *where* it reads it, and do all orchestration (routing → cache
lookup → PCIe copy) in an eager "Phase A" outside the graph.

---

## 2. Goals / Non-Goals

### Goals (Phase 2)
1. `StaticExpertPool` — pre-allocated fixed-address VRAM pools, pre-created
   events, dedicated transfer stream. No allocation / event creation after init.
2. `CudaTransferBackend` — writes into the static pool via pinned-CPU
   `copy_(..., non_blocking=True)` on the transfer stream.
3. `StaticPoolOrchestrator` — host-side "Phase A" adapter: router output →
   cache acquire/prefetch → issue copies → record per-step barrier event.
4. **Standalone interception harness** (not a full SGLang server) proving:
   - captured graph reads the static pool through a static slot-map indirection
     (`expert_id -> slot_id`) with no illegal memory access / capture errors,
   - decode hit rate reproduces the Phase-1 simulator under capacity pressure
     (cache smaller than the key universe — see hit-rate note below) on a fake
     MoE layer (256 experts, 100 decode steps),
   - the per-step barrier handshake works in both eager and captured modes.

> **Corrected hit-rate note (2026-08-02):** an earlier "~98% hit rate" figure
> was measured on a trivially-passing config where the VRAM budget let every
> expert key fit (no eviction ever fired) — that number was meaningless and was
> corrected during review. Under genuine capacity pressure (256 slots vs ~11k
> keys) the honest Phase-1 numbers are: C++ reference prefetch test ≈ 96.7-96.8%
> decode hit rate, and 87-95% on shorter capacity-bound traces (hit rate grows
> with decode length as the hot set stabilizes). The Phase-2 harness asserts
> against these capacity-bound numbers, not 98%.

### Non-Goals (deferred to Phase 3)
- Wiring into SGLang's `DeepSeekV4ForCausalLM` / `DeepseekV2MoE` forward pass.
- Breakable-graph / duplicate-router surgery inside SGLang decode.
- Training / RL refit `update_param` coherence.
- DP/EP multi-rank orchestration.

---

## 3. Architecture

Three new WebAssembly-free, torch-only components in the existing
`python/sglang/srt/layers/moe/expert_cache/` package plus the standalone
harness in `test/registered/unit/layers/moe/`.

### 3.1 `StaticExpertPool`

Owns the fixed VRAM address space and the transfer stream.

- **Twin ML-shaped pools** (approved decision — zero reshape at the GEMM site);
  dtype tracks the model's quantized weights (FP4/FP8 mix; the harness
  exercises FP8 blocks to prove the quantized path):
  - `pool_w13: Tensor[num_slots, H, 2*I]`
  - `pool_w2:  Tensor[num_slots, I, H]`
- **One CUDA event per slot** *plus* a **single reusable `step_event`** —
  all created in `__init__` (outside capture).
- **Transfer stream** leased once at init (`get_stream("offload")` idiom,
  `runtime_context.py:749`), never created inside capture.
- No method allocates or creates an event after `__init__`.

```python
class StaticExpertPool:
    def __init__(self, model: ModelSpec, hw: HardwareSpec,
                 backend: CudaTransferBackend, num_slots: int, /):
        """Allocate pools + events + lease transfer stream. OUTSIDE capture."""

    @property
    def num_slots(self) -> int: ...

    def pool_w13(self) -> torch.Tensor:   # [N, H, 2I] model weight dtype, fixed address
    def pool_w2(self) -> torch.Tensor:    # [N, I, H]  model weight dtype, fixed address

    def copy_in(self, slot_id: int,
                w13: torch.Tensor, w2: torch.Tensor) -> None:
        """Async H2D of one expert (pinned CPU -> static slot) on the transfer
        stream, non_blocking. No allocation, no event creation."""

    def record_step(self) -> bool:
        """Record step_event on the transfer stream after all demand copies for
        this step are enqueued. Returns True if any copy was issued this step."""

    def wait_all(self) -> None:
        """Eager-time host block until in-flight copies complete (tests/shutdown)."""

    def reset(self) -> None:
        """Zero pools + event state; called once before capture warmup so the
        graph trains against the real static addresses."""
```

### 3.2 `CudaTransferBackend` (Phase-1 backend, repurposed)

`load()` now issues `pool.copy_in(...)` instead of allocating ad-hoc tensors.
`Slot.addr` becomes the pool's *slot id* (an int), not a device tensor — the
pool owns the actual storage.

```python
class CudaTransferBackend(TransferBackend):
    def set_pool(self, pool: StaticExpertPool) -> None: ...
    def load(self, key, nbytes, slot, on_done=None) -> float:
        # pool.copy_in(slot.slot_id, w13_cpu, w2_cpu) on transfer stream
    def evict(self, slot) -> None:   # release slot/event, never touch addresses
```

### 3.3 `StaticPoolOrchestrator` (Phase A adapter)

Host-side glue between the **unchanged** Phase-1 `ExpertCache`/policies and the
pool. The engine loop calls these every decode step between graph replays.

```python
class StaticPoolOrchestrator:
    def on_router_output(self, layer, expert_ids, logit, next_ids) -> None:
        """Phase A: cache.acquire for each routed (layer, expert_id); hit ->
        slot_id, miss -> ExpertCache picks the victim slot internally, then
        pool.copy_in(victim_slot, pinned CPU). Prefetches next_ids."""
    def step_commit(self, cache: ExpertCache) -> None:
        """pool.record_step() if any miss was issued this step."""

    def slot_map_tensor(self) -> torch.Tensor:
        """Static (expert_id -> slot_id | -1) int tensor the graph reads,
        written to a pre-replay static buffer. -1 is never committed past
        step_commit (see section 6)."""
```

**Contract:** `ExpertCache` (Phase 1) is untouched — it owns victim selection,
admission, refcounting, and the busy-slot rule. The orchestrator + pool add
the fixed-address and capture-safe layers on top; the orchestrator only maps
slots to copies and exposes the static slot map. The eventual DeepSeek
interception (Phase 3) replaces the FusedMoE weight source with "gather
`pool_w13/2[slot_map[topk_ids]]`".

---

## 4. Data Flow (Phase A / Phase B)

```
decode step N
  PHASE A (eager, host, outside graph)
    router output (from harness's eager pre-pass)     -> expert_ids, logits
    cache.acquire(layer, expert_id) for each
         hit  -> slot_id (resident)
         miss -> policy picks victim slot, pool.copy_in(slot, pinned CPU)
                slot_map[expert_id] = slot_id
    orchestrator.prefetch(next expected experts)      -> pool.copy_in on transfer stream
    for each next-layer candidate : cache.prefetch(...)
    pool.record_step()                                -> step_event.record(transfer_stream)

  PHASE B (captured / replayed CUDA graph)
    ... attention ... etc ...
    captured wait_event(step_event)        # baked once at capture time; waits on
                                           # the LATEST record at every replay
    topk_ids -> slot_ids via slot_map static buffer
    out = bmm(activations, pool_w13[slot_ids])  +  bmm(..., pool_w2[slot_ids])
    ... residual ...
```

**Barrier semantics (approved): single per-step event.** One `step_event`,
re-recorded each Phase A on the transfer stream; one `wait_event(step_event)`
baked into the graph at the start of the pool-consuming region. Replay waits on
the latest record — the standard "producer-stream event re-recorded at runtime,
consumer-baked wait" pattern. If no miss occurred (`record_step()` returned
False), the event is not re-recorded this step, so the wait is a no-op.

---

## 5. Synchronization / Prefetch Overlap

- **Transfer stream** is free during the entire replay → prefetches for the
  *next* expected experts execute concurrently with the current graph.
- **Prefetch targets** are constrained by the Phase-1 busy-slot rule (a slot
  with refcount > 0 or in flight is never a prefetch/eviction victim), so a
  prefetch can never overwrite a slot the current step's graph is about to read.
- **Demand copies** are gated by the per-step barrier: they are enqueued in
  Phase A *before* `record()`, so the graph waits only for what it actually
  needs this step.
  **GPU-verified amendment (2026-08-21):** the barrier wait cannot live
  *inside* the captured graph — a captured stream may not create a dependency
  on uncaptured transfer-stream work (`cudaErrorStreamCaptureIsolation`;
  persists under `capture_error_mode="relaxed"`). Copy→read ordering is
  enforced at **replay time** instead (host waits on the step event /
  `pool.wait_all()` before `g.replay()`; the Phase-A orchestrator's slot-map
  update is host-side anyway). `record_step` remains the producer-side
  barrier primitive for eager paths.
- **Miss stall:** a demand miss is a synchronous stall by design (rare, ~2%).
  Prefetch hides most of it; the barrier never adds latency beyond the copy.

---

## 6. Error Handling

| Case | Behavior |
|------|----------|
| `slot_map[expert_id] == -1` at graph time | Bug — orchestrator must not commit a step until all routed experts are resident or prepped. Harness asserts. |
| No evictable slot (all busy this layer) | Phase-1 `_load_transient` path in `ExpertCache`; orchestrator must not map it into a graph slot for a step it commits. |
| `copy_in` before `set_pool` called | `AssertionError` in `CudaTransferBackend.load`. |
| Out-of-range slot / expert id | Bounds-checked `slot_map` build; raises in orchestrator (host-side). |
| Illegal op during capture | Harness runs a `torch.cuda.graph` capture of a fake MoE step; any allocation/event-creation during capture fails loudly in the test. |

---

## 7. Testing Strategy

Standalone harness (no SGLang server), under `test/registered/unit/layers/moe/
test_static_expert_pool.py` (GPU-gated, `register_cuda_ci`), plus CPU-able unit
tests for the slot-map / orchestrator logic:

1. **Unit, CPU:** slot-map construction, victim selection via `ExpertCache`,
   `record_step` bookkeeping (with `SimBackend`).
2. **Unit, CUDA:** pool allocation, `copy_in` correctness (copy an expert block
   in, read it back, byte-compare vs the pinned source), `wait_all`.
3. **Eager decode harness (no graph):** fake MoE (256 experts, 100 decode
   steps, Zipf trace) through the pool → assert decode hit rate matches the
   Phase-1 simulator under the same capacity-bound config (see hit-rate note),
   not on a guest-of-VRAM config.
4. **Captured decode harness (the key test):** `torch.cuda.graph` capture a
   fake single-token decode that reads `pool_w13/2[slot_map[topk_ids]]` via
   gather + bmm, against real pre-allocated pool addresses; replay multiple
   steps with changing slot maps and re-copied slot contents → assert correct
   outputs (no illegal memory access, no capture errors, `cuda.synchronize`).

---

## 8. Out of Scope (Phase 3 Preview)

- Intercepting `DeepseekV2MoE.forward_normal/dual_stream` (`deepseek_v2.py:970-1006`):
  replace the expert weight source with the pool's `slot_map[topk_ids]` gather.
- Resolving the "router runs inside the graph" problem for real SGLang decode
  (per-layer break point or duplicate-router pre-pass).
- FusedMoE triton-runner integration (`pool_w13[slot_ids]` as the weight input).
- `--moe-cache-size-gb` / `--moe-cache-logit-threshold` server flags.

---

## 9. Decisions Log

| # | Decision | Rationale |
|---|----------|-----------|
| 1 | Twin ML-shaped pools (model weight dtype, FP4/FP8 mix) | Matches real GEMM shapes, zero reshape/view inside capture. |
| 2 | Single per-step barrier event | Per-slot/per-layer events across 43 layers = sync nightmare for negligible latency; PCIe serializes copies anyway. |
| 3 | Phase A eager router pre-pass (harness-controlled) | Only way to know topk before the graph; fine for the standalone harness. |
| 4 | `ExpertCache` unchanged | Phase-1 policy/admission logic already validated; pool+orchestrator add capture-safety without re-deriving policy behavior. |
| 5 | Slot-map indirection, not baked-index gather | The graph must be able to re-point slots between replays; a static int buffer rewritten each step is captured once and re-filled. |
