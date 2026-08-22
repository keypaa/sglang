# MoE Expert Cache Phase 3 M2 (fp8 streaming) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Stream fp8 block-quant expert weights (payload + scale_inv) from the pinned host store into pool rows, lifting M1's quant rejection for the safe CUDA/triton case.

**Architecture:** No new components — scales ride the existing pipeline. Checkpoint layout IS compute layout on the triton path (see spec exploration findings), so `copy_in` gains two optional scale args and everything downstream is byte-plumbing.

**Tech Stack:** as M1 plus `per_block_cast_to_fp8` (`fp8_utils.py`) for offline quantization in tests, `Fp8Config`.

## Global Constraints

- Spec: `docs/superpowers/plans/2026-08-22-expert-cache-phase3-m2-fp8-spec.md`
- bf16 stores must keep working unchanged (backward compat everywhere).
- Guards: fp8 allowed ONLY when block-quant + checkpoint-serialized + CUDA non-fnuz + not flashinfer-trtllm + not deepgemm-runner; reject otherwise with actionable messages.
- Tests CPU-runnable where possible; run via `PYTHONPATH=python python -m pytest ... -q` from repo root. Phase-2 four-file suite stays 21 passed / 7 skipped after every task.
- Surgical edits to deepseek_v2.py only at the established anchor points.
- TDD mandatory in every task: failing test observed before implementation.

---

### Task 1: HostStore scale support

**Files:**
- Modify: `python/sglang/srt/layers/moe/expert_cache/host_store.py`
- Test: extend `test/registered/unit/layers/moe/test_expert_host_store.py`

**Interfaces:**
- Produces: `HostStoreEntry(w13, w2, w13_scale_inv=None, w2_scale_inv=None)`;
  `put(key, w13, w2, w13_scale_inv=None, w2_scale_inv=None)`; `expert_bytes` includes
  scales when present. Backward compat: two-arg puts still work.

- [ ] **Step 1: Failing tests** — (a) put+get with scales round-trips and entries carry them; (b) `total_bytes` includes scale bytes; (c) two-arg put leaves scales None.
- [ ] **Step 2: Run — fail on unexpected keyword/attribute.**
- [ ] **Step 3: Implement** — extend NamedTuple + put signature; pin scales like weights; account bytes only when present.
- [ ] **Step 4: PASS.**
- [ ] **Step 5: Commit** — `feat(expert_cache): host store carries fp8 scale blocks`

### Task 2: Parser scale kinds

**Files:**
- Modify: `python/sglang/srt/models/deepseek_v2.py` (`parse_expert_weight_name`)
- Test: extend parser test in `test_deepseek_moe_expert_cache_wiring.py`

**Interfaces:**
- Produces: parser returns `(idx, "w13_scale")` / `(idx, "w2_scale")` for
  `...w13_weight_scale_inv` / `...w2_weight_scale_inv`; existing kinds unchanged;
  unrelated names still None (e.g. `.weight_scale` without `_inv`, shared_experts).

- [ ] **Step 1: Failing parser tests** (4 new asserts).
- [ ] **Step 2: FAIL.**
- [ ] **Step 3: Implement** — regex alternative `(w13_weight|w2_weight|w13_weight_scale_inv|w2_weight_scale_inv)$`; map suffix → kind.
- [ ] **Step 4: PASS** (+ existing parser tests green).
- [ ] **Step 5: Commit** — `feat(expert_cache): intercept fp8 scale_inv tensors by name`

### Task 3: Pool/runtime scale plumbing

**Files:**
- Modify: `python/sglang/srt/layers/moe/expert_cache/layer_runtime.py`
- Test: extend `test_expert_cache_layer_runtime.py`

**Interfaces:**
- Produces:
  - `RealWeightPool(w13_weight, w2_weight, transfer_stream=None, w13_scale=None, w2_scale=None)` — scale params optional;
  - `copy_in(slot_id, w13, w2, w13_scale_inv=None, w2_scale_inv=None)` — asserts scale shapes/dtypes against pool scale params when provided; raises if scales given but pool has none, or vice versa;
  - `LayerRuntime(..., w13_scale_param=None, w2_scale_param=None)`; `ensure_resident` forwards `entry.w13_scale_inv/w2_scale_inv` through copy_in;
  - residency byte-gate helper extended to scales (used by tests).

- [ ] **Step 1: Failing tests** — fp8 CPU store entry (quantize small tensor via `torch.ops` free approach: build fp8 payload manually or use `per_block_cast_to_fp8` from `sglang.srt.layers.quantization.fp8_utils`) streamed into a pool WITH scale params: weight row AND scale row equal store bytes; mismatched-scale-shape raises; bf16 store into scale-less pool unchanged behavior.
- [ ] **Step 2: FAIL.**
- [ ] **Step 3: Implement** per interfaces above (thread scales through ensure_resident's copy loop and both copy paths — backend-copies path stays as-is since CudaTransferBackend handles only M1 pools; SimBackend path does manual copy_in which now carries scales).
- [ ] **Step 4: PASS** (+ all prior layer-runtime tests green).
- [ ] **Step 5: Commit** — `feat(expert_cache): stream fp8 scales alongside expert payloads`

### Task 4: Guard relaxation + wiring threading

**Files:**
- Modify: `python/sglang/srt/models/deepseek_v2.py` (init cache branch ~673-690; `_finish_expert_cache_wiring` ~3303-3330)
- Test: wiring smoke extension (CUDA-gated)

**Interfaces:**
- Produces: cache-on init accepts `Fp8Config` with block-quant + serialized checkpoint (rejects flashinfer-trtllm format, fnuz/ROCm platform, deepgemm runner combo — each with named reason); `_finish_expert_cache_wiring` passes `experts.w13_weight_scale_inv`/`w2_weight_scale_inv` into LayerRuntime when present; pairing check requires scale tensors intercepted whenever fp8 weights are (else loud ValueError at wiring time).

- [ ] **Step 1: Failing guard unit tests (CPU):** construct guard logic as a pure function if not already (e.g. reuse validate-style helper or test via init smoke with patched args); assert accept/reject matrix: {no quant → OK}, {fp8 block+serialized → OK}, {fp8 per-tensor → raise}, {fp8 block but trtllm-format flag → raise}. If init-level testing is impractical locally, gate on the CUDA smoke like M1 Task 5 did and verify collection locally.
- [ ] **Step 2: FAIL.**
- [ ] **Step 3: Implement** guard relaxation + scale-param threading + pairing validation in wiring.
- [ ] **Step 4: Local suite green (smoke skips).**
- [ ] **Step 5: Commit** — `feat(expert_cache): allow fp8 block-quant checkpoints under expert cache`

### Task 5: fp8 parity gate (GPU)

**Files:**
- Test: extend `test_deepseek_moe_expert_cache_parity.py` (new test class)

**Structure (implement fully):**
- Build tiny config as M1 parity (hidden=inter=128 to exercise multi-block scales? NO — hidden=64/inter=64 keeps blocks at [1,1]; ALSO add a second case hidden=256/inter=128 for real multi-block scales if VRAM trivially allows — it does at these sizes).
- Model A: full-size experts, Fp8Config(block_quant serialized). Model B: same config + cache on, slots=4.
- Weights: take A's UNQUANTIZED expert matrices? No — A must itself be an fp8 model for honest comparison. Construct A with Fp8Config and load RANDOM bf16 weights through its normal loader so A's own process_weights path runs; then extract A's post-load w13/w2/scale rows as "checkpoint" tensors for B (they are already in kernel layout), route through B's interception; forward identical seeded tokens; assert allclose(atol=2e-2, rtol=2e-2) + ≥8 greedy argmax matches; byte-gate store↔A-params per (layer, expert, kind incl. scales).
- Local: collects + skips cleanly.

- [ ] **Step 1: Write test. Step 2: local skip-check. Step 3: commit** — `test(expert_cache): fp8 tiny-model parity gate`
- [ ] **Step 4: controller runs Modal L4 verification; failures come back as fix rounds.**

### Task 6: Docs

- Update `docs/expert-cache.md`: constraints section — fp8 block-quant supported (CUDA/triton/non-fnuz), rejected combos listed; flags unchanged; telemetry unchanged. Note split-name blocker remains.
- [ ] **Commit** — `docs(expert_cache): M2 fp8 streaming constraints`

## Self-Review Notes

- Spec coverage: guards→T4; store→T1; parser→T2; plumbing→T3; parity→T5; docs→T6. Split-name blocker explicitly out of scope (separate milestone).
- Type consistency: kind strings ("w13_scale"/"w2_scale") consistent T2↔T4; HostStoreEntry fields consistent T1↔T3↔T5; copy_in signature consistent T3↔T5.
- Risk flagged: Fp8Config construction details in tests (activation scheme args) may need adjustment — implementer should read `Fp8Config.__init__` signature first.
