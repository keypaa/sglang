# MoE Expert Cache Phase 3 M3 (CUDA-graph-safe) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development or superpowers:executing-plans. Steps use checkbox (`- [ ]`) syntax.

**Goal:** Run the expert cache under segmented CUDA graphs (tc_piecewise): a per-replay eager gap streams routed experts and remaps ids before the captured MoE piece reads pool rows.

**Architecture:** Hook into upstream's existing MoE split boundary (`sglang.moe_forward_piecewise_cuda_graph_impl` region). Guard requires decode backend ∈ {disabled(eager), tc_piecewise} when cache on. Task 0 probes the load-bearing assumption empirically before product code.

**Tech Stack:** as M1/M2 + sglang compilation/piecewise machinery (`compilation/cuda_piecewise_backend.py`, `runner_backend/tc_piecewise_cuda_graph_backend.py`, `register_custom_op`).

## Global Constraints

- Spec: `docs/superpowers/specs/2026-08-22-expert-cache-phase3-m3-cudagraph-design.md`
- Task 0 is GO/NO-GO: if the split-op body does not re-execute per replay, STOP and report — the milestone pivots to appendix A (breakable extension) by decision of the controller/human.
- No weight bytes move inside any captured graph; residency fence stays outside capture.
- bf16 cache-off behavior unchanged; M1/M2 eager paths unchanged.
- Local runs CPU-only (`PYTHONPATH=python python -m pytest ... -q`); GPU work via Modal L4 harness `/tmp/opencode/modal_gpu_verify.py`.
- Phase-1/2 suites stay green after every task.

---

### Task 0: Piecewise replay probe (GO/NO-GO, GPU)

**Files:** throwaway probe script on Modal (no repo changes).

- [ ] Build a minimal tc_piecewise decode run of the tiny DeepSeek config (reuse parity-test construction patterns under an Engine or graph-runner shim; simplest that exercises `moe_forward_piecewise_cuda_graph_impl`).
- [ ] Monkeypatch/instrument a module-level counter + perf_counter around the MoE impl body.
- [ ] Run ≥5 decode steps; assert counter advanced per step; record mean eager-gap cost.
- [ ] Report GO (counter advances per step) or NO-GO to the controller. Commit nothing.

### Task 1: Guard update (CPU)

- Modify `validate_moe_expert_cache` (+ model-init guard): accept cache-on when decode backend is tc_piecewise; reject full/breakable with a message naming allowed options. Pure-helper update + unit tests in `test_expert_cache_server_args.py` / wiring guard tests.
- Steps: failing tests → implement → pass → commit `feat(expert_cache): allow tc_piecewise decode graphs under expert cache`.

### Task 2: Prepare-hook split-op (CPU-testable logic)

- Create `expert_cache/piecewise_hook.py`: `expert_cache_prepare(topk_ids, layer_id) -> slot_ids` custom op (mirror `register_custom_op(out_shape=...)` style of `moe_forward_piecewise_cuda_graph_impl`); body = tolist → ensure_resident → wait fence → `copy_slot_map_into(device_map)` → `remap_topk_ids`. Runtime lookup via a registry the model populates (`_EXPERT_CACHE_RUNTIMES[layer_id]`, set in `_finish_expert_cache_wiring`).
- Unit tests: fake runtime records calls; ids→remap correctness; missing-runtime raises.
- Commit `feat(expert_cache): per-replay prepare hook for piecewise MoE`.

### Task 3: Model wiring for tc_piecewise

- In `DeepseekV2MoE.forward` tc_piecewise branch (where `is_in_tc_piecewise_cuda_graph()`), route topk_ids through the hook when cache enabled; register/unregister runtimes in `_build/_finish/_reset` infra functions.
- CUDA-gated smoke: construct under tc_piecewise flags; assert hook path taken (counter).
- Commit `feat(expert_cache): wire prepare hook into piecewise DeepseekV2MoE forward`.

### Task 4: E2E engine parity gate (GPU, THE gate)

- New test `test_deepseek_moe_expert_cache_engine_parity.py` (CUDA-gated): launch tiny-model `Engine` twice — (a) cache off + tc_piecewise, (b) cache on + tc_piecewise — same prompts; assert identical output token sequences; bf16 AND fp8 variants. Reuse tiny-checkpoint approach from parity tests; engine args include `cuda_graph_backend_decode="tc_piecewise"`.
- Modal run; fix rounds as needed. Commit `test(expert_cache): engine-level parity gate with tc_piecewise graphs`.

### Task 5: Docs + ledger

- Update `docs/expert-cache.md` constraints (graphs-on option), telemetry note, gate results. Commit `docs(expert_cache): M3 cuda-graph-safe constraints`.

## Self-Review Notes

- Spec coverage: §2.1 guards→T1; hook→T2; wiring→T3; §3 gates→T0/T4; docs→T5.
- Risk: Engine-API test complexity is the biggest unknown; fallback = drive the scheduler manually like existing engine CI tests do (grep `run_engine_test` / engine fixtures before writing T4).
- Type consistency: hook signature `(topk_ids, layer_id)` consistent T2↔T3; runtime registry keyed by layer_id consistent T2↔T3↔T4.
