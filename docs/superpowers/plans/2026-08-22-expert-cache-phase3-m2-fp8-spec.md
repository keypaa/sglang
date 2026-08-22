# Phase 3 Milestone 2: fp8 Block-Quant Expert Streaming

**Date:** 2026-08-22
**Status:** Approved (autonomous mandate; supersedes decision-log #5 caution after codebase exploration)
**Builds on:** Phase 3 M1 (pool-as-weights integration, parity-gated)

## Exploration findings (facts driving this design)

1. On CUDA with the triton MoE runner, `Fp8MoEMethod.process_weights_after_loading_block_quant`
   applies **no transform** to plain block-fp8 expert weights (fp8.py:1602-1693 — the only
   call is `requant_block_scale_ue8m0_for_deepgemm`, which is a guaranteed no-op unless
   deepgemm+UE8M0 conditions hold: Blackwell SM100 + JIT deepgemm + bf16 out; and it is
   DeepEPMoE-asserted anyway). Checkpoint layout **is** compute layout.
2. Kernel contract: `w13_weight [E, 2I, H]`, `w2_weight [E, H, I]`, scales
   `[E, ceil(N/128), ceil(K/128)]` fp32 consumed on-the-fly via strides inside the triton
   kernel (dequant never pre-applied).
3. Weight loading is strictly row-local (`param.data[expert_id]` slicing) — per-slot writes
   are exactly how the normal loader works.
4. Hazards that stay rejected: ROCm fnuz normalization + aiter shuffle (elementwise but we
   capture pre-transform bytes), flashinfer-trtllm alignment, mxfp8/fp4 variants,
   per-tensor fp8, deepgemm UE8M0 requant combos.

## Design

Extend M1's pipeline with scale plumbing — no new components:

- **Guards:** init accepts `quant_config` only when it is fp8 block-quant
  (`weight_block_size is not None` and `is_checkpoint_fp8_serialized`); still rejects
  flashinfer-trtllm output-format paths, fnuz/ROCm, and (structurally) deepgemm-runner
  combos. Message names what IS supported.
- **HostStore:** entries carry optional `w13_scale_inv` / `w2_scale_inv` (fp32); byte
  accounting includes them.
- **Parser:** additionally intercepts `w13_weight_scale_inv` / `w2_weight_scale_inv`
  (kinds `w13_scale` / `w2_scale`); pairing validation requires scales when weights are
  fp8.
- **Pool/runtime:** `RealWeightPool.copy_in(slot_id, w13, w2, w13_scale=None,
  w2_scale=None)`; LayerRuntime threads store scales through; residency invariant extends
  to scale rows byte-equal.
- **Parity gate:** tiny-model experts quantized offline per-expert via
  `per_block_cast_to_fp8`; model A runs full-size fp8 uncached, B pool-sized fp8 cached;
  identical quantized bytes through B's real interception; logits + greedy parity as M1.

Out of scope: split HF checkpoint names (separate blocker), deepgemm/UE8M0 targets, ROCm.
