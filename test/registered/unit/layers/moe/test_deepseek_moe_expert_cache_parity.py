"""Tiny-model parity gate: cached (pool-as-weights) vs uncached DeepseekV2.

Builds ONE tiny DeepseekV2ForCausalLM twice from identical seeds:
  A: enable_moe_expert_cache=False (dense routed-expert weights)
  B: enable_moe_expert_cache=True, moe_cache_slots=4 (pool-sized weights +
     host-store streaming through the REAL load_weights interception path)

A's state_dict is routed through B.load_weights so B's fused expert tensors
(``mlp.experts.{i}.w13_weight`` / ``w2_weight``) land in the pinned host store
via interception while everything else loads normally. The test then asserts

1. store bytes are bitwise-equal to A's expert weights (a mismatch here is a
   bug, not tolerance), and
2. final logits agree (allclose atol=rtol=2e-2) and greedy argmax
   continuations match (>= 8 tokens identical) over multiple steps.

The forward drives the real embeddings -> per-layer MoE (through B's
ensure_resident/slot-remap path) -> final norm -> lm_head chain. Only
self_attn is bypassed: its weights and code are byte-identical on both sides,
so any divergence isolates to the expert-cache pool-as-weights path.
"""

import os
import unittest
from types import SimpleNamespace

import torch

from sglang.srt.layers.moe.expert_cache import ExpertKey
from sglang.srt.models.deepseek_v2 import DeepseekV2ForCausalLM
from sglang.srt.runtime_context import get_context, get_parallel
from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.test_utils import CustomTestCase

register_cuda_ci(est_time=240, suite="base-a-test-1-gpu-small")

_NUM_EXPERTS = 8
_NUM_SLOTS = 4
_HIDDEN = 64
_INTER = 64
_NUM_LAYERS = 2
_VOCAB = 128
_TOP_K = 2
_BATCH = 4
_GREEDY_STEPS = 10
_MIN_MATCHED_TOKENS = 8
_SEED_WEIGHTS = 1234
_SEED_TOKENS = 42


def _free_master_port() -> str:
    import socket

    s = socket.socket()
    s.bind(("127.0.0.1", 0))
    port = str(s.getsockname()[1])
    s.close()
    return port


def _tiny_config() -> SimpleNamespace:
    return SimpleNamespace(
        architectures=["DeepseekV2ForCausalLM"],
        hidden_size=_HIDDEN,
        intermediate_size=_INTER,
        moe_intermediate_size=_INTER,
        n_routed_experts=_NUM_EXPERTS,
        num_experts_per_tok=_TOP_K,
        n_shared_experts=None,
        routed_scaling_factor=1.0,
        hidden_act="silu",
        vocab_size=_VOCAB,
        num_hidden_layers=_NUM_LAYERS,
        num_hash_layers=0,
        n_group=1,
        topk_group=1,
        topk_method="noaux_tc",
        scoring_func="sigmoid",
        norm_topk_prob=False,
        first_k_dense_replace=0,
        moe_layer_freq=1,
        q_lora_rank=None,
        kv_lora_rank=32,
        qk_nope_head_dim=16,
        qk_rope_head_dim=16,
        v_head_dim=16,
        num_attention_heads=2,
        rope_theta=10000.0,
        rope_scaling=None,
        max_position_embeddings=512,
        rms_norm_eps=1e-6,
        tie_word_embeddings=False,
        pad_token_id=0,
    )


def _tp1_parallel():
    return get_parallel().override(
        tp_size=1,
        tp_rank=0,
        attn_tp_size=1,
        attn_tp_rank=0,
        moe_tp_size=1,
        moe_tp_rank=0,
        moe_ep_size=1,
        moe_ep_rank=0,
        # Decoder-layer LayerCommunicator reads attn_dp_size; keep it pinned
        # to 1 so it never consults uninitialized DP-attention state.
        attn_dp_size=1,
        attn_dp_rank=0,
    )


def _server_args(cache_enabled: bool):
    # get_exec() config bags are projected from the published ServerArgs, so
    # one override covers get_server_args()/get_exec()/get_model() reads.
    return get_context().override_server_args(
        enable_moe_expert_cache=cache_enabled,
        moe_cache_slots=_NUM_SLOTS,
        disable_cuda_graph=True,
        disable_shared_experts_fusion=True,
        ep_num_redundant_experts=0,
    )


def _build_model(cache_enabled: bool) -> DeepseekV2ForCausalLM:
    torch.manual_seed(_SEED_WEIGHTS)
    with _tp1_parallel(), _server_args(cache_enabled), torch.device("cuda"):
        model = DeepseekV2ForCausalLM(
            config=_tiny_config(),
            quant_config=None,
            prefix="",
        )
    model = model.to(torch.bfloat16)
    # Overwrite every param (some linear layers are torch.empty until load)
    # with deterministic values so A and B start bitwise-identical regardless
    # of allocator garbage or B's extra pool-rebuild RNG consumption.
    torch.manual_seed(_SEED_WEIGHTS + 1)
    for _, param in sorted(model.named_parameters()):
        param.data.copy_(torch.randn_like(param.data))
    return model


def _checkpoint_items(a_state_dict):
    """Split A's state_dict into (items_for_A, items_for_B).

    A receives the canonical split per-expert checkpoint names
    (`mlp.experts.{e}.{gate,up,down}_proj`) so they route through the real
    ``expert_params_mapping`` into its dense fused params. B receives fused
    per-expert names (`mlp.experts.{e}.w13_weight` / `w2_weight`) so they flow
    through B's real load_weights interception into the host store.
    """
    w13_ref, w2_ref = {}, {}
    for name, tensor in a_state_dict.items():
        if name.endswith("mlp.experts.w13_weight"):
            w13_ref[int(name.split(".")[2])] = tensor
        elif name.endswith("mlp.experts.w2_weight"):
            w2_ref[int(name.split(".")[2])] = tensor

    items_a, items_b = [], []
    for name, tensor in a_state_dict.items():
        if name.endswith("mlp.experts.w13_weight") or name.endswith(
            "mlp.experts.w2_weight"
        ):
            kind = "w13_weight" if name.endswith("w13_weight") else "w2_weight"
            layer_id = int(name.split(".")[2])
            for e in range(_NUM_EXPERTS):
                items_b.append(
                    (f"model.layers.{layer_id}.mlp.experts.{e}.{kind}", tensor[e])
                )
        else:
            items_a.append((name, tensor))
            items_b.append((name, tensor))

    for i in range(_NUM_LAYERS):
        for e in range(_NUM_EXPERTS):
            stem = f"model.layers.{i}.mlp.experts.{e}"
            items_a.append((f"{stem}.gate_proj.weight", w13_ref[i][e][:_INTER]))
            items_a.append((f"{stem}.up_proj.weight", w13_ref[i][e][_INTER:]))
            items_a.append((f"{stem}.down_proj.weight", w2_ref[i][e]))
    return items_a, items_b


def _moe_stack_logits(model: DeepseekV2ForCausalLM, tokens: torch.Tensor):
    """embed -> per-layer (post_attention_layernorm -> MoE -> residual add)
    -> final norm -> lm_head. Attention is bypassed (identical weights/code
    on both sides); this still routes every token through the model's real
    MoE stack, which is the entire surface under test."""
    # DeepseekV2MoE.forward reads get_server_args() to pick its dispatch
    # path, so each forward must run under that model's server-args override.
    cache_enabled = model.model.layers[0].mlp.expert_cache_enabled
    hidden = model.model.embed_tokens(tokens)
    with _server_args(cache_enabled):
        for layer in model.model.layers:
            x = layer.post_attention_layernorm(hidden)
            # skip_shared_experts: this tiny config has none (n_shared_experts
            # None); identical on both sides so parity is unaffected.
            hidden = hidden + layer.mlp(x, skip_shared_experts=True)
        final = model.model.norm(hidden)
    return torch.nn.functional.linear(final, model.lm_head.weight)


@unittest.skipUnless(torch.cuda.is_available(), "CUDA required")
class TestTinyModelParity(CustomTestCase):
    @classmethod
    def setUpClass(cls):
        # Minimal single-rank world: FusedMoE.forward_impl reads the TP
        # process group unconditionally via use_symmetric_memory.
        os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
        os.environ.setdefault("MASTER_PORT", _free_master_port())
        os.environ.setdefault("RANK", "0")
        os.environ.setdefault("WORLD_SIZE", "1")
        os.environ.setdefault("LOCAL_RANK", "0")

        from sglang.srt.distributed.parallel_state import (
            init_distributed_environment,
            initialize_model_parallel,
            model_parallel_is_initialized,
        )

        if not torch.distributed.is_initialized():
            init_distributed_environment(
                world_size=1, rank=0, local_rank=0, backend="gloo"
            )
        if not model_parallel_is_initialized():
            initialize_model_parallel(
                tensor_model_parallel_size=1,
                expert_model_parallel_size=1,
                pipeline_model_parallel_size=1,
                backend="gloo",
            )

    def test_cached_matches_uncached_logits(self):
        model_a = _build_model(cache_enabled=False)
        model_b = _build_model(cache_enabled=True)

        # Pool-as-weights took effect on B only.
        self.assertFalse(model_a.model.layers[0].mlp.expert_cache_enabled)
        self.assertTrue(model_b.model.layers[0].mlp.expert_cache_enabled)
        self.assertEqual(
            model_a.model.layers[0].mlp.experts.w13_weight.shape[0], _NUM_EXPERTS
        )
        self.assertEqual(
            model_b.model.layers[0].mlp.experts.w13_weight.shape[0], _NUM_SLOTS
        )

        items_a, items_b = _checkpoint_items(model_a.state_dict())
        with _tp1_parallel(), _server_args(cache_enabled=False):
            model_a.load_weights(items_a)
        with _tp1_parallel(), _server_args(cache_enabled=True):
            model_b.load_weights(items_b)

        # Byte gate BEFORE tolerance: every store entry must be bitwise-equal
        # to A's expert weights (interception fidelity). Every cache-layer
        # runtime must be wired (flag-on-but-unwired would raise on forward).
        w13_ref = {
            i: model_a.state_dict()[f"model.layers.{i}.mlp.experts.w13_weight"]
            for i in range(_NUM_LAYERS)
        }
        w2_ref = {
            i: model_a.state_dict()[f"model.layers.{i}.mlp.experts.w2_weight"]
            for i in range(_NUM_LAYERS)
        }
        for i, layer in enumerate(model_b.model.layers):
            runtime = getattr(layer.mlp, "expert_cache_runtime", None)
            self.assertIsNotNone(runtime)
            for e in range(_NUM_EXPERTS):
                entry = runtime.store.get(ExpertKey(i, e))
                self.assertEqual(entry.w13.dtype, w13_ref[i].dtype)
                self.assertTrue(
                    torch.equal(entry.w13.cpu(), w13_ref[i][e].cpu()),
                    f"store w13 bytes != checkpoint bytes (layer {i}, expert {e})",
                )
                self.assertTrue(
                    torch.equal(entry.w2.cpu(), w2_ref[i][e].cpu()),
                    f"store w2 bytes != checkpoint bytes (layer {i}, expert {e})",
                )

        # Parity: identical tokens -> identical logits -> matching greedy chain.
        model_a.eval()
        model_b.eval()
        with torch.no_grad():
            torch.manual_seed(_SEED_TOKENS)
            tokens_a = torch.randint(0, _VOCAB, (_BATCH,), device="cuda")
            tokens_b = tokens_a.clone()
            matched = 0
            total = 0
            for step in range(_GREEDY_STEPS):
                logits_a = _moe_stack_logits(model_a, tokens_a)
                logits_b = _moe_stack_logits(model_b, tokens_b)
                self.assertEqual(logits_a.shape, logits_b.shape)
                self.assertTrue(
                    torch.allclose(logits_a, logits_b, atol=2e-2, rtol=2e-2),
                    f"logits diverge at step {step}: max abs diff "
                    f"{(logits_a - logits_b).abs().max().item():.3e}",
                )
                next_a = logits_a.argmax(dim=-1)
                next_b = logits_b.argmax(dim=-1)
                matched += int((next_a == next_b).sum())
                total += next_a.numel()
                tokens_a = next_a
                tokens_b = next_b
        self.assertGreaterEqual(
            matched,
            _MIN_MATCHED_TOKENS,
            f"greedy continuations diverged: {matched}/{total} argmax tokens equal",
        )


if __name__ == "__main__":
    unittest.main()
