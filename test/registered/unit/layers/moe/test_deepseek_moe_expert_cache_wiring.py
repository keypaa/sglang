"""Wiring smoke: tiny DeepseekV2MoE constructs with the cache enabled."""

import os
import unittest
from types import SimpleNamespace

import torch

from sglang.srt.layers.moe.expert_cache import (
    ExpertCache,
    ExpertHostStore,
    ExpertKey,
    HardwareSpec,
    ModelSpec,
    SimBackend,
    make_policy,
)
from sglang.srt.layers.moe.expert_cache.layer_runtime import LayerRuntime
from sglang.srt.models.deepseek_v2 import DeepseekV2MoE
from sglang.srt.runtime_context import get_context, get_parallel
from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.test_utils import CustomTestCase

register_cuda_ci(est_time=60, suite="base-a-test-1-gpu-small")

_NUM_EXPERTS = 8
_NUM_SLOTS = 4
_HIDDEN = 32
_INTER = 32


def _free_master_port() -> str:
    import socket

    s = socket.socket()
    s.bind(("127.0.0.1", 0))
    port = str(s.getsockname()[1])
    s.close()
    return port


def _tiny_config() -> SimpleNamespace:
    # Minimal DeepseekV2Config-like namespace for DeepseekV2MoE.__init__.
    return SimpleNamespace(
        hidden_size=_HIDDEN,
        intermediate_size=_INTER,
        moe_intermediate_size=_INTER,
        n_routed_experts=_NUM_EXPERTS,
        num_experts_per_tok=2,
        n_shared_experts=None,
        routed_scaling_factor=1.0,
        hidden_act="silu",
        vocab_size=128,
        num_hash_layers=0,
        n_group=1,
        topk_group=1,
        topk_method="noaux_tc",
        scoring_func="sigmoid",
        norm_topk_prob=False,
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
    )


def _cache_server_args():
    # get_exec() config bags (moe.disable_shared_experts_fusion etc.) are
    # projected from the published ServerArgs, so one override covers both
    # get_server_args() and get_exec() reads inside DeepseekV2MoE.
    return get_context().override_server_args(
        enable_moe_expert_cache=True,
        moe_cache_slots=_NUM_SLOTS,
        disable_cuda_graph=True,
        disable_shared_experts_fusion=True,
        ep_num_redundant_experts=0,
    )


def _build_tiny_test_moe() -> DeepseekV2MoE:
    # torch.device("cuda") places every created param on GPU; forward_normal
    # feeds CUDA hidden states through the gate, which would otherwise hit a
    # cpu-vs-cuda matmul error.
    with _tp1_parallel(), _cache_server_args(), torch.device("cuda"):
        return DeepseekV2MoE(
            config=_tiny_config(),
            layer_id=0,
            quant_config=None,
            prefix="model.layers.0.mlp",
        )


class TestParseExpertWeightName(CustomTestCase):
    def test_matches(self):
        from sglang.srt.models.deepseek_v2 import parse_expert_weight_name

        self.assertEqual(
            parse_expert_weight_name("model.layers.3.mlp.experts.17.w13_weight"),
            (17, "w13"),
        )
        self.assertEqual(
            parse_expert_weight_name("model.layers.3.mlp.experts.0.w2_weight"),
            (0, "w2"),
        )
        self.assertEqual(
            parse_expert_weight_name(
                "model.layers.3.mlp.experts.17.w13_weight_scale_inv"
            ),
            (17, "w13_scale"),
        )
        self.assertEqual(
            parse_expert_weight_name(
                "model.layers.3.mlp.experts.0.w2_weight_scale_inv"
            ),
            (0, "w2_scale"),
        )

    def test_non_expert_names(self):
        from sglang.srt.models.deepseek_v2 import parse_expert_weight_name

        self.assertIsNone(
            parse_expert_weight_name(
                "model.layers.3.mlp.shared_experts.gate_up_proj.weight"
            )
        )
        self.assertIsNone(
            parse_expert_weight_name("model.layers.3.self_attn.qkv_proj.weight")
        )
        self.assertIsNone(
            parse_expert_weight_name("model.layers.3.mlp.experts.17.w13_weight_scale")
        )
        self.assertIsNone(
            parse_expert_weight_name("model.layers.3.mlp.experts.17.w2_weight_scale")
        )


@unittest.skipUnless(torch.cuda.is_available(), "CUDA required")
class TestWiringSmoke(CustomTestCase):
    @classmethod
    def setUpClass(cls):
        # FusedMoE.forward_impl reads the TP process group unconditionally
        # (use_symmetric_memory(get_tp_group(), ...)), so a bare module
        # forward needs a minimal single-rank world.
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

    def test_forward_normal_uses_pool_weights(self):
        with _tp1_parallel(), _cache_server_args():
            moe = _build_tiny_test_moe()

            # Pool-as-weights: the experts layer is rebuilt sized to the pool.
            self.assertTrue(moe.expert_cache_enabled)
            self.assertEqual(moe.moe_cache_num_slots, _NUM_SLOTS)
            self.assertEqual(moe.experts.w13_weight.shape[0], _NUM_SLOTS)

            # Task 6 attaches the real shared runtime; for this smoke we
            # attach a minimal LayerRuntime by hand and seed the store with
            # every expert so ensure_resident can stream slots 0..3.
            moe_bf16 = moe.bfloat16()
            w13 = moe_bf16.experts.w13_weight.data
            w2 = moe_bf16.experts.w2_weight.data
            spec = ModelSpec(
                num_layers=1,
                num_experts=_NUM_EXPERTS,
                top_k=2,
                shared_experts=0,
                hidden_size=_HIDDEN,
                intermediate_size=_INTER,
            )
            spec.bytes_on_disk = spec.num_layers * spec.num_experts * 1024
            cache = ExpertCache(
                spec, HardwareSpec(), make_policy("lru", _NUM_SLOTS), SimBackend(1e12)
            )
            store = ExpertHostStore(num_layers=1, num_experts=_NUM_EXPERTS)
            for e in range(_NUM_EXPERTS):
                store.put(
                    ExpertKey(0, e),
                    torch.randn_like(w13[0]),
                    torch.randn_like(w2[0]),
                )
            runtime = LayerRuntime(
                0, cache, store, w13, w2, num_experts=_NUM_EXPERTS, num_layers=1
            )
            moe_bf16.expert_cache_runtime = runtime
            runtime.ensure_resident(list(range(_NUM_SLOTS)))

            torch.manual_seed(0)
            hidden_states = torch.randn(3, _HIDDEN, device="cuda", dtype=torch.bfloat16)
            output = moe_bf16.forward_normal(hidden_states, skip_shared_experts=True)
            self.assertEqual(output.shape, (3, _HIDDEN))


if __name__ == "__main__":
    unittest.main()
