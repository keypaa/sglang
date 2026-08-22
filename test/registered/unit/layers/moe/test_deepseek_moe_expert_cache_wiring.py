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
from sglang.srt.layers.quantization.base_config import QuantizationConfig
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


def _tiny_config(hidden: int = _HIDDEN, inter: int = _INTER) -> SimpleNamespace:
    # Minimal DeepseekV2Config-like namespace for DeepseekV2MoE.__init__.
    return SimpleNamespace(
        hidden_size=hidden,
        intermediate_size=inter,
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


class _DummyQuantConfig(QuantizationConfig):
    """Minimal non-fp8 quant config for the guard's unsupported-type branch."""

    def get_name(self):
        return "dummy"

    def get_supported_act_dtypes(self):
        return [torch.bfloat16]

    @classmethod
    def get_min_capability(cls):
        return 0

    @staticmethod
    def get_config_filenames():
        return []

    @classmethod
    def from_config(cls, config):
        return cls()

    def get_quant_method(self, layer, prefix):
        return None

    def get_scaled_act_names(self):
        return []


def _fp8_block_config(**kwargs):
    from sglang.srt.layers.quantization.fp8 import Fp8Config

    return Fp8Config(
        is_checkpoint_fp8_serialized=True, weight_block_size=[128, 128], **kwargs
    )


def _fp8_per_tensor_config():
    from sglang.srt.layers.quantization.fp8 import Fp8Config

    # Serialized checkpoint, no weight_block_size -> per-tensor scales.
    return Fp8Config(is_checkpoint_fp8_serialized=True)


class TestExpertCacheQuantGuard(CustomTestCase):
    """Accept/reject matrix of the pure init guard `_expert_cache_quant_ok`."""

    KW = dict(flashinfer_trtllm_runner=False, fnuz_platform=False, deep_gemm_runner=False)

    def _call(self, quant_config, **overrides):
        from sglang.srt.models.deepseek_v2 import _expert_cache_quant_ok

        kwargs = dict(self.KW)
        kwargs.update(overrides)
        return _expert_cache_quant_ok(quant_config, **kwargs)

    def test_bf16_no_quant_ok(self):
        self._call(None)

    def test_fp8_block_serialized_ok(self):
        self._call(_fp8_block_config())

    def test_fp8_per_tensor_raises(self):
        with self.assertRaises(ValueError) as cm:
            self._call(_fp8_per_tensor_config())
        self.assertIn("block-wise", str(cm.exception))

    def test_fp8_not_serialized_raises(self):
        from sglang.srt.layers.quantization.fp8 import Fp8Config

        with self.assertRaises(ValueError):
            self._call(Fp8Config())

    def test_mxfp8_variant_raises(self):
        from sglang.srt.layers.quantization.fp8 import Fp8Config

        mxfp8 = Fp8Config(is_checkpoint_fp8_serialized=True, use_mxfp8=True)
        with self.assertRaises(ValueError) as cm:
            self._call(mxfp8)
        self.assertIn("mxfp8", str(cm.exception))

    def test_fp4_packed_variant_raises(self):
        fp4 = _fp8_block_config(is_fp4_experts=True)
        with self.assertRaises(ValueError) as cm:
            self._call(fp4)
        self.assertIn("fp4", str(cm.exception))

    def test_non_fp8_quant_raises(self):
        with self.assertRaises(ValueError) as cm:
            self._call(_DummyQuantConfig())
        self.assertIn("DummyQuantConfig", str(cm.exception))

    def test_trtllm_runner_raises_with_and_without_quant(self):
        for quant in (None, _fp8_block_config()):
            with self.assertRaises(ValueError) as cm:
                self._call(quant, flashinfer_trtllm_runner=True)
            self.assertIn("TRTLLM", str(cm.exception))

    def test_fnuz_platform_raises_for_fp8(self):
        with self.assertRaises(ValueError) as cm:
            self._call(_fp8_block_config(), fnuz_platform=True)
        self.assertIn("fnuz", str(cm.exception))

    def test_deep_gemm_runner_raises_for_fp8(self):
        with self.assertRaises(ValueError) as cm:
            self._call(_fp8_block_config(), deep_gemm_runner=True)
        self.assertIn("deep_gemm", str(cm.exception))

    def test_bf16_fnuz_deep_gemm_still_ok(self):
        # The fnuz/deepgemm hazards are fp8-specific; bf16 cache runs keep
        # their pre-M2 acceptance.
        self._call(None, fnuz_platform=True, deep_gemm_runner=True)


def _expert_entry(w13_dtype=torch.bfloat16, w2_dtype=torch.bfloat16, scales=False):
    entry = {
        "w13": torch.randn(4, 6).to(w13_dtype),
        "w2": torch.randn(6, 4).to(w2_dtype),
    }
    if scales:
        entry["w13_scale"] = torch.rand(1, 1, dtype=torch.float32)
        entry["w2_scale"] = torch.rand(1, 1, dtype=torch.float32)
    return entry


class TestExpertCacheEntryPairing(CustomTestCase):
    """Weights<->scales presence parity enforced at wiring time."""

    def _validate(self, layer_id=0, expert_index=0, entry=None):
        from sglang.srt.models.deepseek_v2 import (
            _expert_cache_entry_pairing_ok,
        )

        _expert_cache_entry_pairing_ok(layer_id, expert_index, entry)

    def test_bf16_weights_without_scales_ok(self):
        self._validate(entry=_expert_entry())

    def test_fp8_weights_with_scales_ok(self):
        fp8 = torch.float8_e4m3fn
        self._validate(entry=_expert_entry(fp8, fp8, scales=True))

    def test_fp8_weights_without_scales_raises(self):
        fp8 = torch.float8_e4m3fn
        with self.assertRaises(ValueError) as cm:
            self._validate(entry=_expert_entry(fp8, fp8))
        self.assertIn("scale_inv", str(cm.exception))
        self.assertIn("expert 0", str(cm.exception))

    def test_stray_scale_without_fp8_weights_raises(self):
        with self.assertRaises(ValueError) as cm:
            self._validate(entry=_expert_entry(scales=True))
        self.assertIn("non-fp8", str(cm.exception))

    def test_half_paired_scales_raises(self):
        fp8 = torch.float8_e4m3fn
        entry = _expert_entry(fp8, fp8, scales=True)
        del entry["w2_scale"]
        with self.assertRaises(ValueError):
            self._validate(entry=entry)

    def test_missing_fused_weight_raises(self):
        entry = _expert_entry()
        del entry["w2"]
        with self.assertRaises(ValueError) as cm:
            self._validate(layer_id=3, expert_index=7, entry=entry)
        self.assertIn("layer 3 expert 7", str(cm.exception))


def _fp8_block_quant_config():
    from sglang.srt.layers.quantization.fp8 import Fp8Config

    return Fp8Config(is_checkpoint_fp8_serialized=True, weight_block_size=[128, 128])


def _fp8_tiny_config():
    """fp8 needs expert dims divisible by the 128x128 quant blocks:
    w13 N=2*inter and K=hidden, w2 N=hidden and K=inter."""
    return _tiny_config(hidden=256, inter=128)


def _causal_lm_wiring_shim(moe):
    """Minimal DeepseekV2ForCausalLM shell exposing the expert-cache wiring
    methods over one pre-built MoE layer (no full model construction)."""
    from sglang.srt.models.deepseek_v2 import DeepseekV2ForCausalLM

    shim = object.__new__(DeepseekV2ForCausalLM)
    shim.config = SimpleNamespace(
        num_hidden_layers=1,
        n_routed_experts=_NUM_EXPERTS,
        num_experts_per_tok=2,
        hidden_size=_HIDDEN,
        moe_intermediate_size=_INTER,
    )
    shim.model = SimpleNamespace(
        start_layer=0, end_layer=1, layers=[SimpleNamespace(mlp=moe)]
    )
    return shim


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

    def test_fp8_block_wiring_threads_scales(self):
        with _tp1_parallel(), _cache_server_args(), torch.device("cuda"):
            moe = DeepseekV2MoE(
                config=_fp8_tiny_config(),
                layer_id=0,
                quant_config=_fp8_block_quant_config(),
                prefix="model.layers.0.mlp",
            )
            self.assertTrue(moe.expert_cache_enabled)

            # Pool-sized fp8 experts carry slot-major scale_inv params.
            self.assertEqual(moe.experts.w13_weight.dtype, torch.float8_e4m3fn)
            w13_scale_param = getattr(moe.experts, "w13_weight_scale_inv", None)
            w2_scale_param = getattr(moe.experts, "w2_weight_scale_inv", None)
            self.assertIsNotNone(w13_scale_param)
            self.assertIsNotNone(w2_scale_param)
            self.assertEqual(w13_scale_param.dtype, torch.float32)

            # Simulate interception of one expert's fp8 weights + scales
            # (the interceptor stores raw CPU tensors, as at load time).
            expert_index = 5
            inter, hidden = 128, 256  # matches _fp8_tiny_config()
            w13 = torch.randn(2 * inter, hidden).to(torch.float8_e4m3fn)
            w2 = torch.randn(hidden, inter).to(torch.float8_e4m3fn)
            w13_scale_inv = torch.rand(2 * inter // 128, hidden // 128)
            w2_scale_inv = torch.rand(hidden // 128, inter // 128)
            moe._moe_expert_cache_pending = {
                (0, expert_index): {
                    "w13": w13,
                    "w2": w2,
                    "w13_scale": w13_scale_inv,
                    "w2_scale": w2_scale_inv,
                }
            }

            shim = _causal_lm_wiring_shim(moe)
            shim._build_expert_cache_infra()
            pool = moe.expert_cache_runtime.pool
            # LayerRuntime was threaded the fp8 scale params.
            self.assertIsNotNone(pool.w13_scale)
            self.assertIsNotNone(pool.w2_scale)
            self.assertEqual(
                pool.w13_scale.data_ptr(), w13_scale_param.data_ptr()
            )
            self.assertEqual(
                pool.w2_scale.data_ptr(), w2_scale_param.data_ptr()
            )

            shim._finish_expert_cache_wiring()
            entry = moe.expert_cache_runtime.store.get(ExpertKey(0, expert_index))
            self.assertTrue(torch.equal(entry.w13, w13))
            self.assertTrue(torch.equal(entry.w2, w2))
            self.assertTrue(torch.equal(entry.w13_scale_inv, w13_scale_inv))
            self.assertTrue(torch.equal(entry.w2_scale_inv, w2_scale_inv))

    def test_fp8_entry_without_scales_raises_at_wiring(self):
        with _tp1_parallel(), _cache_server_args(), torch.device("cuda"):
            moe = DeepseekV2MoE(
                config=_fp8_tiny_config(),
                layer_id=0,
                quant_config=_fp8_block_quant_config(),
                prefix="model.layers.0.mlp",
            )
            moe._moe_expert_cache_pending = {
                (0, 1): {
                    "w13": torch.randn(256, 256).to(torch.float8_e4m3fn),
                    "w2": torch.randn(256, 128).to(torch.float8_e4m3fn),
                }
            }
            shim = _causal_lm_wiring_shim(moe)
            shim._build_expert_cache_infra()
            with self.assertRaises(ValueError) as cm:
                shim._finish_expert_cache_wiring()
            self.assertIn("weight_scale_inv", str(cm.exception))


if __name__ == "__main__":
    unittest.main()
