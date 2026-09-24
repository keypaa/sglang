"""Engine-level parity gate: expert-cache on vs off under tc_piecewise graphs.

Fabricates a tiny loadable DeepseekV2ForCausalLM checkpoint twice per dtype
(bf16 and block-fp8) from ONE deterministically filled reference model, then
launches a real `Engine` per arm with `cuda_graph_backend_decode="tc_piecewise"`:

  A: enable_moe_expert_cache off (dense routed-expert weights, canonical
     gate/up/down checkpoint layout)
  B: enable_moe_expert_cache on, moe_cache_slots=4 (per-expert fused checkpoint
     layout streamed into the pinned host store via the real load_weights
     interception path)

Both arms consume bitwise-identical weight bytes, so greedy decoding must emit
byte-identical output_ids sequences. Any divergence isolates to the
expert-cache pool-as-weights path under segmented decode graphs.

Checkpoints are written to temp dirs (config.json + model.safetensors) and
loaded through the standard model loader, exactly like production serving.
The fabricated config carries one shared expert like production DeepSeek
models, emitted in the canonical split layout in both checkpoint dirs.
Engines run with skip_tokenizer_init=True so no tokenizer assets are needed.
"""

import json
import os
import shutil
import tempfile
import threading
import unittest
from contextlib import contextmanager

import torch
from safetensors.torch import save_file
from sglang.srt.models.deepseek_v2 import DeepseekV2ForCausalLM
from sglang.srt.runtime_context import get_context, get_parallel
from sglang.srt.server_args import validate_moe_expert_cache
from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.test_utils import CustomTestCase

register_cuda_ci(est_time=1500, suite="base-a-test-1-gpu-small")

_NUM_EXPERTS = 8
_NUM_SLOTS = 4
_NUM_LAYERS = 2
_VOCAB = 128
_TOP_K = 2
_HIDDEN = 64
_INTER = 64
_FP_HIDDEN = 128
_FP_INTER = 128
_FP_BLOCK = 128
_SEED_WEIGHTS = 1234

_PROMPT_TOKEN_IDS = [[7, 11, 13], [42, 5], [90, 3, 21, 57]]
_NEW_TOKENS = 16
_LAUNCH_DEADLINE_S = 600
_GENERATE_DEADLINE_S = 300


def _with_deadline(label: str, deadline_s: int, fn):
    result = {}

    def _worker():
        try:
            result["value"] = fn()
        except BaseException as exc:
            result["error"] = exc

    thread = threading.Thread(target=_worker, daemon=True)
    thread.start()
    thread.join(deadline_s)
    if thread.is_alive():
        raise TimeoutError(f"{label} exceeded its {deadline_s}s deadline")
    if "error" in result:
        raise result["error"]
    return result["value"]


def _config_dict(hidden: int, inter: int) -> dict:
    return {
        "architectures": ["DeepseekV2ForCausalLM"],
        "model_type": "deepseek_v2",
        "torch_dtype": "bfloat16",
        "hidden_size": hidden,
        "intermediate_size": inter,
        "moe_intermediate_size": inter,
        "n_routed_experts": _NUM_EXPERTS,
        "num_experts_per_tok": _TOP_K,
        "n_shared_experts": 1,
        "routed_scaling_factor": 1.0,
        "hidden_act": "silu",
        "vocab_size": _VOCAB,
        "num_hidden_layers": _NUM_LAYERS,
        "num_hash_layers": 0,
        "n_group": 1,
        "topk_group": 1,
        "topk_method": "noaux_tc",
        "scoring_func": "sigmoid",
        "norm_topk_prob": False,
        "first_k_dense_replace": 0,
        "moe_layer_freq": 1,
        "q_lora_rank": None,
        # Stock Triton grouped decode needs next_pow2(kv_lora + rope) ==
        # next_pow2(kv_lora) in its MLA branch (both 64 here; 32/16 gave 64 vs 32).
        "kv_lora_rank": 48,
        "qk_nope_head_dim": 16,
        "qk_rope_head_dim": 16,
        "v_head_dim": 16,
        "num_attention_heads": 2,
        "rope_theta": 10000.0,
        "rope_scaling": None,
        "max_position_embeddings": 512,
        "rms_norm_eps": 1e-6,
        "tie_word_embeddings": False,
        "pad_token_id": 0,
    }


def _quantization_config_dict() -> dict:
    return {
        "quant_method": "fp8",
        "activation_scheme": "dynamic",
        "weight_block_size": [_FP_BLOCK, _FP_BLOCK],
        "fmt": "e4m3",
    }


def _engine_kwargs(cache_enabled: bool) -> dict:
    kwargs = {
        "skip_tokenizer_init": True,
        "cuda_graph_backend_decode": "tc_piecewise",
        "enable_moe_expert_cache": bool(cache_enabled),
        "disable_shared_experts_fusion": True,
        "ep_num_redundant_experts": 0,
        "mem_fraction_static": 0.25,
        "cuda_graph_max_bs_decode": len(_PROMPT_TOKEN_IDS) + 4,
        "random_seed": 42,
    }
    if cache_enabled:
        kwargs["moe_cache_slots"] = _NUM_SLOTS
    return kwargs


def _setup_distributed() -> None:
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


def _free_master_port() -> str:
    import socket

    s = socket.socket()
    s.bind(("127.0.0.1", 0))
    port = str(s.getsockname()[1])
    s.close()
    return port


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
        attn_dp_size=1,
        attn_dp_rank=0,
    )


@contextmanager
def _bf16_default_dtype():
    # Quantized models must be BORN in their final dtypes: Module.to(bf16)
    # would upcast fp8 weight params and destroy the quantized layout.
    prev = torch.get_default_dtype()
    torch.set_default_dtype(torch.bfloat16)
    try:
        yield
    finally:
        torch.set_default_dtype(prev)


def _build_reference(quant_config, hidden: int, inter: int):
    """Build ONE cache-off reference model whose params feed both checkpoints."""
    from sglang.srt.layers.quantization.fp8 import Fp8Config

    if quant_config == "fp8":
        quant_config = Fp8Config(
            is_checkpoint_fp8_serialized=True,
            activation_scheme="dynamic",
            weight_block_size=[_FP_BLOCK, _FP_BLOCK],
        )
    torch.manual_seed(_SEED_WEIGHTS)
    overrides = {"enable_moe_expert_cache": False, "disable_shared_experts_fusion": True}
    with (
        _tp1_parallel(),
        get_context().override_server_args(**overrides),
        torch.device("cuda"),
    ):
        if quant_config is not None:
            with _bf16_default_dtype():
                model = DeepseekV2ForCausalLM(
                    config=_SimpleNamespaceConfig(hidden, inter),
                    quant_config=quant_config,
                    prefix="",
                )
        else:
            model = DeepseekV2ForCausalLM(
                config=_SimpleNamespaceConfig(hidden, inter),
                quant_config=None,
                prefix="",
            )
    _fill_params_deterministically(model)
    return model.to(torch.bfloat16) if quant_config is None else model


def _fill_params_deterministically(model) -> None:
    torch.manual_seed(_SEED_WEIGHTS + 1)
    for _, param in sorted(model.named_parameters()):
        if param.dtype == torch.float8_e4m3fn:
            flat = torch.arange(param.numel(), dtype=torch.float32)
            vals = ((flat % 60) + 20) * 0.25
        elif param.dtype.is_floating_point:
            vals = torch.randn(param.numel(), dtype=torch.float32)
        else:
            continue
        param.data.copy_(vals.reshape(param.shape).to(param.dtype))
    for name, param in sorted(model.named_parameters()):
        if "weight_scale" in name:
            flat = torch.arange(param.numel(), dtype=torch.float32)
            vals = ((flat % 40) + 10) * 0.125
            param.data.copy_(vals.reshape(param.shape).to(param.dtype))


class _SimpleNamespaceConfig:
    """Attribute view over _config_dict for in-process model construction."""

    def __init__(self, hidden: int, inter: int):
        self.__dict__.update(_config_dict(hidden, inter))


def _shared_expert_items(
    layer_id: int, gate_up, down, gate_up_scale, down_scale, inter: int, block: int
):
    """Canonical shared-expert checkpoint entries for one layer.

    The reference model fuses gate+up into gate_up_proj; the loader's
    stacked_params_mapping consumes canonical gate_proj/up_proj/down_proj
    names (plus weight_scale_inv for block-fp8). Shared experts are
    dense/non-routed, so both layouts carry these identical entries.
    """
    stem = f"model.layers.{layer_id}.mlp.shared_experts"
    items = [
        (f"{stem}.gate_proj.weight", gate_up[:inter].clone()),
        (f"{stem}.up_proj.weight", gate_up[inter:].clone()),
        (f"{stem}.down_proj.weight", down.clone()),
    ]
    if gate_up_scale is not None:
        blocks_per_gate = inter // block
        assert 2 * blocks_per_gate == gate_up_scale.shape[0]
        items.extend(
            [
                (
                    f"{stem}.gate_proj.weight_scale_inv",
                    gate_up_scale[:blocks_per_gate].clone(),
                ),
                (
                    f"{stem}.up_proj.weight_scale_inv",
                    gate_up_scale[blocks_per_gate:].clone(),
                ),
                (f"{stem}.down_proj.weight_scale_inv", down_scale.clone()),
            ]
        )
    return items


def _write_checkpoints(root: str, quant_config, hidden: int, inter: int):
    """Derive two loadable checkpoint dirs from one filled reference model.

    off/: canonical per-expert layout (gate/up/down_proj[.weight_scale_inv])
          consumed by A's dense fused-expert loader.
    on/:  per-expert fused layout (experts.{e}.w13_weight[...]) intercepted by
          B's expert-cache host-store streaming.
    Shared experts are dense: both layouts carry the identical canonical
    entries, which the loader consumes without interception.
    Bytes are identical across layouts by construction.
    """
    ref = _build_reference(quant_config, hidden, inter)
    items_off, items_on = [], []
    for name, param in sorted(ref.named_parameters()):
        if (
            ".mlp.experts.w13_weight" in name
            or ".mlp.experts.w2_weight" in name
            or ".mlp.shared_experts.gate_up_proj" in name
            or ".mlp.shared_experts.down_proj" in name
        ):
            continue
        cpu = param.data.detach().cpu()
        items_off.append((name, cpu.clone()))
        items_on.append((name, cpu.clone()))

    for i in range(_NUM_LAYERS):
        shared = ref.model.layers[i].mlp.shared_experts
        gate_up = shared.gate_up_proj.weight.data.detach().cpu()
        down = shared.down_proj.weight.data.detach().cpu()
        gate_up_scale = getattr(shared.gate_up_proj, "weight_scale_inv", None)
        down_scale = getattr(shared.down_proj, "weight_scale_inv", None)
        if gate_up_scale is not None:
            gate_up_scale = gate_up_scale.data.detach().cpu()
        if down_scale is not None:
            down_scale = down_scale.data.detach().cpu()
        shared_items = _shared_expert_items(
            i, gate_up, down, gate_up_scale, down_scale, inter, _FP_BLOCK
        )
        items_off.extend(shared_items)
        items_on.extend([(name, tensor.clone()) for name, tensor in shared_items])

    block = _FP_BLOCK
    for i in range(_NUM_LAYERS):
        experts = ref.model.layers[i].mlp.experts
        w13 = experts.w13_weight.data.detach().cpu()
        w2 = experts.w2_weight.data.detach().cpu()
        s13 = getattr(experts, "w13_weight_scale_inv", None)
        s2 = getattr(experts, "w2_weight_scale_inv", None)
        if s13 is not None:
            s13 = s13.data.detach().cpu()
            blocks_per_gate = inter // block
            assert 2 * blocks_per_gate == s13.shape[1]
        if s2 is not None:
            s2 = s2.data.detach().cpu()

        for e in range(_NUM_EXPERTS):
            items_on.append((f"model.layers.{i}.mlp.experts.{e}.w13_weight", w13[e].clone()))
            items_on.append((f"model.layers.{i}.mlp.experts.{e}.w2_weight", w2[e].clone()))
            if s13 is not None:
                items_on.append(
                    (f"model.layers.{i}.mlp.experts.{e}.w13_weight_scale_inv", s13[e].clone())
                )
                items_on.append(
                    (f"model.layers.{i}.mlp.experts.{e}.w2_weight_scale_inv", s2[e].clone())
                )
            items_off.extend(
                [
                    (f"model.layers.{i}.mlp.experts.{e}.gate_proj.weight", w13[e][:inter].clone()),
                    (f"model.layers.{i}.mlp.experts.{e}.up_proj.weight", w13[e][inter:].clone()),
                    (f"model.layers.{i}.mlp.experts.{e}.down_proj.weight", w2[e].clone()),
                ]
            )
            if s13 is not None:
                items_off.extend(
                    [
                        (
                            f"model.layers.{i}.mlp.experts.{e}.gate_proj.weight_scale_inv",
                            s13[e][:blocks_per_gate].clone(),
                        ),
                        (
                            f"model.layers.{i}.mlp.experts.{e}.up_proj.weight_scale_inv",
                            s13[e][blocks_per_gate:].clone(),
                        ),
                        (
                            f"model.layers.{i}.mlp.experts.{e}.down_proj.weight_scale_inv",
                            s2[e].clone(),
                        ),
                    ]
                )

    cfg_common = {}
    if quant_config == "fp8":
        cfg_common["quantization_config"] = _quantization_config_dict()

    off_dir, on_dir = os.path.join(root, "ckpt_off"), os.path.join(root, "ckpt_on")
    for path, items in ((off_dir, items_off), (on_dir, items_on)):
        os.makedirs(path, exist_ok=True)
        config = _config_dict(hidden, inter)
        config.update(cfg_common)
        with open(os.path.join(path, "config.json"), "w") as f:
            json.dump(config, f, indent=2)
        save_file(
            {name: t.contiguous() for name, t in items},
            os.path.join(path, "model.safetensors"),
            metadata={"format": "pt"},
        )
    return off_dir, on_dir


def _generate_token_ids(ckpt_dir: str, cache_enabled: bool):
    from sglang import Engine

    engine = _with_deadline(
        "engine launch",
        _LAUNCH_DEADLINE_S,
        lambda: Engine(model_path=ckpt_dir, **_engine_kwargs(cache_enabled)),
    )
    try:
        outputs = _with_deadline(
            "generation",
            _GENERATE_DEADLINE_S,
            lambda: engine.generate(
                input_ids=_PROMPT_TOKEN_IDS,
                sampling_params={"temperature": 0, "max_new_tokens": _NEW_TOKENS},
            ),
        )
    finally:
        engine.shutdown()
    return [list(out["output_ids"]) for out in outputs]


class TestExpertCacheEngineArgValidation(CustomTestCase):
    """Offline guard checks: the exact startup-path validator must accept the
    engine configs used above, on any machine (no CUDA required)."""

    def test_guard_accepts_cache_on_with_tc_piecewise(self):
        kwargs = _engine_kwargs(cache_enabled=True)
        slots = validate_moe_expert_cache(kwargs)
        self.assertEqual(slots, _NUM_SLOTS)

    def test_guard_ignores_cache_off_arm(self):
        kwargs = _engine_kwargs(cache_enabled=False)
        self.assertEqual(validate_moe_expert_cache(kwargs), 0)

    def test_guard_rejects_cache_on_without_tc_piecewise(self):
        kwargs = _engine_kwargs(cache_enabled=True)
        del kwargs["cuda_graph_backend_decode"]
        with self.assertRaises(ValueError):
            validate_moe_expert_cache(kwargs)

    def test_config_carries_one_shared_expert(self):
        self.assertEqual(_config_dict(_HIDDEN, _INTER)["n_shared_experts"], 1)

    def test_config_mla_dims_satisfy_decode_kernel(self):
        def _next_pow2(n):
            p = 1
            while p < n:
                p *= 2
            return p

        for hidden, inter in ((_HIDDEN, _INTER), (_FP_HIDDEN, _FP_INTER)):
            cfg = _config_dict(hidden, inter)
            self.assertEqual(
                _next_pow2(cfg["kv_lora_rank"] + cfg["qk_rope_head_dim"]),
                _next_pow2(cfg["kv_lora_rank"]),
            )

    def test_shared_expert_items_split_names_and_bytes(self):
        inter, block = 128, 128
        gate_up = torch.arange(2 * inter * 64, dtype=torch.float32).reshape(2 * inter, 64)
        down = torch.arange(64 * inter, dtype=torch.float32).reshape(64, inter)
        gate_up_scale = torch.arange(2, dtype=torch.float32).reshape(2, 1)
        down_scale = torch.arange(1, dtype=torch.float32).reshape(1, 1)
        items = _shared_expert_items(3, gate_up, down, gate_up_scale, down_scale, inter, block)
        by_name = dict(items)
        stem = "model.layers.3.mlp.shared_experts"
        self.assertEqual(
            sorted(by_name),
            sorted(
                f"{stem}.{p}.{s}"
                for p in ("gate_proj", "up_proj", "down_proj")
                for s in ("weight", "weight_scale_inv")
            ),
        )
        self.assertTrue(torch.equal(by_name[f"{stem}.gate_proj.weight"], gate_up[:inter]))
        self.assertTrue(torch.equal(by_name[f"{stem}.up_proj.weight"], gate_up[inter:]))
        self.assertTrue(torch.equal(by_name[f"{stem}.down_proj.weight"], down))
        self.assertTrue(
            torch.equal(
                by_name[f"{stem}.gate_proj.weight_scale_inv"], gate_up_scale[:1]
            )
        )
        self.assertTrue(
            torch.equal(
                by_name[f"{stem}.up_proj.weight_scale_inv"], gate_up_scale[1:]
            )
        )
        self.assertTrue(
            torch.equal(by_name[f"{stem}.down_proj.weight_scale_inv"], down_scale)
        )

    def test_shared_expert_items_bf16_has_no_scales(self):
        gate_up = torch.zeros(2 * _INTER, _HIDDEN)
        down = torch.zeros(_HIDDEN, _INTER)
        items = _shared_expert_items(0, gate_up, down, None, None, _INTER, _FP_BLOCK)
        self.assertEqual(len(items), 3)
        self.assertTrue(all(name.endswith(".weight") for name, _ in items))


class _EngineParityBase(CustomTestCase):
    quant_variant = None

    @classmethod
    def setUpClass(cls):
        _setup_distributed()
        cls._root = tempfile.mkdtemp(prefix="expert_cache_engine_parity_")
        if cls.quant_variant == "fp8":
            cls.off_dir, cls.on_dir = _write_checkpoints(
                cls._root, "fp8", _FP_HIDDEN, _FP_INTER
            )
        else:
            cls.off_dir, cls.on_dir = _write_checkpoints(
                cls._root, None, _HIDDEN, _INTER
            )

    @classmethod
    def tearDownClass(cls):
        root = getattr(cls, "_root", None)
        if root:
            shutil.rmtree(root, ignore_errors=True)

    def _run_parity(self):
        off_ids = _generate_token_ids(self.off_dir, cache_enabled=False)
        on_ids = _generate_token_ids(self.on_dir, cache_enabled=True)
        self.assertEqual(
            off_ids,
            on_ids,
            f"cache-on token sequences diverge from cache-off:\noff={off_ids}\non={on_ids}",
        )
        self.assertEqual(len(off_ids), len(_PROMPT_TOKEN_IDS))
        for seq in off_ids:
            self.assertEqual(len(seq), _NEW_TOKENS)
        # Non-vacuity: distinct prompts must not collapse to one sequence.
        self.assertGreaterEqual(len({tuple(seq) for seq in off_ids}), 2)


@unittest.skipUnless(torch.cuda.is_available(), "CUDA required")
class TestEngineParityBf16(_EngineParityBase):
    def test_cache_on_matches_cache_off_tokens(self):
        self._run_parity()


@unittest.skipUnless(torch.cuda.is_available(), "CUDA required")
class TestEngineParityFp8(_EngineParityBase):
    quant_variant = "fp8"

    def test_fp8_cache_on_matches_cache_off_tokens(self):
        self._run_parity()


if __name__ == "__main__":
    unittest.main()
