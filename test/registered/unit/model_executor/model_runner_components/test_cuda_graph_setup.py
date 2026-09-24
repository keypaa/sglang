import sys
from types import SimpleNamespace

import pytest
from torch import nn

from sglang.srt.model_executor.model_runner_components import cuda_graph_setup
from sglang.srt.model_executor.model_runner_components.cuda_graph_setup import (
    _populate_attention_and_moe_layers,
    capture_decode_graph,
)
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


def test_model_runner_can_override_decode_graph_runner(monkeypatch):
    class CustomGraphRunner:
        def __init__(self, model_runner):
            self.model_runner = model_runner

    class TestModelRunner:
        is_generation = True
        device = "cuda"
        gpu_id = 0
        is_draft_worker = False
        spec_algorithm = SimpleNamespace(is_speculative=lambda: False)
        server_args = SimpleNamespace(
            model_impl="auto",
            cuda_graph_config=SimpleNamespace(
                decode=SimpleNamespace(backend="default")
            ),
        )

        def _decode_cuda_graph_runner_cls(self):
            return CustomGraphRunner

    model_runner = TestModelRunner()
    monkeypatch.setattr(cuda_graph_setup, "check_cuda_graph_backend", lambda *_: False)
    monkeypatch.setattr(cuda_graph_setup, "get_available_gpu_memory", lambda *_: 10.0)
    monkeypatch.setattr(
        cuda_graph_setup, "get_batch_sizes_to_capture", lambda *_: ([1], None)
    )
    monkeypatch.setattr(
        cuda_graph_setup.current_platform, "is_out_of_tree", lambda: False
    )

    capture = capture_decode_graph(model_runner=model_runner)

    assert isinstance(capture.runner, CustomGraphRunner)
    assert capture.runner.model_runner is model_runner


def _decode_model_runner(monkeypatch, model):
    class CustomGraphRunner:
        def __init__(self, model_runner):
            self.model_runner = model_runner

    class TestModelRunner:
        is_generation = True
        device = "cuda"
        gpu_id = 0
        is_draft_worker = False
        spec_algorithm = SimpleNamespace(is_speculative=lambda: False)
        server_args = SimpleNamespace(
            model_impl="auto",
            cuda_graph_config=SimpleNamespace(
                decode=SimpleNamespace(backend="default")
            ),
        )

        def _decode_cuda_graph_runner_cls(self):
            return CustomGraphRunner

    model_runner = TestModelRunner()
    model_runner.model = model
    monkeypatch.setattr(cuda_graph_setup, "check_cuda_graph_backend", lambda *_: False)
    monkeypatch.setattr(cuda_graph_setup, "get_available_gpu_memory", lambda *_: 10.0)
    monkeypatch.setattr(
        cuda_graph_setup, "get_batch_sizes_to_capture", lambda *_: ([1], None)
    )
    monkeypatch.setattr(
        cuda_graph_setup.current_platform, "is_out_of_tree", lambda: False
    )
    return model_runner


def test_capture_decode_graph_populates_missing_layer_lists(monkeypatch):
    experts = SimpleNamespace()
    model_runner = _decode_model_runner(
        monkeypatch,
        SimpleNamespace(
            model=SimpleNamespace(
                layers=[
                    SimpleNamespace(
                        self_attn=SimpleNamespace(attn=SimpleNamespace()),
                        mlp=SimpleNamespace(experts=experts),
                    )
                ]
            )
        ),
    )
    assert getattr(model_runner, "moe_layers", None) is None

    capture_decode_graph(model_runner=model_runner)

    assert model_runner.moe_layers == [experts]


def test_capture_decode_graph_preserves_existing_layer_lists(monkeypatch):
    model_runner = _decode_model_runner(
        monkeypatch, SimpleNamespace(model=SimpleNamespace())
    )
    sentinel = [SimpleNamespace()]
    model_runner.moe_layers = sentinel

    capture_decode_graph(model_runner=model_runner)

    assert model_runner.moe_layers is sentinel


def _causal_lm_runner(layers):
    inner = SimpleNamespace(layers=layers)
    return SimpleNamespace(model=SimpleNamespace(model=inner)), inner


def test_populate_moe_layers_collected_in_order_with_nones_elsewhere():
    experts = SimpleNamespace()
    mlp_moe = SimpleNamespace(experts=experts)
    attn0, attn1 = SimpleNamespace(), SimpleNamespace()
    runner, inner = _causal_lm_runner(
        [
            SimpleNamespace(self_attn=SimpleNamespace(attn=attn0), mlp=mlp_moe),
            SimpleNamespace(
                self_attn=SimpleNamespace(attn=attn1), mlp=SimpleNamespace()
            ),
        ]
    )

    language_model = _populate_attention_and_moe_layers(runner)

    assert language_model is inner
    assert runner.model.model is inner
    assert runner.attention_layers == [attn0, attn1]
    assert runner.moe_layers == [experts, None]
    assert runner.moe_fusions == [mlp_moe, None]
    assert runner.dsa_indexers == [None, None]
    assert runner.mha_companion_layers == [None, None]


def test_populate_dense_model_yields_all_none_entries():
    runner, _ = _causal_lm_runner(
        [SimpleNamespace(self_attn=SimpleNamespace(attn=SimpleNamespace()))] * 2
    )

    _populate_attention_and_moe_layers(runner)

    assert runner.moe_layers == [None, None]
    assert runner.moe_layers is not None


def test_populate_moduledict_layers_variant():
    class _Layer(nn.Module):
        pass

    experts = SimpleNamespace()
    moe_layer, dense_layer = _Layer(), _Layer()
    moe_layer.self_attn = SimpleNamespace(attn=SimpleNamespace())
    moe_layer.mlp = SimpleNamespace(experts=experts)
    dense_layer.self_attn = SimpleNamespace(attn=SimpleNamespace())
    dense_layer.mlp = SimpleNamespace()
    inner = SimpleNamespace(
        layers=nn.ModuleDict({"l0": moe_layer, "l1": dense_layer})
    )
    runner = SimpleNamespace(model=SimpleNamespace(model=inner))

    _populate_attention_and_moe_layers(runner)

    assert runner.moe_layers == [experts, None]


def test_populate_vlm_wrapper_descends_model_chain():
    experts = SimpleNamespace()
    inner = SimpleNamespace(
        layers=[
            SimpleNamespace(
                self_attn=SimpleNamespace(attn=SimpleNamespace()),
                mlp=SimpleNamespace(experts=experts),
            )
        ]
    )
    outer = SimpleNamespace(model=inner)
    runner = SimpleNamespace(model=SimpleNamespace(language_model=outer))

    language_model = _populate_attention_and_moe_layers(runner)

    assert language_model is outer
    assert not hasattr(runner.model, "model")
    assert runner.moe_layers == [experts]


def test_populate_undiscoverable_model_returns_none_and_assigns_nothing():
    runner = SimpleNamespace(model=SimpleNamespace())

    assert _populate_attention_and_moe_layers(runner) is None
    for attr in (
        "attention_layers",
        "moe_layers",
        "moe_fusions",
        "dsa_indexers",
        "mha_companion_layers",
    ):
        assert not hasattr(runner, attr)


def test_populate_language_model_without_layers_assigns_nothing():
    language_model = SimpleNamespace()
    runner = SimpleNamespace(
        model=SimpleNamespace(language_model=language_model)
    )

    assert _populate_attention_and_moe_layers(runner) is language_model
    assert not hasattr(runner, "moe_layers")


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
