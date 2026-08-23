"""CPU units for decode-phase backend resolution and MoE split-op registration.

Covers the two wiring gaps that make an explicit
``cuda_graph_backend_decode="tc_piecewise"`` produce segmented decode graphs:

  * ``resolve_decode_backend`` must return ``TcPiecewiseCudaGraphBackend``
    instead of silently falling back to FULL.
  * ``TcPiecewiseCudaGraphBackend.build_compilation_config`` must register
    ``sglang.moe_forward_piecewise_cuda_graph_impl`` as a split op for
    expert-cache configs (plain TP, a2a none), while default configs stay
    unregistered and deepep/mooncake stays registered exactly once.
"""

import unittest
from types import SimpleNamespace
from unittest.mock import patch

from sglang.srt.model_executor.cuda_graph_config import (
    Backend,
    CudaGraphConfig,
    PhaseConfig,
)
from sglang.srt.model_executor.runner_backend.utils import resolve_decode_backend
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="base-a-test-cpu")

_MOE_SPLIT_OP = "sglang.moe_forward_piecewise_cuda_graph_impl"

_UTILS = "sglang.srt.model_executor.runner_backend.utils"


def _make_runner(decode_backend):
    server_args = SimpleNamespace(
        cuda_graph_config=CudaGraphConfig(
            decode=PhaseConfig(backend=decode_backend),
            prefill=PhaseConfig(
                backend=Backend.BREAKABLE, bs=[32, 64], tc_compiler="eager"
            ),
        ),
        enable_memory_saver=False,
        debug_cuda_graph=False,
    )
    return SimpleNamespace(
        model_runner=SimpleNamespace(server_args=server_args, device="cuda")
    )


def _make_server_args(
    enable_moe_expert_cache, decode_backend=Backend.TC_PIECEWISE
):
    return SimpleNamespace(
        cuda_graph_config=CudaGraphConfig(
            decode=PhaseConfig(backend=decode_backend),
            prefill=PhaseConfig(
                backend=Backend.TC_PIECEWISE, bs=[32, 64], tc_compiler="eager"
            ),
        ),
        enable_torch_compile_debug_mode=False,
        enable_moe_expert_cache=enable_moe_expert_cache,
    )


def _build_config(
    mock_a2a,
    *,
    deepep=False,
    mooncake=False,
    expert_cache=False,
    decode_backend=Backend.TC_PIECEWISE,
):
    mock_a2a.return_value = SimpleNamespace(
        is_deepep=lambda: deepep, is_mooncake=lambda: mooncake
    )
    from sglang.srt.model_executor.runner_backend import (
        tc_piecewise_cuda_graph_backend as tc_backend_mod,
    )

    return tc_backend_mod.TcPiecewiseCudaGraphBackend.build_compilation_config(
        _make_server_args(expert_cache, decode_backend)
    )


class TestResolveDecodeBackend(CustomTestCase):
    def test_tc_piecewise_returns_tc_piecewise_backend(self):
        runner = _make_runner(Backend.TC_PIECEWISE)
        with patch(f"{_UTILS}.TcPiecewiseCudaGraphBackend") as mock_tc, patch(
            f"{_UTILS}.FullCudaGraphBackend"
        ) as mock_full:
            backend = resolve_decode_backend(runner)

        self.assertIs(backend, mock_tc.return_value)
        mock_tc.assert_called_once_with(runner)
        mock_full.assert_not_called()

    def test_full_default_returns_full_backend(self):
        runner = _make_runner(Backend.FULL)
        sentinel = object()
        with patch(f"{_UTILS}.FullCudaGraphBackend") as mock_backend_cls:
            mock_backend_cls.return_value = sentinel
            backend = resolve_decode_backend(runner)

        self.assertIs(backend, sentinel)
        mock_backend_cls.assert_called_once_with(runner, enable_memory_saver=False)

    def test_breakable_returns_breakable_backend(self):
        runner = _make_runner(Backend.BREAKABLE)
        sentinel = object()
        with patch(f"{_UTILS}.BreakableCudaGraphBackend") as mock_backend_cls:
            mock_backend_cls.return_value = sentinel
            backend = resolve_decode_backend(runner)

        self.assertIs(backend, sentinel)


class TestBuildCompilationConfigMoeSplitOp(CustomTestCase):
    @patch(
        "sglang.srt.model_executor.runner_backend."
        "tc_piecewise_cuda_graph_backend.get_moe_a2a_backend"
    )
    def test_expert_cache_enables_split_op(self, mock_a2a):
        config = _build_config(
            mock_a2a, expert_cache=True, decode_backend=Backend.FULL
        )
        self.assertIn(_MOE_SPLIT_OP, config.split_ops)

    @patch(
        "sglang.srt.model_executor.runner_backend."
        "tc_piecewise_cuda_graph_backend.get_moe_a2a_backend"
    )
    def test_deepep_still_enables_split_op(self, mock_a2a):
        config = _build_config(mock_a2a, deepep=True, expert_cache=False)
        self.assertIn(_MOE_SPLIT_OP, config.split_ops)

    @patch(
        "sglang.srt.model_executor.runner_backend."
        "tc_piecewise_cuda_graph_backend.get_moe_a2a_backend"
    )
    def test_mooncake_still_enables_split_op(self, mock_a2a):
        config = _build_config(mock_a2a, mooncake=True, expert_cache=False)
        self.assertIn(_MOE_SPLIT_OP, config.split_ops)

    @patch(
        "sglang.srt.model_executor.runner_backend."
        "tc_piecewise_cuda_graph_backend.get_moe_a2a_backend"
    )
    def test_no_duplicate_registration_when_both_true(self, mock_a2a):
        config = _build_config(mock_a2a, deepep=True, expert_cache=True)
        self.assertEqual(config.split_ops.count(_MOE_SPLIT_OP), 1)

    @patch(
        "sglang.srt.model_executor.runner_backend."
        "tc_piecewise_cuda_graph_backend.get_moe_a2a_backend"
    )
    def test_tc_piecewise_decode_opt_in_enables_split_op(self, mock_a2a):
        config = _build_config(mock_a2a, expert_cache=False)
        self.assertIn(_MOE_SPLIT_OP, config.split_ops)

    @patch(
        "sglang.srt.model_executor.runner_backend."
        "tc_piecewise_cuda_graph_backend.get_moe_a2a_backend"
    )
    def test_prefill_only_piecewise_config_stays_unregistered(self, mock_a2a):
        config = _build_config(
            mock_a2a,
            expert_cache=False,
            decode_backend=Backend.FULL,
        )
        self.assertNotIn(_MOE_SPLIT_OP, config.split_ops)


if __name__ == "__main__":
    unittest.main()
