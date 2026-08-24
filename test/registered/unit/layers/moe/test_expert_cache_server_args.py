"""CPU units for --enable-moe-expert-cache startup guards."""

import pytest
from sglang.srt.server_args import validate_moe_expert_cache
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


def _args(**kw):
    defaults = dict(
        enable_moe_expert_cache=True,
        disable_cuda_graph=False,
        tp_size=1,
        moe_cache_slots=128,
    )
    defaults.update(kw)
    return defaults


class TestMoeExpertCacheGuards(CustomTestCase):
    def test_requires_disable_cuda_graph(self):
        with pytest.raises(ValueError, match="disable-cuda-graph"):
            validate_moe_expert_cache(_args())

    def test_accepts_tc_piecewise_decode(self):
        assert (
            validate_moe_expert_cache(
                _args(cuda_graph_backend_decode="tc_piecewise")
            )
            == 128
        )

    def test_accepts_tc_piecewise_with_disable_cuda_graph(self):
        assert (
            validate_moe_expert_cache(
                _args(
                    disable_cuda_graph=True,
                    cuda_graph_backend_decode="tc_piecewise",
                )
            )
            == 128
        )

    def test_rejects_full_backend(self):
        with pytest.raises(ValueError, match="tc_piecewise"):
            validate_moe_expert_cache(_args(cuda_graph_backend_decode="full"))

    def test_rejects_breakable_backend(self):
        with pytest.raises(ValueError, match="tc_piecewise"):
            validate_moe_expert_cache(
                _args(cuda_graph_backend_decode="breakable")
            )

    def test_rejects_disabled_backend(self):
        with pytest.raises(ValueError, match="disable-cuda-graph"):
            validate_moe_expert_cache(
                _args(cuda_graph_backend_decode="disabled")
            )

    def test_bootable_cache_on_decodes_are_eager_or_tc_piecewise(self):
        """decode=full with cache-on is rejected at startup, so the
        ``enable_moe_expert_cache`` term in
        TcPiecewiseCudaGraphBackend.build_compilation_config's split-op
        predicate only ever evaluates on bootable configs."""
        with pytest.raises(ValueError, match="tc_piecewise"):
            validate_moe_expert_cache(_args(cuda_graph_backend_decode="full"))
        assert validate_moe_expert_cache(_args(disable_cuda_graph=True)) == 128
        assert (
            validate_moe_expert_cache(
                _args(cuda_graph_backend_decode="tc_piecewise")
            )
            == 128
        )

    def test_rejects_tp(self):
        with pytest.raises(ValueError, match="tp-size"):
            validate_moe_expert_cache(_args(disable_cuda_graph=True, tp_size=2))

    def test_rejects_speculative_decoding(self):
        with pytest.raises(ValueError, match="speculative"):
            validate_moe_expert_cache(
                _args(disable_cuda_graph=True, speculative_algorithm="EAGLE")
            )

    def test_slots_clamped_to_routed_experts(self):
        slots = validate_moe_expert_cache(
            _args(disable_cuda_graph=True, moe_cache_slots=999)
        )
        assert slots == 256

    def test_ok_config_returns_slots(self):
        assert (
            validate_moe_expert_cache(_args(disable_cuda_graph=True)) == 128
        )
