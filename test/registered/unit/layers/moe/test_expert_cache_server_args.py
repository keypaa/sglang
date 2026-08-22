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
