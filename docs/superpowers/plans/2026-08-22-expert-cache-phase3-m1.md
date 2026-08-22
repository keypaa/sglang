# MoE Expert Cache Phase 3 (Milestone 1) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Wire the GPU-verified expert cache into the real `DeepseekV2MoE` forward path — expert weights stream from a pinned host store into a pool that IS the FusedMoE weight tensor, with slot-id remap between topk and the kernel.

**Architecture:** Pool-as-weights (spec §2): each layer's `FusedMoE` is constructed with `num_experts = num_slots`; its w13/w2 tensors are the static pool. A per-layer runtime guarantees routed-expert residency before the kernel reads; a device gather remaps `topk_ids → slot_ids`. Eager-only (`--disable-cuda-graph` required), TP1/EP1, bf16 parity gate first.

**Tech Stack:** PyTorch (pinned CPU tensors, non_blocking H2D), existing Phase-2 `expert_cache` package (`ExpertCache`, `StaticPoolOrchestrator`, `make_policy`, `CudaTransferBackend`, `HardwareSpec`, `ModelSpec`), SGLang ServerArgs, pytest + CustomTestCase.

## Global Constraints

- Spec: `docs/superpowers/specs/2026-08-22-expert-cache-phase3-design.md`
- Milestone 1 requires `--disable-cuda-graph`; error at startup otherwise.
- TP=1 and EP=1 only; startup error otherwise.
- `num_hash_layers > 0` configs are rejected when the cache is on.
- `num_fused_shared_experts` forced to 0 when the cache is on.
- bf16/fp16 unquantized checkpoints only in milestone 1 (fp8 = follow-up).
- All new code goes in `python/sglang/srt/layers/moe/expert_cache/` plus surgical edits to `server_args.py` and `models/deepseek_v2.py`. NO other model files.
- Tests live in `test/registered/unit/layers/moe/` and follow existing conventions: `register_cpu_ci(...)` / `register_cuda_ci(...)`, `CustomTestCase`.
- Local machine has NO CUDA: CUDA-gated tests must skip cleanly. Run CPU tests with:
  `PYTHONPATH=python python -m pytest <files> -q`
  (real package works locally; do NOT use any shim).
- Phase-2 suite must stay green after every task.

---

### Task 1: Server flags + startup validation

**Files:**
- Modify: `python/sglang/srt/server_args.py`
- Test: `test/registered/unit/layers/moe/test_expert_cache_server_args.py`

**Interfaces:**
- Consumes: nothing new.
- Produces: `ServerArgs.enable_moe_expert_cache: bool = False`, `ServerArgs.moe_cache_slots: int = 128`; module-level function `validate_moe_expert_cache(args) -> None` in `server_args.py` (raises `ValueError`), called from the existing ServerArgs validation path.

- [ ] **Step 1: Write the failing test**

```python
"""CPU units for --enable-moe-expert-cache startup guards."""

import pytest

from sglang.srt.server_args import ServerArgs, validate_moe_expert_cache
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

    def test_slots_clamped_to_routed_experts(self):
        slots = validate_moe_expert_cache(
            _args(disable_cuda_graph=True, moe_cache_slots=999)
        )
        assert slots == 256

    def test_ok_config_returns_slots(self):
        assert (
            validate_moe_expert_cache(_args(disable_cuda_graph=True)) == 128
        )
```

- [ ] **Step 2: Run test to verify it fails**

Run: `PYTHONPATH=python python -m pytest test/registered/unit/layers/moe/test_expert_cache_server_args.py -q`
Expected: FAIL — `ImportError: cannot import name 'validate_moe_expert_cache'`

- [ ] **Step 3: Implement**

In `server_args.py`, add fields next to the other MoE flags (follow the annotated style of `enable_eplb` around line 2309):

```python
    enable_moe_expert_cache: A[bool, "Stream MoE experts from pinned host RAM into a fixed VRAM pool (DeepSeek MoE)", NS("exec.moe")] = False
    moe_cache_slots: A[int, "Expert-cache pool slots per layer", NS("exec.moe")] = 128
```

Add the validator near the other validation helpers:

```python
def validate_moe_expert_cache(args) -> int:
    """Validate --enable-moe-expert-cache combos; returns clamped slot count."""
    if not args.enable_moe_expert_cache:
        return 0
    if not getattr(args, "disable_cuda_graph", False):
        raise ValueError(
            "--enable-moe-expert-cache milestone 1 requires "
            "--disable-cuda-graph (the router runs inside the captured decode "
            "graph; eager-only until that design lands)."
        )
    if getattr(args, "tp_size", 1) > 1 or getattr(args, "ep_size", 1) > 1:
        raise ValueError(
            "--enable-moe-expert-cache currently requires TP=1 and EP=1."
        )
    if args.moe_cache_slots < 1:
        raise ValueError("--moe-cache-slots must be >= 1.")
    # Clamp against the largest plausible routed-expert count; the model layer
    # re-clamps to config.n_routed_experts at init.
    n_routed = getattr(args, "moe_num_routed_experts_override", 0) or 256
    if args.moe_cache_slots > n_routed:
        logger.warning(
            "--moe-cache-slots %d > routed experts %d; clamping.",
            args.moe_cache_slots, n_routed,
        )
        return n_routed
    return args.moe_cache_slots
```

Call it from the existing ServerArgs validation method (same place other
feature guards raise, ~line 3754 region): `validate_moe_expert_cache(self)`.

Note: if `logger` is not already imported in server_args.py, use the module's
existing logging setup; check with grep before adding an import.

- [ ] **Step 4: Run test to verify it passes**

Run: `PYTHONPATH=python python -m pytest test/registered/unit/layers/moe/test_expert_cache_server_args.py -q`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add python/sglang/srt/server_args.py test/registered/unit/layers/moe/test_expert_cache_server_args.py
git commit -m "feat(expert_cache): server flags + startup guards for expert cache"
```

---

### Task 2: ExpertHostStore

**Files:**
- Create: `python/sglang/srt/layers/moe/expert_cache/host_store.py`
- Modify: `python/sglang/srt/layers/moe/expert_cache/__init__.py` (export)
- Test: `test/registered/unit/layers/moe/test_expert_host_store.py`

**Interfaces:**
- Consumes: `ExpertKey` from types.
- Produces:
  - `class HostStoreEntry(NamedTuple): w13: torch.Tensor; w2: torch.Tensor`
  - `class ExpertHostStore`: `__init__(self, num_layers: int, num_experts: int)`; `put(key: ExpertKey, w13: torch.Tensor, w2: torch.Tensor) -> None` (pins + stores; raises `RuntimeError` on duplicate key); `get(key: ExpertKey) -> HostStoreEntry`; `contains(key: ExpertKey) -> bool`; `expert_bytes(key) -> int`; property `total_bytes -> int`.

- [ ] **Step 1: Write the failing test**

```python
"""CPU units for the pinned expert host store."""

import torch

from sglang.srt.layers.moe.expert_cache import ExpertKey, ExpertHostStore
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


class TestExpertHostStore(CustomTestCase):
    def test_put_get_roundtrip_pins_memory(self):
        store = ExpertHostStore(num_layers=2, num_experts=4)
        w13 = torch.randn(8, 16)
        w2 = torch.randn(4, 8)
        store.put(ExpertKey(0, 1), w13, w2)
        entry = store.get(ExpertKey(0, 1))
        self.assertTrue(torch.equal(entry.w13, w13))
        self.assertTrue(torch.equal(entry.w2, w2))
        if torch.cuda.is_available():
            self.assertTrue(entry.w13.is_pinned())

    def test_duplicate_put_raises(self):
        store = ExpertHostStore(1, 2)
        e = (torch.randn(2, 2), torch.randn(2, 2))
        store.put(ExpertKey(0, 0), *e)
        with self.assertRaises(RuntimeError):
            store.put(ExpertKey(0, 0), *e)

    def test_missing_key_raises(self):
        store = ExpertHostStore(1, 2)
        with self.assertRaises(KeyError):
            store.get(ExpertKey(0, 7))

    def test_bytes_accounting(self):
        store = ExpertHostStore(1, 2)
        store.put(ExpertKey(0, 0), torch.zeros(8, 4), torch.zeros(4, 8))
        self.assertEqual(store.total_bytes, 8 * 4 * 4 + 4 * 8 * 4)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `PYTHONPATH=python python -m pytest test/registered/unit/layers/moe/test_expert_host_store.py -q`
Expected: FAIL — `ImportError: cannot import name 'ExpertHostStore'`

- [ ] **Step 3: Implement** (`host_store.py`)

```python
"""Pinned host-RAM store for all MoE expert weights (spec §5).

Holds every layer's expert blocks so device-side pools can be streamed
arbitrarily. Pinned memory is mandatory: pageable H2D measured ~3x slower
(benchmark/moe_expert_cache_bench.py).
"""
from __future__ import annotations

from typing import Dict, NamedTuple

import torch

from .types import ExpertKey


class HostStoreEntry(NamedTuple):
    w13: torch.Tensor
    w2: torch.Tensor


class ExpertHostStore:
    def __init__(self, num_layers: int, num_experts: int):
        self._num_layers = num_layers
        self._num_experts = num_experts
        self._entries: Dict[ExpertKey, HostStoreEntry] = {}
        self._total_bytes = 0

    def put(self, key: ExpertKey, w13: torch.Tensor, w2: torch.Tensor) -> None:
        if key in self._entries:
            raise RuntimeError(f"duplicate expert weight put: {key}")
        if not key.valid() or key.layer >= self._num_layers or key.expert >= self._num_experts:
            raise KeyError(f"key out of range for store: {key}")
        entry = HostStoreEntry(w13=self._pin(w13), w2=self._pin(w2))
        self._entries[key] = entry
        self._total_bytes += entry.w13.numel() * entry.w13.element_size()
        self._total_bytes += entry.w2.numel() * entry.w2.element_size()

    @staticmethod
    def _pin(t: torch.Tensor) -> torch.Tensor:
        t = t.detach().contiguous().cpu()
        return t.pin_memory() if torch.cuda.is_available() else t

    def get(self, key: ExpertKey) -> HostStoreEntry:
        return self._entries[key]  # raises KeyError on missing

    def contains(self, key: ExpertKey) -> bool:
        return key in self._entries

    def expert_bytes(self, key: ExpertKey) -> int:
        e = self._entries[key]
        return (e.w13.numel() + e.w2.numel()) * e.w13.element_size()

    @property
    def total_bytes(self) -> int:
        return self._total_bytes
```

Export `ExpertHostStore` and `HostStoreEntry` in `expert_cache/__init__.py`
imports + `__all__` (alphabetical order).

- [ ] **Step 4: Run test to verify it passes**

Run: `PYTHONPATH=python python -m pytest test/registered/unit/layers/moe/test_expert_host_store.py -q`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add python/sglang/srt/layers/moe/expert_cache/ test/registered/unit/layers/moe/test_expert_host_store.py
git commit -m "feat(expert_cache): pinned host store for all-layer expert weights"
```

---

### Task 3: Slot-id remap helper

**Files:**
- Create: `python/sglang/srt/layers/moe/expert_cache/slot_remap.py`
- Modify: `python/sglang/srt/layers/moe/expert_cache/__init__.py` (export)
- Test: `test/registered/unit/layers/moe/test_slot_remap.py`

**Interfaces:**
- Produces: `remap_topk_ids(topk_ids: torch.Tensor, device_slot_map: torch.Tensor) -> torch.Tensor` — takes `[num_tokens, top_k]` ids (int32 or int64) and an int32 device tensor `[num_experts]`; returns same-shape/dtype slot ids via `torch.index_select` on the flattened input.

- [ ] **Step 1: Write the failing test**

```python
"""CPU units for the topk->slot id remap."""

import torch

from sglang.srt.layers.moe.expert_cache import remap_topk_ids
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


class TestSlotRemap(CustomTestCase):
    def test_remaps_each_id_independently(self):
        # experts 0..7 -> slots 5,3,1,0,7,6,4,2
        m = torch.tensor([5, 3, 1, 0, 7, 6, 4, 2], dtype=torch.int32)
        ids = torch.tensor([[0, 1], [2, 3]], dtype=torch.int64)
        out = remap_topk_ids(ids, m)
        self.assertEqual(out.tolist(), [[5, 3], [1, 0]])
        self.assertEqual(out.dtype, torch.int64)

    def test_preserves_int32(self):
        m = torch.arange(10, dtype=torch.int32)
        ids = torch.tensor([[4, 9]], dtype=torch.int32)
        self.assertEqual(remap_topk_ids(ids, m).dtype, torch.int32)

    def test_layer_maps_are_independent_inputs(self):
        # caller picks which layer's map to pass; helper is stateless
        m0 = torch.tensor([9, 8], dtype=torch.int32)
        m1 = torch.tensor([1, 2], dtype=torch.int32)
        ids = torch.tensor([[0, 1]], dtype=torch.int64)
        self.assertEqual(remap_topk_ids(ids, m0).tolist(), [[9, 8]])
        self.assertEqual(remap_topk_ids(ids, m1).tolist(), [[1, 2]])
```

- [ ] **Step 2: Run test to verify it fails**

Run: `PYTHONPATH=python python -m pytest test/registered/unit/layers/moe/test_slot_remap.py -q`
Expected: FAIL — ImportError on `remap_topk_ids`

- [ ] **Step 3: Implement** (`slot_remap.py`)

```python
"""topk expert-id -> pool-slot-id remap (spec §4).

The MoE kernel gathers weight rows by "expert id"; passing it slot ids makes
it read exactly the streamed rows. Pure gather, no allocation beyond output.
"""
from __future__ import annotations

import torch


def remap_topk_ids(topk_ids: torch.Tensor, device_slot_map: torch.Tensor) -> torch.Tensor:
    flat = topk_ids.reshape(-1).long()
    out = torch.index_select(device_slot_map, 0, flat)
    return out.reshape(topk_ids.shape).to(topk_ids.dtype)
```

Export `remap_topk_ids` in `__init__.py`.

- [ ] **Step 4: Run test to verify it passes**

Run: `PYTHONPATH=python python -m pytest test/registered/unit/layers/moe/test_slot_remap.py -q`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add python/sglang/srt/layers/moe/expert_cache/ test/registered/unit/layers/moe/test_slot_remap.py
git commit -m "feat(expert_cache): slot-id remap for pool-as-weights MoE"
```

---

### Task 4: Per-layer runtime (residency + copies into arbitrary weight params)

**Files:**
- Create: `python/sglang/srt/layers/moe/expert_cache/layer_runtime.py`
- Test: `test/registered/unit/layers/moe/test_expert_cache_layer_runtime.py`

**Interfaces:**
- Consumes: `ExpertCache`, `StaticPoolOrchestrator`, `ExpertHostStore` (Task 2).
- Produces:
  - `class LayerRuntime`: `__init__(self, layer_id: int, cache: ExpertCache, orchestrator: StaticPoolOrchestrator, store: ExpertHostStore, w13_weight: torch.Tensor, w2_weight: torch.Tensor)`; `ensure_resident(expert_ids: Sequence[int]) -> None` (per-expert: if not resident, H2D `store.get(...)` into `w13_weight[slot]/w2_weight[slot]` via the orchestrator's normal acquire path); `stats -> CacheStats`.

  DESIGN NOTE for implementer: the orchestrator's backend already writes into
  `pool.copy_in(slot_id, w13, w2)`. The real FusedMoE weights are plain
  tensors, not a StaticExpertPool. Provide a minimal adapter object exposing
  `copy_in(slot_id, w13, w2)` / `record_step()` / `num_slots` /
  `wait_all()` over the weight tensors (the FakePool contract from Phase 2),
  implemented as `RealWeightPool` in this file:

```python
class RealWeightPool:
    """Adapts FusedMoE weight params to the pool surface copy_in/record_step."""

    def __init__(self, w13_weight: torch.Tensor, w2_weight: torch.Tensor,
                 transfer_stream: Optional[torch.cuda.Stream] = None):
        assert w13_weight.dim() == 3 and w2_weight.dim() == 3
        self.w13 = w13_weight
        self.w2 = w2_weight
        self.num_slots = w13_weight.shape[0]
        self._stream = transfer_stream or (
            torch.cuda.Stream() if torch.cuda.is_available() else None
        )
        self._pending = 0

    def copy_in(self, slot_id: int, w13: torch.Tensor, w2: torch.Tensor) -> None:
        assert 0 <= slot_id < self.num_slots
        assert self.w13[slot_id].shape == w13.shape and self.w13.dtype == w13.dtype
        assert self.w2[slot_id].shape == w2.shape and self.w2.dtype == w2.dtype
        if self._stream is not None:
            with torch.cuda.stream(self._stream):
                self.w13[slot_id].copy_(w13, non_blocking=True)
                self.w2[slot_id].copy_(w2, non_blocking=True)
        else:
            self.w13[slot_id].copy_(w13)
            self.w2[slot_id].copy_(w2)
        self._pending += 1

    def record_step(self) -> bool:
        if self._pending == 0:
            return False
        self._pending = 0
        return True

    def wait_all(self) -> None:
        if self._pending == 0:
            return
        if self._stream is not None:
            self._stream.synchronize()
```

  And `LayerRuntime.ensure_resident` wires it together:

```python
class LayerRuntime:
    def __init__(self, layer_id, cache, store, w13_weight, w2_weight,
                 num_experts, num_layers=1):
        self.layer_id = layer_id
        self.pool = RealWeightPool(w13_weight, w2_weight)
        self.cache = cache          # shared across layers
        self.store = store
        self.orch = StaticPoolOrchestrator(cache, self.pool, num_experts, 1)
        self.device_map = None      # lazily created device mirror (CUDA)

    def ensure_resident(self, expert_ids):
        logits = [0.9] * len(expert_ids)
        self.orch.on_router_output(0, list(expert_ids), logits, next_ids=[])
        self.orch.step_commit()
        self.pool.wait_all()

    def slot_map_device(self, device):
        if self.device_map is None:
            self.device_map = torch.full((self.orch._num_experts,), -1,
                                         dtype=torch.int32, device=device)
        self.orch.copy_slot_map_into(self.device_map)
        return self.device_map
```

  NOTE: `cache.prefetch_budget()`/prefetch stay available through the shared
  cache; cross-layer prefetch hints are milestone-2 (single-layer runtime here;
  `num_layers=1` in the orchestrator keeps the slot map per-layer by design).

- Test plan (CPU): build a small `ExpertCache`+`SimBackend`+policy, a
  `FakePool`-style `RealWeightPool` on CPU (stream=None path), a store with
  known weights; call `ensure_resident([e])` and assert the mapped slot row
  equals the stored weights and `slot_id_of(0, e) >= 0`.

- [ ] **Step 1: Write the failing test** — implement the test file with one test class covering: roundtrip content correctness, second call is a hit (no reload: `cache.stats.loads` unchanged), and slot map mirror matches `slot_id_of`. Use `torch.manual_seed(0)` weights shaped `(4, 6)` / `(6, 4)`, num_slots=4, num_experts=8.
- [ ] **Step 2: Run — expect ImportError on `LayerRuntime`.**
- [ ] **Step 3: Implement** `layer_runtime.py` with `RealWeightPool` + `LayerRuntime` exactly as above (add imports: `Optional`, `Sequence`, `torch`, and the expert_cache imports).
- [ ] **Step 4: Run — PASS.**
- [ ] **Step 5: Commit** — `feat(expert_cache): per-layer runtime streaming experts into FusedMoE weight params`

---

### Task 5: DeepseekV2MoE wiring — init + forward hook

**Files:**
- Modify: `python/sglang/srt/models/deepseek_v2.py` (~line 551 init, ~line 1052 forward_normal)
- Test: `test/registered/unit/layers/moe/test_deepseek_moe_expert_cache_wiring.py` (CUDA-gated smoke; full behavior gated by Task 7 parity)

**Interfaces:**
- Consumes: Tasks 1–4.
- Produces: on `DeepseekV2MoE` when enabled: `self.expert_cache_runtime: LayerRuntime`, `self.expert_cache_enabled: bool`; forward returns identical-shape outputs with slot-remapped experts call.

- [ ] **Step 1: Write the failing smoke test (CUDA-gated)**

```python
"""Wiring smoke: tiny DeepseekV2MoE constructs with the cache enabled."""

import unittest

import torch

from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.test_utils import CustomTestCase

register_cuda_ci(est_time=60, suite="base-a-test-gpu")


@unittest.skipUnless(torch.cuda.is_available(), "CUDA required")
class TestWiringSmoke(CustomTestCase):
    def test_forward_normal_uses_pool_weights(self):
        # Full construction needs a PretrainedConfig + quant_config=None and
        # get_server_args patched to enable the cache. Build a minimal
        # DeepseekV2Config-like namespace with n_routed_experts=8,
        # num_experts_per_tok=2, moe_intermediate_size=32, hidden_size=32,
        # num_hash_layers=0, n_shared_experts=0, routed_scaling_factor=1.0,
        # hidden_act="silu".
        #
        # Patch sglang.srt.models.deepseek_v2.get_server_args to return an
        # object with enable_moe_expert_cache=True, moe_cache_slots=4,
        # disable_cuda_graph=True.
        #
        # Assert: moe.expert_cache_enabled is True;
        #         moe.experts.w13_weight.shape[0] == 4 (pool-sized);
        #         a fixed hidden_states [3, 32] through moe.forward_normal
        #         returns shape [3, 32] without error after
        #         runtime.ensure_resident seeded experts 0..3 manually into
        #         the store.
        raise NotImplementedError("see comments above — implemented in Step 3")
```

Implementer note: write the real body following the comments; if
construction pulls in too much engine state, factor the construction into a
helper `_build_tiny_test_moe()` inside the test file and keep assertions as
listed. If some engine-global (e.g. `get_exec().moe.*`) resists patching,
patch it the same way via `unittest.mock.patch.object`.

- [ ] **Step 2: Run — expect FAIL (NotImplementedError / attribute missing).**
- [ ] **Step 3: Implement in deepseek_v2.py:**

In `DeepseekV2MoE.__init__` after `self.topk` is set (~line 707):

```python
        _sa = get_server_args()
        self.expert_cache_enabled = bool(
            getattr(_sa, "enable_moe_expert_cache", False)
        )
        if self.expert_cache_enabled:
            if getattr(config, "num_hash_layers", 0) > 0:
                raise ValueError("expert cache unsupported with hash layers")
            if self.num_fused_shared_experts != 0:
                raise ValueError("expert cache requires fused shared experts off")
            num_slots = min(int(getattr(_sa, "moe_cache_slots", 128)),
                            num_experts_for_moe)
            # Pool-as-weights: rebuild experts sized to the pool.
            self.experts = get_moe_impl_class(quant_config)(
                num_experts=num_slots,
                num_fused_shared_experts=0,
                top_k=top_k_for_moe,
                hidden_size=config.hidden_size,
                intermediate_size=config.moe_intermediate_size,
                layer_id=self.layer_id,
                quant_config=quant_config,
                routed_scaling_factor=self.routed_scaling_factor,
                routing_method_type=getattr(
                    config, "routing_method_type", RoutingMethodType.DeepSeekV3
                ),
                swiglu_limit=getattr(config, "swiglu_limit", None),
                prefix=add_prefix("experts", prefix),
            )
            self.moe_cache_num_slots = num_slots
```

(The shared `ExpertCache` + `LayerRuntime` objects are attached later, in
Task 6/7 once the host store exists — the model exposes
`moe_cache_num_slots` for that.)

In `forward_normal`, immediately before `final_hidden_states = self.experts(...)` (~line 1140), insert the remap:

```python
        if self.expert_cache_enabled:
            from sglang.srt.layers.moe.expert_cache import remap_topk_ids

            ids_host = topk_output.topk_ids.to(torch.long).tolist()
            self.expert_cache_runtime.ensure_resident(sorted(set(ids_host)))
            slot_ids = remap_topk_ids(
                topk_output.topk_ids,
                self.expert_cache_runtime.slot_map_device(
                    topk_output.topk_ids.device
                ),
            )
            topk_output = topk_output._replace(topk_ids=slot_ids)
```

- [ ] **Step 4: Run smoke — PASS on CUDA; SKIPS locally. Verify collection locally:**
  `PYTHONPATH=python python -m pytest test/registered/unit/layers/moe/test_deepseek_moe_expert_cache_wiring.py -q` → 1 skipped.
- [ ] **Step 5: Commit** — `feat(expert_cache): DeepseekV2MoE pool-as-weights wiring (init + eager remap)`

---

### Task 6: load_weights interception (bf16)

**Files:**
- Modify: `python/sglang/srt/models/deepseek_v2.py` (load_weights path, line ~3112 region; follow `do_load_weights`)
- Test: extend `test/registered/unit/layers/moe/test_deepseek_moe_expert_cache_wiring.py` (CUDA-gated)

**Interfaces:**
- Produces: module-level pure function in deepseek_v2.py:
  `parse_expert_weight_name(name: str) -> Optional[Tuple[int, str]]` returning `(expert_index, kind)` where kind ∈ {"w13", "w2"} for names matching `mlp.experts.{i}.w13_weight` / `.w2_weight` (no scale suffixes), else `None`. When the cache is on, these tensors go to the shared `ExpertHostStore` instead of device params; pool rows start uninitialized.

- [ ] **Step 1: Write failing unit (CPU-safe) for the parser:**

```python
class TestParseExpertWeightName(CustomTestCase):
    def test_matches(self):
        from sglang.srt.models.deepseek_v2 import parse_expert_weight_name
        self.assertEqual(parse_expert_weight_name(
            "model.layers.3.mlp.experts.17.w13_weight"), (17, "w13"))
        self.assertEqual(parse_expert_weight_name(
            "model.layers.3.mlp.experts.0.w2_weight"), (0, "w2"))

    def test_non_expert_names(self):
        from sglang.srt.models.deepseek_v2 import parse_expert_weight_name
        self.assertIsNone(parse_expert_weight_name(
            "model.layers.3.mlp.shared_experts.gate_up_proj.weight"))
        self.assertIsNone(parse_expert_weight_name(
            "model.layers.3.self_attn.qkv_proj.weight"))
        self.assertIsNone(parse_expert_weight_name(
            "model.layers.3.mlp.experts.17.w13_weight_scale_inv"))
```

- [ ] **Step 2: Run — FAIL (function missing).**
- [ ] **Step 3: Implement parser + interception.** Parser with a compiled regex
  `r"mlp\.experts\.(\d+)\.(w13_weight|w2_weight)$"`. Interception: in the
  weight-loading loop where params are matched by name, if
  `enable_moe_expert_cache` and the parser matches, write
  `(w13|w2)` into the model-level `ExpertHostStore`
  (`self.moe_expert_host_store[expert][kind] = tensor.cpu()`) and `continue`
  (skip param loading). The store + per-layer `LayerRuntime`s are created
  lazily on first expert tensor (all layers share one `ExpertCache` built with
  `ModelSpec(num_layers=config.num_hidden_layers, num_experts=n_routed,
  top_k=num_experts_per_tok, shared_experts=0, hidden_size,
  intermediate_size)` and `make_policy("logitgds", num_slots)`;
  `CudaTransferBackend` is NOT used here — `SimBackend(25e9)` because
  RealWeightPool does the actual copies). After all weights are loaded, pin
  every store entry (reuse `ExpertHostStore.put`) — see note: collect raw
  CPU tensors during iteration, then bulk-`put` at end to keep pinning off
  the hot loop.
- [ ] **Step 4: Run parser test PASS; wiring smoke still skips locally.**
- [ ] **Step 5: Commit** — `feat(expert_cache): intercept expert weights into pinned host store`

---

### Task 7: Tiny-model parity gate (GPU) — THE milestone gate

**Files:**
- Test: `test/registered/unit/layers/moe/test_deepseek_moe_expert_cache_parity.py`

**Interfaces:**
- Consumes: Tasks 1–6.
- Produces: the milestone gate — cached vs uncached forward parity.

- [ ] **Step 1: Write the parity test (CUDA-gated)**

Structure (implement fully; no placeholders):

```python
@unittest.skipUnless(torch.cuda.is_available(), "CUDA required")
class TestTinyModelParity(CustomTestCase):
    def test_cached_matches_uncached_logits(self):
        # 1. Build ONE tiny DeepseekV2Config (n_routed_experts=8,
        #    num_experts_per_tok=2, moe_intermediate_size=64, hidden_size=64,
        #    num_hidden_layers=2, num_hash_layers=0, n_shared_experts=0,
        #    routed_scaling_factor=1.0, hidden_act="silu") and construct
        #    DeepseekV2ForCausalLM twice under different patched server args:
        #      A: enable_moe_expert_cache=False
        #      B: enable_moe_expert_cache=True, moe_cache_slots=4,
        #         disable_cuda_graph=True
        #    (use unittest.mock.patch.object on
        #     sglang.srt.models.deepseek_v2.get_server_args and any
        #     get_exec().moe globals that resist default construction.)
        # 2. Seed BOTH models identically: iterate A.state_dict(), copy each
        #    param into B where shapes match; for B's expert layers (shapes
        #    differ: [slots,...]), route A's expert weights through B's
        #    host-store interception path (call the Task 6 store population
        #    directly with A's expert params).
        # 3. Fixed input: torch.manual_seed(42); tokens = torch.randint(
        #    0, vocab_size, (4,)); run both models' forward (eval, no_grad)
        #    on the same embeddings path — compare final logits.
        # 4. Assert torch.allclose(a_logits, b_logits, atol=2e-2, rtol=2e-2)
        #    AND argmax token equality over a short greedy continuation
        #    (>= 8 tokens identical).
```

Tolerance rationale: cached path recomputes the same GEMMs from copied
bytes — bitwise-equal weights give bitwise-close outputs; the tolerance
absorbs only non-deterministic kernel dispatch differences. If mismatches
appear, FIRST verify pool-row bytes equal store bytes (`torch.equal`) before
loosening tolerance — a byte mismatch is a bug, not tolerance.

- [ ] **Step 2: Run on CUDA host (Modal): expect PASS. Locally: SKIPS.**
- [ ] **Step 3: Extend `/tmp/opencode/modal_gpu_verify.py` TEST_FILES with the two new CUDA-gated files and run the Modal verification; require all-pass.**
- [ ] **Step 4: Commit** — `test(expert_cache): tiny-model parity gate for pool-as-weights integration`

---

### Task 8: Telemetry + docs

**Files:**
- Modify: `python/sglang/srt/layers/moe/expert_cache/layer_runtime.py` (stall accumulator)
- Modify: `docs/expert-cache.md` (Phase-3 section)
- Test: extends Task 4 test file (stall accounting)

**Interfaces:**
- Produces: `LayerRuntime.telemetry() -> dict` with keys `hits, misses, loads, stall_ms_total, stall_ms_last` (stall = wall time inside `ensure_resident` when copies were pending).

- [ ] **Step 1: Failing test:** after seeding, wrap `ensure_resident` timing: first call (misses) has `stall_ms_last >= 0` and `loads >= 1`; second call (hits) leaves `loads` unchanged and resets `stall_ms_last == 0`.
- [ ] **Step 2: Run — FAIL.**
- [ ] **Step 3: Implement** using `time.perf_counter()` around the residency block; record into instance fields; expose `telemetry()`. Add a Phase-3 section to `docs/expert-cache.md`: what's wired, flags, constraints (bf16/eager/TP1/big-RAM host), telemetry keys.
- [ ] **Step 4: Run — PASS.**
- [ ] **Step 5: Commit** — `feat(expert_cache): per-layer stall/hit telemetry + docs`

---

## Self-Review Notes

- Spec coverage: §3 architecture → Tasks 1/2/4/5; §4 data flow → Tasks 3/5; §5 loading → Task 6; §6 errors → Tasks 1/5 (guards); §7 testing → Tasks 4/5/6/7 units + parity gate; telemetry spec §7.3 → Task 8. fp8 deferred per decision-log #5 (documented in Task list header + spec).
- Type consistency: `RealWeightPool.copy_in/record_step/wait_all/num_slots` matches the FakePool contract the orchestrator already consumes; `remap_topk_ids(topk_ids, device_slot_map)` used identically in Tasks 3/5; `ExpertHostStore.put/get` signatures consistent between Tasks 2/4/6.
- Risk flagged for executors: Tasks 5–7 touch `deepseek_v2.py`, a large frozen-ish model file — keep edits surgical, follow large-class-style conventions, and prefer patching engine globals in tests over editing them.
