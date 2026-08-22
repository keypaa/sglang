"""Locality-aware expert caching for MoE inference engines.

A Python port of the validated C++ MoE-LRU simulator
(/home/keypaa/Projects/DSV4-0731/MoE-LRU). Implements the LogitGDS eviction
policy (router-confidence x Greedy-Dual aging), the fixed-slot ExpertCache
lifecycle (LOADING/READY/TRANSIENT, refcounting, prefetch budget), and a
discrete-event simulator used to validate the port against the C++ reference.

Phase 1 scope: policy + cache + transfer backends + simulator, all in pure
Python. A later phase wires these into the DeepSeek-V4 MoE forward pass.
"""

from .backends import CudaTransferBackend, SimBackend, TransferBackend
from .expert_cache import ExpertCache, Slot
from .freq_sketch import FreqSketch
from .host_store import ExpertHostStore, HostStoreEntry
from .orchestrator import StaticPoolOrchestrator
from .policies import (
    EvictionPolicy,
    KeyNode,
    LRUPolicy,
    LogitGDSPolicy,
    TinyLFUPolicy,
    make_policy,
)
from .simulator import SimResult, TraceGenerator, run_simulation
from .slot_remap import remap_topk_ids
from .static_pool import StaticExpertPool
from .types import (
    CacheStats,
    ExpertKey,
    HardwareSpec,
    ModelSpec,
    RouterChoice,
    SlotState,
)

__all__ = [
    "CacheStats",
    "CudaTransferBackend",
    "EvictionPolicy",
    "ExpertCache",
    "ExpertHostStore",
    "ExpertKey",
    "FreqSketch",
    "HardwareSpec",
    "HostStoreEntry",
    "KeyNode",
    "LRUPolicy",
    "LogitGDSPolicy",
    "ModelSpec",
    "RouterChoice",
    "SimBackend",
    "SimResult",
    "Slot",
    "SlotState",
    "StaticExpertPool",
    "StaticPoolOrchestrator",
    "TinyLFUPolicy",
    "TraceGenerator",
    "TransferBackend",
    "make_policy",
    "remap_topk_ids",
    "run_simulation",
]
