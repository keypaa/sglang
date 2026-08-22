"""Physics benchmark for the MoE expert cache (Phase-3 de-risk).

Measures, with DeepSeek-V4-Flash-0731-realistic shapes, the numbers that decide
whether expert streaming is viable on a 24 GB consumer GPU:

  A. Pinned vs pageable H2D copy latency/bandwidth for one expert block
     (w13 [H,2I] + w2 [I,H], fp8) and for a top_k burst.
  B. Transfer/compute stream overlap: copies on the transfer stream while the
     compute stream runs decode-shaped GEMMs — concurrent vs serial.
  C. Captured-graph decode step latency through StaticExpertPool +
     StaticPoolOrchestrator: all-hit steady state vs demand-miss steps
     (the per-miss stall a Phase-3 integration would pay).

Run on a CUDA host:
    python benchmark/moe_expert_cache_bench.py
"""

import time

import torch

from sglang.srt.layers.moe.expert_cache import (
    CudaTransferBackend,
    ExpertCache,
    ExpertKey,
    HardwareSpec,
    ModelSpec,
    StaticPoolOrchestrator,
    make_policy,
)
from sglang.srt.layers.moe.expert_cache.static_pool import StaticExpertPool


def bench_copy_bandwidth(m: ModelSpec, num_experts: int):
    """A: one-expert-block H2D latency + achieved bandwidth, pinned vs pageable."""
    w13_shape, w2_shape = m.expert_shapes()
    block_bytes = (
        w13_shape[0] * w13_shape[1] + w2_shape[0] * w2_shape[1]
    )  # fp8 = 1 byte/elem, w13 + w2

    pinned = {
        e: (
            torch.randn(w13_shape).to(torch.float8_e4m3fn).pin_memory(),
            torch.randn(w2_shape).to(torch.float8_e4m3fn).pin_memory(),
        )
        for e in range(num_experts)
    }
    pageable = {
        e: (t[0].clone(), t[1].clone()) for e, t in pinned.items()  # un-pinned
    }
    dst = (
        torch.empty(w13_shape, device="cuda", dtype=torch.float8_e4m3fn),
        torch.empty(w2_shape, device="cuda", dtype=torch.float8_e4m3fn),
    )

    def copy_one(src):
        dst[0].copy_(src[0], non_blocking=True)
        dst[1].copy_(src[1], non_blocking=True)

    results = {}
    for name, store in (("pinned", pinned), ("pageable", pageable)):
        # warmup
        for e in range(4):
            copy_one(store[e])
        torch.cuda.synchronize()

        # single block latency (avg over experts)
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        for e in range(num_experts):
            copy_one(store[e])
        end.record()
        torch.cuda.synchronize()
        total_ms = start.elapsed_time(end)
        per_block_ms = total_ms / num_experts
        results[name] = (per_block_ms, block_bytes / per_block_ms / 1e3)  # ms, GB/s

        # top_k burst: k blocks back-to-back, one sync
        k = m.top_k
        start.record()
        for e in range(k):
            copy_one(store[e])
        end.record()
        torch.cuda.synchronize()
        results[name + f"_top{k}_burst_ms"] = start.elapsed_time(end)

    return block_bytes, results


def bench_overlap(m: ModelSpec, num_experts: int, iters: int = 20):
    """B: do transfer-stream copies overlap compute-stream GEMMs?"""
    w13_shape, w2_shape = m.expert_shapes()
    H, I = m.hidden_size, m.intermediate_size
    pinned = {
        e: (
            torch.randn(w13_shape).to(torch.float8_e4m3fn).pin_memory(),
            torch.randn(w2_shape).to(torch.float8_e4m3fn).pin_memory(),
        )
        for e in range(num_experts)
    }
    dst = (
        torch.empty(num_experts, *w13_shape, device="cuda", dtype=torch.float8_e4m3fn),
        torch.empty(num_experts, *w2_shape, device="cuda", dtype=torch.float8_e4m3fn),
    )
    transfer_stream = torch.cuda.Stream()
    compute_stream = torch.cuda.Stream()

    # decode-shaped compute: top_k experts' GEMMs for one token, bf16
    acts = torch.randn(1, H, device="cuda", dtype=torch.bfloat16)
    w13_g = torch.randn(m.top_k, H, 2 * I, device="cuda", dtype=torch.bfloat16)
    w2_g = torch.randn(m.top_k, I, H, device="cuda", dtype=torch.bfloat16)

    def compute_step():
        up = torch.matmul(acts, w13_g)
        gate, act = up.chunk(2, dim=-1)
        mid = gate * torch.nn.functional.gelu(act)
        return torch.bmm(mid, w2_g)

    def copies(step):
        with torch.cuda.stream(transfer_stream):
            for e in range(m.top_k):
                dst[0][e].copy_(pinned[e][0], non_blocking=True)
                dst[1][e].copy_(pinned[e][1], non_blocking=True)

    def time_wall(fn, iters):
        """Wall-clock per-iteration time; fn must leave streams drained."""
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        for i in range(iters):
            fn(i)
        torch.cuda.synchronize()
        return (time.perf_counter() - t0) / iters * 1e3

    # Barrier-every-step: copies + compute, full sync each iteration.
    def barrier_step(i):
        copies(i)
        compute_step()
        torch.cuda.synchronize()

    # Pipelined: enqueue every iteration's copies + compute, drain once.
    # This models steady-state decode where layer L+1's copies overlap
    # layer L's compute (spec section 5 intent).
    def pipelined(i):
        copies(i)
        compute_step()

    barrier_ms = time_wall(barrier_step, iters)
    pipelined_ms = time_wall(pipelined, iters)
    compute_only_ms = time_wall(lambda i: compute_step(), iters)
    return {"barrier_every_step_ms": barrier_ms, "pipelined_ms": pipelined_ms,
            "compute_only_ms": compute_only_ms}


def bench_orchestrated_steps(
    m: ModelSpec, hw: HardwareSpec, num_slots: int = 128, steps: int = 50
):
    """C: orchestrated decode step latency — all-hit vs demand-miss, captured."""
    pool = StaticExpertPool(m, hw, None, num_slots, dtype=torch.float8_e4m3fn)
    backend = CudaTransferBackend(h2d_bw=hw.h2d_bw)
    backend.set_pool(pool)
    cache = ExpertCache(m, hw, make_policy("logitgds", num_slots), backend)
    orch = StaticPoolOrchestrator(cache, pool, m.num_experts, 1)

    H, I = m.hidden_size, m.intermediate_size
    w13_shape, w2_shape = m.expert_shapes()
    store = {
        ExpertKey(0, e): (
            torch.randn(w13_shape).to(torch.float8_e4m3fn).pin_memory(),
            torch.randn(w2_shape).to(torch.float8_e4m3fn).pin_memory(),
        )
        for e in range(m.num_experts)
    }
    backend.set_expert_source(lambda k: store[k])

    top_k = m.top_k
    activations = torch.randn(H, device="cuda", dtype=torch.bfloat16)
    topk_ids_buf = torch.zeros(1, top_k, dtype=torch.int64, device="cuda")
    slot_map_buf = orch.slot_map_tensor().to(dtype=torch.int64, device="cuda")
    out_buf = torch.zeros(top_k, H, device="cuda", dtype=torch.bfloat16)

    def ffn(w13, w2):
        up = torch.matmul(activations, w13.to(torch.bfloat16))
        gate, act = up.chunk(2, dim=-1)
        mid = gate * torch.nn.functional.gelu(act)
        return torch.bmm(mid.unsqueeze(1), w2.to(torch.bfloat16)).squeeze(1)

    def forward():
        ids = topk_ids_buf.view(-1)
        slot_ids = slot_map_buf.index_select(0, ids)
        out_buf.copy_(ffn(pool.pool_w13()[slot_ids], pool.pool_w2()[slot_ids]))

    # warmup (cuBLAS init) then capture
    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        forward()
    torch.cuda.current_stream().wait_stream(s)
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g, stream=torch.cuda.Stream()):
        forward()

    def run_step(ids):
        orch.on_router_output(0, ids, [0.9] * len(ids), next_ids=[])
        orch.step_commit()
        pool.wait_all()  # Phase-A replay-time barrier (see spec §5 amendment)
        topk_ids_buf.copy_(torch.tensor([ids], dtype=torch.int64, device="cuda"))
        slot_map_buf.copy_(
            orch.slot_map_tensor().to(dtype=torch.int64, device="cuda")
        )
        g.replay()
        torch.cuda.synchronize()

    # warm the cache: first num_slots experts resident
    for start_e in range(0, m.num_experts, top_k):
        ids = [(start_e + j) % m.num_experts for j in range(top_k)]
        run_step(ids)
    torch.cuda.synchronize()

    # all-hit: repeatedly route the same resident set
    resident_ids = list(range(top_k))
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    n_hit = 30
    for _ in range(n_hit):
        run_step(resident_ids)
    hit_ms = (time.perf_counter() - t0) / n_hit * 1e3

    # demand-miss: cycle through ALL experts with capacity 128/256 -> misses
    t0 = time.perf_counter()
    for i in range(steps):
        ids = [(i * top_k + j) % m.num_experts for j in range(top_k)]
        run_step(ids)
    miss_ms = (time.perf_counter() - t0) / steps * 1e3

    stats = cache.stats
    misses_per_step = stats.misses / max(1, steps)
    return {
        "all_hit_step_ms": hit_ms,
        "miss_step_ms": miss_ms,
        "misses_per_step": misses_per_step,
        "stall_per_miss_us": (miss_ms - hit_ms) / max(misses_per_step, 1e-9) * 1e3,
        "hit_rate_steps": stats.hit_rate,
    }


def main():
    assert torch.cuda.is_available(), "requires CUDA"
    m = ModelSpec()  # 43 layers, 256 experts, top_k 6, H=3840, I=1536
    hw = HardwareSpec()
    print(f"model: {m.num_layers}L x {m.num_experts}E top_k={m.top_k} "
          f"H={m.hidden_size} I={m.intermediate_size}")
    print(f"assumed H2D bw: {hw.h2d_bw/1e9:.0f} GB/s\n")

    block_bytes, copy = bench_copy_bandwidth(m, num_experts=32)
    print(f"== A. H2D copy, one expert block ({block_bytes/1e6:.1f} MB, fp8) ==")
    for k, v in copy.items():
        if isinstance(v, tuple):
            print(f"  {k:>10}: {v[0]:7.3f} ms/block  ({v[1]:6.2f} GB/s)")
        else:
            print(f"  {k:>10}: {v:7.3f} ms")
    print()

    ov = bench_overlap(m, num_experts=32)
    print("== B. transfer/compute streams, top_k copies + decode GEMMs/iter ==")
    print(f"  compute only:         {ov['compute_only_ms']:8.3f} ms")
    print(f"  barrier every step:   {ov['barrier_every_step_ms']:8.3f} ms")
    print(f"  pipelined (no sync):  {ov['pipelined_ms']:8.3f} ms")

    st = bench_orchestrated_steps(m, hw)
    print("== C. orchestrated captured decode step (128 slots / 256 experts) ==")
    for k, v in st.items():
        print(f"  {k:>18}: {v:8.3f}")


if __name__ == "__main__":
    main()
