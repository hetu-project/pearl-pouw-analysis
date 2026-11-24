#!/usr/bin/env python3
"""
Where does the HOST-side fixed cost of one pearl_gemm_noisy call go?

kernel_breakdown.py showed the GPU sits idle ~0.6-0.7 ms per mining call while the GPU
kernels themselves only add ~0.2 ms. That idle time is host work: Python, allocations,
pageable host-to-device copies and synchronisation. This script wraps every step inside
pearl_gemm_noisy with a host-side timer and reports where the milliseconds actually go.

Same safety setup as kernel_overhead.py: mock gateway, block submission skipped,
nothing connects to the Pearl network.

Usage: python host_breakdown.py --pearl /workspace/pearl --out host_breakdown.json
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import os
import statistics
import time
from collections import defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent
P05 = Path("/workspace/pearl-pouw-analysis/benchmarks/P05-result-accuracy/gpu/p05_gpu_check.py")

SHAPES = [
    ("o_proj m=256", 256, 4096, 4096),
    ("o_proj m=1024 (threshold)", 1024, 4096, 4096),
    ("o_proj m=4096", 4096, 4096, 4096),
    ("gate_up m=1024", 1024, 28672, 4096),
]

timings: dict[str, list[float]] = defaultdict(list)
_recording = False


def load_p05():
    spec = importlib.util.spec_from_file_location("p05_gpu_check", P05)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def wrap(obj, name, label):
    """Replace obj.name with a host-timed version."""
    orig = getattr(obj, name)

    def timed(*a, **kw):
        if not _recording:
            return orig(*a, **kw)
        t0 = time.perf_counter()
        try:
            return orig(*a, **kw)
        finally:
            timings[label].append((time.perf_counter() - t0) * 1000)

    setattr(obj, name, timed)
    return orig


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pearl", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--iters", type=int, default=50)
    args = ap.parse_args()

    os.environ.update({
        "MINER_NO_MINING": "false",
        "MINER_NO_GATEWAY": "false",
        "MINER_SKIP_BLOCK_SUBMISSION": "true",
    })
    load_p05().start_mock_gateway(Path(args.pearl))

    import torch
    from miner_base.settings import MinerSettings
    from vllm_miner import gemm_operators as GO
    from vllm_miner.mining_state import get_async_manager, init_async_manager, init_pinned_pool
    from vllm_miner.quantization_operators import quant_7bit

    settings = MinerSettings()
    assert not settings.no_mining and settings.skip_block_submission
    init_async_manager(settings)
    init_pinned_pool(get_async_manager()._conf.pinned_pool_size)

    # --- instrument every step inside pearl_gemm_noisy ---
    wrap(GO, "tensor_hash", "1. tensor_hash (A and B Merkle hash)")
    wrap(GO, "commitment_hash_from_merkle_roots", "2. commitment_hash_from_merkle_roots")
    wrap(GO, "generate_noise_factors", "3. generate_noise_factors")
    wrap(GO, "noisy_gemm", "4. noisy_gemm (the mining GEMM launch)")
    wrap(GO, "make_pow_target_tensor", "5. make_pow_target_tensor")
    wrap(GO, "get_required_scratchpad_bytes", "6. get_required_scratchpad_bytes")
    wrap(GO, "get_host_signal_sync_size", "7. get_host_signal_sync_size")
    wrap(GO, "get_async_manager", "8. get_async_manager / get_mining_job")
    wrap(GO, "get_pinned_pool", "9. get_pinned_pool")
    wrap(GO.CommitmentHasher, "get_key", "10. CommitmentHasher.get_key (host BLAKE3)")
    wrap(GO.GPUMatmulConfigFactory, "create", "11. GPUMatmulConfigFactory.create")

    # allocations and the pageable host->device copy
    wrap(torch, "empty", "12. torch.empty (all allocations)")
    wrap(torch, "zeros", "13. torch.zeros")
    wrap(torch, "frombuffer", "14. torch.frombuffer (hash key on host)")
    orig_to = torch.Tensor.to

    def timed_to(self, *a, **kw):
        if not _recording:
            return orig_to(self, *a, **kw)
        t0 = time.perf_counter()
        try:
            return orig_to(self, *a, **kw)
        finally:
            timings["15. Tensor.to (incl. pageable HtoD copy)"].append(
                (time.perf_counter() - t0) * 1000)

    torch.Tensor.to = timed_to

    global _recording
    torch.manual_seed(0)
    results = []

    for label, m, n, k in SHAPES:
        w_q, w_s, _ = quant_7bit(torch.randn(n, k, dtype=torch.bfloat16, device="cuda"))
        x_q, x_s, _ = quant_7bit(torch.randn(m, k, dtype=torch.bfloat16, device="cuda"))
        w_q, w_s = w_q.contiguous(), w_s.squeeze(-1)
        x_q, x_s = x_q.contiguous(), x_s.squeeze(-1)

        def noisy():
            return GO.pearl_gemm_noisy(x_q, w_q, scale_a=x_s, scale_b=w_s,
                                       out_dtype=torch.bfloat16, submit_block=False)

        def vanilla():
            return GO.pearl_gemm_vanilla(x_q, w_q, scale_a=x_s, scale_b=w_s,
                                         out_dtype=torch.bfloat16)

        for _ in range(5):
            noisy(); vanilla()
        torch.cuda.synchronize()

        # vanilla reference wall time
        t0 = time.perf_counter()
        for _ in range(args.iters):
            vanilla()
        torch.cuda.synchronize()
        vanilla_ms = (time.perf_counter() - t0) * 1000 / args.iters

        timings.clear()
        _recording = True
        t0 = time.perf_counter()
        for _ in range(args.iters):
            noisy()
        host_only_ms = (time.perf_counter() - t0) * 1000 / args.iters  # no sync: host-side cost
        torch.cuda.synchronize()
        wall_ms = (time.perf_counter() - t0) * 1000 / args.iters
        _recording = False

        steps = []
        for name, vals in timings.items():
            total = sum(vals) / args.iters
            steps.append({
                "step": name,
                "ms_per_call": round(total, 4),
                "calls_per_gemm": round(len(vals) / args.iters, 2),
            })
        steps.sort(key=lambda s: -s["ms_per_call"])
        accounted = sum(s["ms_per_call"] for s in steps)

        row = {
            "shape": label, "m": m, "n": n, "k": k,
            "vanilla_wall_ms": round(vanilla_ms, 4),
            "noisy_host_ms": round(host_only_ms, 4),
            "noisy_wall_ms": round(wall_ms, 4),
            "accounted_host_ms": round(accounted, 4),
            "unaccounted_host_ms": round(host_only_ms - accounted, 4),
            "steps": steps,
        }
        results.append(row)

        print(f"\n===== {label}  (n={n}, k={k}) =====")
        print(f"  plain GEMM wall       {vanilla_ms:8.3f} ms")
        print(f"  mining GEMM wall      {wall_ms:8.3f} ms")
        print(f"  mining HOST time      {host_only_ms:8.3f} ms   <-- CPU side, before GPU sync")
        print(f"  {'step':52s} {'ms/call':>9s} {'calls':>6s}")
        for s in steps:
            if s["ms_per_call"] >= 0.0005:
                print(f"  {s['step']:52s} {s['ms_per_call']:9.4f} {s['calls_per_gemm']:6.1f}")
        print(f"  {'(unaccounted host time)':52s} {row['unaccounted_host_ms']:9.4f}")

    Path(args.out).write_text(json.dumps(
        {"gpu": torch.cuda.get_device_name(0), "iters": args.iters, "rows": results}, indent=1))
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
