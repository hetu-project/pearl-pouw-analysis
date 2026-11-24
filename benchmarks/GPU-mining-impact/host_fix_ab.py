#!/usr/bin/env python3
"""
A/B test: two host-side fixes for Pearl's mining GEMM.

host_breakdown.py found that most of the fixed ~0.68 ms host cost of one pearl_gemm_noisy
call is two things that are recomputed on every call although their inputs do not change:

  FIX 1  GPUMatmulConfigFactory.create (~0.35 ms, constant)
         Rebuilds MinerSettings() from the environment and re-parses the 64-entry
         cols_pattern on every GEMM. Inputs are (k, noise_rank, moe) -> constant per layer.
         Fix: memoise.

  FIX 2  make_pow_target_tensor (~0.09-0.42 ms)
         Writes 8 uint32 words into a CUDA tensor one at a time from a Python loop,
         i.e. 8 separate host-to-device operations. The value only changes per block.
         Fix: build the 8 words on the host, transfer once, and cache by value.

Both fixes are pure plumbing: the numbers fed to the kernels are identical, so the GEMM
result must be bit-identical. The script asserts that.

Safety: same as kernel_overhead.py -- mock gateway, block submission skipped, no network.
Nothing in Pearl's source tree is modified; the fixes are applied as in-process patches.

Usage: python host_fix_ab.py --pearl /workspace/pearl --out host_fix_ab.json
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import os
import statistics
import time
from functools import lru_cache
from pathlib import Path

P05 = Path("/workspace/pearl-pouw-analysis/benchmarks/P05-result-accuracy/gpu/p05_gpu_check.py")

SHAPES = [
    ("o_proj", 4096, 4096),
    ("gate_up_proj", 28672, 4096),
]
MS = [256, 512, 1024, 2048, 4096, 8192]


def load_p05():
    spec = importlib.util.spec_from_file_location("p05_gpu_check", P05)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def time_ms(fn, warmup, iters):
    import torch
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    times = []
    for _ in range(iters):
        start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        start.record(); fn(); end.record(); end.synchronize()
        times.append(start.elapsed_time(end))
    return statistics.median(times)


def wall_ms(fn, warmup, iters):
    import torch
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(iters):
        fn()
    torch.cuda.synchronize()
    return (time.perf_counter() - t0) * 1000 / iters


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

    orig_create = GO.GPUMatmulConfigFactory.create
    orig_pow = GO.make_pow_target_tensor

    # ---- FIX 1: memoise the per-layer config object ----
    @lru_cache(maxsize=None)
    def _cached_create(k, noise_rank):
        return orig_create(k=k, noise_rank=noise_rank, moe=None)

    class PatchedFactory:
        @staticmethod
        def create(k, noise_rank, moe=None):
            if moe is None:
                return _cached_create(k, noise_rank)
            return orig_create(k=k, noise_rank=noise_rank, moe=moe)

    # ---- FIX 2: build the pow target on the host, one transfer, cached by value ----
    _pow_cache: dict[int, "torch.Tensor"] = {}

    def fast_pow_target(value: int, device="cuda"):
        hit = _pow_cache.get(value)
        if hit is not None:
            return hit
        words = [(value >> (32 * i)) & 0xFFFFFFFF for i in range(8)]
        t = torch.tensor(words, dtype=torch.uint32, device=device)
        if len(_pow_cache) < 64:
            _pow_cache[value] = t
        return t

    torch.manual_seed(0)
    rows = []

    for name, n, k in SHAPES:
        w_q, w_s, _ = quant_7bit(torch.randn(n, k, dtype=torch.bfloat16, device="cuda"))
        w_q, w_s = w_q.contiguous(), w_s.squeeze(-1)
        for m in MS:
            x_q, x_s, _ = quant_7bit(torch.randn(m, k, dtype=torch.bfloat16, device="cuda"))
            x_q, x_s = x_q.contiguous(), x_s.squeeze(-1)

            vanilla = lambda: GO.pearl_gemm_vanilla(x_q, w_q, scale_a=x_s, scale_b=w_s,
                                                    out_dtype=torch.bfloat16)
            noisy = lambda: GO.pearl_gemm_noisy(x_q, w_q, scale_a=x_s, scale_b=w_s,
                                                out_dtype=torch.bfloat16, submit_block=False)

            # ---------- BEFORE (stock code paths) ----------
            base_out = noisy().clone()
            t_v = time_ms(vanilla, 5, args.iters)
            t_before = time_ms(noisy, 5, args.iters)
            w_before = wall_ms(noisy, 5, args.iters)

            # ---------- AFTER ----------
            saved_factory, saved_pow = GO.GPUMatmulConfigFactory, GO.make_pow_target_tensor
            GO.GPUMatmulConfigFactory = PatchedFactory
            GO.make_pow_target_tensor = fast_pow_target
            try:
                fixed_out = noisy()
                identical = torch.equal(base_out, fixed_out)
                correct = torch.equal(vanilla(), fixed_out)
                t_after = time_ms(noisy, 5, args.iters)
                w_after = wall_ms(noisy, 5, args.iters)
            finally:
                GO.GPUMatmulConfigFactory, GO.make_pow_target_tensor = saved_factory, saved_pow

            row = {
                "layer": name, "m": m, "n": n, "k": k,
                "vanilla_ms": round(t_v, 4),
                "noisy_before_ms": round(t_before, 4),
                "noisy_after_ms": round(t_after, 4),
                "wall_before_ms": round(w_before, 4),
                "wall_after_ms": round(w_after, 4),
                "saved_ms": round(w_before - w_after, 4),
                "saved_pct": round(100 * (w_before - w_after) / w_before, 1),
                "overhead_before": round(w_before / t_v - 1, 3),
                "overhead_after": round(w_after / t_v - 1, 3),
                "output_identical_to_unpatched": bool(identical),
                "output_equals_plain_gemm": bool(correct),
            }
            rows.append(row)
            print(f"{name:13s} m={m:5d}  plain {t_v:7.3f}  mining before {w_before:7.3f} -> "
                  f"after {w_after:7.3f} ms  saved {row['saved_pct']:5.1f}%   "
                  f"overhead {100*row['overhead_before']:7.0f}% -> {100*row['overhead_after']:6.0f}%   "
                  f"identical={identical} correct={correct}", flush=True)

    bad = [r for r in rows if not r["output_identical_to_unpatched"]]
    Path(args.out).write_text(json.dumps({
        "gpu": torch.cuda.get_device_name(0), "iters": args.iters,
        "all_outputs_identical": not bad, "rows": rows}, indent=1))
    print(f"\nall outputs identical to unpatched: {not bad}")
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
