#!/usr/bin/env python3
"""
How much extra GPU time does Pearl's mining GEMM cost compared with the plain GEMM it replaces?

Times pearl_gemm_noisy (hash A and B, generate noise, noisy GEMM, denoise, PoW check) against
pearl_gemm_vanilla on the 7-bit layer shapes of Llama-3.1-8B, for m (tokens in one step) from
256 to 8192. Calling the operators directly bypasses the plugin's m >= 1024 gate, so the cost
below the threshold is measured too. Compare with the 2r/m estimate in docs/problems/P02.

Needs an H100/H200 and Pearl's miner environment. Block submission is skipped and the gateway is
fed Pearl's mock block template (see p05_gpu_check.py); nothing connects to a network.

Usage: python kernel_overhead.py --pearl <pearl_dir> --out kernel_overhead.json
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import os
import statistics
from pathlib import Path

HERE = Path(__file__).resolve().parent
P05 = HERE.parent / "P05-result-accuracy/gpu/p05_gpu_check.py"

# 7-bit (mining) layers of Llama-3.1-8B: (name, n = output dim, k = input dim)
SHAPES = [("qkv_proj", 6144, 4096), ("o_proj", 4096, 4096), ("gate_up_proj", 28672, 4096)]
MS = [256, 512, 1024, 2048, 4096, 8192]


def load_p05():
    spec = importlib.util.spec_from_file_location("p05_gpu_check", P05)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def time_ms(fn, warmup: int, iters: int) -> float:
    import torch

    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    times = []
    for _ in range(iters):
        start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        start.record()
        fn()
        end.record()
        end.synchronize()
        times.append(start.elapsed_time(end))
    return statistics.median(times)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pearl", required=True, help="Pearl checkout at 156ef9a8")
    ap.add_argument("--out", required=True)
    ap.add_argument("--iters", type=int, default=30)
    args = ap.parse_args()

    os.environ.update({"MINER_NO_MINING": "false", "MINER_NO_GATEWAY": "false", "MINER_SKIP_BLOCK_SUBMISSION": "true"})
    load_p05().start_mock_gateway(Path(args.pearl))

    import torch
    from miner_base.settings import MinerSettings
    from vllm_miner.config import config
    from vllm_miner.gemm_operators import pearl_gemm_noisy, pearl_gemm_vanilla
    from vllm_miner.mining_state import get_async_manager, init_async_manager, init_pinned_pool
    from vllm_miner.quantization_operators import quant_7bit

    settings = MinerSettings()
    assert not settings.no_mining and settings.skip_block_submission
    init_async_manager(settings)
    init_pinned_pool(get_async_manager()._conf.pinned_pool_size)
    r = config.settings.noise_rank

    torch.manual_seed(0)
    rows = []
    for name, n, k in SHAPES:
        w_q, w_s, _ = quant_7bit(torch.randn(n, k, dtype=torch.bfloat16, device="cuda"))
        w_q, w_s = w_q.contiguous(), w_s.squeeze(-1)
        for m in MS:
            x_q, x_s, _ = quant_7bit(torch.randn(m, k, dtype=torch.bfloat16, device="cuda"))
            x_q, x_s = x_q.contiguous(), x_s.squeeze(-1)
            vanilla = lambda: pearl_gemm_vanilla(x_q, w_q, scale_a=x_s, scale_b=w_s, out_dtype=torch.bfloat16)
            noisy = lambda: pearl_gemm_noisy(
                x_q, w_q, scale_a=x_s, scale_b=w_s, out_dtype=torch.bfloat16, submit_block=False
            )
            same = torch.equal(vanilla(), noisy())
            t_v = time_ms(vanilla, 5, args.iters)
            t_n = time_ms(noisy, 5, args.iters)
            row = {
                "layer": name, "m": m, "n": n, "k": k, "r": r,
                "vanilla_ms": round(t_v, 4), "noisy_ms": round(t_n, 4),
                "overhead": round(t_n / t_v - 1, 4), "estimate_2r_over_m": round(2 * r / m, 4),
                "noisy_equals_vanilla": same, "gated_in": config.should_use_noisy_gemm(m, n, k),
            }
            rows.append(row)
            print(f"{name:13s} m={m:5d}  vanilla {t_v:8.3f} ms  noisy {t_n:8.3f} ms  "
                  f"overhead {100 * row['overhead']:7.1f}%  (2r/m {100 * row['estimate_2r_over_m']:6.1f}%)  "
                  f"identical output {same}  plugin mines {row['gated_in']}")

    Path(args.out).write_text(json.dumps({"gpu": torch.cuda.get_device_name(0), "rows": rows}, indent=1))


if __name__ == "__main__":
    main()
