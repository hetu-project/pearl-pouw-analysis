#!/usr/bin/env python3
"""
Where does the fixed cost of one pearl_gemm_noisy call go?

kernel_overhead.py shows that a mining GEMM costs a roughly constant extra 0.7-1.2 ms per call,
whatever m is. This profiles single calls (o_proj and gate_up_proj shapes of Llama-3.1-8B) with
torch.profiler and splits the wall time into GPU kernel time, by kernel, and time the GPU waits
on the host (Python, host-to-device copies, synchronisation).

Same environment and safety setup as kernel_overhead.py.
Usage: python kernel_breakdown.py --pearl <pearl_dir> --out kernel_breakdown.txt
"""
from __future__ import annotations

import argparse
import os
import time
from pathlib import Path

from kernel_overhead import load_p05

SHAPES = [("o_proj", 4096, 4096, 4096), ("gate_up_proj", 28672, 4096, 4096), ("narrow_n", 1024, 4096, 4096)]  # (label, n, k, m)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pearl", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--iters", type=int, default=20)
    args = ap.parse_args()

    os.environ.update({"MINER_NO_MINING": "false", "MINER_NO_GATEWAY": "false", "MINER_SKIP_BLOCK_SUBMISSION": "true"})
    load_p05().start_mock_gateway(Path(args.pearl))

    import torch
    from torch.profiler import ProfilerActivity, profile

    from miner_base.settings import MinerSettings
    from vllm_miner.gemm_operators import pearl_gemm_noisy, pearl_gemm_vanilla
    from vllm_miner.mining_state import get_async_manager, init_async_manager, init_pinned_pool
    from vllm_miner.quantization_operators import quant_7bit

    init_async_manager(MinerSettings())
    init_pinned_pool(get_async_manager()._conf.pinned_pool_size)

    lines = []
    for name, n, k, m in SHAPES:
        w_q, w_s, _ = quant_7bit(torch.randn(n, k, dtype=torch.bfloat16, device="cuda"))
        x_q, x_s, _ = quant_7bit(torch.randn(m, k, dtype=torch.bfloat16, device="cuda"))
        w_q, w_s, x_q, x_s = w_q.contiguous(), w_s.squeeze(-1), x_q.contiguous(), x_s.squeeze(-1)
        calls = {
            "vanilla": lambda: pearl_gemm_vanilla(x_q, w_q, scale_a=x_s, scale_b=w_s, out_dtype=torch.bfloat16),
            "noisy": lambda: pearl_gemm_noisy(x_q, w_q, scale_a=x_s, scale_b=w_s, out_dtype=torch.bfloat16,
                                              submit_block=False),
        }
        for label, fn in calls.items():
            for _ in range(5):
                fn()
            torch.cuda.synchronize()
            with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
                t0 = time.perf_counter()
                for _ in range(args.iters):
                    fn()
                torch.cuda.synchronize()
                wall_ms = (time.perf_counter() - t0) * 1000 / args.iters
            kernels = {}
            for e in prof.events():
                if e.device_type.name == "CUDA":
                    kernels[e.name] = kernels.get(e.name, 0.0) + e.device_time / 1000 / args.iters
            gpu_ms = sum(kernels.values())
            lines.append(f"\n{name} m={m} n={n} k={k}  {label}: wall {wall_ms:.3f} ms/call, "
                         f"GPU kernels {gpu_ms:.3f} ms/call, GPU idle {wall_ms - gpu_ms:.3f} ms/call")
            for kname, ms in sorted(kernels.items(), key=lambda kv: -kv[1])[:12]:
                lines.append(f"    {ms:8.3f} ms  {kname[:110]}")
            print("\n".join(lines[-13:]), flush=True)
    Path(args.out).write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
