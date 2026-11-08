#!/usr/bin/env python3
"""
Pure-mining baseline: how fast can this GPU do mining work when it does nothing else?

Expected block rewards are proportional to mined multiply-adds: each h x w output tile is one
lottery ticket, and the PoW target is scaled by h * w * k (zk-pow sanity_checks.rs:91,
extract_difficulty_bound; applied by the gateway's adjust_target), so a GEMM of m x n x k has
(m/h)(n/w) tickets each worth h*w*k, i.e. m*n*k in total. Mined MACs per second is therefore the
mining yield, and comparing it with the mined MACs per second of an AI service (serving_impact.py,
online_impact.py) gives how much yield "useful" mining gives up.

This loops Pearl's own pearl_gemm_noisy (the operator the vLLM plugin uses) on random int7 data,
for several shapes, for a fixed time each, and reports mined MACs per second. A dedicated
third-party miner may be faster than Pearl's plugin operator, so this is a lower bound on pure
mining. Same environment and safety setup as kernel_overhead.py.
Usage: python pure_mining.py --pearl <pearl_dir> --out pure_mining.json
"""
from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

from kernel_overhead import load_p05

SHAPES = [(8192, 28672, 4096), (16384, 28672, 4096), (32768, 28672, 4096), (16384, 16384, 8192),
          (8192, 4096, 4096)]  # (m, n, k)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pearl", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--seconds", type=float, default=20)
    args = ap.parse_args()

    os.environ.update({"MINER_NO_MINING": "false", "MINER_NO_GATEWAY": "false", "MINER_SKIP_BLOCK_SUBMISSION": "true"})
    load_p05().start_mock_gateway(Path(args.pearl))

    import torch
    from miner_base.settings import MinerSettings
    from vllm_miner.gemm_operators import pearl_gemm_noisy
    from vllm_miner.mining_state import get_async_manager, init_async_manager, init_pinned_pool
    from vllm_miner.quantization_operators import quant_7bit

    init_async_manager(MinerSettings())
    init_pinned_pool(get_async_manager()._conf.pinned_pool_size)

    rows = []
    for m, n, k in SHAPES:
        w_q, w_s, _ = quant_7bit(torch.randn(n, k, dtype=torch.bfloat16, device="cuda"))
        x_q, x_s, _ = quant_7bit(torch.randn(m, k, dtype=torch.bfloat16, device="cuda"))
        w_q, w_s, x_q, x_s = w_q.contiguous(), w_s.squeeze(-1), x_q.contiguous(), x_s.squeeze(-1)
        call = lambda: pearl_gemm_noisy(x_q, w_q, scale_a=x_s, scale_b=w_s, out_dtype=torch.bfloat16,
                                        submit_block=False)
        for _ in range(3):
            call()
        torch.cuda.synchronize()
        calls, t0 = 0, time.perf_counter()
        while time.perf_counter() - t0 < args.seconds:
            call()
            calls += 1
            if calls % 16 == 0:
                torch.cuda.synchronize()
        torch.cuda.synchronize()
        elapsed = time.perf_counter() - t0
        rate = calls * m * n * k / elapsed
        rows.append({"m": m, "n": n, "k": k, "calls": calls, "seconds": round(elapsed, 2), "mined_macs_per_s": rate})
        print(f"m={m:6d} n={n:6d} k={k:5d}: {calls} calls in {elapsed:.1f} s, "
              f"{rate / 1e12:.1f} TMAC/s mined", flush=True)
        del w_q, w_s, x_q, x_s
        torch.cuda.empty_cache()

    best = max(rows, key=lambda r: r["mined_macs_per_s"])
    Path(args.out).write_text(json.dumps({"gpu": torch.cuda.get_device_name(0), "rows": rows, "best": best}, indent=1))
    print(f"best: {best['mined_macs_per_s'] / 1e12:.1f} TMAC/s at m={best['m']} n={best['n']} k={best['k']}")


if __name__ == "__main__":
    main()
