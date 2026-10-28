#!/usr/bin/env python3
"""
P03 benchmark, part 3: expected share rate of a useful miner vs a pure miner on the same GPU.
ALL OUTPUTS ARE ESTIMATES (CPU simulation with published H100/H200 specs; no GPU was used).

R = (lottery multiply-adds per second of the useful miner) / (same for a pure miner on the same GPU).
Expected shares per second are proportional to lottery multiply-adds per second, because the
jackpot target is scaled by tile size x dot-product length only (sanity_checks.rs:191,195).

Reuses the model of benchmarks/P02-mining-coverage/p02_workload_sim.py:
  * useful miner: vLLM-like scheduler, mines only steps with m >= 1024 on int7 layers, pays the
    per-call overhead of pearl_gemm_noisy (noise A and B, denoise, B-side traffic);
  * pure miner: same kernel efficiency, fixed B per job (B noise once per block), no denoising,
    picks large n; residual overhead PURE_OVH = 2%.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "P02-mining-coverage"))
from p02_workload_sim import (  # noqa: E402
    EFF_BW, EFF_TC, GPUS, LLAMA_8B, LLAMA_70B, PURE_OVH, WORKLOADS, best_case, simulate,
)

SETUPS = ((LLAMA_8B, "H100-SXM-80G", (1, 4, 16, 64)), (LLAMA_70B, "H200-141G", (0.25, 1, 4, 16)))


def main():
    print(__doc__.strip().splitlines()[0])
    print(f"assumptions: GEMM efficiency {EFF_TC}, HBM efficiency {EFF_BW}, pure-miner overhead {PURE_OVH:.0%}\n")

    print("Table 1. Best case for the useful miner: GPU saturated with prefill, 8192 tokens per step")
    print(f"  {'model / simulated GPU':28s} {'prompt len':>10s} {'R':>7s} {'pure/useful':>12s} {'R (5 ms/step)':>14s}")
    for model, g, _ in SETUPS:
        for L in (512, 1024, 4096, 8192):
            r0, _ = best_case(model, GPUS[g], 8192, L, 0.0)
            r5, _ = best_case(model, GPUS[g], 8192, L, 0.005)
            print(f"  {model.name + ' / ' + g:28s} {L:10d} {r0:7.1%} {1 / r0:11.1f}x {r5:14.1%}")
    print()

    print("Table 2. Real workloads: R including idle time (R_wall), by request rate")
    for model, g, rates in SETUPS:
        print(f"  {model.name} / simulated {g}")
        print(f"    {'workload':24s} " + " ".join(f"{str(r) + ' req/s':>11s}" for r in rates))
        for wl in WORKLOADS:
            cells = []
            for lam in rates:
                r = simulate(model, GPUS[g], wl, lam)["share_ratio_wall"]
                cells.append(f"{r:11.1%}")
            print(f"    {wl:24s} " + " ".join(cells))
    print("\n  pure miner advantage = 1 / R; R = 0 means the useful miner earns no shares at that load")


if __name__ == "__main__":
    main()
