#!/usr/bin/env python3
"""
P02 benchmark, part 2: useful-mining coverage under public workload length distributions, and
"useful miner vs pure miner" share-rate ratio on the same GPU.   ALL OUTPUTS ARE ESTIMATES.

Builds on experiments/miner/coverage.py (model shapes, trigger rule, checkpoint layer split) and
experiments/miner/gemm_overhead.py (per-call extra tensor-core work). Part 1
(p02_trigger_table.py) checks the trigger rule and layer split against Pearl's own code.  New here:
  * time-domain simulator of a vLLM-V1-like scheduler (chunked prefill, budget 8192 tok/step,
    max_num_seqs 1024, KV-capacity admission, optional prefix-cache hit rate);
  * a roofline step-time model for H100 SXM / H200 so coverage can be weighted by GPU time;
  * lognormal input/output lengths matched to published statistics (sources in WORKLOADS);
  * share rate: expected shares/s is proportional to sum of m*n*dot_len over mined GEMMs
    (target is scaled by tile_size*dot_len/r*128, sanity_checks.rs:194-196), so we compare
    "lottery MAC/s" of the useful miner with that of a pure miner on the same GPU.

Key assumptions (all stated, none measured):
  A1 int8 and fp8 dense tensor-core peak equal (H100/H200 SXM: 1979 TOPS => 989.5 TMAC/s);
     bf16 half of that.  Achieved GEMM efficiency EFF_TC=0.6 for every GEMM (useful and pure).
  A2 HBM efficiency EFF_BW=0.8; H100 3.35 TB/s 80 GB, H200 4.8 TB/s 141 GB.
  A3 step time = max(linear compute, weight read) + attention + lm_head + mining extra + T_FIXED.
     T_FIXED (eager-mode launch/scheduler overhead per step) default 0; sensitivity run with 5 ms.
  A4 mining extra per mined GEMM: tensor-core-equivalent extra = 2r/n + 2r/m + 2*(2r/k)
     (gemm_overhead.py) at the same efficiency, plus 4*n*k bytes of B-side traffic.
  A5 pure miner: same kernel efficiency, picks its own shapes (n large, B fixed per job so B noise
     is computed once per job, no denoise); overhead PURE_OVH=2% (≈2r/n for n=16384 + A hashing).
  A6 lengths ~ lognormal(sigma=1.0) with the published mean (or median) matched; clipped to
     [1, 32768] input, [1, 4096] output.
"""
from __future__ import annotations

import math
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "experiments" / "miner"))
from coverage import LLAMA_8B, LLAMA_70B, Model, triggers  # noqa: E402

R = 128
VERIFIER_MIN_K = 16 * R
EFF_TC, EFF_BW = 0.6, 0.8
PURE_OVH = 0.02
BUDGET, MAX_SEQS = 8192, 1024

GPUS = {
    "H100-SXM-80G": dict(tmac_int8=989.5e12, tmac_bf16=494.7e12, bw=3.35e12, mem=80e9),
    "H200-141G": dict(tmac_int8=989.5e12, tmac_bf16=494.7e12, bw=4.8e12, mem=141e9),
}

# name: (input stat, output stat, stat kind, source)
WORKLOADS = {
    "Alpaca": (19.31, 58.45, "mean", "vLLM paper (Kwon et al., SOSP'23, arXiv 2309.06180) Fig.11"),
    "LMSYS-Chat-1M": (69.5, 214.5, "mean", "LMSYS-Chat-1M (arXiv 2309.11998) Table 1, per turn"),
    "ShareGPT": (161.31, 337.99, "mean", "vLLM paper Fig.11"),
    "Azure-conv (Splitwise)": (1020, 129, "median", "Splitwise (arXiv 2311.18677) §III, Azure 2023-11-11"),
    "Azure-code (Splitwise)": (1500, 13, "median", "Splitwise §III"),
    "Mooncake/Kimi": (8590, 182, "mean", "Mooncake (arXiv 2407.00079) §4, trace avg"),
}
SIGMA = 1.0


def lognormal_sampler(stat, kind, rng, lo, hi):
    mu = math.log(stat) - (SIGMA**2 / 2 if kind == "mean" else 0.0)
    return lambda: int(min(hi, max(lo, round(rng.lognormvariate(mu, SIGMA)))))


# ------------------------------------------------------------------ per-model constants
def model_consts(model: Model):
    per_tok_linear = 0.0
    weight_bytes = 0.0
    mineable_bytes = 0.0
    for _n, n, k, ml in model.linears(1):
        per_tok_linear += n * k * model.L
        weight_bytes += n * k * model.L  # int7-in-int8 or fp8: 1 byte/param
        if ml and triggers(1024, n, k):
            mineable_bytes += n * k * ml
    head_bytes = model.V * model.H * 2
    return per_tok_linear, weight_bytes + head_bytes, mineable_bytes


def step_cost(model: Model, gpu, m, n_logits, prefill_chunks, decode_ctx_sum, t_fixed):
    """Return (step_time_s, mined_lottery_macs, linear_macs, attn_macs, head_macs)."""
    per_tok_linear, wbytes, mineable_bytes = model_consts(model)
    linear_macs = m * per_tok_linear
    mined = 0.0
    extra_tc = 0.0
    extra_bytes = 0.0
    for _n, n, k, ml in model.linears(1):
        if ml and triggers(m, n, k):
            macs = m * n * k * ml
            if k >= VERIFIER_MIN_K:
                mined += macs * ((k - k % R) / k)
            extra_tc += macs * (2 * R / n + 2 * R / m + 2 * (2 * R / k))
            extra_bytes += 4 * n * k * ml
    d_attn = model.nh * model.hd
    attn_macs = 0.0
    for (start, length) in prefill_chunks:  # causal attention of a chunk against prefix
        attn_macs += model.L * 2 * d_attn * (length * start + length * (length + 1) / 2)
    attn_macs += model.L * 2 * d_attn * decode_ctx_sum
    head_macs = n_logits * model.V * model.H
    tc = gpu["tmac_int8"] * EFF_TC
    bw = gpu["bw"] * EFF_BW
    t_lin = max(linear_macs / tc, wbytes / bw)
    t_attn = attn_macs / (gpu["tmac_bf16"] * EFF_TC) + decode_ctx_sum * model.kv_bytes_per_token / bw
    t_head = head_macs / (gpu["tmac_bf16"] * EFF_TC)
    t_mine = extra_tc / tc + extra_bytes / bw
    return t_lin + t_attn + t_head + t_mine + t_fixed, mined, linear_macs, attn_macs, head_macs


def kv_capacity_tokens(model, gpu, util=0.9, act_reserve=5e9):
    _, wbytes, _ = model_consts(model)
    return int((gpu["mem"] * util - wbytes - act_reserve) / model.kv_bytes_per_token)


# ------------------------------------------------------------------ simulator
def simulate(model, gpu, wl, rate, horizon=300.0, t_fixed=0.0, prefix_hit=0.0, seed=1):
    rng = random.Random(seed)
    i_stat, o_stat, kind, _ = WORKLOADS[wl]
    s_in = lognormal_sampler(i_stat, kind, rng, 1, 32768)
    s_out = lognormal_sampler(o_stat, kind, rng, 1, 4096)
    kv_cap = kv_capacity_tokens(model, gpu)
    arrivals = []
    t = 0.0
    while t < horizon:
        t += rng.expovariate(rate)
        arrivals.append(t)
    ai = 0
    waiting = []  # [prompt_len_to_compute, prompt_done, out_len, total_ctx_reserved]
    running = []  # [ctx, remaining_out]
    kv_used = 0
    now = 0.0
    busy = 0.0
    acc = dict(mined=0.0, lin=0.0, attn=0.0, head=0.0, tokens=0, mining_time=0.0, out_tokens=0,
               steps=0, mining_steps=0, step_time=0.0)
    while now < horizon:
        while ai < len(arrivals) and arrivals[ai] <= now:
            p = s_in()
            o = s_out()
            p_eff = max(1, int(round(p * (1 - prefix_hit))))
            waiting.append([p_eff, p - p_eff, o, p + o, False])
            ai += 1
        if not running and not waiting:
            if ai >= len(arrivals):
                break
            now = arrivals[ai]
            continue
        m = len(running)
        decode_ctx = sum(r[0] for r in running)
        left = BUDGET - m
        chunks = []
        finished_prefill = []
        nseq = len(running)
        idx = 0
        while idx < len(waiting) and left > 0:
            w = waiting[idx]
            # admission (first chunk) needs KV room and a sequence slot
            if not w[4]:
                if nseq >= MAX_SEQS or kv_used + w[3] > kv_cap:
                    break
                kv_used += w[3]
                nseq += 1
                w[4] = True
            take = min(w[0], left)
            chunks.append((w[1], take))
            w[0] -= take
            w[1] += take
            left -= take
            m += take
            if w[0] == 0:
                finished_prefill.append(idx)
            idx += 1
        if m == 0:
            # blocked on KV: should not happen when running is empty; jump time
            now = arrivals[ai] if ai < len(arrivals) else horizon
            continue
        n_logits = len(running) + len(finished_prefill)
        dt, mined, lin, attn, head = step_cost(model, gpu, m, n_logits, chunks, decode_ctx, t_fixed)
        acc["mined"] += mined
        acc["lin"] += lin
        acc["attn"] += attn
        acc["head"] += head
        acc["tokens"] += m
        acc["steps"] += 1
        acc["step_time"] += dt
        if mined > 0:
            acc["mining_steps"] += 1
            acc["mining_time"] += dt
        acc["out_tokens"] += n_logits
        busy += dt
        now += dt
        # advance state
        new_running = []
        for ctx, rem, res in running:
            if rem - 1 > 0:
                new_running.append([ctx + 1, rem - 1, res])
            else:
                kv_used -= res
        for j in reversed(finished_prefill):
            w = waiting.pop(j)
            if w[2] - 1 > 0:
                new_running.append([w[1] + 1, w[2] - 1, w[3]])
            else:
                kv_used -= w[3]
        running = new_running
        kv_used = max(kv_used, 0)
    wall = max(now, 1e-9)
    pure_rate = gpu["tmac_int8"] * EFF_TC / (1 + PURE_OVH)
    tot = acc["lin"] + acc["attn"] + acc["head"]
    return dict(
        out_tok_s=acc["out_tokens"] / wall,
        busy=busy / wall,
        mined_over_linear=acc["mined"] / max(acc["lin"], 1),
        mined_over_all=acc["mined"] / max(tot, 1),
        steps_mining=acc["mining_steps"] / max(acc["steps"], 1),
        time_mining=acc["mining_time"] / max(acc["step_time"], 1e-12),
        share_ratio_busy=(acc["mined"] / max(busy, 1e-12)) / pure_rate,
        share_ratio_wall=(acc["mined"] / wall) / pure_rate,
    )


def best_case(model, gpu, m=8192, L=1024, t_fixed=0.0):
    """Saturated prefill: m tokens per step made of m/L prompts of length L."""
    B = m // L
    chunks = [(0, L)] * B
    dt, mined, lin, attn, head = step_cost(model, gpu, m, B, chunks, 0, t_fixed)
    pure_rate = gpu["tmac_int8"] * EFF_TC / (1 + PURE_OVH)
    return mined / dt / pure_rate, mined / (lin + attn + head)


def main():
    print(__doc__.split("Key assumptions")[0].strip().splitlines()[0])
    print(f"EFF_TC={EFF_TC} EFF_BW={EFF_BW} PURE_OVH={PURE_OVH} budget={BUDGET} max_seqs={MAX_SEQS} sigma={SIGMA}\n")
    setups = ((LLAMA_8B, "H100-SXM-80G"), (LLAMA_70B, "H200-141G"))
    print("== D11 static: useful miner in best case (saturated prefill, m=8192) vs pure miner, same GPU")
    for model, g in setups:
        for L in (512, 1024, 4096, 8192):
            for tf in (0.0, 0.005):
                r, cov = best_case(model, GPUS[g], 8192, L, tf)
                print(f"  {model.name:14s} {g:13s} prompt L={L:5d} T_FIXED={tf*1e3:3.0f}ms : "
                      f"lottery-MAC/s ratio useful/pure = {r:6.1%}   (mined/all MACs {cov:5.1%})")
    for model, g in setups:
        print(f"  KV capacity {model.name} on {g}: {kv_capacity_tokens(model, GPUS[g]):,} tokens")
    print()
    print("== D10/D11 workload simulation (time domain, 300 s, Poisson arrivals)")
    hdr = (f"{'model/GPU':24s} {'workload':24s} {'req/s':>6s} {'outTok/s':>9s} {'busy':>6s} "
           f"{'stepsMine':>9s} {'timeMine':>8s} {'mined/lin':>9s} {'mined/all':>9s} {'R_busy':>7s} {'R_wall':>7s}")
    for tf in (0.0, 0.005):
        print(f"\n-- T_FIXED = {tf*1e3:.0f} ms/step, prefix-cache hit 0")
        print(hdr)
        for model, g in setups:
            for wl in WORKLOADS:
                rates = (1, 4, 16, 64) if model is LLAMA_8B else (0.25, 1, 4, 16)
                for lam in rates:
                    r = simulate(model, GPUS[g], wl, lam, t_fixed=tf)
                    print(f"{model.name + '/' + g[:4]:24s} {wl:24s} {lam:6.2f} {r['out_tok_s']:9.0f} {r['busy']:6.1%} "
                          f"{r['steps_mining']:9.1%} {r['time_mining']:8.1%} {r['mined_over_linear']:9.1%} "
                          f"{r['mined_over_all']:9.1%} {r['share_ratio_busy']:7.1%} {r['share_ratio_wall']:7.1%}")
    print("\n-- prefix-cache sensitivity (T_FIXED=0): Mooncake 50% hit, ShareGPT 50% hit")
    print(hdr)
    for model, g in setups:
        for wl in ("Mooncake/Kimi", "ShareGPT"):
            rates = (1, 4, 16, 64) if model is LLAMA_8B else (0.25, 1, 4, 16)
            for lam in rates:
                r = simulate(model, GPUS[g], wl, lam, prefix_hit=0.5)
                print(f"{model.name + '/' + g[:4]:24s} {wl + ' hit50':24s} {lam:6.2f} {r['out_tok_s']:9.0f} {r['busy']:6.1%} "
                      f"{r['steps_mining']:9.1%} {r['time_mining']:8.1%} {r['mined_over_linear']:9.1%} "
                      f"{r['mined_over_all']:9.1%} {r['share_ratio_busy']:7.1%} {r['share_ratio_wall']:7.1%}")
    print("\nSources:")
    for wl, (_i, _o, kind, src) in WORKLOADS.items():
        print(f"  {wl}: input {kind} {_i}, output {kind} {_o} -- {src}")


if __name__ == "__main__":
    main()
