#!/usr/bin/env python3
"""
Pearl vllm-miner "useful mining coverage" estimator (component 7).

Estimates, for Llama-3.1-8B and Llama-3.3-70B as shipped by pearl-ai (mixed W7A7 / FP8
checkpoints), what fraction of GEMM multiply-adds (MACs) in a vLLM forward step go
through pearl_gemm_noisy (i.e. are "mined").

Code facts this model encodes (pearl commit 156ef9a8):
  * trigger: vllm_miner/config.py:36-44  ->  m>=1024 and n>=256 and k>=1024 (and none ==1)
    with m = x_q.shape[0] (tokens in this forward), n = w_q.shape[0] (local out dim),
    k = x_q.shape[1] (local in dim)   (vllm_kernels.py:203-208)
  * only 7-bit layers (PearlConfig._is_mining_layer, vllm_config.py:67-84) can mine.
  * pearl-ai checkpoints (HF config.json quantization_config, fetched 2026-10-27):
      int7 (mineable): o_proj (all layers), gate_proj+up_proj (all layers),
                       q/k/v_proj in the 2nd half of layers (8B: 16-31, 70B: 40-79)
      fp8 block (never mined): down_proj (all), q/k/v_proj in the 1st half of layers
      lm_head: ignored (bf16, never mined)
  * attention QK^T / PV run inside vLLM's attention backend, not a LinearMethod -> never mined.
  * verifier needs k >= 16*r = 2048 (zk-pow/src/api/sanity_checks.rs:36) -- reported separately.

Assumptions (estimates!):
  * vLLM fuses q/k/v into qkv_proj and gate/up into gate_up_proj (standard LlamaForCausalLM).
  * logits are computed only for sampled positions (1 per sequence per step).
  * attention MACs: prefill (causal, full prompt of length L) per layer = 2 * L*(L+1)/2 * d_attn,
    decode at context c per layer = 2 * c * d_attn,  d_attn = n_heads*head_dim.
  * MACs counted, not time: int8 and fp8/bf16 MACs are weighted equally.
"""
from __future__ import annotations

import random
from dataclasses import dataclass

MIN_M, MIN_N, MIN_K = 1024, 256, 1024
NOISE_RANK = 128
VERIFIER_MIN_K = 16 * NOISE_RANK  # 2048


@dataclass
class Model:
    name: str
    H: int
    I: int
    L: int
    nh: int
    nkv: int
    hd: int
    V: int
    int7_qkv_from_layer: int  # q/k/v int7 for layers >= this index
    kv_bytes_per_token: int = 0

    def linears(self, tp: int = 1):
        """(name, n_local, k_local, mineable_layers) per transformer layer."""
        qkv_n = (self.nh + 2 * self.nkv) * self.hd
        return [
            ("qkv_proj", qkv_n // tp, self.H, self.L - self.int7_qkv_from_layer),
            ("o_proj", self.H, (self.nh * self.hd) // tp, self.L),
            ("gate_up_proj", 2 * self.I // tp, self.H, self.L),
            ("down_proj", self.H, self.I // tp, 0),  # fp8, never mined
        ]


LLAMA_8B = Model("Llama-3.1-8B", H=4096, I=14336, L=32, nh=32, nkv=8, hd=128, V=128256,
                 int7_qkv_from_layer=16)
LLAMA_70B = Model("Llama-3.3-70B", H=8192, I=28672, L=80, nh=64, nkv=8, hd=128, V=128256,
                  int7_qkv_from_layer=40)
for _m in (LLAMA_8B, LLAMA_70B):
    _m.kv_bytes_per_token = 2 * _m.L * _m.nkv * _m.hd * 2  # K+V, bf16


def triggers(m, n, k):
    if 1 in (m, n, k):
        return False
    return m >= MIN_M and n >= MIN_N and k >= MIN_K


def step_macs(model: Model, m: int, attn_macs: float, n_logits: int, tp: int = 1):
    """Return (mined_macs, linear_macs, total_macs) for one forward with m tokens (whole model,
    summed over TP ranks)."""
    mined = 0.0
    linear = 0.0
    for _name, n, k, mineable_layers in model.linears(tp):
        per_layer = m * n * k * tp  # all ranks
        linear += per_layer * model.L
        if triggers(m, n, k):
            mined += per_layer * mineable_layers
    head = n_logits * model.V * model.H
    total = linear + head + attn_macs
    return mined, linear, total


def static_fraction(model: Model, tp=1):
    tot = sum(n * k for _, n, k, _ in model.linears(tp)) * model.L
    mine = sum(n * k * ml for _, n, k, ml in model.linears(tp) if triggers(MIN_M, n, k))
    return mine / tot


def prefill_table(model: Model):
    rows = []
    for L in (128, 512, 1024, 2048, 4096, 8192):
        for B in (1, 2, 4, 8, 16, 64):
            m = B * L
            if m > 8192:  # vLLM API-server default max_num_batched_tokens on H100/H200 = 8192
                continue
            d_attn = model.nh * model.hd
            attn = B * model.L * 2 * (L * (L + 1) / 2) * d_attn
            mined, linear, total = step_macs(model, m, attn, n_logits=B)
            rows.append((L, B, m, mined / linear, mined / total))
    return rows


def decode_table(model: Model):
    rows = []
    for B in (1, 32, 128, 256, 512, 1023, 1024):
        for c in (512, 2048, 8192):
            d_attn = model.nh * model.hd
            attn = B * model.L * 2 * c * d_attn
            mined, linear, total = step_macs(model, B, attn, n_logits=B)
            rows.append((B, c, mined / linear, mined / total))
    return rows


def max_decode_batch(model: Model, gpu_mem_gb: float, weights_gb: float, ctx: int,
                     util=0.9, act_reserve_gb=5.0):
    free = gpu_mem_gb * util - weights_gb - act_reserve_gb
    return int(free * 1e9 / (model.kv_bytes_per_token * ctx))


# ---------------------------------------------------------------- continuous batching sim
def simulate(model: Model, arrival_per_step: float, prompt: int, out: int,
             budget=8192, max_seqs=1024, steps=20000, seed=0):
    """Toy vLLM-V1-like scheduler with chunked prefill. Each step: all running decodes get
    1 token; remaining token budget goes to waiting prefills (FCFS, chunked). Returns the
    fraction of linear-layer MACs that are mined (= mined_layer_fraction x fraction of tokens
    processed in steps with m>=1024), and the fraction of steps that mine."""
    rng = random.Random(seed)
    waiting = []  # remaining prompt tokens per request
    decoding = []  # remaining output tokens per request
    tok_total = tok_mined = 0
    mining_steps = 0
    import math

    def poisson(lam):
        L, k, p = math.exp(-lam), 0, 1.0
        while True:
            p *= rng.random()
            if p <= L:
                return k
            k += 1

    for _ in range(steps):
        for _a in range(poisson(arrival_per_step)):
            waiting.append(prompt)
        m = len(decoding)
        left = budget - m
        new_dec = []
        i = 0
        while i < len(waiting) and left > 0 and len(decoding) + len(new_dec) < max_seqs:
            take = min(waiting[i], left)
            waiting[i] -= take
            left -= take
            m += take
            if waiting[i] == 0:
                new_dec.append(out)
                waiting.pop(i)
            else:
                i += 1
        tok_total += m
        if m >= MIN_M:
            tok_mined += m
            mining_steps += 1
        decoding = [d - 1 for d in decoding if d - 1 > 0] + new_dec
    frac_tokens = tok_mined / max(tok_total, 1)
    return frac_tokens, mining_steps / steps, tok_total / steps


def main():
    for model in (LLAMA_8B, LLAMA_70B):
        print(f"\n===== {model.name} =====")
        for tp in (1, 2, 4, 8):
            print(f"TP={tp}: mineable share of linear MACs when m>=1024: {static_fraction(model, tp):.1%}")
            for name, n, k, ml in model.linears(tp):
                flag = "MINE" if (ml and triggers(MIN_M, n, k)) else "no"
                warn = " (k<2048: verifier would reject proof)" if flag == "MINE" and k < VERIFIER_MIN_K else ""
                print(f"    {name:13s} n={n:6d} k={k:6d} int7_layers={ml:3d}/{model.L} -> {flag}{warn}")
        print("\nPREFILL (one step, B prompts of length L batched; m=B*L <= 8192)")
        print("   L     B      m   mined/linear  mined/all(incl attn+lm_head)")
        for L, B, m, a, b in prefill_table(model):
            print(f"{L:5d} {B:5d} {m:6d}   {a:10.1%}   {b:10.1%}")
        print("\nDECODE (one step, B concurrent sequences, context c; m=B)")
        print("    B     c   mined/linear  mined/all")
        for B, c, a, b in decode_table(model):
            print(f"{B:5d} {c:5d}   {a:10.1%}   {b:10.1%}")

    print("\nKV-cache bound on concurrent decodes (rough):")
    for model, gpu, w in ((LLAMA_8B, 80, 9.1), (LLAMA_70B, 141, 72.7)):
        for ctx in (512, 2048, 8192):
            print(f"  {model.name} on 1x{gpu}GB (weights {w}GB), ctx={ctx}: B_max ~ {max_decode_batch(model, gpu, w, ctx)}")

    print("\nCONTINUOUS BATCHING toy sim (budget 8192 tok/step, max_num_seqs 1024)")
    print(" model         prompt out  arrivals/step  tok/step  steps_mining  tokens_in_mining_steps  mined/linear")
    for model in (LLAMA_8B, LLAMA_70B):
        sf = static_fraction(model)
        for prompt, out in ((512, 256), (2048, 256), (256, 1024), (4096, 512)):
            for lam in (0.05, 0.2, 1.0):
                ft, fs, tps = simulate(model, lam, prompt, out, steps=6000)
                print(f" {model.name:13s} {prompt:5d} {out:4d}  {lam:8.2f}   {tps:9.0f}   {fs:8.1%}   {ft:12.1%}   {ft*sf:10.1%}")


if __name__ == "__main__":
    main()
