#!/usr/bin/env python3
"""
On a Mixture-of-Experts model, how much of the inference does Pearl mine, and how much of
that mining can never reach the chain?

Three buckets of linear-layer multiply-adds, measured live inside vLLM:

  dense     attention projections (qkv, o). Mining these produces an ordinary dense proof,
            which consensus accepts.
  expert1   the expert gate/up GEMM. Pearl mines this, but the resulting certificate is a
            grouped-GEMM (MoE) certificate, and node/blockchain/validate.go:542 rejects
            every MoE certificate at or after DenseOnlyForkHeight (mainnet 91,630).
            So this work is burnt: mined, submitted, rejected, and the miner is not told.
  expert2   the expert down projection. gemm_operators never mines it at all
            ("# GEMM2 (down projection): not mined, kept in the original fp8 block").

Safety: mock gateway, block submission skipped, nothing connects to the Pearl network.

Usage: python moe_coverage.py --pearl /workspace/pearl --model <moe model> --out moe_coverage.json
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import os
import random
from collections import defaultdict
from pathlib import Path

P05 = Path("/workspace/pearl-pouw-analysis/benchmarks/P05-result-accuracy/gpu/p05_gpu_check.py")

WORKLOADS = {
    "chat": {"n": 128, "inp": 200, "out": 64, "seed": 1},
    "long": {"n": 64, "inp": 2000, "out": 16, "seed": 2},
}


def load_p05():
    spec = importlib.util.spec_from_file_location("p05_gpu_check", P05)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pearl", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--gpu-util", type=float, default=0.5)
    args = ap.parse_args()

    os.environ.update({
        "MINER_NO_MINING": "false",
        "MINER_NO_GATEWAY": "false",
        "MINER_SKIP_BLOCK_SUBMISSION": "true",
        "VLLM_ENABLE_V1_MULTIPROCESSING": "0",
    })
    load_p05().start_mock_gateway(Path(args.pearl))

    import torch
    from miner_base.settings import MinerSettings
    from vllm import LLM, SamplingParams
    from vllm.inputs import TokensPrompt

    import vllm_miner.pearl_moe_experts as moe_mod
    import vllm_miner.vllm_kernels as kernels
    from vllm_miner.config import config
    from vllm_miner.mining_state import get_async_manager, init_async_manager, init_pinned_pool

    settings = MinerSettings()
    assert not settings.no_mining and settings.skip_block_submission
    init_async_manager(settings)
    init_pinned_pool(get_async_manager()._conf.pinned_pool_size)

    # bucket -> {"macs": int, "mined_macs": int, "calls": int, "mined_calls": int}
    acc: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))

    def note(bucket, macs, mined):
        a = acc[bucket]
        a["calls"] += 1
        a["macs"] += macs
        if mined:
            a["mined_calls"] += 1
            a["mined_macs"] += macs

    orig_dense = kernels.PearlKernel._apply_weights_mining

    def dense_recorded(self, layer, x, w_q, *rest, **kw):
        m, k, n = x.reshape(-1, x.shape[-1]).shape[0], x.shape[-1], w_q.shape[0]
        mined = config.should_use_noisy_gemm(m, n, k) and not config.settings.no_mining
        note("dense_attention", m * n * k, mined)
        return orig_dense(self, layer, x, w_q, *rest, **kw)

    kernels.PearlKernel._apply_weights_mining = dense_recorded

    orig_apply = moe_mod.PearlMoEExperts.apply

    def moe_recorded(self, output, hidden_states, w1, w2, topk_weights, topk_ids, *rest, **kw):
        _, tokens, n, k, topk = self.moe_problem_size(hidden_states, w1, w2, topk_ids)
        mined = config.should_use_noisy_gemm(tokens, n, k) and not config.settings.no_mining
        # GEMM1 (gate/up): n is the fused 2*moe_intermediate output width
        note("expert_gate_up", tokens * topk * n * k, mined)
        # GEMM2 (down): never mined -- half the width of GEMM1, reversed shape
        note("expert_down", tokens * topk * (n // 2) * k, False)
        return orig_apply(self, output, hidden_states, w1, w2, topk_weights, topk_ids, *rest, **kw)

    moe_mod.PearlMoEExperts.apply = moe_recorded

    llm = LLM(model=args.model, max_model_len=4096, gpu_memory_utilization=args.gpu_util,
              max_num_seqs=1024, enforce_eager=True, disable_log_stats=True, seed=0)
    hf = llm.llm_engine.model_config.hf_config
    vocab = hf.vocab_size

    def prompts(n, length, seed):
        rng = random.Random(seed)
        return [TokensPrompt(prompt_token_ids=[rng.randrange(1000, vocab - 1000) for _ in range(length)])
                for _ in range(n)]

    llm.generate(prompts(4, 64, 999), SamplingParams(max_tokens=4, ignore_eos=True), use_tqdm=False)

    results = {}
    for name, w in WORKLOADS.items():
        acc.clear()
        llm.generate(prompts(w["n"], w["inp"], w["seed"]),
                     SamplingParams(temperature=0.0, max_tokens=w["out"], ignore_eos=True),
                     use_tqdm=False)
        buckets = {b: {kk: v.get(kk, 0) for kk in ("calls", "mined_calls", "macs", "mined_macs")}
                   for b, v in acc.items()}
        total = sum(v["macs"] for v in buckets.values())
        mined = sum(v["mined_macs"] for v in buckets.values())
        wasted = buckets.get("expert_gate_up", {}).get("mined_macs", 0)
        results[name] = {
            **w, "buckets": buckets,
            "total_linear_macs": total,
            "mined_macs": mined,
            "mined_share": round(mined / total, 4) if total else None,
            "mined_but_rejected_macs": wasted,
            "wasted_share_of_mining": round(wasted / mined, 4) if mined else None,
            "never_mineable_down_share": round(
                buckets.get("expert_down", {}).get("macs", 0) / total, 4) if total else None,
        }
        print(f"\n===== {name} =====")
        print(f"  {'bucket':18} {'MACs':>16} {'share':>7} {'mined MACs':>16} {'mined%':>7}")
        for b in ("dense_attention", "expert_gate_up", "expert_down"):
            v = buckets.get(b)
            if not v:
                continue
            print(f"  {b:18} {v['macs']:16d} {100*v['macs']/total:6.1f}% "
                  f"{v['mined_macs']:16d} {100*v['mined_macs']/v['macs']:6.1f}%")
        print(f"  mined overall                 {100*mined/total:6.1f}% of linear MACs")
        print(f"  of that, MoE certs consensus REJECTS: {100*wasted/mined if mined else 0:6.1f}%")
        print(f"  down projection, never mineable:      "
              f"{100*buckets.get('expert_down',{}).get('macs',0)/total:6.1f}% of linear MACs")

    Path(args.out).write_text(json.dumps(
        {"model": args.model, "gpu": torch.cuda.get_device_name(0),
         "hf": {k: getattr(hf, k, None) for k in
                ("hidden_size", "num_hidden_layers", "moe_intermediate_size",
                 "num_experts", "num_experts_per_tok", "num_attention_heads",
                 "num_key_value_heads", "head_dim")},
         "workloads": results}, indent=1))
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
