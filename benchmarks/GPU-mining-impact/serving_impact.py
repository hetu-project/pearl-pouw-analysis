#!/usr/bin/env python3
"""
What does mining cost an AI service running Pearl's vLLM plugin: latency, throughput, coverage, power?

Three configurations of the same Pearl checkpoint, one process each (the miner reads its
settings once at import):
  graph       mining off, CUDA graphs on for decode -- the closest Pearl gets to normal serving.
              vLLM's default torch.compile fails on the plugin's quantisation kernel at 156ef9a8,
              so compilation is off and only decode steps are captured (FULL_DECODE_ONLY).
  eager       mining off, --enforce-eager -- what Pearl requires so that mining can run
  eager_mine  mining on,  --enforce-eager -- Pearl's mining setup
graph -> eager is the cost of giving up CUDA graphs; eager -> eager_mine is the cost of mining.

Workloads use random token prompts of fixed length and ignore EOS, so every configuration
does exactly the same amount of work:
  chat    512 requests, 200 tokens in, 256 out   (short prompts, decode-heavy)
  long    256 requests, 2000 tokens in, 64 out   (long prompts, prefill-heavy)
  single  5 requests one at a time, 200 in, 256 out (one user, latency only)
All requests of a workload are submitted at once (offline batch), so TTFT includes queueing.

In eager configurations every call of a 7-bit layer is recorded as (m, n, k, mined), which gives
how much of the linear-layer work was mined. Under CUDA graphs Python is not re-run on replay, so
nothing is recorded there. GPU power and memory are sampled with nvidia-smi once per second.

Usage (Pearl's miner environment, H100/H200):
  python serving_impact.py run --pearl <pearl_dir> --config graph      --out graph.json
  python serving_impact.py run --pearl <pearl_dir> --config eager      --out eager.json
  python serving_impact.py run --pearl <pearl_dir> --config eager_mine --out eager_mine.json
  python serving_impact.py report graph.json eager.json eager_mine.json
"""
from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import os
import random
import statistics
import subprocess
import tempfile
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
P05 = HERE.parent / "P05-result-accuracy/gpu/p05_gpu_check.py"
DEFAULT_MODEL = "pearl-ai/Llama-3.1-8B-Instruct-pearl"
CONFIGS = {"graph": (False, False), "eager": (False, True), "eager_mine": (True, True)}  # (mining, eager)
WORKLOADS = {
    "chat": {"n": 512, "inp": 200, "out": 256, "sequential": False, "seed": 1},
    "long": {"n": 256, "inp": 2000, "out": 64, "sequential": False, "seed": 2},
    "single": {"n": 5, "inp": 200, "out": 256, "sequential": True, "seed": 3},
}


def load_p05():
    spec = importlib.util.spec_from_file_location("p05_gpu_check", P05)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def pct(values, q):
    values = sorted(values)
    return values[min(len(values) - 1, int(q * len(values)))] if values else None


def linear_macs_per_token(hf) -> int:
    """Multiply-adds of all transformer linear layers per token (Llama layout, lm_head excluded)."""
    h, inter, layers = hf.hidden_size, hf.intermediate_size, hf.num_hidden_layers
    kv = hf.num_key_value_heads * (h // hf.num_attention_heads)
    return layers * (h * (h + 2 * kv) + h * h + h * 2 * inter + inter * h)


class SmiSampler:
    def __enter__(self):
        self.path = tempfile.mktemp(suffix=".csv")
        self.proc = subprocess.Popen(
            ["nvidia-smi", "--query-gpu=power.draw,memory.used,utilization.gpu", "--format=csv,noheader,nounits",
             "-l", "1", "-f", self.path])
        return self

    def __exit__(self, *exc):
        self.proc.terminate()
        self.proc.wait()
        with open(self.path) as f:
            rows = [[float(v) for v in r] for r in csv.reader(f) if len(r) == 3]
        os.unlink(self.path)
        self.summary = {
            "power_w_mean": round(statistics.mean(r[0] for r in rows), 1) if rows else None,
            "memory_mib_max": max(r[1] for r in rows) if rows else None,
            "util_pct_mean": round(statistics.mean(r[2] for r in rows), 1) if rows else None,
        }


def run(args):
    mining, eager = CONFIGS[args.config]
    os.environ.update({
        "MINER_NO_MINING": "false" if mining else "true",
        "MINER_NO_GATEWAY": "false",
        "MINER_SKIP_BLOCK_SUBMISSION": "true",
        "VLLM_ENABLE_V1_MULTIPROCESSING": "0",  # keep the model in this process so the recorder sees it
    })
    load_p05().start_mock_gateway(Path(args.pearl))

    import torch
    from miner_base.settings import MinerSettings
    from vllm import LLM, SamplingParams
    from vllm.inputs import TokensPrompt

    import vllm_miner.vllm_kernels as kernels
    from vllm_miner.config import config
    from vllm_miner.mining_state import get_async_manager, init_async_manager, init_pinned_pool

    settings = MinerSettings()
    assert settings.no_mining == (not mining) and settings.skip_block_submission
    init_async_manager(settings)
    init_pinned_pool(get_async_manager()._conf.pinned_pool_size)

    calls = []  # (m, n, k, mined) per 7-bit layer call
    original = kernels.PearlKernel._apply_weights_mining

    def recorded(self, layer, x, w_q, *rest, **kw):
        m, k, n = x.reshape(-1, x.shape[-1]).shape[0], x.shape[-1], w_q.shape[0]
        calls.append((m, n, k, config.should_use_noisy_gemm(m, n, k) and not config.settings.no_mining))
        return original(self, layer, x, w_q, *rest, **kw)

    kernels.PearlKernel._apply_weights_mining = recorded

    graph_kw = {} if eager else {"compilation_config": {"mode": 0, "cudagraph_mode": "FULL_DECODE_ONLY"}}
    llm = LLM(model=args.model, max_model_len=4096, gpu_memory_utilization=0.9, max_num_seqs=1024,
              enforce_eager=eager, disable_log_stats=False, seed=0, **graph_kw)
    hf = llm.llm_engine.model_config.hf_config
    vocab = hf.vocab_size
    per_token = linear_macs_per_token(hf)

    def prompts(n, length, seed):
        rng = random.Random(seed)
        return [TokensPrompt(prompt_token_ids=[rng.randrange(1000, vocab - 1000) for _ in range(length)])
                for _ in range(n)]

    llm.generate(prompts(8, 64, 999), SamplingParams(max_tokens=8, ignore_eos=True), use_tqdm=False)  # warm-up
    results = {}
    for name, w in WORKLOADS.items():
        if args.only and name not in args.only:
            continue
        ps = prompts(w["n"], w["inp"], w["seed"])
        sp = SamplingParams(temperature=0.0, max_tokens=w["out"], ignore_eos=True)
        calls.clear()
        with SmiSampler() as smi:
            t0 = time.perf_counter()
            if w["sequential"]:
                outs = [llm.generate([p], sp, use_tqdm=False)[0] for p in ps]
            else:
                outs = llm.generate(ps, sp, use_tqdm=False)
            elapsed = time.perf_counter() - t0
        ttft, tpot = [], []
        for o in outs:
            s = o.metrics
            if s is None:
                continue
            ttft.append(s.first_token_latency)
            if s.num_generation_tokens > 1:
                tpot.append((s.last_token_ts - s.first_token_ts) / (s.num_generation_tokens - 1))
        out_tokens = sum(len(o.outputs[0].token_ids) for o in outs)
        entry = {
            **w, "seconds": round(elapsed, 2),
            "output_tok_per_s": round(out_tokens / elapsed, 1),
            "total_tok_per_s": round((out_tokens + w["n"] * w["inp"]) / elapsed, 1),
            "ttft_ms_p50": round(1000 * pct(ttft, 0.5), 1) if ttft else None,
            "ttft_ms_p90": round(1000 * pct(ttft, 0.9), 1) if ttft else None,
            "tpot_ms_p50": round(1000 * pct(tpot, 0.5), 2) if tpot else None,
            "tpot_ms_p90": round(1000 * pct(tpot, 0.9), 2) if tpot else None,
            **smi.summary,
        }
        if calls:
            o_proj = [c for c in calls if c[1] == hf.hidden_size and c[2] == hf.hidden_size]
            tokens = sum(c[0] for c in o_proj) / hf.num_hidden_layers
            mined_macs = sum(m * n * k for m, n, k, mined in calls if mined)
            seven_bit_macs = sum(m * n * k for m, n, k, _ in calls)
            entry.update({
                "layer_calls": len(calls), "mined_calls": sum(c[3] for c in calls),
                "steps": len(o_proj) // hf.num_hidden_layers, "tokens_processed": int(tokens),
                "mined_share_of_7bit_macs": round(mined_macs / seven_bit_macs, 4),
                "mined_share_of_linear_macs": round(mined_macs / (tokens * per_token), 4),
            })
        results[name] = entry
        print(f"[{args.config}] {name}: {json.dumps(entry)}", flush=True)

    Path(args.out).write_text(json.dumps(
        {"config": args.config, "mining": mining, "eager": eager, "model": args.model,
         "gpu": torch.cuda.get_device_name(0), "workloads": results}, indent=1))


def report(args):
    runs = [json.loads(Path(p).read_text()) for p in args.files]
    keys = [("seconds", "s"), ("output_tok_per_s", "out tok/s"), ("ttft_ms_p50", "TTFT p50 ms"),
            ("ttft_ms_p90", "TTFT p90 ms"), ("tpot_ms_p50", "TPOT p50 ms"), ("power_w_mean", "power W"),
            ("memory_mib_max", "mem MiB"), ("mined_share_of_linear_macs", "mined/linear")]
    print(f"model {runs[0]['model']} on {runs[0]['gpu']}")
    for wl in WORKLOADS:
        print(f"\n{wl}")
        print("  " + "config".ljust(12) + "".join(label.rjust(14) for _, label in keys))
        for r in runs:
            e = r["workloads"].get(wl)
            if e:
                print("  " + r["config"].ljust(12) + "".join(str(e.get(k, "-")).rjust(14) for k, _ in keys))


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run")
    r.add_argument("--pearl", required=True, help="Pearl checkout at 156ef9a8")
    r.add_argument("--config", choices=list(CONFIGS), required=True)
    r.add_argument("--out", required=True)
    r.add_argument("--model", default=DEFAULT_MODEL)
    r.add_argument("--only", nargs="*", choices=list(WORKLOADS))
    p = sub.add_parser("report")
    p.add_argument("files", nargs="+")
    args = ap.parse_args()
    run(args) if args.cmd == "run" else report(args)


if __name__ == "__main__":
    main()
