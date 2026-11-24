#!/usr/bin/env python3
"""
Online (Poisson arrival) chat traffic with the mining threshold lowered to min_m=16,
stock Pearl code vs the fast mining path, with and without CUDA graphs.

  graph_nomine      CUDA graphs, mining off           (best case without mining)
  eager_nomine      --enforce-eager, mining off       (what Pearl requires, before mining)
  eager_stock16     --enforce-eager, mining on, min_m=16, Pearl's code as shipped
  eager_fast16      --enforce-eager, mining on, min_m=16, fast_noisy.py
  graph_fast16      CUDA graphs, mining on, min_m=16, fast_noisy.py -- COST PROBE: replayed
                    decode steps run the mining kernels but no win check, so not valid mining

Same harness as online_impact.py (vllm serve + vllm bench serve, ShareGPT, EOS ignored).
config.yaml's min_m is edited per configuration and restored with git checkout at the end.
Mock gateway, block submission skipped, nothing connects to a network.
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import signal
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

HERE = Path(__file__).resolve().parent
MODEL = "pearl-ai/Llama-3.1-8B-Instruct-pearl"
PORT = 8000
GRAPH = ["--compilation-config",
         '{"mode": 0, "cudagraph_mode": "FULL_DECODE_ONLY", '
         '"cudagraph_capture_sizes": [1, 2, 4, 8, 16, 32, 64, 128, 256, 512]}']
EAGER = ["--enforce-eager"]
CONFIGS = {  # name: (MINER_NO_MINING, min_m, PEARL_FAST, flags)
    "graph_nomine": ("true", 1024, "0", GRAPH),
    "eager_nomine": ("true", 1024, "0", EAGER),
    "eager_stock16": ("false", 16, "0", EAGER),
    "eager_fast16": ("false", 16, "1", EAGER),
    "graph_fast16": ("false", 16, "1", GRAPH),
}
RUNS = [("sharegpt", 4, 400), ("sharegpt", 16, 1000)]
CFG = "miner/vllm-miner/src/vllm_miner/config.yaml"


def set_min_m(pearl: Path, value: int):
    p = pearl / CFG
    lines = p.read_text().splitlines(keepends=True)
    out = [l[: len(l) - len(l.lstrip())] + f"min_m: {value}\n" if l.strip().startswith("min_m:") else l for l in lines]
    p.write_text("".join(out))


def wait_ready(proc, timeout=900):
    for _ in range(timeout):
        if proc.poll() is not None:
            return False
        try:
            urllib.request.urlopen(f"http://localhost:{PORT}/health", timeout=2)
            return True
        except Exception:
            time.sleep(1)
    return False


def read_totals(d: Path):
    t = {}
    for f in d.glob("*.json"):
        try:
            for k, v in json.loads(f.read_text()).items():
                t[k] = t.get(k, 0) + v
        except Exception:
            pass
    return t


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pearl", required=True)
    ap.add_argument("--sharegpt", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--work", default="/workspace/res/online_fast")
    ap.add_argument("--configs", nargs="*", default=list(CONFIGS))
    ap.add_argument("--gpu-util", default="0.5")
    args = ap.parse_args()
    pearl = Path(args.pearl)
    work = Path(args.work)
    work.mkdir(parents=True, exist_ok=True)

    gateway = subprocess.Popen([sys.executable, str(HERE / "mock_gateway.py"), "--pearl", args.pearl],
                               stdout=open(work / "gateway.log", "w"), stderr=subprocess.STDOUT)
    time.sleep(15)
    results = []
    try:
        for name in args.configs:
            no_mining, min_m, fast, flags = CONFIGS[name]
            set_min_m(pearl, min_m)
            rec = work / f"rec_{name}"
            shutil.rmtree(rec, ignore_errors=True)
            rec.mkdir()
            env = {**os.environ, "MINER_NO_MINING": no_mining, "MINER_NO_GATEWAY": "false",
                   "MINER_SKIP_BLOCK_SUBMISSION": "true", "PEARL_RECORD_DIR": str(rec),
                   "PEARL_FAST": fast,
                   "PYTHONPATH": f"{HERE / 'fastsite'}:{os.environ.get('PYTHONPATH', '')}"}
            server = subprocess.Popen(
                ["vllm", "serve", MODEL, "--port", str(PORT), "--max-model-len", "4096",
                 "--gpu-memory-utilization", args.gpu_util, "--max-num-seqs", "1024", "--seed", "0", *flags],
                env=env, stdout=open(work / f"server_{name}.log", "w"), stderr=subprocess.STDOUT,
                start_new_session=True)
            try:
                if not wait_ready(server):
                    print(json.dumps({"config": name, "error": "server did not start"}), flush=True)
                    results.append({"config": name, "error": "server did not start"})
                    continue
                for workload, rate, n in RUNS:
                    tag = f"{name}_{workload}_{rate}"
                    time.sleep(3)
                    before = read_totals(rec)
                    r = subprocess.run(
                        ["vllm", "bench", "serve", "--backend", "vllm", "--base-url", f"http://localhost:{PORT}",
                         "--model", MODEL, "--dataset-name", "sharegpt", "--dataset-path", args.sharegpt,
                         "--num-prompts", str(n), "--request-rate", str(rate),
                         "--ignore-eos", "--seed", "0", "--temperature", "0",
                         "--percentile-metrics", "ttft,tpot,itl,e2el", "--metric-percentiles", "50,90,99",
                         "--save-result", "--result-dir", str(work), "--result-filename", f"{tag}.json"],
                        stdout=open(work / f"{tag}.log", "w"), stderr=subprocess.STDOUT)
                    time.sleep(3)
                    if r.returncode != 0:
                        row = {"config": name, "rate": rate, "error": f"bench exited {r.returncode}"}
                    else:
                        after = read_totals(rec)
                        d = {k: after.get(k, 0) - before.get(k, 0) for k in after}
                        b = json.loads((work / f"{tag}.json").read_text())
                        row = {"config": name, "rate": rate, "num_prompts": n,
                               **{k: b.get(k) for k in (
                                   "duration", "completed", "failed", "total_output_tokens", "output_throughput",
                                   "median_ttft_ms", "p90_ttft_ms", "median_tpot_ms", "p90_tpot_ms",
                                   "median_e2el_ms", "p90_e2el_ms")}}
                        if d.get("calls"):
                            row.update({"recorded_calls": d["calls"], "mined_calls": d["mined_calls"],
                                        "mined_share_of_7bit_macs": round(d["mined_macs"] / d["macs"], 4)})
                    results.append(row)
                    print(json.dumps(row), flush=True)
                    Path(args.out).write_text(json.dumps(results, indent=1))
            finally:
                try:
                    os.killpg(server.pid, signal.SIGTERM)
                    server.wait(timeout=120)
                except Exception:
                    pass
                time.sleep(10)
    finally:
        gateway.terminate()
        subprocess.run(["git", "checkout", "--", CFG], cwd=pearl)
        n = subprocess.run(["git", "status", "--porcelain"], cwd=pearl, capture_output=True, text=True).stdout
        print(f"REVERTED: {len(n.splitlines())} changes", flush=True)
        print("ONLINE_FAST_DONE", flush=True)


if __name__ == "__main__":
    main()
