#!/usr/bin/env python3
"""
Mining under online traffic: requests arrive over time (Poisson) instead of all at once.

For each configuration (graph / eager / eager_mine, as in serving_impact.py) this starts
`vllm serve` with Pearl's plugin, then drives it with `vllm bench serve` at several request rates:
  sharegpt  real chat prompts and answer lengths from ShareGPT, at 1, 4 and 16 requests/s
  long      random 2000-token prompts, 64-token answers, at 0.5 and 2 requests/s
EOS is ignored so all configurations do the same work. TTFT, TPOT and throughput come from
vllm bench; how much of the linear-layer work was mined comes from recorder/sitecustomize.py,
which counts the plugin's 7-bit layer calls inside the engine process (eager configurations only:
CUDA graph replays do not run Python). The mock gateway (mock_gateway.py) runs for the whole
benchmark; block submission is skipped and nothing connects to a network.

Usage (Pearl's miner environment, H100/H200):
  python online_impact.py --pearl <pearl_dir> --sharegpt ShareGPT_V3_unfiltered_cleaned_split.json --out online.json
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
CONFIGS = {
    "graph": ("true", ["--compilation-config", '{"mode": 0, "cudagraph_mode": "FULL_DECODE_ONLY"}']),
    "eager": ("true", ["--enforce-eager"]),
    "eager_mine": ("false", ["--enforce-eager"]),
}  # name: (MINER_NO_MINING, vllm serve flags)
RUNS = [
    ("sharegpt", 1, 120), ("sharegpt", 4, 400), ("sharegpt", 16, 1000),
    ("long", 0.5, 60), ("long", 2, 200),
]  # (workload, requests per second, number of requests)
LINEAR_MACS_PER_TOKEN = 32 * (4096 * (4096 + 2 * 1024) + 4096 * 4096 + 4096 * 2 * 14336 + 14336 * 4096)  # Llama-3.1-8B
LAYERS = 32


def wait_ready(proc, timeout=900):
    for _ in range(timeout):
        if proc.poll() is not None:
            sys.exit(f"vllm serve exited with {proc.returncode}")
        try:
            urllib.request.urlopen(f"http://localhost:{PORT}/health", timeout=2)
            return
        except Exception:
            time.sleep(1)
    sys.exit("vllm serve did not become ready")


def read_totals(record_dir: Path):
    totals = {}
    for f in record_dir.glob("*.json"):
        for key, value in json.loads(f.read_text()).items():
            totals[key] = totals.get(key, 0) + value
    return totals


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pearl", required=True)
    ap.add_argument("--sharegpt", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--work", default="/workspace/res/online")
    ap.add_argument("--configs", nargs="*", default=list(CONFIGS))
    args = ap.parse_args()

    work = Path(args.work)
    work.mkdir(parents=True, exist_ok=True)
    gateway = subprocess.Popen([sys.executable, str(HERE / "mock_gateway.py"), "--pearl", args.pearl],
                               stdout=open(work / "gateway.log", "w"), stderr=subprocess.STDOUT)
    time.sleep(15)
    results = []
    try:
        for name in args.configs:
            no_mining, flags = CONFIGS[name]
            record_dir = work / f"rec_{name}"
            shutil.rmtree(record_dir, ignore_errors=True)
            record_dir.mkdir()
            env = {**os.environ, "MINER_NO_MINING": no_mining, "MINER_NO_GATEWAY": "false",
                   "MINER_SKIP_BLOCK_SUBMISSION": "true", "PEARL_RECORD_DIR": str(record_dir),
                   "PYTHONPATH": f"{HERE / 'recorder'}:{os.environ.get('PYTHONPATH', '')}"}
            server = subprocess.Popen(
                ["vllm", "serve", MODEL, "--port", str(PORT), "--max-model-len", "4096",
                 "--gpu-memory-utilization", "0.9", "--max-num-seqs", "1024", "--seed", "0", *flags],
                env=env, stdout=open(work / f"server_{name}.log", "w"), stderr=subprocess.STDOUT,
                start_new_session=True)
            try:
                wait_ready(server)
                for workload, rate, n in RUNS:
                    tag = f"{name}_{workload}_{rate}"
                    dataset = (["--dataset-name", "sharegpt", "--dataset-path", args.sharegpt]
                               if workload == "sharegpt" else
                               ["--dataset-name", "random", "--random-input-len", "2000", "--random-output-len", "64"])
                    time.sleep(3)
                    before = read_totals(record_dir)
                    subprocess.run(
                        ["vllm", "bench", "serve", "--backend", "vllm", "--base-url", f"http://localhost:{PORT}",
                         "--model", MODEL, *dataset, "--num-prompts", str(n), "--request-rate", str(rate),
                         "--ignore-eos", "--seed", "0", "--temperature", "0",
                         "--percentile-metrics", "ttft,tpot,itl,e2el", "--metric-percentiles", "50,90,99",
                         "--save-result", "--result-dir", str(work), "--result-filename", f"{tag}.json"],
                        check=True, stdout=open(work / f"{tag}.log", "w"), stderr=subprocess.STDOUT)
                    time.sleep(3)
                    after = read_totals(record_dir)
                    delta = {k: after.get(k, 0) - before.get(k, 0) for k in after}
                    bench = json.loads((work / f"{tag}.json").read_text())
                    row = {"config": name, "workload": workload, "rate": rate, "num_prompts": n,
                           **{k: bench.get(k) for k in (
                               "duration", "completed", "total_input_tokens", "total_output_tokens",
                               "request_throughput", "output_throughput", "total_token_throughput",
                               "median_ttft_ms", "p90_ttft_ms", "p99_ttft_ms",
                               "median_tpot_ms", "p90_tpot_ms", "p99_tpot_ms", "median_itl_ms", "p90_itl_ms",
                               "median_e2el_ms", "p90_e2el_ms")}}
                    if delta.get("square_calls"):
                        tokens = delta["square_m"] / LAYERS
                        row.update({
                            "steps": delta["square_calls"] // LAYERS, "tokens_processed": int(tokens),
                            "mined_calls": delta["mined_calls"], "mined_macs": delta["mined_macs"],
                            "mined_share_of_7bit_macs": round(delta["mined_macs"] / delta["macs"], 4),
                            "mined_share_of_linear_macs": round(delta["mined_macs"] / (tokens * LINEAR_MACS_PER_TOKEN), 4),
                            "mined_macs_per_s": delta["mined_macs"] / bench["duration"],
                        })
                    results.append(row)
                    print(json.dumps(row), flush=True)
                    Path(args.out).write_text(json.dumps(results, indent=1))
            finally:
                os.killpg(server.pid, signal.SIGTERM)
                server.wait(timeout=120)
                time.sleep(10)
    finally:
        gateway.terminate()


if __name__ == "__main__":
    main()
