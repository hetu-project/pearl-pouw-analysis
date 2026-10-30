#!/usr/bin/env python3
"""
P05 GPU check: does mining change the useful output?

Needs an H100/H200 and Pearl's miner environment (vllm, vllm-miner, pearl-gateway, pearl-mining
installed as in Pearl's miner README), plus the `datasets` package for GSM8K.
WRITTEN BUT NOT YET RUN: we have no Hopper GPU, and Pearl's mining kernel builds only for sm_90a.

This is the comparison Pearl's own test (miner/vllm-miner/tests/test_vllm_execution.py,
test_consistency) intends, set up so that mining really happens:
  * same checkpoint, same prompts, greedy decoding; the only difference between the two runs is
    MINER_NO_MINING, which selects noisy or vanilla GEMM on the 7-bit layers (vllm_kernels.py);
  * all GSM8K test questions are sent in one batch with max_num_seqs=1024, so prefill steps and
    most decode steps reach the plugin's 1024-token threshold;
  * pearl_gemm_noisy is wrapped to count how often it really runs: it must run with mining on and
    never with mining off, so a run that never mined cannot pass as a comparison;
  * the runs are compared by accuracy and per-question agreement, not by the exact text of one
    short answer.
Block submission is skipped (MINER_SKIP_BLOCK_SUBMISSION) and the gateway is fed Pearl's own mock
block template from miner/conftest.py; nothing connects to a real network.
Dense checkpoints only: MoE expert GEMMs go through pearl_moe_experts.py, which is not counted.

Usage (on the GPU machine, inside Pearl's miner environment; one process per mode, because the
miner reads MINER_NO_MINING once at import):
  python p05_gpu_check.py run --pearl <pearl_dir> --mode no_mining --out off.json
  python p05_gpu_check.py run --pearl <pearl_dir> --mode mining    --out on.json
  python p05_gpu_check.py compare off.json on.json
"""
from __future__ import annotations

import argparse
import ast
import asyncio
import json
import os
import re
import sys
import threading
import time
from pathlib import Path

DEFAULT_MODEL = "pearl-ai/Llama-3.1-8B-Instruct-pearl"
SOCKET = "/tmp/pearlgw.sock"  # pearl-gateway default, as in Pearl's test fixture
INSTRUCTION = "Solve the problem step by step. End with a final line of the form '#### <number>'."


def conftest_literal(pearl: Path, fixture: str):
    """Return value of a fixture in Pearl's miner/conftest.py, read from source (no pytest)."""
    tree = ast.parse((pearl / "miner/conftest.py").read_text())
    fn = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == fixture)
    ret = next(n for n in ast.walk(fn) if isinstance(n, ast.Return))
    return ast.literal_eval(ret.value)


def start_mock_gateway(pearl: Path):
    """Same as Pearl's pearl_gateway_process fixture: a local gateway with a mock block template."""
    from pearl_gateway.comm.dataclasses import BlockTemplate
    from pearl_gateway.pearl_gateway import PearlGateway
    from pearl_gateway.rpc_types import GetBlockTemplateResponse

    address = conftest_literal(pearl, "mining_address")
    template = BlockTemplate.from_get_block_template(
        GetBlockTemplateResponse.model_validate(conftest_literal(pearl, "sample_block_template_data")),
        mining_address=address,
    )
    os.environ["PEARLD_MINING_ADDRESS"] = address
    if os.path.exists(SOCKET):
        os.unlink(SOCKET)

    async def serve():
        gateway = PearlGateway(debug_mode=True)
        await gateway.start()
        await gateway.work_cache.update_template(template)
        while True:
            await asyncio.sleep(1)

    threading.Thread(target=lambda: asyncio.run(serve()), daemon=True).start()
    for _ in range(40):
        if os.path.exists(SOCKET):
            break
        time.sleep(0.5)
    else:
        raise RuntimeError(f"gateway socket {SOCKET} not created")
    time.sleep(2)


def final_number(text: str) -> str | None:
    m = re.search(r"####\s*(-?[\d,]*\.?\d+)", text)
    if m is None:
        nums = re.findall(r"-?[\d,]*\.?\d+", text)
        if not nums:
            return None
        value = nums[-1]
    else:
        value = m.group(1)
    value = value.replace(",", "").rstrip(".")
    try:
        return str(float(value)).rstrip("0").rstrip(".")
    except ValueError:
        return None


def run(args):
    no_mining = args.mode == "no_mining"
    os.environ.update({
        "MINER_NO_MINING": "true" if no_mining else "false",
        "MINER_NO_GATEWAY": "false",
        "MINER_SKIP_BLOCK_SUBMISSION": "true",
        "VLLM_ENABLE_V1_MULTIPROCESSING": "0",  # keep the model in this process so the counter sees it
    })
    pearl = Path(args.pearl)
    start_mock_gateway(pearl)

    import torch
    from datasets import load_dataset
    from miner_base.settings import MinerSettings
    from vllm import LLM, SamplingParams

    import vllm_miner.vllm_kernels as kernels
    from vllm_miner.mining_state import init_async_manager

    settings = MinerSettings()
    assert settings.no_mining == no_mining and settings.skip_block_submission
    init_async_manager(settings)

    stats = {"calls": 0, "rows": 0, "min_m": None}
    original = kernels.pearl_gemm_noisy

    def counted(a, *rest, **kw):
        m = a.shape[0]
        stats["calls"] += 1
        stats["rows"] += m
        stats["min_m"] = m if stats["min_m"] is None else min(stats["min_m"], m)
        return original(a, *rest, **kw)

    kernels.pearl_gemm_noisy = counted

    ds = load_dataset("openai/gsm8k", "main", split="test")
    if args.n:
        ds = ds.select(range(args.n))
    llm = LLM(
        model=args.model,
        max_model_len=4096,
        enforce_eager=True,  # as in Pearl's test and README
        gpu_memory_utilization=0.9,
        attention_config={"backend": "TRITON_ATTN"},  # bitwise-reproducible, as in Pearl's test
        max_num_seqs=1024,
        seed=0,
    )
    tok = llm.get_tokenizer()
    prompts = [
        tok.apply_chat_template([{"role": "user", "content": f"{q}\n{INSTRUCTION}"}], tokenize=False, add_generation_prompt=True)
        for q in ds["question"]
    ]
    t0 = time.time()
    outs = llm.generate(prompts, SamplingParams(temperature=0.0, max_tokens=args.max_tokens, seed=42))
    elapsed = time.time() - t0

    rows = []
    for ex, out in zip(ds, outs):
        text = out.outputs[0].text
        gold = final_number(ex["answer"])
        pred = final_number(text)
        rows.append({"text": text, "gold": gold, "pred": pred, "correct": pred is not None and pred == gold})
    result = {
        "mode": args.mode, "model": args.model, "n": len(rows), "max_tokens": args.max_tokens,
        "gpu": torch.cuda.get_device_name(0), "seconds": round(elapsed, 1),
        "noisy_gemm_calls": stats["calls"], "noisy_gemm_rows": stats["rows"], "noisy_gemm_min_m": stats["min_m"],
        "accuracy": sum(r["correct"] for r in rows) / len(rows), "rows": rows,
    }
    Path(args.out).write_text(json.dumps(result, ensure_ascii=False, indent=1))
    print(f"{args.mode}: accuracy {result['accuracy']:.2%} on {len(rows)} questions, "
          f"noisy GEMM calls {stats['calls']} (min m {stats['min_m']}), {elapsed:.0f} s on {result['gpu']}")
    if no_mining and stats["calls"] != 0:
        sys.exit("noisy GEMM ran although mining was switched off")
    if not no_mining and stats["calls"] == 0:
        sys.exit("mining run never executed a noisy GEMM: the comparison would be meaningless")


def compare(args):
    off = json.loads(Path(args.off).read_text())
    on = json.loads(Path(args.on).read_text())
    assert off["mode"] == "no_mining" and on["mode"] == "mining" and off["n"] == on["n"]
    assert off["noisy_gemm_calls"] == 0 < on["noisy_gemm_calls"], "mining did not run in the mining run"
    pairs = list(zip(off["rows"], on["rows"]))
    same_text = sum(a["text"] == b["text"] for a, b in pairs) / len(pairs)
    same_pred = sum(a["pred"] == b["pred"] for a, b in pairs) / len(pairs)
    lost = sum(a["correct"] and not b["correct"] for a, b in pairs)
    gained = sum(b["correct"] and not a["correct"] for a, b in pairs)
    print(f"model {on['model']}, {on['n']} GSM8K questions, {on['gpu']}")
    print(f"  noisy GEMM calls with mining on: {on['noisy_gemm_calls']} (smallest m {on['noisy_gemm_min_m']})")
    print(f"  accuracy  mining off {off['accuracy']:.2%}   mining on {on['accuracy']:.2%}   "
          f"difference {100 * (on['accuracy'] - off['accuracy']):+.2f} points")
    print(f"  identical generated text: {same_text:.1%}   identical final answer: {same_pred:.1%}")
    print(f"  questions right only with mining off: {lost}   right only with mining on: {gained}")


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run")
    r.add_argument("--pearl", required=True, help="Pearl checkout at 156ef9a8")
    r.add_argument("--mode", choices=["mining", "no_mining"], required=True)
    r.add_argument("--out", required=True)
    r.add_argument("--model", default=DEFAULT_MODEL)
    r.add_argument("--n", type=int, default=0, help="number of GSM8K test questions (0 = all 1319)")
    r.add_argument("--max-tokens", type=int, default=512)
    c = sub.add_parser("compare")
    c.add_argument("off")
    c.add_argument("on")
    args = ap.parse_args()
    run(args) if args.cmd == "run" else compare(args)


if __name__ == "__main__":
    main()
