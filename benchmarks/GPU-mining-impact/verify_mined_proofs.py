#!/usr/bin/env python3
"""
Are the proofs mined during inference valid? Checked locally with Pearl's own prover and verifier.

All other benchmarks here only show that the noisy GEMM ran. With the mock template's difficulty
(2**180) no tile ever meets the target, so no proof is ever produced or checked. This script:

  * runs Pearl's gateway in this process with the same mock block template, but with an easy
    target (--bits), so that tiles meet it during a short run;
  * lets the plugin hand each found proof to that gateway (MINER_SKIP_BLOCK_SUBMISSION=false);
  * the gateway, in debug mode, builds the ZK certificate and checks it with
    verify_proof_for_cert_version (pearl_gateway/proof_generator.py) -- Pearl's verifier;
  * the gateway's node client is replaced by a stub, so no block is sent anywhere. Nothing
    connects to a network; the template is a fixed fake at height 100000.

Run once with --enforce-eager (Pearl's supported setup, the control) and once with --graph
(CUDA graphs for decode, which Pearl's README does not allow):
  python verify_mined_proofs.py --pearl <pearl_dir> --out eager.json
  python verify_mined_proofs.py --pearl <pearl_dir> --graph --out graph.json
"""
from __future__ import annotations

import argparse
import asyncio
import json
import os
import random
import threading
import time
from pathlib import Path

from kernel_overhead import load_p05

MODEL = "pearl-ai/Llama-3.1-8B-Instruct-pearl"
SOCKET = "/tmp/pearlgw.sock"
COUNTS = {"received": 0, "verified": 0, "failed": 0, "errors": []}


class NoNode:
    """Stands in for PearlNodeClient: accepts the finished block and sends it nowhere."""

    async def submit_block(self, block_hex: str) -> str:
        return "local_test_not_submitted"


def start_verifying_gateway(pearl: Path, bits: str):
    p05 = load_p05()
    import pearl_gateway.comm.dataclasses as dc
    from pearl_gateway.comm.dataclasses import BlockTemplate
    from pearl_gateway.pearl_gateway import PearlGateway
    from pearl_gateway.proof_generator import ProofGenerator
    from pearl_gateway.rpc_types import GetBlockTemplateResponse

    data = dict(p05.conftest_literal(pearl, "sample_block_template_data"))
    target = dc.bits_to_target(int(bits, 16))
    data.update({"bits": bits, "target": f"{target:x}"})
    address = p05.conftest_literal(pearl, "mining_address")
    template = BlockTemplate.from_get_block_template(GetBlockTemplateResponse.model_validate(data),
                                                     mining_address=address)
    os.environ["PEARLD_MINING_ADDRESS"] = address
    os.environ["PEARLD_RPC_URL"] = "http://127.0.0.1:9"  # discard port; the client is replaced below anyway

    original = ProofGenerator.generate_block.__func__

    def counted(cls, plain_proof, tpl, debug_mode=False):
        COUNTS["received"] += 1
        try:
            block = original(cls, plain_proof, tpl, True)  # True: always verify
        except Exception as e:  # generation or verification failed
            COUNTS["failed"] += 1
            if len(COUNTS["errors"]) < 5:
                COUNTS["errors"].append(repr(e)[:300])
            raise
        COUNTS["verified"] += 1
        return block

    ProofGenerator.generate_block = classmethod(counted)

    if os.path.exists(SOCKET):
        os.unlink(SOCKET)

    async def serve():
        gateway = PearlGateway(debug_mode=True)
        gateway.submission_service.pearl_client = NoNode()
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
    return target


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pearl", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--graph", action="store_true")
    ap.add_argument("--bits", default="1f00ffff", help="compact target; 1f00ffff is about 2**240")
    ap.add_argument("--n", type=int, default=64, help="number of 2000-token prompts")
    ap.add_argument("--max-tokens", type=int, default=32)
    ap.add_argument("--gpu-mem", type=float, default=0.5)
    ap.add_argument("--model", default=MODEL)
    ap.add_argument("--wait", type=int, default=900, help="seconds to wait for the gateway to finish checking")
    args = ap.parse_args()

    os.environ.update({"MINER_NO_MINING": "false", "MINER_NO_GATEWAY": "false",
                       "MINER_SKIP_BLOCK_SUBMISSION": "false", "VLLM_ENABLE_V1_MULTIPROCESSING": "0"})
    target = start_verifying_gateway(Path(args.pearl), args.bits)

    import torch
    from miner_base.settings import MinerSettings
    from vllm import LLM, SamplingParams
    from vllm.inputs import TokensPrompt

    import vllm_miner.callbacks as callbacks
    import vllm_miner.vllm_kernels as kernels
    from vllm_miner.mining_state import get_async_manager, init_async_manager, init_pinned_pool

    settings = MinerSettings()
    assert not settings.no_mining and not settings.skip_block_submission
    init_async_manager(settings)
    init_pinned_pool(get_async_manager()._conf.pinned_pool_size)

    stats = {"noisy_calls": 0, "found": 0}
    noisy = kernels.pearl_gemm_noisy

    def counted_noisy(*a, **kw):
        stats["noisy_calls"] += 1
        return noisy(*a, **kw)

    kernels.pearl_gemm_noisy = counted_noisy
    check = callbacks.StatusCheckCallback.__call__

    def counted_check(self, handle_submit_block):
        def counting_submit(*a, **kw):
            stats["found"] += 1
            return handle_submit_block(*a, **kw)
        return check(self, counting_submit)

    callbacks.StatusCheckCallback.__call__ = counted_check

    mode = ({"compilation_config": {"mode": 0, "cudagraph_mode": "FULL_DECODE_ONLY"}} if args.graph
            else {"enforce_eager": True})
    llm = LLM(model=args.model, max_model_len=4096, gpu_memory_utilization=args.gpu_mem, max_num_seqs=1024, seed=0, **mode)
    vocab = len(llm.get_tokenizer())
    rng = random.Random(7)
    prompts = [TokensPrompt(prompt_token_ids=[rng.randrange(1000, vocab - 1000) for _ in range(2000)])
               for _ in range(args.n)]
    t0 = time.time()
    llm.generate(prompts, SamplingParams(temperature=0.0, max_tokens=args.max_tokens, ignore_eos=True), use_tqdm=False)
    generate_s = time.time() - t0

    deadline = time.time() + args.wait
    last, stable = None, 0
    while time.time() < deadline:
        time.sleep(5)
        now = (stats["found"], COUNTS["received"], COUNTS["verified"] + COUNTS["failed"])
        stable = stable + 1 if now == last and now[1] == now[2] else 0
        last = now
        if stable >= 6:  # nothing new for 30 s and nothing in progress
            break
    result = {"mode": "graph" if args.graph else "eager", "bits": args.bits, "target_log2": target.bit_length() - 1,
              "model": args.model, "prompts": args.n, "generate_s": round(generate_s, 1), "gpu": torch.cuda.get_device_name(0),
              "noisy_gemm_calls": stats["noisy_calls"], "proofs_found_by_miner": stats["found"],
              "proofs_received_by_gateway": COUNTS["received"], "proofs_verified": COUNTS["verified"],
              "proofs_failed": COUNTS["failed"], "errors": COUNTS["errors"]}
    Path(args.out).write_text(json.dumps(result, indent=1))
    print("VERIFY_RESULT " + json.dumps(result), flush=True)
    os._exit(0)


if __name__ == "__main__":
    main()
