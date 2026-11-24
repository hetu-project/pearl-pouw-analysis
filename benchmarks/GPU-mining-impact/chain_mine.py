#!/usr/bin/env python3
"""
Mine on a private regtest chain while serving inference, and let Pearl's own node decide
whether the blocks are valid.

Expects a local pearld (--regtest, no peers) and pearl-gateway already running; see
run_chain.sh. Block submission is ON here -- the only place in this benchmark suite where
it is -- and it goes to that private node only.

  --fast         use fast_noisy.py instead of Pearl's pearl_gemm_noisy
  --m-range a b  mine only GEMMs whose row count m is in [a, b]; everything else runs the
                 plain kernel. Used to make every block come from small decode-step batches.
                 (config.yaml's min_m must be <= a; run_chain.sh sets it.)
  --batch N      concurrent sequences (= m of a decode step)
  --seconds S    keep generating for about S seconds

Prints one JSON line with what was mined; chain height is read by run_chain.sh via prlctl.
"""
from __future__ import annotations

import argparse
import json
import os
import random
import sys
import time
from collections import Counter
from pathlib import Path

HERE = Path(__file__).resolve().parent


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="pearl-ai/Llama-3.1-8B-Instruct-pearl")
    ap.add_argument("--fast", action="store_true")
    ap.add_argument("--m-range", type=int, nargs=2, default=None)
    ap.add_argument("--batch", type=int, default=24)
    ap.add_argument("--inp", type=int, default=200)
    ap.add_argument("--out-tokens", type=int, default=64)
    ap.add_argument("--seconds", type=float, default=120)
    ap.add_argument("--gpu-util", type=float, default=0.5)
    ap.add_argument("--tag", required=True)
    args = ap.parse_args()

    os.environ.update({
        "MINER_NO_MINING": "false",
        "MINER_NO_GATEWAY": "false",
        "MINER_SKIP_BLOCK_SUBMISSION": "false",
        "VLLM_ENABLE_V1_MULTIPROCESSING": "0",
    })

    from miner_base.settings import MinerSettings
    from vllm import LLM, SamplingParams
    from vllm.inputs import TokensPrompt

    import vllm_miner.gemm_operators as GO
    import vllm_miner.vllm_kernels as kernels
    from vllm_miner.config import config
    from vllm_miner.mining_state import get_async_manager, init_async_manager, init_pinned_pool

    settings = MinerSettings()
    assert not settings.no_mining and not settings.skip_block_submission and not settings.no_gateway
    init_async_manager(settings)
    init_pinned_pool(get_async_manager()._conf.pinned_pool_size)

    if args.fast:
        sys.path.insert(0, str(HERE))
        import fast_noisy
        fast_noisy.install(GO)
        for name, m in list(sys.modules.items()):
            if name.startswith("vllm_miner") and m is not GO and getattr(m, "pearl_gemm_noisy", None) is not None:
                m.pearl_gemm_noisy = GO.pearl_gemm_noisy
        print("CHAIN: fast path installed", flush=True)

    if args.m_range:
        lo, hi = args.m_range
        orig_should = type(config).should_use_noisy_gemm

        def should(self, m, n, k):
            return lo <= m <= hi and orig_should(self, m, n, k)

        type(config).should_use_noisy_gemm = should
        print(f"CHAIN: mining restricted to {lo} <= m <= {hi}", flush=True)

    mined_m = Counter()
    plain_calls = [0]
    original = kernels.PearlKernel._apply_weights_mining

    def recorded(self, layer, x, w_q, *rest, **kw):
        m, k, n = x.reshape(-1, x.shape[-1]).shape[0], x.shape[-1], w_q.shape[0]
        if config.should_use_noisy_gemm(m, n, k):
            mined_m[m] += 1
        else:
            plain_calls[0] += 1
        return original(self, layer, x, w_q, *rest, **kw)

    kernels.PearlKernel._apply_weights_mining = recorded

    llm = LLM(model=args.model, max_model_len=4096, gpu_memory_utilization=args.gpu_util,
              max_num_seqs=1024, enforce_eager=True, disable_log_stats=True, seed=0)
    vocab = llm.llm_engine.model_config.hf_config.vocab_size
    rng = random.Random(7)

    def prompts():
        return [TokensPrompt(prompt_token_ids=[rng.randrange(1000, vocab - 1000) for _ in range(args.inp)])
                for _ in range(args.batch)]

    sp = SamplingParams(temperature=0.0, max_tokens=args.out_tokens, ignore_eos=True)
    llm.generate(prompts()[:2], SamplingParams(max_tokens=2, ignore_eos=True), use_tqdm=False)
    mined_m.clear()
    plain_calls[0] = 0

    t0 = time.time()
    rounds = tokens = 0
    while time.time() - t0 < args.seconds:
        outs = llm.generate(prompts(), sp, use_tqdm=False)
        tokens += sum(len(o.outputs[0].token_ids) for o in outs)
        rounds += 1
    elapsed = time.time() - t0
    print("CHAIN: generation finished, draining proofs for 90 s", flush=True)
    time.sleep(90)  # let queued ZK proofs finish and be submitted

    ms = sorted(mined_m)
    print("CHAIN_RESULT " + json.dumps({
        "tag": args.tag, "fast": args.fast, "m_range": args.m_range, "batch": args.batch,
        "min_m_config": config.matrix_multiplication_config["use_simplified_gemm"]["min_m"],
        "gen_seconds": round(elapsed, 1), "rounds": rounds, "output_tokens": tokens,
        "mined_calls": sum(mined_m.values()), "plain_calls": plain_calls[0],
        "mined_m_min": ms[0] if ms else None, "mined_m_max": ms[-1] if ms else None,
        "mined_m_hist": {str(k): mined_m[k] for k in ms[:12]},
    }), flush=True)
    os._exit(0)


if __name__ == "__main__":
    main()
