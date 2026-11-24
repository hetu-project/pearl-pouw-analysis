#!/usr/bin/env python3
"""
Can the weight matrix's Merkle hash be computed once per block instead of once per GEMM?

Inside pearl_gemm_noisy, tensor_hash() is called twice: first over A (the activations),
then over B (the weights). B's hash depends only on
  - B itself, which is a weight and never changes, and
  - key_tensor = CommitmentHasher.get_key(block_header_bytes, mining_config),
which changes once per block (mainnet target block time: 194 s).

So B's hash is constant for a given (layer, block) yet Pearl rescans the whole weight
matrix -- 117 MB for Llama's gate_up -- on every single call. In one block that is
repeated thousands of times.

This measures what caching it would buy. Three configurations per shape:
  stock      unmodified
  cached     B's hash cached per (weight pointer, shape, key); A is never cached
  no_b_hash  B's hash skipped entirely -- NOT correct, only an upper bound on the saving

Correctness: 'cached' must produce bit-identical output to 'stock'. The script asserts it.
Safety: mock gateway, block submission skipped, nothing connects to the Pearl network.

Usage: python bhash_cache_ab.py --pearl /workspace/pearl --out bhash_cache.json
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import os
import statistics
import time
from pathlib import Path

P05 = Path("/workspace/pearl-pouw-analysis/benchmarks/P05-result-accuracy/gpu/p05_gpu_check.py")

SHAPES = [
    ("o_proj", 4096, 4096),
    ("qkv_proj", 6144, 4096),
    ("gate_up_proj", 28672, 4096),
]
MS = [256, 512, 1024, 2048, 4096, 8192]


def load_p05():
    spec = importlib.util.spec_from_file_location("p05_gpu_check", P05)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def time_ms(fn, warmup, iters):
    import torch
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(iters):
        fn()
    torch.cuda.synchronize()
    return (time.perf_counter() - t0) * 1000 / iters


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pearl", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--iters", type=int, default=50)
    args = ap.parse_args()

    os.environ.update({
        "MINER_NO_MINING": "false",
        "MINER_NO_GATEWAY": "false",
        "MINER_SKIP_BLOCK_SUBMISSION": "true",
    })
    load_p05().start_mock_gateway(Path(args.pearl))

    import torch
    from miner_base.settings import MinerSettings
    from vllm_miner import gemm_operators as GO
    from vllm_miner.mining_state import get_async_manager, init_async_manager, init_pinned_pool
    from vllm_miner.quantization_operators import quant_7bit

    settings = MinerSettings()
    assert not settings.no_mining and settings.skip_block_submission
    init_async_manager(settings)
    init_pinned_pool(get_async_manager()._conf.pinned_pool_size)

    orig_tensor_hash = GO.tensor_hash
    orig_get_key = GO.CommitmentHasher.get_key

    state = {"key": None, "which": 0, "mode": "stock"}
    cache: dict = {}

    class KeyRecordingHasher:
        @staticmethod
        def get_key(header_bytes, mining_config):
            k = orig_get_key(header_bytes, mining_config)
            state["key"] = bytes(k)
            state["which"] = 0  # next tensor_hash in this call is A
            return k

    def patched_tensor_hash(x, key_tensor, out, scratch):
        state["which"] += 1
        is_b = state["which"] == 2  # second call inside pearl_gemm_noisy is the weights
        if not is_b or state["mode"] == "stock":
            return orig_tensor_hash(x, key_tensor, out, scratch)
        if state["mode"] == "no_b_hash":
            return None  # deliberately wrong: upper bound on the saving only
        ck = (x.data_ptr(), tuple(x.shape), state["key"])
        hit = cache.get(ck)
        if hit is not None:
            out.copy_(hit)
            return None
        orig_tensor_hash(x, key_tensor, out, scratch)
        cache[ck] = out.clone()
        return None

    GO.CommitmentHasher = KeyRecordingHasher
    GO.tensor_hash = patched_tensor_hash

    torch.manual_seed(0)
    rows = []

    for name, n, k in SHAPES:
        w_q, w_s, _ = quant_7bit(torch.randn(n, k, dtype=torch.bfloat16, device="cuda"))
        w_q, w_s = w_q.contiguous(), w_s.squeeze(-1)
        for m in MS:
            x_q, x_s, _ = quant_7bit(torch.randn(m, k, dtype=torch.bfloat16, device="cuda"))
            x_q, x_s = x_q.contiguous(), x_s.squeeze(-1)

            def noisy():
                return GO.pearl_gemm_noisy(x_q, w_q, scale_a=x_s, scale_b=w_s,
                                           out_dtype=torch.bfloat16, submit_block=False)

            def vanilla():
                return GO.pearl_gemm_vanilla(x_q, w_q, scale_a=x_s, scale_b=w_s,
                                             out_dtype=torch.bfloat16)

            cache.clear()
            state["mode"] = "stock"
            ref = noisy().clone()
            t_plain = time_ms(vanilla, 5, args.iters)
            t_stock = time_ms(noisy, 5, args.iters)

            state["mode"] = "cached"
            cache.clear()
            noisy()                      # populate
            got = noisy()                # served from cache
            identical = torch.equal(ref, got)
            t_cached = time_ms(noisy, 5, args.iters)

            state["mode"] = "no_b_hash"
            t_skip = time_ms(noisy, 5, args.iters)
            state["mode"] = "stock"

            row = {
                "layer": name, "m": m, "n": n, "k": k,
                "weight_mib": round(n * k / 2**20, 1),
                "plain_ms": round(t_plain, 4),
                "stock_ms": round(t_stock, 4),
                "cached_ms": round(t_cached, 4),
                "skip_b_hash_ms": round(t_skip, 4),
                "saved_ms": round(t_stock - t_cached, 4),
                "saved_pct_of_mining_overhead": round(
                    100 * (t_stock - t_cached) / (t_stock - t_plain), 1) if t_stock > t_plain else None,
                "output_identical": bool(identical),
            }
            rows.append(row)
            print(f"{name:13s} m={m:5d} B={row['weight_mib']:6.1f}MiB  plain {t_plain:7.3f}  "
                  f"stock {t_stock:7.3f} -> cached {t_cached:7.3f} (skip {t_skip:7.3f})  "
                  f"saved {row['saved_ms']:6.3f} ms = {row['saved_pct_of_mining_overhead']}% of overhead  "
                  f"identical={identical}", flush=True)

    bad = [r for r in rows if not r["output_identical"]]
    Path(args.out).write_text(json.dumps({
        "gpu": torch.cuda.get_device_name(0), "iters": args.iters,
        "all_identical": not bad, "rows": rows}, indent=1))
    print(f"\nall cached outputs bit-identical to stock: {not bad}")
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
