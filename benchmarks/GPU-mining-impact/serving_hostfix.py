#!/usr/bin/env python3
"""
Runs serving_impact.py with (or without) the two host-side mining fixes applied in-process.

The fixes (see host_fix_ab.py) are pure plumbing and leave every number fed to the CUDA
kernels unchanged:
  FIX 1  memoise GPUMatmulConfigFactory.create  -- it rebuilds MinerSettings() from the
         environment and re-parses the 64-entry cols_pattern on every single GEMM.
  FIX 2  make_pow_target_tensor writes 8 uint32 words into a CUDA tensor one at a time
         from a Python loop; build them on the host, transfer once, cache by value.

They are applied by a daemon thread that waits for vllm_miner.gemm_operators to appear in
sys.modules (the same pattern recorder/sitecustomize.py uses), so Pearl's source tree is
never modified.

Enable with PEARL_HOSTFIX=1; without it this is exactly serving_impact.py.

Usage:
  PEARL_HOSTFIX=1 python serving_hostfix.py run --pearl /workspace/pearl \
      --config eager_mine --only chat long --out out.json
"""
from __future__ import annotations

import os
import runpy
import sys
import threading
import time
from functools import lru_cache
from pathlib import Path

BENCH = Path("/workspace/pearl-pouw-analysis/benchmarks/GPU-mining-impact")
sys.path.insert(0, str(BENCH))

ENABLED = os.environ.get("PEARL_HOSTFIX") == "1"


def _patch_when_ready():
    mod = None
    for _ in range(9000):  # give up after fifteen minutes
        cand = sys.modules.get("vllm_miner.gemm_operators")
        if cand is not None and hasattr(cand, "pearl_gemm_noisy"):
            mod = cand
            break
        time.sleep(0.1)
    if mod is None:
        print("HOSTFIX: vllm_miner.gemm_operators never imported, no patch applied", flush=True)
        return

    import torch

    orig_create = mod.GPUMatmulConfigFactory.create
    orig_pow = mod.make_pow_target_tensor

    @lru_cache(maxsize=None)
    def _cached_create(k, noise_rank):
        return orig_create(k=k, noise_rank=noise_rank, moe=None)

    class CachedFactory:
        @staticmethod
        def create(k, noise_rank, moe=None):
            if moe is None:
                return _cached_create(k, noise_rank)
            return orig_create(k=k, noise_rank=noise_rank, moe=moe)

    _pow_cache: dict = {}

    def fast_pow_target(value: int, device="cuda"):
        hit = _pow_cache.get(value)
        if hit is not None:
            return hit
        words = [(value >> (32 * i)) & 0xFFFFFFFF for i in range(8)]
        t = torch.tensor(words, dtype=torch.uint32, device=device)
        if len(_pow_cache) < 256:
            _pow_cache[value] = t
        return t

    mod.GPUMatmulConfigFactory = CachedFactory
    mod.make_pow_target_tensor = fast_pow_target
    print("HOSTFIX: applied (config memoised, pow target single-transfer + cached)", flush=True)


def _patch_weight_hash_cache():
    """Cache the weight matrix's Merkle hash per (weight, block).

    pearl_gemm_noisy hashes A then B on every call. B is a weight and never changes, and the
    hash key only changes when the block header does (194 s on mainnet), so B's hash is constant
    for a (layer, block) yet the whole weight matrix is rescanned every call. Cache it.
    Only the second tensor_hash of each call (B) is cached; A is always recomputed.
    """
    mod = None
    for _ in range(9000):
        cand = sys.modules.get("vllm_miner.gemm_operators")
        if cand is not None and hasattr(cand, "pearl_gemm_noisy"):
            mod = cand
            break
        time.sleep(0.1)
    if mod is None:
        print("BHASH: gemm_operators never imported, no patch applied", flush=True)
        return

    orig_tensor_hash = mod.tensor_hash
    orig_get_key = mod.CommitmentHasher.get_key
    state = {"key": None, "which": 0}
    cache: dict = {}

    class KeyRecordingHasher:
        @staticmethod
        def get_key(header_bytes, mining_config):
            k = orig_get_key(header_bytes, mining_config)
            state["key"] = bytes(k)
            state["which"] = 0
            return k

    def cached_tensor_hash(x, key_tensor, out, scratch):
        state["which"] += 1
        if state["which"] != 2:  # A: always recompute
            return orig_tensor_hash(x, key_tensor, out, scratch)
        ck = (x.data_ptr(), tuple(x.shape), state["key"])
        hit = cache.get(ck)
        if hit is not None:
            out.copy_(hit)
            return None
        orig_tensor_hash(x, key_tensor, out, scratch)
        if len(cache) < 4096:
            cache[ck] = out.clone()
        return None

    mod.CommitmentHasher = KeyRecordingHasher
    mod.tensor_hash = cached_tensor_hash
    print("BHASH: applied (weight Merkle hash cached per weight+block)", flush=True)


def _cap_gpu_memory():
    """serving_impact.py hard-codes gpu_memory_utilization=0.9. Another tenant holds ~40 GiB
    of this card, so allow an override via PEARL_GPU_UTIL without editing that file."""
    util = os.environ.get("PEARL_GPU_UTIL")
    graph = os.environ.get("PEARL_FORCE_GRAPH") == "1"
    if not util and not graph:
        return
    import vllm

    orig_init = vllm.LLM.__init__

    def capped_init(self, *a, **kw):
        if util:
            kw["gpu_memory_utilization"] = float(util)
        if graph:
            # COST PROBE: CUDA graphs with mining on. Replayed decode steps run the mining
            # kernels but no Python, so wins in those steps are never checked or submitted.
            kw["enforce_eager"] = False
            cc = {"mode": 0, "cudagraph_mode": "FULL_DECODE_ONLY"}
            sizes = os.environ.get("PEARL_GRAPH_SIZES")
            if sizes:
                cc["cudagraph_capture_sizes"] = [int(s) for s in sizes.split(",")]
            kw["compilation_config"] = cc
        return orig_init(self, *a, **kw)

    vllm.LLM.__init__ = capped_init
    print(f"HOSTFIX: gpu_memory_utilization={util} force_graph={graph}", flush=True)


if ENABLED:
    threading.Thread(target=_patch_when_ready, daemon=True).start()
    print("HOSTFIX: enabled, waiting for vllm_miner.gemm_operators", flush=True)
else:
    print("HOSTFIX: disabled (stock Pearl code paths)", flush=True)

if os.environ.get("PEARL_BHASH") == "1":
    threading.Thread(target=_patch_weight_hash_cache, daemon=True).start()
    print("BHASH: enabled, waiting for vllm_miner.gemm_operators", flush=True)


def _patch_no_weight_noise():
    """COST PROBE, NOT A WORKING SCHEME.

    Turns off noising of the weight matrix (run_noising_B=False) using pearl-gemm's own switch.
    The GEMM result is then NOT valid Pearl mining -- this exists only to measure what a design
    that never perturbs the weights would cost, using Pearl's own kernels as the yardstick.
    Combine with PEARL_BHASH=1 and the .to->.view edit to remove every weight-sized pass that
    is not intrinsic to the matmul itself.
    """
    mod = None
    for _ in range(9000):
        cand = sys.modules.get("vllm_miner.gemm_operators")
        if cand is not None and hasattr(cand, "noisy_gemm"):
            mod = cand
            break
        time.sleep(0.1)
    if mod is None:
        print("NOBNOISE: gemm_operators never imported, no patch applied", flush=True)
        return

    orig_noisy_gemm = mod.noisy_gemm

    def no_b_noise(*a, **kw):
        kw["run_noising_B"] = False
        return orig_noisy_gemm(*a, **kw)

    mod.noisy_gemm = no_b_noise
    print("NOBNOISE: applied -- weights are NOT noised; output is not valid mining, "
          "this run measures cost only", flush=True)


if os.environ.get("PEARL_NOBNOISE") == "1":
    threading.Thread(target=_patch_no_weight_noise, daemon=True).start()
    print("NOBNOISE: enabled (cost probe, output invalid by design)", flush=True)

def _patch_fast_path():
    """Replace pearl_gemm_noisy with fast_noisy.py's version (round-2 host work removal).
    Supersedes PEARL_HOSTFIX / PEARL_BHASH and needs no source edit; do not combine them."""
    mod = None
    for _ in range(9000):
        cand = sys.modules.get("vllm_miner.gemm_operators")
        if cand is not None and hasattr(cand, "pearl_gemm_noisy") and hasattr(cand, "noisy_gemm"):
            mod = cand
            break
        time.sleep(0.1)
    if mod is None:
        print("FAST: gemm_operators never imported, no patch applied", flush=True)
        return
    import fast_noisy

    fast_noisy.install(mod, reuse_buffers=os.environ.get("PEARL_FAST_REUSE", "1") == "1")
    # callers may have bound the name at import time
    for name, m in list(sys.modules.items()):
        if name.startswith("vllm_miner") and m is not mod and getattr(m, "pearl_gemm_noisy", None) is not None:
            m.pearl_gemm_noisy = mod.pearl_gemm_noisy
            print(f"FAST: rebound in {name}", flush=True)
    print("FAST: applied (per-block host values, reused work buffers, no int8 copy)", flush=True)


if os.environ.get("PEARL_FAST") == "1":
    threading.Thread(target=_patch_fast_path, daemon=True).start()
    print("FAST: enabled, waiting for vllm_miner.gemm_operators", flush=True)

_cap_gpu_memory()

runpy.run_path(str(BENCH / "serving_impact.py"), run_name="__main__")
