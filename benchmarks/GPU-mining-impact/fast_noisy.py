#!/usr/bin/env python3
"""
A drop-in replacement for vllm_miner.gemm_operators.pearl_gemm_noisy that does the same
mining with less per-call host work, plus a standalone profiler / A-B harness for it.

Nothing fed to the CUDA kernels changes: same matrices, same keys, same noise, same PoW
target. Only *where and how often* host-side values are produced changes:

  per block (was: per GEMM)   matmul config, adjusted PoW target, hash key, key tensor,
                              PoW target tensor, weight Merkle root
  reused    (was: allocated)  scratchpad and noise work buffers, keyed by shape
  no copy   (was: copy)       int8 -> uint8 reinterpretation of A and B

Per-call and never cached: A's Merkle root, both commitment hashes, the noise itself,
the host-signal buffers and the win-check callback.

`install(mod)` swaps the function into an imported gemm_operators module.
Run as a script for the profiler:  python fast_noisy.py --pearl /workspace/pearl --out x.json
"""
from __future__ import annotations

import time
from collections import defaultdict

import torch

PROFILE = False
T: dict[str, float] = defaultdict(float)
N = {"calls": 0}


def _tick(name, t0):
    if PROFILE:
        t1 = time.perf_counter()
        T[name] += t1 - t0
        return t1
    return t0


def build(mod, reuse_buffers=True):
    """Returns fast_noisy bound to the given (imported) gemm_operators module."""
    config = mod.config
    get_async_manager = mod.get_async_manager
    get_pinned_pool = mod.get_pinned_pool
    tensor_hash = mod.tensor_hash
    commit = mod.commitment_hash_from_merkle_roots
    noise_gen = mod.noise_gen
    noisy_gemm = mod.noisy_gemm
    StatusCheckCallback = mod.StatusCheckCallback
    scratch_bytes = mod.get_required_scratchpad_bytes
    create_config = mod.GPUMatmulConfigFactory.create
    get_key = mod.CommitmentHasher.get_key
    sync_size = mod.get_host_signal_sync_size()

    cfg_cache: dict = {}        # (k, r) -> matmul_config
    job_cache: dict = {}        # k -> (job, adjusted_target, hash_key, key_tensor, pow_tensor)
    bhash_cache: dict = {}      # (ptr, shape, hash_key) -> 32-byte root on GPU
    bufs: dict = {}             # (tag, shape) -> tensor

    def buf(tag, shape, dtype, device):
        if not reuse_buffers:
            return torch.empty(shape, dtype=dtype, device=device)
        key = (tag, shape)
        t = bufs.get(key)
        if t is None:
            t = torch.empty(shape, dtype=dtype, device=device)
            if len(bufs) < 512:
                bufs[key] = t
        return t

    def fast_noisy(a, b, scale_a, scale_b, out_dtype, layer=None, submit_block=True):
        t0 = time.perf_counter() if PROFILE else 0.0
        N["calls"] += 1
        m, n, k = a.shape[0], b.shape[0], a.shape[1]
        s = config.settings
        r = s.noise_rank
        dev = a.device

        mc = cfg_cache.get((k, r))
        if mc is None:
            mc = cfg_cache[(k, r)] = create_config(k=k, noise_rank=r)
        t0 = _tick("config", t0)

        job = get_async_manager().get_mining_job()
        t0 = _tick("get_job", t0)

        ent = job_cache.get(k)
        if ent is None or ent[0] is not job:
            target = job.adjust_target(mining_config=mc.mining_config)
            hash_key = bytes(get_key(job.incomplete_header_bytes, mc.mining_config))
            words = [(target >> (32 * i)) & 0xFFFFFFFF for i in range(8)]
            # Persistent device tensors, refreshed in place from pinned staging memory:
            # the addresses never change, which is what a captured CUDA graph needs.
            if ent is None:
                key_tensor = torch.empty(32, dtype=torch.uint8, device="cuda")
                pow_t = torch.empty(8, dtype=torch.uint32, device="cuda")
            else:
                key_tensor, pow_t = ent[3], ent[4]
            # non-blocking: a blocking copy synchronises the stream, which a capture forbids.
            # The staging tensors are kept alive in the cache entry until the next block.
            stage_key = torch.frombuffer(bytearray(hash_key), dtype=torch.uint8).pin_memory()
            stage_pow = torch.tensor(words, dtype=torch.uint32).pin_memory()
            key_tensor.copy_(stage_key, non_blocking=True)
            pow_t.copy_(stage_pow, non_blocking=True)
            ent = job_cache[k] = (job, target, hash_key, key_tensor, pow_t, stage_key, stage_pow)
            if len(bhash_cache) > 8192:
                bhash_cache.clear()
        hash_key, key_tensor, pow_t = ent[2], ent[3], ent[4]
        t0 = _tick("per_block", t0)

        C = torch.empty((m, n), dtype=out_dtype, device=dev)
        scratch = buf("scratch", (scratch_bytes(max(m * k, n * k)),), torch.uint8, dev)
        t0 = _tick("alloc", t0)

        a_hash = torch.empty(32, device="cuda", dtype=torch.uint8)
        tensor_hash(a.view(torch.uint8), key_tensor, a_hash, scratch)
        t0 = _tick("hash_A", t0)

        bk = (b.data_ptr(), n, k, hash_key)
        b_hash = bhash_cache.get(bk)
        if b_hash is None:
            b_hash = torch.empty(32, device="cuda", dtype=torch.uint8)
            tensor_hash(b.view(torch.uint8), key_tensor, b_hash, scratch)
            bhash_cache[bk] = b_hash
        t0 = _tick("hash_B", t0)

        ch_a = torch.empty(32, device="cuda", dtype=torch.uint8)
        ch_b = torch.empty(32, device="cuda", dtype=torch.uint8)
        commit(a_hash, b_hash, key_tensor, ch_a, ch_b,
               salted_dims=(m, n) if job.cert_version.uses_salted_seeds else None)
        t0 = _tick("commit", t0)

        EAL = buf("EAL", (m, r), torch.int8, dev)
        EBR = buf("EBR", (n, r), torch.int8, dev)
        EAR_R = buf("EAR_R", (k, r), torch.int8, dev)
        EBL_R = buf("EBL_R", (k, r), torch.int8, dev)
        EAR_K = buf("EAR_K", (r, k), torch.int8, dev)
        EBL_K = buf("EBL_K", (r, k), torch.int8, dev)
        EAL16 = buf("EAL16", (m, r), torch.float16, dev)
        EBR16 = buf("EBR16", (n, r), torch.float16, dev)
        BpEB = buf("BpEB", (n, k), torch.int8, dev)
        EARxBpEB = buf("EARxBpEB", (n, r), torch.float16, dev)
        ApEA = buf("ApEA", (m, k), torch.int8, dev)
        A_E_BL = buf("A_E_BL", (m, r), torch.float16, dev)
        t0 = _tick("alloc", t0)

        noise_gen(R=r, EAL=EAL, EAL_fp16=EAL16, EAR_R_major=EAR_R, EAR_K_major=EAR_K,
                  EBL_R_major=EBL_R, EBL_K_major=EBL_K, EBR=EBR, EBR_fp16=EBR16,
                  key_A=ch_a, key_B=ch_b)
        t0 = _tick("noise_gen", t0)

        sync = torch.zeros((sync_size,), dtype=torch.int8, device="cuda")
        pinned = get_pinned_pool().acquire()
        t0 = _tick("signal_bufs", t0)

        noisy_gemm(
            A=a, B=b, EAL=EAL, EAL_fp16=EAL16, EBR=EBR, EBR_fp16=EBR16,
            EAR_R_major=EAR_R, EBL_R_major=EBL_R, EAR_K_major=EAR_K, EBL_K_major=EBL_K,
            AxEBL_fp16=A_E_BL, EARxBpEB_fp16=EARxBpEB, ApEA=ApEA, BpEB=BpEB,
            A_scales=scale_a, B_scales=scale_b, C=C,
            host_signal_header_pinned=pinned, host_signal_sync=sync,
            pow_target=pow_t, pow_key=ch_a.view(torch.uint32),
            tile_size_m=s.tile_size_m, tile_size_n=s.tile_size_n, tile_size_k=s.tile_size_k,
            run_noising_A=True, run_noising_B=True, skip_reduction=False, skip_denoising=False,
        )
        t0 = _tick("noisy_gemm", t0)

        if submit_block and not torch.cuda.is_current_stream_capturing():
            ev = torch.cuda.Event()
            ev.record()
            cb = StatusCheckCallback(
                host_signal_header_pinned=pinned, commitment_hash_A_tensor=ch_a,
                commitment_hash_B_tensor=ch_b, A=a, B=b, mining_job=job)
            get_async_manager().schedule_status_check(ev, cb)
        else:
            get_pinned_pool().release(pinned)
        _tick("callback", t0)
        return C

    return fast_noisy


def install(mod, reuse_buffers=True):
    mod.pearl_gemm_noisy = build(mod, reuse_buffers=reuse_buffers)


# ----------------------------------------------------------------------------- profiler
def _main():
    import argparse
    import importlib.util
    import json
    import os
    from pathlib import Path

    global PROFILE
    ap = argparse.ArgumentParser()
    ap.add_argument("--pearl", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--iters", type=int, default=300)
    ap.add_argument("--m", type=int, nargs="*", default=[16, 128, 512])
    args = ap.parse_args()

    os.environ.update({"MINER_NO_MINING": "false", "MINER_NO_GATEWAY": "false",
                       "MINER_SKIP_BLOCK_SUBMISSION": "true"})
    p05 = Path("/workspace/pearl-pouw-analysis/benchmarks/P05-result-accuracy/gpu/p05_gpu_check.py")
    spec = importlib.util.spec_from_file_location("p05_gpu_check", p05)
    p05m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(p05m)
    p05m.start_mock_gateway(Path(args.pearl))

    from miner_base.settings import MinerSettings
    from vllm_miner import gemm_operators as GO
    from vllm_miner.mining_state import get_async_manager, init_async_manager, init_pinned_pool
    from vllm_miner.quantization_operators import quant_7bit

    settings = MinerSettings()
    assert not settings.no_mining and settings.skip_block_submission
    init_async_manager(settings)
    init_pinned_pool(get_async_manager()._conf.pinned_pool_size)

    stock = GO.pearl_gemm_noisy
    fast = build(GO, reuse_buffers=True)
    fast_noreuse = build(GO, reuse_buffers=False)

    def wall(fn, iters):
        for _ in range(20):
            fn()
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(iters):
            fn()
        host = time.perf_counter() - t0          # time to *issue* the work
        torch.cuda.synchronize()
        total = time.perf_counter() - t0         # time until the GPU has finished it
        return host * 1000 / iters, total * 1000 / iters

    shapes = [("o_proj", 4096, 4096), ("qkv_proj", 6144, 4096),
              ("gate_up_proj", 28672, 4096), ("down_proj", 4096, 14336)]
    torch.manual_seed(0)
    rows = []
    for name, n, k in shapes:
        w_q, w_s, _ = quant_7bit(torch.randn(n, k, dtype=torch.bfloat16, device="cuda"))
        w_q, w_s = w_q.contiguous(), w_s.squeeze(-1)
        for m in args.m:
            x_q, x_s, _ = quant_7bit(torch.randn(m, k, dtype=torch.bfloat16, device="cuda"))
            x_q, x_s = x_q.contiguous(), x_s.squeeze(-1)
            kw = dict(scale_a=x_s, scale_b=w_s, out_dtype=torch.bfloat16)

            ref = stock(x_q, w_q, submit_block=False, **kw).clone()
            same = bool(torch.equal(ref, fast(x_q, w_q, submit_block=False, **kw))
                        and torch.equal(ref, fast(x_q, w_q, submit_block=False, **kw)))

            plain = wall(lambda: GO.pearl_gemm_vanilla(x_q, w_q, **kw), args.iters)
            res = {}
            for tag, fn in (("stock", stock), ("fast_noreuse", fast_noreuse), ("fast", fast)):
                res[tag] = wall(lambda: fn(x_q, w_q, submit_block=True, **kw), args.iters)
                torch.cuda.synchronize()
                time.sleep(0.2)          # let queued status checks drain

            PROFILE = True
            T.clear()
            for _ in range(args.iters):
                fast(x_q, w_q, submit_block=True, **kw)
            torch.cuda.synchronize()
            PROFILE = False
            segs = {kk: round(v * 1000 / args.iters, 4) for kk, v in sorted(T.items(), key=lambda z: -z[1])}

            row = {"layer": name, "m": m, "n": n, "k": k, "identical": same,
                   "plain_host_ms": round(plain[0], 4), "plain_total_ms": round(plain[1], 4),
                   **{f"{t}_host_ms": round(v[0], 4) for t, v in res.items()},
                   **{f"{t}_total_ms": round(v[1], 4) for t, v in res.items()},
                   "fast_segments_ms": segs}
            rows.append(row)
            print(f"{name:13s} m={m:4d} same={same} | host ms: plain {plain[0]:.3f} "
                  f"stock {res['stock'][0]:.3f} noreuse {res['fast_noreuse'][0]:.3f} "
                  f"fast {res['fast'][0]:.3f} | total ms: plain {plain[1]:.3f} "
                  f"stock {res['stock'][1]:.3f} fast {res['fast'][1]:.3f}", flush=True)
            print("      segments:", segs, flush=True)

    Path(args.out).write_text(json.dumps({"gpu": torch.cuda.get_device_name(0),
                                          "iters": args.iters, "rows": rows}, indent=1))
    print("all identical:", all(r["identical"] for r in rows))


if __name__ == "__main__":
    _main()
