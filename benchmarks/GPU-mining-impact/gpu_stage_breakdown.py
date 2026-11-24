#!/usr/bin/env python3
"""
Where does the GPU-side cost of one mined GEMM go?

Each stage is run in a tight loop and timed to completion (synchronised), with all host-side
inputs prepared once, so the numbers are GPU work plus bare launch cost:
  plain            ordinary quantised GEMM (what inference needs anyway)
  hash_A, hash_B   Merkle roots of activations / weights
  commit           commitment hashes from the two roots
  noise_gen        low-rank noise factors
  noisy_full       Pearl's noisy GEMM kernel sequence, everything on
  noisy_-X         the same with one of Pearl's own switches turned off (cost probes; the
                   output of these is not valid mining, and with denoising off not valid inference)
Mock gateway, no submission, no network.
"""
from __future__ import annotations

import importlib.util
import json
import os
import sys
import time
from pathlib import Path


def main():
    pearl, outp = sys.argv[1], sys.argv[2]
    os.environ.update({"MINER_NO_MINING": "false", "MINER_NO_GATEWAY": "false",
                       "MINER_SKIP_BLOCK_SUBMISSION": "true"})
    p05 = Path("/workspace/pearl-pouw-analysis/benchmarks/P05-result-accuracy/gpu/p05_gpu_check.py")
    spec = importlib.util.spec_from_file_location("p05_gpu_check", p05)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    mod.start_mock_gateway(Path(pearl))

    import torch
    from miner_base.settings import MinerSettings
    from vllm_miner import gemm_operators as GO
    from vllm_miner.mining_state import (get_async_manager, get_pinned_pool,
                                         init_async_manager, init_pinned_pool)
    from vllm_miner.quantization_operators import quant_7bit

    settings = MinerSettings()
    init_async_manager(settings)
    init_pinned_pool(get_async_manager()._conf.pinned_pool_size)
    s = GO.config.settings
    r = s.noise_rank
    sync_size = GO.get_host_signal_sync_size()

    def timed(fn, iters=200):
        for _ in range(20):
            fn()
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(iters):
            fn()
        torch.cuda.synchronize()
        return round((time.perf_counter() - t0) * 1000 / iters, 4)

    shapes = [("o_proj", 4096, 4096), ("qkv_proj", 6144, 4096),
              ("gate_up_proj", 28672, 4096), ("down_proj", 4096, 14336)]
    rows = []
    torch.manual_seed(0)
    for name, n, k in shapes:
        w_q, w_s, _ = quant_7bit(torch.randn(n, k, dtype=torch.bfloat16, device="cuda"))
        w_q, w_s = w_q.contiguous(), w_s.squeeze(-1)
        for m in (24, 512):
            x_q, x_s, _ = quant_7bit(torch.randn(m, k, dtype=torch.bfloat16, device="cuda"))
            x_q, x_s = x_q.contiguous(), x_s.squeeze(-1)
            dev = x_q.device
            mc = GO.GPUMatmulConfigFactory.create(k=k, noise_rank=r)
            job = get_async_manager().get_mining_job()
            key = bytes(GO.CommitmentHasher.get_key(job.incomplete_header_bytes, mc.mining_config))
            key_t = torch.frombuffer(bytearray(key), dtype=torch.uint8).to("cuda")
            target = job.adjust_target(mining_config=mc.mining_config)
            pow_t = torch.tensor([(target >> (32 * i)) & 0xFFFFFFFF for i in range(8)],
                                 dtype=torch.uint32, device="cuda")
            scratch = torch.empty(GO.get_required_scratchpad_bytes(max(m * k, n * k)), dtype=torch.uint8, device=dev)
            ha, hb, ca, cb = (torch.empty(32, device="cuda", dtype=torch.uint8) for _ in range(4))
            salted = (m, n) if job.cert_version.uses_salted_seeds else None
            xa, wb = x_q.view(torch.uint8), w_q.view(torch.uint8)
            GO.tensor_hash(xa, key_t, ha, scratch)
            GO.tensor_hash(wb, key_t, hb, scratch)
            GO.commitment_hash_from_merkle_roots(ha, hb, key_t, ca, cb, salted_dims=salted)
            E = dict(
                EAL=torch.empty((m, r), dtype=torch.int8, device=dev),
                EBR=torch.empty((n, r), dtype=torch.int8, device=dev),
                EAR_R_major=torch.empty((k, r), dtype=torch.int8, device=dev),
                EBL_R_major=torch.empty((k, r), dtype=torch.int8, device=dev),
                EAR_K_major=torch.empty((r, k), dtype=torch.int8, device=dev),
                EBL_K_major=torch.empty((r, k), dtype=torch.int8, device=dev),
                EAL_fp16=torch.empty((m, r), dtype=torch.float16, device=dev),
                EBR_fp16=torch.empty((n, r), dtype=torch.float16, device=dev))
            GO.noise_gen(R=r, key_A=ca, key_B=cb, **E)
            BpEB = torch.empty((n, k), dtype=torch.int8, device=dev)
            EARxBpEB = torch.empty((n, r), dtype=torch.float16, device=dev)
            ApEA = torch.empty((m, k), dtype=torch.int8, device=dev)
            A_E_BL = torch.empty((m, r), dtype=torch.float16, device=dev)
            C = torch.empty((m, n), dtype=torch.bfloat16, device=dev)
            sync = torch.zeros((sync_size,), dtype=torch.int8, device="cuda")
            pinned = get_pinned_pool().acquire()

            def noisy(**sw):
                flags = dict(run_noising_A=True, run_noising_B=True, skip_reduction=False, skip_denoising=False)
                flags.update(sw)
                GO.noisy_gemm(A=x_q, B=w_q, AxEBL_fp16=A_E_BL, EARxBpEB_fp16=EARxBpEB, ApEA=ApEA, BpEB=BpEB,
                              A_scales=x_s, B_scales=w_s, C=C, host_signal_header_pinned=pinned,
                              host_signal_sync=sync, pow_target=pow_t, pow_key=ca.view(torch.uint32),
                              tile_size_m=s.tile_size_m, tile_size_n=s.tile_size_n, tile_size_k=s.tile_size_k,
                              **E, **flags)

            noisy()  # make sure BpEB etc. are populated before switch-off probes
            row = {"layer": name, "m": m, "n": n, "k": k,
                   "plain": timed(lambda: GO.pearl_gemm_vanilla(x_q, w_q, scale_a=x_s, scale_b=w_s,
                                                                out_dtype=torch.bfloat16)),
                   "hash_A": timed(lambda: GO.tensor_hash(xa, key_t, ha, scratch)),
                   "hash_B": timed(lambda: GO.tensor_hash(wb, key_t, hb, scratch)),
                   "commit": timed(lambda: GO.commitment_hash_from_merkle_roots(ha, hb, key_t, ca, cb,
                                                                                salted_dims=salted)),
                   "noise_gen": timed(lambda: GO.noise_gen(R=r, key_A=ca, key_B=cb, **E)),
                   "noisy_full": timed(noisy),
                   "noisy_-noiseB": timed(lambda: noisy(run_noising_B=False)),
                   "noisy_-noiseA": timed(lambda: noisy(run_noising_A=False)),
                   "noisy_-reduction": timed(lambda: noisy(skip_reduction=True)),
                   "noisy_-denoise": timed(lambda: noisy(skip_denoising=True))}
            get_pinned_pool().release(pinned)
            rows.append(row)
            print(json.dumps(row), flush=True)
    Path(outp).write_text(json.dumps(rows, indent=1))


if __name__ == "__main__":
    main()
    os._exit(0)
