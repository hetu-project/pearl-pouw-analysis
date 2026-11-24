#!/usr/bin/env python3
"""
Which step of Pearl's mining GEMM cannot be recorded into a CUDA graph?

Captures each step on its own inside torch.cuda.graph and reports whether capture
succeeds, then captures the full fast path (with and without the win-check callback)
and compares replayed output with eager output.

Each capture attempt runs in a fresh subprocess, because a failed capture poisons
the CUDA context. Mock gateway, block submission skipped, no network.
"""
from __future__ import annotations

import importlib.util
import json
import os
import subprocess
import sys
import time
from pathlib import Path

STEPS = ["vanilla", "hash_A", "hash_B", "commit", "noise_gen", "noisy_gemm",
         "event_record", "full_nosubmit", "full_submit"]


def child(step: str, pearl: str):
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

    sys.path.insert(0, str(Path(__file__).parent))
    import fast_noisy

    settings = MinerSettings()
    assert not settings.no_mining and settings.skip_block_submission
    init_async_manager(settings)
    init_pinned_pool(get_async_manager()._conf.pinned_pool_size)

    m, n, k, r = 512, 4096, 4096, GO.config.settings.noise_rank
    torch.manual_seed(0)
    w_q, w_s, _ = quant_7bit(torch.randn(n, k, dtype=torch.bfloat16, device="cuda"))
    x_q, x_s, _ = quant_7bit(torch.randn(m, k, dtype=torch.bfloat16, device="cuda"))
    w_q, w_s, x_q, x_s = w_q.contiguous(), w_s.squeeze(-1), x_q.contiguous(), x_s.squeeze(-1)
    kw = dict(scale_a=x_s, scale_b=w_s, out_dtype=torch.bfloat16)
    fast = fast_noisy.build(GO)

    # everything the single-step captures need, prepared eagerly
    mc = GO.GPUMatmulConfigFactory.create(k=k, noise_rank=r)
    job = get_async_manager().get_mining_job()
    key = bytes(GO.CommitmentHasher.get_key(job.incomplete_header_bytes, mc.mining_config))
    key_t = torch.frombuffer(bytearray(key), dtype=torch.uint8).to("cuda")
    scratch = torch.empty(GO.get_required_scratchpad_bytes(max(m * k, n * k)), dtype=torch.uint8, device="cuda")
    ha = torch.empty(32, device="cuda", dtype=torch.uint8)
    hb = torch.empty(32, device="cuda", dtype=torch.uint8)
    ca = torch.empty(32, device="cuda", dtype=torch.uint8)
    cb = torch.empty(32, device="cuda", dtype=torch.uint8)
    GO.tensor_hash(x_q.view(torch.uint8), key_t, ha, scratch)
    GO.tensor_hash(w_q.view(torch.uint8), key_t, hb, scratch)
    salted = (m, n) if job.cert_version.uses_salted_seeds else None
    GO.commitment_hash_from_merkle_roots(ha, hb, key_t, ca, cb, salted_dims=salted)
    ref = fast(x_q, w_q, submit_block=False, **kw).clone()
    torch.cuda.synchronize()

    out = {}

    def body():
        if step == "vanilla":
            out["C"] = GO.pearl_gemm_vanilla(x_q, w_q, **kw)
        elif step == "hash_A":
            GO.tensor_hash(x_q.view(torch.uint8), key_t, ha, scratch)
        elif step == "hash_B":
            GO.tensor_hash(w_q.view(torch.uint8), key_t, hb, scratch)
        elif step == "commit":
            GO.commitment_hash_from_merkle_roots(ha, hb, key_t, ca, cb, salted_dims=salted)
        elif step == "noise_gen":
            GO.generate_noise_factors(m, n, k, r, ca, cb, x_q.device)
        elif step == "noisy_gemm":
            out["C"] = fast(x_q, w_q, submit_block=False, **kw)   # includes the steps above
        elif step == "event_record":
            ev = torch.cuda.Event()
            ev.record()
        elif step == "full_nosubmit":
            out["C"] = fast(x_q, w_q, submit_block=False, **kw)
        elif step == "full_submit":
            out["C"] = fast(x_q, w_q, submit_block=True, **kw)

    res = {"step": step}
    try:
        g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g):
            body()
        torch.cuda.synchronize()
        res["capture"] = "ok"
        if "C" in out:
            # replay on new input written into the same memory
            x2, _, _ = quant_7bit(torch.randn(m, k, dtype=torch.bfloat16, device="cuda"))
            x_q.copy_(x2)
            g.replay()
            torch.cuda.synchronize()
            got = out["C"].clone()
            want = GO.pearl_gemm_vanilla(x_q, w_q, **kw)
            res["replay_matches_plain_gemm_on_new_input"] = bool(torch.equal(got, want))
            t0 = time.perf_counter()
            for _ in range(300):
                g.replay()
            torch.cuda.synchronize()
            res["replay_ms"] = round((time.perf_counter() - t0) * 1000 / 300, 4)
    except Exception as e:  # noqa: BLE001
        res["capture"] = "FAILED"
        res["error"] = f"{type(e).__name__}: {str(e).splitlines()[0][:200]}"
    print("PROBE_RESULT " + json.dumps(res), flush=True)
    os._exit(0)


def main():
    pearl, outp = sys.argv[1], sys.argv[2]
    rows = []
    for s in STEPS:
        p = subprocess.run([sys.executable, __file__, "--child", s, pearl],
                           capture_output=True, text=True, timeout=300)
        line = [l for l in p.stdout.splitlines() if l.startswith("PROBE_RESULT ")]
        row = json.loads(line[-1][13:]) if line else {"step": s, "capture": "CRASHED",
                                                       "error": (p.stderr.strip().splitlines() or ["?"])[-1][:200]}
        rows.append(row)
        print(row, flush=True)
    Path(outp).write_text(json.dumps(rows, indent=1))


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--child":
        child(sys.argv[2], sys.argv[3])
    else:
        main()
