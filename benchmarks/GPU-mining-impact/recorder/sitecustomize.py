"""
Counts Pearl's 7-bit layer calls inside vLLM's engine process, for online_impact.py.

Put this directory on PYTHONPATH when starting `vllm serve`. In every Python process a daemon
thread waits for vllm_miner.vllm_kernels to be imported (only the engine process imports it),
then wraps PearlKernel._apply_weights_mining to accumulate (m, n, k, mined) totals, written once
a second to $PEARL_RECORD_DIR/<pid>.json. Totals are cumulative; the harness takes differences.
"""
import json
import os
import sys
import threading
import time

_DIR = os.environ.get("PEARL_RECORD_DIR")


def _install():
    for _ in range(6000):  # give up after ten minutes
        mod = sys.modules.get("vllm_miner.vllm_kernels")
        if mod is not None and hasattr(mod, "PearlKernel"):
            break
        time.sleep(0.1)
    else:
        return
    from vllm_miner.config import config

    totals = {"calls": 0, "mined_calls": 0, "macs": 0, "mined_macs": 0, "square_m": 0, "square_calls": 0}
    original = mod.PearlKernel._apply_weights_mining

    def recorded(self, layer, x, w_q, *rest, **kw):
        m, k, n = x.reshape(-1, x.shape[-1]).shape[0], x.shape[-1], w_q.shape[0]
        mined = config.should_use_noisy_gemm(m, n, k) and not config.settings.no_mining
        totals["calls"] += 1
        totals["macs"] += m * n * k
        if mined:
            totals["mined_calls"] += 1
            totals["mined_macs"] += m * n * k
        if n == k:  # o_proj: one call per layer per step, m = tokens in the step
            totals["square_calls"] += 1
            totals["square_m"] += m
        return original(self, layer, x, w_q, *rest, **kw)

    mod.PearlKernel._apply_weights_mining = recorded

    def patch_moe():
        # Mixture-of-experts models: the expert gate/up GEMM goes through PearlMoEExperts.apply,
        # which mines when the whole step has enough tokens (not per expert).
        moe = sys.modules.get("vllm_miner.pearl_moe_experts")
        if moe is None or getattr(moe, "_recorded", False):
            return
        moe._recorded = True
        original_apply = moe.PearlMoEExperts.apply

        def recorded_apply(self, output, hidden_states, w1, w2, topk_weights, topk_ids, *rest, **kw):
            _, tokens, n, k, topk = self.moe_problem_size(hidden_states, w1, w2, topk_ids)
            mined = config.should_use_noisy_gemm(tokens, n, k) and not config.settings.no_mining
            macs = tokens * topk * n * k
            totals["calls"] += 1
            totals["macs"] += macs
            if mined:
                totals["mined_calls"] += 1
                totals["mined_macs"] += macs
            return original_apply(self, output, hidden_states, w1, w2, topk_weights, topk_ids, *rest, **kw)

        moe.PearlMoEExperts.apply = recorded_apply
    path = os.path.join(_DIR, f"{os.getpid()}.json")
    while True:
        patch_moe()
        tmp = path + ".tmp"
        with open(tmp, "w") as f:
            json.dump(totals, f)
        os.replace(tmp, path)
        time.sleep(1)


if _DIR:
    threading.Thread(target=_install, daemon=True).start()
