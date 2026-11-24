"""
sitecustomize for online_fast.py: optionally installs the fast mining path (PEARL_FAST=1)
inside vLLM's engine process, then chains to recorder/sitecustomize.py for the call counters.
Pearl's source tree is not modified.
"""
import os
import runpy
import sys
import threading
import time

_HERE = os.path.dirname(os.path.abspath(__file__))
_BENCH = os.path.dirname(_HERE)


def _install_fast():
    for _ in range(6000):
        mod = sys.modules.get("vllm_miner.gemm_operators")
        if mod is not None and hasattr(mod, "pearl_gemm_noisy") and hasattr(mod, "noisy_gemm"):
            break
        time.sleep(0.1)
    else:
        return
    sys.path.insert(0, _BENCH)
    import fast_noisy

    fast_noisy.install(mod)
    for name, m in list(sys.modules.items()):
        if name.startswith("vllm_miner") and m is not mod and getattr(m, "pearl_gemm_noisy", None) is not None:
            m.pearl_gemm_noisy = mod.pearl_gemm_noisy
    print("FAST: applied in pid", os.getpid(), flush=True)


if os.environ.get("PEARL_FAST") == "1":
    threading.Thread(target=_install_fast, daemon=True).start()

runpy.run_path(os.path.join(_BENCH, "recorder", "sitecustomize.py"))
