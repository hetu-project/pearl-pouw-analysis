#!/usr/bin/env python3
"""
P02 benchmark, part 1: which GEMMs of Pearl's published checkpoints actually mine.

Everything in this script is determined by code and published configuration, not estimated:
  * the trigger rule is Pearl's own Config.should_use_noisy_gemm, loaded from
    $PEARL_DIR/miner/vllm-miner/src/vllm_miner/config.py together with its config.yaml;
  * m = tokens in this forward step, n = output dim, k = input dim (vllm_kernels.py:203-206);
  * a layer can mine only if its checkpoint scheme is 7-bit weights + 7-bit per-token dynamic
    activations (PearlConfig._is_mining_layer, vllm_config.py:67-84), read from the checkpoint's
    config.json on Hugging Face (pinned revisions, see run.sh);
  * vLLM's LlamaForCausalLM fuses q/k/v into qkv_proj and gate/up into gate_up_proj; tensor
    parallelism splits n of column-parallel layers (qkv, gate_up) and k of row-parallel layers
    (o, down) by the TP degree. lm_head is in the checkpoint's ignore list; attention (QK^T, PV)
    runs in vLLM's attention backend, not through the plugin. Neither can mine.

Usage: p02_trigger_table.py <pearl_dir> <checkpoint_config_dir>
"""
from __future__ import annotations

import importlib.util
import json
import re
import sys
import types
from pathlib import Path

M_VALUES = [1, 8, 64, 256, 512, 1023, 1024, 2048, 8192]
TP_VALUES = [1, 2, 4, 8]
CHECKPOINTS = ["Llama-3.1-8B-Instruct-pearl", "Llama-3.3-70B-Instruct-pearl"]


def load_pearl_trigger(pearl_dir: Path):
    """Load Pearl's Config.should_use_noisy_gemm without installing the miner package.

    config.py imports miner_base.settings.MinerSettings only to read runtime flags
    (no_mining, ...), which the trigger rule does not use; a stub stands in for it.
    """
    src = pearl_dir / "miner/vllm-miner/src/vllm_miner/config.py"
    miner_base = types.ModuleType("miner_base")
    settings = types.ModuleType("miner_base.settings")
    settings.MinerSettings = type("MinerSettings", (), {})
    sys.modules["miner_base"] = miner_base
    sys.modules["miner_base.settings"] = settings
    spec = importlib.util.spec_from_file_location("pearl_vllm_miner_config", src)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    thresholds = mod.config.matrix_multiplication_config["use_simplified_gemm"]
    return mod.config.should_use_noisy_gemm, thresholds, src


def is_mining_scheme(group: dict) -> bool:
    """Same conditions as PearlConfig._is_mining_layer (vllm_config.py:67-84)."""
    w, a = group.get("weights"), group.get("input_activations")
    if w is None or a is None:
        return False
    return (
        w["num_bits"] == a["num_bits"] == 7
        and w["strategy"] in ("tensor", "channel")
        and a["strategy"] == "token"
        and w["symmetric"]
        and a["dynamic"]
        and not w["dynamic"]
    )


def scheme_of(module: str, qcfg: dict) -> str:
    if module in qcfg.get("ignore", []):
        return "bf16"
    for group in qcfg["config_groups"].values():
        for t in group["targets"]:
            pat = t[3:] if t.startswith("re:") else re.escape(t) + "$"
            if re.match(pat, module):
                return "int7" if is_mining_scheme(group) else f"{group['weights']['type']}{group['weights']['num_bits']}"
    return "bf16"


def model_linears(cfg: dict):
    """Per transformer layer: [(fused name, n, k, parallel kind, [checkpoint modules])]."""
    H, I = cfg["hidden_size"], cfg["intermediate_size"]
    nh, nkv = cfg["num_attention_heads"], cfg["num_key_value_heads"]
    hd = cfg.get("head_dim") or H // nh
    return [
        ("qkv_proj", (nh + 2 * nkv) * hd, H, "col", ["self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj"]),
        ("o_proj", H, nh * hd, "row", ["self_attn.o_proj"]),
        ("gate_up_proj", 2 * I, H, "col", ["mlp.gate_proj", "mlp.up_proj"]),
        ("down_proj", H, I, "row", ["mlp.down_proj"]),
    ]


def layer_schemes(cfg: dict):
    """{fused name: [scheme per layer index]}; fused parts must share one scheme."""
    qcfg = cfg["quantization_config"]
    out = {}
    for name, _n, _k, _p, parts in model_linears(cfg):
        per_layer = []
        for i in range(cfg["num_hidden_layers"]):
            s = {scheme_of(f"model.layers.{i}.{p}", qcfg) for p in parts}
            assert len(s) == 1, f"layer {i} {name}: parts use different schemes {s}"
            per_layer.append(s.pop())
        out[name] = per_layer
    return out


def ranges(idx):
    """[0,1,2,5,6] -> '0-2,5-6'."""
    if not idx:
        return "-"
    runs = [[idx[0], idx[0]]]
    for i in idx[1:]:
        if i == runs[-1][1] + 1:
            runs[-1][1] = i
        else:
            runs.append([i, i])
    return ",".join(f"{a}-{b}" if a != b else f"{a}" for a, b in runs)


def main():
    pearl_dir, ckpt_dir = Path(sys.argv[1]), Path(sys.argv[2])
    trigger, thr, src = load_pearl_trigger(pearl_dir)
    print(f"Trigger rule: Pearl {src.relative_to(pearl_dir)} -> should_use_noisy_gemm(m, n, k)")
    print(f"  thresholds from config.yaml: min_m={thr['min_m']} min_n={thr['min_n']} min_k={thr['min_k']}")
    print("  m = tokens scheduled in this forward step (prefill chunk tokens + one per decoding sequence)\n")

    for ck in CHECKPOINTS:
        cfg = json.loads((ckpt_dir / f"{ck}.config.json").read_text())
        L = cfg["num_hidden_layers"]
        schemes = layer_schemes(cfg)
        lins = model_linears(cfg)
        print(f"=== {ck}: {L} layers, hidden {cfg['hidden_size']}, intermediate {cfg['intermediate_size']}")
        print(f"    lm_head scheme: {scheme_of('lm_head', cfg['quantization_config'])} (never mines); attention QK^T/PV: not a plugin GEMM (never mines)\n")

        print("  Table 1. Linear layers (TP=1): checkpoint scheme and whether Pearl's trigger fires at m")
        hdr = f"  {'layer':13s} {'n':>6s} {'k':>6s}  {'int7 (mineable) layers':24s} {'other layers':16s} " + " ".join(f"{'m=' + str(m):>7s}" for m in M_VALUES)
        print(hdr)
        for name, n, k, _p, _parts in lins:
            int7 = [i for i, s in enumerate(schemes[name]) if s == "int7"]
            other = [i for i, s in enumerate(schemes[name]) if s != "int7"]
            other_s = f"{ranges(other)} ({schemes[name][other[0]]})" if other else "-"
            cells = []
            for m in M_VALUES:
                fires = trigger(m, n, k)
                cells.append(("mine" if int7 else "(n/a)") if fires else "no")
            print(f"  {name:13s} {n:6d} {k:6d}  {ranges(int7):24s} {other_s:16s} " + " ".join(f"{c:>7s}" for c in cells))
        print("  ('mine': trigger fires and the layer is int7; '(n/a)': trigger would fire but the layer is not int7)\n")

        def mined_share(m, tp):
            mined = total = 0
            for name, n, k, par, _parts in lins:
                n_l, k_l = (n // tp, k) if par == "col" else (n, k // tp)
                for i in range(L):
                    total += n * k
                    if schemes[name][i] == "int7" and trigger(m, n_l, k_l):
                        mined += n * k
            return mined / total

        print("  Table 2. Share of transformer linear-layer MACs that go through noisy GEMM, by m and TP degree")
        print("  " + f"{'TP':>4s} " + " ".join(f"{'m=' + str(m):>7s}" for m in M_VALUES))
        for tp in TP_VALUES:
            print("  " + f"{tp:4d} " + " ".join(f"{mined_share(m, tp):7.1%}" for m in M_VALUES))
        print("  (lm_head and attention are excluded from the denominator; including them lowers every value)\n")


if __name__ == "__main__":
    main()
