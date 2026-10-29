#!/usr/bin/env python3
"""
P04 benchmark, part 2: share of the official plugin's mining work that can never yield a valid
block, per published checkpoint and tensor-parallel degree.

Everything is determined by code and published configuration, not estimated:
  * plugin rule: Pearl's own Config.should_use_noisy_gemm with its config.yaml thresholds
    (miner/vllm-miner/src/vllm_miner/config.py), applied with m = tokens in the step (m >= 1024
    here; the result does not depend on m once the step mines);
  * verifier rule: k >= C * r, with C read from zk-pow/src/api/sanity_checks.rs and r from
    MinerSettings.noise_rank (miner/miner-base/src/miner_base/settings.py);
  * dense-only rule: after mainnet height 91,630 the node rejects every MoE certificate
    (node/blockchain/validate.go), while the plugin still mines MoE expert GEMMs
    (miner/vllm-miner/src/vllm_miner/pearl_moe_experts.py);
  * which layers are 7-bit (mineable): each checkpoint's quantization_config, same conditions as
    PearlConfig._is_mining_layer (vllm_config.py);
  * vLLM sharding: qkv / gate_up are column-parallel (n / TP), o_proj / down are row-parallel
    (k / TP); MoE experts without expert parallelism are split on n.
A mined GEMM's chance to win is proportional to its multiply-adds (the jackpot target scales with
tile size x dot length), so the share of wasted wins equals the share of mined multiply-adds that
sit in GEMMs the verifier rejects.

Usage: p04_wasted_mining.py <pearl_dir> <checkpoint_config_dir>
"""
from __future__ import annotations

import ast
import importlib.util
import json
import re
import sys
import types
from pathlib import Path

TP_VALUES = [1, 2, 4, 8]
M_STEP = 8192  # any m >= min_m gives the same result


def load_pearl_trigger(pearl_dir: Path):
    """Pearl's Config.should_use_noisy_gemm; MinerSettings is stubbed (the rule does not use it)."""
    miner_base = types.ModuleType("miner_base")
    settings = types.ModuleType("miner_base.settings")
    settings.MinerSettings = type("MinerSettings", (), {})
    sys.modules["miner_base"] = miner_base
    sys.modules["miner_base.settings"] = settings
    src = pearl_dir / "miner/vllm-miner/src/vllm_miner/config.py"
    spec = importlib.util.spec_from_file_location("pearl_vllm_miner_config", src)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.config.should_use_noisy_gemm, mod.config.matrix_multiplication_config["use_simplified_gemm"]


def load_verifier_rule(pearl_dir: Path):
    src = (pearl_dir / "zk-pow/src/api/sanity_checks.rs").read_text()
    factor = int(re.search(r"ensure!\(k >= (\d+) \* r,", src).group(1))
    settings = (pearl_dir / "miner/miner-base/src/miner_base/settings.py").read_text()
    cls = next(n for n in ast.parse(settings).body if isinstance(n, ast.ClassDef) and n.name == "MinerSettings")
    rank = next(ast.literal_eval(n.value) for n in cls.body
                if isinstance(n, ast.AnnAssign) and getattr(n.target, "id", "") == "noise_rank")
    return factor, rank


def is_mining_scheme(group: dict) -> bool:
    """Same conditions as PearlConfig._is_mining_layer (vllm_config.py:67-84)."""
    w, a = group.get("weights"), group.get("input_activations")
    if w is None or a is None:
        return False
    return (w["num_bits"] == a["num_bits"] == 7 and w["strategy"] in ("tensor", "channel")
            and a["strategy"] == "token" and w["symmetric"] and a["dynamic"] and not w["dynamic"])


def is_int7(module: str, qcfg: dict) -> bool:
    """Regex targets win over the class-name target "Linear" (compressed-tensors precedence)."""
    if module in qcfg.get("ignore", []):
        return False
    groups = qcfg["config_groups"].values()
    for group in groups:
        for t in group["targets"]:
            if t.startswith("re:") and re.match(t[3:], module):
                return is_mining_scheme(group)
    for group in groups:
        if "Linear" in group["targets"]:
            return is_mining_scheme(group)
    return False


def linears(name: str, cfg: dict):
    """Yield (label, n, k, parallel, MoE?, int7?, multiply-adds per token) for every layer."""
    t = cfg.get("text_config", cfg)
    q = cfg.get("quantization_config") or t["quantization_config"]
    H, nh, L = t["hidden_size"], t["num_attention_heads"], t["num_hidden_layers"]
    for i in range(L):
        p = f"model.layers.{i}."
        if name.startswith("Llama"):
            I, nkv = t["intermediate_size"], t["num_key_value_heads"]
            hd = t.get("head_dim") or H // nh
            yield "qkv_proj", (nh + 2 * nkv) * hd, H, "col", False, is_int7(p + "self_attn.q_proj", q), (nh + 2 * nkv) * hd * H
            yield "o_proj", H, nh * hd, "row", False, is_int7(p + "self_attn.o_proj", q), H * nh * hd
            yield "gate_up_proj", 2 * I, H, "col", False, is_int7(p + "mlp.gate_proj", q), 2 * I * H
            yield "down_proj", H, I, "row", False, is_int7(p + "mlp.down_proj", q), H * I
        elif name.startswith("Gemma"):
            I = t["intermediate_size"]
            full = t["layer_types"][i] == "full_attention"
            hd = t["global_head_dim"] if full else t["head_dim"]
            nkv = t["num_global_key_value_heads"] if full else t["num_key_value_heads"]
            kind = "full" if full else "sliding"
            yield f"qkv_proj[{kind}]", (nh + 2 * nkv) * hd, H, "col", False, is_int7(p + "self_attn.q_proj", q), (nh + 2 * nkv) * hd * H
            yield f"o_proj[{kind}]", H, nh * hd, "row", False, is_int7(p + "self_attn.o_proj", q), H * nh * hd
            yield "gate_up_proj", 2 * I, H, "col", False, is_int7(p + "mlp.gate_proj", q), 2 * I * H
            yield "down_proj", H, I, "row", False, is_int7(p + "mlp.down_proj", q), H * I
        else:  # Qwen3 MoE
            Im, nkv, hd, topk = t["moe_intermediate_size"], t["num_key_value_heads"], t["head_dim"], t["num_experts_per_tok"]
            yield "qkv_proj", (nh + 2 * nkv) * hd, H, "col", False, is_int7(p + "self_attn.q_proj", q), (nh + 2 * nkv) * hd * H
            yield "o_proj", H, nh * hd, "row", False, is_int7(p + "self_attn.o_proj", q), H * nh * hd
            yield f"experts.gate_up (x{topk} active)", 2 * Im, H, "col", True, is_int7(p + "mlp.experts.0.gate_proj", q), topk * 2 * Im * H
            yield f"experts.down (x{topk} active)", H, Im, "col", True, is_int7(p + "mlp.experts.0.down_proj", q), topk * Im * H


def main():
    pearl_dir, ckpt_dir = Path(sys.argv[1]), Path(sys.argv[2])
    trigger, thr = load_pearl_trigger(pearl_dir)
    factor, rank = load_verifier_rule(pearl_dir)
    k_min = factor * rank
    print(f"Plugin rule (config.yaml): mine iff m >= {thr['min_m']}, n >= {thr['min_n']}, k >= {thr['min_k']}")
    print(f"Verifier rule (sanity_checks.rs): k >= {factor}r = {k_min} at r = {rank} (MinerSettings.noise_rank)")
    print("Dense-only rule (validate.go, mainnet height 91,630): every MoE certificate is rejected\n")

    checkpoints = ["Llama-3.1-8B-Instruct-pearl", "Llama-3.3-70B-Instruct-pearl",
                   "Gemma-4-31B-it-pearl", "Qwen3-30B-A3B-Instruct-2507-pearl"]
    summary = []
    for ck in checkpoints:
        cfg = json.loads((ckpt_dir / f"{ck}.config.json").read_text())
        rows = list(linears(ck, cfg))
        print(f"=== {ck}")
        for tp in TP_VALUES:
            mined = bad_k = bad_moe = 0
            seen = {}
            for label, n, k, par, moe, int7, macs in rows:
                n_l, k_l = (n // tp, k) if par == "col" else (n, k // tp)
                mines = int7 and trigger(M_STEP, n_l, k_l)
                if mines:
                    mined += macs
                    if k_l < k_min:
                        bad_k += macs
                    elif moe:
                        bad_moe += macs
                if not int7:
                    status = "not 7-bit, not mined"
                elif not mines:
                    status = "below plugin threshold, not mined"
                elif k_l < k_min:
                    status = f"MINED, REJECTED (k={k_l} < {k_min})"
                elif moe:
                    status = "MINED, REJECTED (MoE, dense-only)"
                else:
                    status = "mined, valid"
                seen.setdefault((label, n_l, k_l, status), 0)
                seen[(label, n_l, k_l, status)] += 1
            print(f"  TP={tp}")
            for (label, n_l, k_l, status), cnt in seen.items():
                print(f"    {label:28s} x{cnt:3d}  n={n_l:6d} k={k_l:6d}  {status}")
            summary.append((ck, tp, bad_k / mined if mined else 0.0, (bad_k + bad_moe) / mined if mined else 0.0))
        print()

    print("Share of the plugin's mining work (= share of its expected wins) that can never become a valid block")
    print(f"  {'checkpoint':36s} {'TP':>3s}  {'k < ' + str(k_min):>10s}  {'+ MoE rejected':>15s}")
    for ck, tp, f1, f2 in summary:
        print(f"  {ck:36s} {tp:3d}  {f1:10.1%}  {f2:15.1%}")


if __name__ == "__main__":
    main()
