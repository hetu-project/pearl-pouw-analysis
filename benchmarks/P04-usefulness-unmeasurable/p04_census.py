#!/usr/bin/env python3
"""
P04 benchmark, part 2: shape census. How many mainnet blocks could have come from Pearl's
official useful miner (the vLLM plugin), judged only by the public mining parameters?

The official useful miner can only produce certificates with all of:
  S1  rank r = 128                       miner_base/settings.py: noise_rank = 128
  S2  tile rows [0, 8], 64 fixed columns miner_base/settings.py: rows_pattern / cols_pattern
  S3  no MoE                             MoE blocks are rejected after the dense-only fork (91630)
  S4  m >= 1024                          vllm_miner/config.yaml: min_m = 1024 (see P02)
  S5  n and k are layer dimensions of a published Pearl model, possibly split by tensor
      parallelism TP in {1, 2, 4, 8}. Deliberately loose: every q/k/v/o/gate/up/down width,
      fused qkv / gate_up widths and hidden/intermediate sizes of all four checkpoints count,
      for both n and k. A tighter rule would only lower the bound.

A block that fails any condition was certainly not mined by the official useful miner. A block
that passes all of them is only "possibly useful": a pure miner can copy these parameters. So the
share of passing blocks is an UPPER BOUND on official useful mining.

Usage: p04_census.py <pearl_dir> <certs.jsonl> <checkpoint_config_dir>
"""
from __future__ import annotations

import ast
import json
import math
import sys
from collections import Counter
from pathlib import Path

CHECKPOINTS = ["Llama-3.1-8B-Instruct-pearl", "Llama-3.3-70B-Instruct-pearl",
               "Gemma-4-31B-it-pearl", "Qwen3-30B-A3B-Instruct-2507-pearl"]
TP_VALUES = (1, 2, 4, 8)


def official_settings(pearl_dir: Path):
    """Read noise_rank, rows_pattern and cols_pattern defaults from Pearl's MinerSettings."""
    src = (pearl_dir / "miner/miner-base/src/miner_base/settings.py").read_text()
    cls = next(n for n in ast.parse(src).body if isinstance(n, ast.ClassDef) and n.name == "MinerSettings")
    vals = {}
    for node in cls.body:
        if isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
            if node.target.id in ("noise_rank", "rows_pattern", "cols_pattern"):
                vals[node.target.id] = ast.literal_eval(node.value)
    return vals["noise_rank"], vals["rows_pattern"], vals["cols_pattern"]


def model_dims(cfg: dict) -> set[int]:
    t = cfg.get("text_config", cfg)
    H, nh = t["hidden_size"], t["num_attention_heads"]
    hd = t.get("head_dim") or H // nh
    dims = {H, nh * hd}
    for heads_kv, head_dim in ((t["num_key_value_heads"], hd),
                               (t.get("num_global_key_value_heads"), t.get("global_head_dim"))):
        if heads_kv and head_dim:
            dims |= {nh * head_dim, heads_kv * head_dim, (nh + 2 * heads_kv) * head_dim}
    for inter in (t.get("intermediate_size"), t.get("moe_intermediate_size")):
        if inter:
            dims |= {inter, 2 * inter}
    return dims


def wilson_upper(k: int, n: int, z: float = 1.96) -> float:
    p = k / n
    centre = p + z * z / (2 * n)
    margin = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return (centre + margin) / (1 + z * z / n)


def main():
    pearl_dir, certs_path, ckpt_dir = Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3])
    rank, rows_pat, cols_pat = official_settings(pearl_dir)
    dims = set()
    for ck in CHECKPOINTS:
        dims |= model_dims(json.loads((ckpt_dir / f"{ck}.config.json").read_text()))
    allowed = {d // tp for d in dims for tp in TP_VALUES if d % tp == 0}

    blocks = [json.loads(line) for line in certs_path.read_text().splitlines() if line.strip()]
    blocks.sort(key=lambda b: b["height"])
    N = len(blocks)
    print(f"Blocks analysed: {N} (heights {blocks[0]['height']}..{blocks[-1]['height']}), "
          f"certificate versions {dict(Counter(b['cert_version'] for b in blocks))}")
    print(f"Official useful miner (Pearl MinerSettings): r={rank}, rows={rows_pat}, "
          f"cols=<{len(cols_pat)} columns, {cols_pat[:4]}...{cols_pat[-2:]}>")
    print(f"Allowed n/k values (layer widths of {len(CHECKPOINTS)} checkpoints / TP {TP_VALUES}): "
          f"{len(allowed)} values, max {max(allowed)}\n")

    checks = [
        ("S1 r = 128", lambda b: b["r"] == rank),
        ("S2 official tile pattern (2 x 64)", lambda b: b["rows"] == rows_pat and b["cols"] == cols_pat),
        ("S3 no MoE", lambda b: b["moe_e"] == 0),
        ("S4 m >= 1024", lambda b: b["m"] >= 1024),
        ("S5 n and k are model layer widths", lambda b: b["n"] in allowed and b["k"] in allowed),
    ]
    print("Table 1. Blocks passing each condition (cumulative)")
    remaining = blocks
    for name, fn in checks:
        alone = sum(fn(b) for b in blocks)
        remaining = [b for b in remaining if fn(b)]
        print(f"  {name:38s} alone {alone:5d} ({alone / N:6.1%})   cumulative {len(remaining):5d} ({len(remaining) / N:6.1%})")
    k = len(remaining)
    print(f"\n  Possibly official useful mining: {k} / {N} = {k / N:.2%}  "
          f"(95% upper bound {wilson_upper(k, N):.2%})\n")

    def key(b):
        return (b["k"], b["r"], f"{b['h']}x{b['w']}", b["m"], b["n"])

    print("Table 2. Most common parameter sets (k, r, tile h x w, m, n)")
    for (kk, r, tile, m, n), c in Counter(key(b) for b in blocks).most_common(12):
        print(f"  k={kk:6d} r={r:4d} tile={tile:6s} m={m:6d} n={n:6d}   {c:5d} blocks ({c / N:6.1%})")
    print()

    print("Table 3. Distribution of single parameters")
    for field in ("k", "r"):
        print(f"  {field}: " + ", ".join(f"{v} ({c / N:.1%})" for v, c in Counter(b[field] for b in blocks).most_common(6)))
    tiles = Counter(f"{b['h']}x{b['w']}" for b in blocks)
    print("  tile h x w: " + ", ".join(f"{v} ({c / N:.1%})" for v, c in tiles.most_common(6)))
    print(f"  m: min {min(b['m'] for b in blocks)}, median {sorted(b['m'] for b in blocks)[N // 2]}, max {max(b['m'] for b in blocks)}")
    print(f"  n: min {min(b['n'] for b in blocks)}, median {sorted(b['n'] for b in blocks)[N // 2]}, max {max(b['n'] for b in blocks)}")
    print(f"  k equal to a model layer width: {sum(b['k'] in allowed for b in blocks) / N:.1%}; "
          f"n equal to a model layer width: {sum(b['n'] in allowed for b in blocks) / N:.1%}")
    print(f"  proof size: {dict(Counter(b['proof_len'] for b in blocks))}")


if __name__ == "__main__":
    main()
