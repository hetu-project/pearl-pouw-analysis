#!/usr/bin/env python3
"""
P03 benchmark, part 1: the code facts behind "useful mining earns fewer shares than pure mining".

Locates each fact in the Pearl source by pattern (not by hard-coded line number) and prints the
file, line and text, so a reader can check them against any checkout. Exits non-zero if a pattern
is missing, i.e. if the checkout no longer contains the code this analysis relies on.

Usage: p03_code_facts.py <pearl_dir>
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

GEMM_OPS = "miner/vllm-miner/src/vllm_miner/gemm_operators.py"

# (group, claim, file, regex) -- every regex must match at least one line
FACTS = [
    ("A. Useful miner: extra work on every mined GEMM call (pearl_gemm_noisy)",
     "entry point of a mined GEMM", GEMM_OPS, r"^def pearl_gemm_noisy\("),
    ("A", "copies and BLAKE3-hashes the activations A", GEMM_OPS, r"^\s+A\.to\(torch\.uint8\),"),
    ("A", "copies and BLAKE3-hashes the whole weight shard B, on every call", GEMM_OPS, r"^\s+B\.to\(torch\.uint8\),"),
    ("A", "regenerates noise factors from the new commitments, on every call", GEMM_OPS, r"= generate_noise_factors\("),
    ("A", "noises B again, on every call", GEMM_OPS, r"run_noising_B=True"),
    ("A", "denoises the output (the useful result), on every call", GEMM_OPS, r"skip_denoising=False"),
    ("B. Pure miner: what it can skip",
     "B-side seed depends only on job key and B's Merkle root, not on A",
     "miner/miner-base/src/miner_base/commitment_hash.py", r"return blake3\(key \+ B_merkle_root\)"),
    ("B", "A-side seed is chained after B: changing A does not change B's noise",
     "miner/miner-base/src/miner_base/commitment_hash.py", r"return blake3\(commitment_B \+ A_merkle_root\)"),
    ("B", "gateway replaces the job only when the previous block hash changes (once per block)",
     "miner/pearl-gateway/src/pearl_gateway/work_cache.py", r"template\.header\.previous_block_hash"),
    ("B", "the denoised product C is never checked by the verifier (see P01): a pure miner can skip denoising",
     "zk-pow/src/api/verify.rs", r"pub fn verify_plain_proof"),
    ("C. Shares count multiply-adds only",
     "jackpot target is scaled by tile size x dot-product length, nothing else",
     "zk-pow/src/api/sanity_checks.rs", r"tile_size\(config\) \* config\.dot_product_length\(\)"),
    ("C", "after the rank-penalty fork: tile size x (dot length / r) x 128",
     "zk-pow/src/api/sanity_checks.rs", r"tile_size\(config\) \* \(config\.dot_product_length\(\) / config\.rank as usize\) \* PENALTY_BASE_RANK"),
    ("D. Hardware gate for the official useful miner",
     "CUDA kernels are compiled for one target only: Hopper sm_90a (H100/H200)",
     "miner/pearl-gemm/setup.py", r'COMPUTE_CAPABILITY = "arch=compute_90a,code=sm_90a"'),
    ("D", "the vLLM plugin refuses GPUs below compute capability 9",
     "miner/vllm-miner/src/vllm_miner/vllm_scheme.py", r"return 9\b"),
    ("D", "README: only sm90 (H100 / H200) is supported",
     "miner/README.md", r"only \*\*sm90\*\* \(H100 / H200\)"),
]


def main():
    pearl = Path(sys.argv[1])
    missing = 0
    group = None
    for g, claim, rel, pattern in FACTS:
        if g != group and len(g) > 1:
            print(f"\n{g}")
            group = g
        lines = (pearl / rel).read_text().splitlines()
        hits = [(i + 1, line.strip()) for i, line in enumerate(lines) if re.search(pattern, line)]
        if not hits:
            print(f"  [MISSING] {claim}\n            {rel}: no line matches /{pattern}/")
            missing += 1
            continue
        ln, text = hits[0]
        print(f"  - {claim}\n      {rel}:{ln}  {text}")
    print()
    if missing:
        print(f"{missing} fact(s) not found in this checkout")
        sys.exit(1)
    print(f"All {len(FACTS)} facts found.")


if __name__ == "__main__":
    main()
