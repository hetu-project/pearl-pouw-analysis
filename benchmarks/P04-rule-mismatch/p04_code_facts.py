#!/usr/bin/env python3
"""
P04 benchmark, part 3: the code facts behind "miner, pool and consensus apply different rules".

Locates each fact in the Pearl source by pattern (not by hard-coded line number) and prints the
file, line and text, so a reader can check them against any checkout. Exits non-zero if a pattern
is missing, i.e. if the checkout no longer contains the code this analysis relies on.

Usage: p04_code_facts.py <pearl_dir>
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

VALIDATE_GO = "node/blockchain/validate.go"
SANITY_RS = "zk-pow/src/api/sanity_checks.rs"

# (group, claim, file, regex) -- every regex must match at least one line
FACTS = [
    ("A. Dimension rule: the plugin mines what the verifier rejects",
     "plugin mines when k >= 1024", "miner/vllm-miner/src/vllm_miner/config.yaml", r"^\s+min_k: 1024"),
    ("A", "verifier requires k >= 16r (2048 at r = 128)", SANITY_RS, r'ensure!\(k >= 16 \* r, "k must be >= 16r'),
    ("A", "official noise rank r = 128", "miner/miner-base/src/miner_base/settings.py", r"noise_rank: int = 128"),
    ("A", "gateway's ZK proving only checks the private witness against the public data, not the k rule",
     "zk-pow/src/api/prove.rs", r"public\.sanity_check_private_params\(&private\)"),
    ("A", "gateway answers 'submitted' before the proof is built or checked, so the miner never sees the rejection",
     "miner/pearl-gateway/src/pearl_gateway/miner_rpc/server.py", r'return self\._jsonrpc_success\("submitted", request_id\)'),
    ("B. MoE rule: the node rejects MoE, the plugin still mines MoE experts",
     "mainnet dense-only fork height", "node/chaincfg/params.go", r"DenseOnlyForkHeight: 91630"),
    ("B", "node rejects every MoE certificate once the dense-only fork is active", VALIDATE_GO,
     r"cert\.IsMoE\(\) && params\.IsDenseOnlyForkActive\(height\)"),
    ("B", "plugin decides to mine MoE expert GEMMs with the ordinary size rule, no dense-only check",
     "miner/vllm-miner/src/vllm_miner/pearl_moe_experts.py", r"pearl_config\.should_use_noisy_gemm\(num_tokens, N, K\)"),
    ("C. Rank penalty: enforced by the node only",
     "the Rust verifier has no notion of block height, so height-dependent rules are left to the caller",
     SANITY_RS, r"this crate has no notion of\s*$"),
    ("C", "pool share check (verify_plain_proof) uses the unpenalized bound tile size x dot length",
     SANITY_RS, r"let jackpot_hash_bound = extract_difficulty_bound\(nbits, &public_params\.mining_config\);"),
    ("C", "node applies the rank penalty separately, after fork height 96,251", VALIDATE_GO,
     r"zkpow\.CheckRankPenalty\(header\.Bits, cert\.PublicDataBytes\(\)\)"),
    ("C", "mainnet rank-penalty fork height", "node/chaincfg/params.go", r"RankPenaltyForkHeight: 96251"),
    ("C", "official pool example validates shares with verify_plain_proof_for_cert_version only",
     "py-pearl-mining/examples/v1_v2_gateway_example.py", r"ok, _ = pm\.verify_plain_proof_for_cert_version\("),
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
    rs = (pearl / "zk-pow/src/api/verify.rs").read_text()
    print("\n  - verify.rs never calls check_rank_penalty:", "yes" if "check_rank_penalty" not in rs else "NO")
    if "check_rank_penalty" in rs:
        missing += 1
    print()
    if missing:
        print(f"{missing} fact(s) not found in this checkout")
        sys.exit(1)
    print(f"All {len(FACTS) + 1} facts found.")


if __name__ == "__main__":
    main()
