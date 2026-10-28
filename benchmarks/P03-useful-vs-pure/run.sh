#!/usr/bin/env bash
# Run the P03 benchmark (CPU only, Python standard library):
# (1) locate the code facts in Pearl, (2) per-call overhead of a mined GEMM,
# (3) useful-vs-pure share-rate ratio (estimates, simulated H100/H200).
set -euo pipefail

HERE="$(cd "$(dirname "$0")" && pwd)"
ROOT="$(cd "$HERE/../.." && pwd)"
PEARL_DIR="${PEARL_DIR:-$ROOT/third_party/pearl}"

if [ ! -d "$PEARL_DIR/zk-pow" ]; then
  echo "Pearl not found at $PEARL_DIR; run scripts/setup-pearl.sh first" >&2
  exit 1
fi

mkdir -p "$HERE/results"
STAMP="$(date +%Y%m%d-%H%M%S)"

echo "== Part 1: code facts (located in the Pearl source)"
{
  echo "# pearl @$(git -C "$PEARL_DIR" rev-parse --short HEAD 2>/dev/null || echo unknown), $(date)"
  python3 -I "$HERE/p03_code_facts.py" "$PEARL_DIR"
} | tee "$HERE/results/run-$STAMP-code-facts.txt"

echo
echo "== Part 2: extra work per mined GEMM call (operation counts from the code)"
python3 -I "$ROOT/experiments/miner/gemm_overhead.py" | tee "$HERE/results/run-$STAMP-overhead.txt"

echo
echo "== Part 3: useful vs pure miner share rate (estimates, ~10 s)"
python3 -I "$HERE/p03_share_ratio.py" | tee "$HERE/results/run-$STAMP-share-ratio.txt"

echo
echo "Logs written to $HERE/results/run-$STAMP-*.txt"
