#!/usr/bin/env bash
# Run the P04 benchmark: (1) proofs through miner, pool and consensus checks (Rust, CPU),
# (2) share of the plugin's mining work that can never yield a block (Python), (3) code facts.
# Pass "plain" as the first argument to skip ZK proving (seconds instead of ~2 min, ~11 GB RAM).
# Needs network once, to fetch four checkpoint config.json files and PyYAML.
set -euo pipefail

HERE="$(cd "$(dirname "$0")" && pwd)"
ROOT="$(cd "$HERE/../.." && pwd)"
PEARL_DIR="${PEARL_DIR:-$ROOT/third_party/pearl}"
CKPT_DIR="${CHECKPOINT_DIR:-$ROOT/third_party/hf-configs}"
VENV="$ROOT/third_party/venv-p04"
MODE="${1:-zk}"

# Pearl checkpoints at the revisions analysed (sha256 of config.json in the last column)
CHECKPOINTS=(
  "Llama-3.1-8B-Instruct-pearl 5dcb348a9f6d26fc42c0db0d8fbab2a0708796c0 35085ae9ece492af51cfa439cee510461bc64e35d7e90e0fca689bb7875c3bdc"
  "Llama-3.3-70B-Instruct-pearl 6cc401caab46ffa688dea3553b2f55d3dfc1d0aa 7a76c1e91a3d57773d2d8cfec07a728d6b6577f64dd6756f36579f65b47758a8"
  "Gemma-4-31B-it-pearl f1dfba688ce6343b0433de57ca4dc0f3d1c5baa5 600c651d1a9358b92b15f8a0333d1427f1646d927344c1b0f8b668b50e6aab79"
  "Qwen3-30B-A3B-Instruct-2507-pearl 751fd1052d4b25faf16072173493b15097a12901 ea51e79273d4a87bcf3d22c9bb6a1af0371f97377b6fb42aaedad3cf3370efce"
)

if [ ! -d "$PEARL_DIR/zk-pow" ]; then
  echo "Pearl not found at $PEARL_DIR; run scripts/setup-pearl.sh first" >&2
  exit 1
fi

mkdir -p "$CKPT_DIR"
for entry in "${CHECKPOINTS[@]}"; do
  read -r name rev sha <<<"$entry"
  file="$CKPT_DIR/$name.config.json"
  if [ ! -f "$file" ]; then
    curl -sfL -o "$file" "https://huggingface.co/pearl-ai/$name/resolve/$rev/config.json"
  fi
  if [ "$(shasum -a 256 "$file" | cut -d' ' -f1)" != "$sha" ]; then
    echo "WARNING: $file does not match the analysed revision ($rev)" >&2
  fi
done

if [ ! -x "$VENV/bin/python" ]; then
  python3 -m venv "$VENV"
  "$VENV/bin/pip" install -q --disable-pip-version-check pyyaml==6.0.2
fi

mkdir -p "$HERE/results" "$PEARL_DIR/zk-pow/examples"
STAMP="$(date +%Y%m%d-%H%M%S)"
LOG1="$HERE/results/run-$STAMP-proofs.log"
LOG2="$HERE/results/run-$STAMP-wasted.txt"
LOG3="$HERE/results/run-$STAMP-code-facts.txt"

ARGS=()
[ "$MODE" = "zk" ] && ARGS=(zk)
cp "$HERE/p04_rule_mismatch.rs" "$PEARL_DIR/zk-pow/examples/"
(cd "$PEARL_DIR/zk-pow" && cargo +1.88.0 build --release --example p04_rule_mismatch)

echo "== Part 1: proofs through miner, pool and consensus checks"
{
  echo "# cmd: p04_rule_mismatch ${ARGS[*]-}  ($(rustc +1.88.0 --version), pearl @$(git -C "$PEARL_DIR" rev-parse --short HEAD 2>/dev/null || echo unknown), $(uname -sm), $(date))"
  (cd "$PEARL_DIR/zk-pow" && cargo +1.88.0 run -q --release --example p04_rule_mismatch -- "${ARGS[@]+"${ARGS[@]}"}")
} 2>&1 | tee "$LOG1"

echo
echo "== Part 2: share of mining work that can never yield a block (determined by code)"
"$VENV/bin/python" -I "$HERE/p04_wasted_mining.py" "$PEARL_DIR" "$CKPT_DIR" | tee "$LOG2"

echo
echo "== Part 3: code facts"
python3 -I "$HERE/p04_code_facts.py" "$PEARL_DIR" | tee "$LOG3"

echo
echo "Logs written to $LOG1, $LOG2 and $LOG3"
