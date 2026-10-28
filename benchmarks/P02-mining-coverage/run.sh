#!/usr/bin/env bash
# Run the P02 benchmark: (1) Pearl's own trigger rule applied to the published checkpoints,
# (2) the workload simulation (estimates). Needs network once, to fetch two checkpoint
# config.json files from Hugging Face and PyYAML (which Pearl's config.py imports).
set -euo pipefail

HERE="$(cd "$(dirname "$0")" && pwd)"
ROOT="$(cd "$HERE/../.." && pwd)"
PEARL_DIR="${PEARL_DIR:-$ROOT/third_party/pearl}"
CKPT_DIR="${CHECKPOINT_DIR:-$ROOT/third_party/hf-configs}"
VENV="$ROOT/third_party/venv-p02"

# Pearl checkpoints at the revisions analysed (sha256 of config.json in the last column)
CHECKPOINTS=(
  "Llama-3.1-8B-Instruct-pearl 5dcb348a9f6d26fc42c0db0d8fbab2a0708796c0 35085ae9ece492af51cfa439cee510461bc64e35d7e90e0fca689bb7875c3bdc"
  "Llama-3.3-70B-Instruct-pearl 6cc401caab46ffa688dea3553b2f55d3dfc1d0aa 7a76c1e91a3d57773d2d8cfec07a728d6b6577f64dd6756f36579f65b47758a8"
)

if [ ! -f "$PEARL_DIR/miner/vllm-miner/src/vllm_miner/config.py" ]; then
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

mkdir -p "$HERE/results"
STAMP="$(date +%Y%m%d-%H%M%S)"
LOG1="$HERE/results/run-$STAMP-trigger.txt"
LOG2="$HERE/results/run-$STAMP-workload.txt"

echo "== Part 1: trigger rule x checkpoint layers (determined by code)"
{
  echo "# pearl @$(git -C "$PEARL_DIR" rev-parse --short HEAD 2>/dev/null || echo unknown), $(date)"
  "$VENV/bin/python" -I "$HERE/p02_trigger_table.py" "$PEARL_DIR" "$CKPT_DIR"
} | tee "$LOG1"

echo
echo "== Part 2: workload simulation (estimates, ~15 s)"
python3 -I "$HERE/p02_workload_sim.py" | tee "$LOG2"

echo
echo "Logs written to $LOG1 and $LOG2"
