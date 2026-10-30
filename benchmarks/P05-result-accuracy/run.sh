#!/usr/bin/env bash
# Run the CPU part of the P05 benchmark: what Pearl's own test and published data can show.
# The GPU check (gpu/p05_gpu_check.py) needs an H100/H200; see README.md.
# Needs network once, to fetch the pinned 8B model card from Hugging Face.
set -euo pipefail

HERE="$(cd "$(dirname "$0")" && pwd)"
ROOT="$(cd "$HERE/../.." && pwd)"
PEARL_DIR="${PEARL_DIR:-$ROOT/third_party/pearl}"
CKPT_DIR="${CHECKPOINT_DIR:-$ROOT/third_party/hf-configs}"

# Llama-3.1-8B-Instruct-pearl model card at the revision analysed
CARD_REV="5dcb348a9f6d26fc42c0db0d8fbab2a0708796c0"
CARD="$CKPT_DIR/Llama-3.1-8B-Instruct-pearl.README.md"

if [ ! -f "$PEARL_DIR/miner/vllm-miner/tests/test_vllm_execution.py" ]; then
  echo "Pearl not found at $PEARL_DIR; run scripts/setup-pearl.sh first" >&2
  exit 1
fi

mkdir -p "$CKPT_DIR" "$HERE/results"
if [ ! -f "$CARD" ]; then
  curl -sfL -o "$CARD" "https://huggingface.co/pearl-ai/Llama-3.1-8B-Instruct-pearl/resolve/$CARD_REV/README.md"
fi

LOG="$HERE/results/run-$(date +%Y%m%d-%H%M%S)-code-facts.txt"
python3 -I "$HERE/p05_code_facts.py" "$PEARL_DIR" "$CARD" | tee "$LOG"
echo
echo "Log written to $LOG"
