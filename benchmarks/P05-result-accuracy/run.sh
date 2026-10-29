#!/usr/bin/env bash
# Run the P05 benchmark: (1) what Pearl's tests and published data can show (code facts),
# (2) CPU emulation of the fp16 denoising error (~30 s). Needs network once, to fetch the pinned
# 8B model card from Hugging Face and NumPy.
set -euo pipefail

HERE="$(cd "$(dirname "$0")" && pwd)"
ROOT="$(cd "$HERE/../.." && pwd)"
PEARL_DIR="${PEARL_DIR:-$ROOT/third_party/pearl}"
CKPT_DIR="${CHECKPOINT_DIR:-$ROOT/third_party/hf-configs}"
VENV="$ROOT/third_party/venv-p05"

# Llama-3.1-8B-Instruct-pearl model card at the revision analysed
CARD_REV="5dcb348a9f6d26fc42c0db0d8fbab2a0708796c0"
CARD="$CKPT_DIR/Llama-3.1-8B-Instruct-pearl.README.md"

if [ ! -f "$PEARL_DIR/miner/pearl-gemm/tests/test_pearl_gemm.py" ]; then
  echo "Pearl not found at $PEARL_DIR; run scripts/setup-pearl.sh first" >&2
  exit 1
fi

mkdir -p "$CKPT_DIR"
if [ ! -f "$CARD" ]; then
  curl -sfL -o "$CARD" "https://huggingface.co/pearl-ai/Llama-3.1-8B-Instruct-pearl/resolve/$CARD_REV/README.md"
fi

if [ ! -x "$VENV/bin/python" ]; then
  python3 -m venv "$VENV"
  "$VENV/bin/pip" install -q --disable-pip-version-check numpy==2.1.3
fi

mkdir -p "$HERE/results"
STAMP="$(date +%Y%m%d-%H%M%S)"
LOG1="$HERE/results/run-$STAMP-code-facts.txt"
LOG2="$HERE/results/run-$STAMP-denoise.txt"

echo "== Part 1: what Pearl's tests and published data can show"
python3 -I "$HERE/p05_code_facts.py" "$PEARL_DIR" "$CARD" | tee "$LOG1"

echo
echo "== Part 2: CPU emulation of the fp16 denoising error"
"$VENV/bin/python" -I "$HERE/p05_denoise_error.py" | tee "$LOG2"

echo
echo "Logs written to $LOG1 and $LOG2"
