#!/usr/bin/env bash
# Run the P04 shape census (CPU only, Python standard library).
#   ./run.sh          analyse the committed sample (results/certs-mainnet.jsonl)
#   ./run.sh fetch    first re-read the sample from the public Blockbook (~10 min, read-only)
set -euo pipefail

HERE="$(cd "$(dirname "$0")" && pwd)"
ROOT="$(cd "$HERE/../.." && pwd)"
PEARL_DIR="${PEARL_DIR:-$ROOT/third_party/pearl}"
CKPT_DIR="${CHECKPOINT_DIR:-$ROOT/third_party/hf-configs}"
CERTS="$HERE/results/certs-mainnet.jsonl"

# Pearl checkpoints at the revisions analysed (sha256 of config.json in the last column)
CHECKPOINTS=(
  "Llama-3.1-8B-Instruct-pearl 5dcb348a9f6d26fc42c0db0d8fbab2a0708796c0 35085ae9ece492af51cfa439cee510461bc64e35d7e90e0fca689bb7875c3bdc"
  "Llama-3.3-70B-Instruct-pearl 6cc401caab46ffa688dea3553b2f55d3dfc1d0aa 7a76c1e91a3d57773d2d8cfec07a728d6b6577f64dd6756f36579f65b47758a8"
  "Gemma-4-31B-it-pearl f1dfba688ce6343b0433de57ca4dc0f3d1c5baa5 600c651d1a9358b92b15f8a0333d1427f1646d927344c1b0f8b668b50e6aab79"
  "Qwen3-30B-A3B-Instruct-2507-pearl 751fd1052d4b25faf16072173493b15097a12901 ea51e79273d4a87bcf3d22c9bb6a1af0371f97377b6fb42aaedad3cf3370efce"
)

if [ ! -f "$PEARL_DIR/miner/miner-base/src/miner_base/settings.py" ]; then
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

if [ "${1:-}" = "fetch" ]; then
  CERTS="$HERE/results/run-$(date +%Y%m%d-%H%M%S)-certs.jsonl"
  echo "== Fetching block certificates into $CERTS"
  python3 -I "$HERE/p04_fetch_certs.py" "$CERTS" 99000 134800 50 200
fi

echo "== Shape census of $CERTS"
python3 -I "$HERE/p04_census.py" "$PEARL_DIR" "$CERTS" "$CKPT_DIR" | tee "$HERE/results/run-$(date +%Y%m%d-%H%M%S)-census.txt"
