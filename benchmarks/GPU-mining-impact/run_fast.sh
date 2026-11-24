#!/bin/bash
# Round 2: the fast mining path (fast_noisy.py), chat workload, min_m=16 (100% coverage).
# Only config.yaml is touched (min_m); it is reverted on exit.
set -u
cd /workspace/pearl || exit 1
B=/workspace/pearl-pouw-analysis/benchmarks/GPU-mining-impact
M=pearl-ai/Llama-3.1-8B-Instruct-pearl
CFG=miner/vllm-miner/src/vllm_miner/config.yaml
RES=/workspace/res

cleanup() { git checkout -- "$CFG" 2>/dev/null; echo "REVERTED: $(git status --porcelain | wc -l) changes"; }
trap cleanup EXIT

sed -i "s/^\( *\)min_m: .*/\1min_m: 16/" "$CFG"; grep -n "min_m" "$CFG"

for rep in 1 2 3; do
    echo "##### rep=$rep cfg=D"
    PEARL_FAST=1 PEARL_GPU_UTIL=0.5 uv run --no-sync python "$B/serving_hostfix.py" run \
        --pearl /workspace/pearl --config eager_mine --model "$M" \
        --only chat --out "$RES/rep${rep}_D.json" 2>&1 | grep -E "FAST|\[eager|Traceback|Error:"
done
echo "##### long workload, fast path at default threshold is not needed; run long at min_m=16"
PEARL_FAST=1 PEARL_GPU_UTIL=0.5 uv run --no-sync python "$B/serving_hostfix.py" run \
    --pearl /workspace/pearl --config eager_mine --model "$M" \
    --only long --out "$RES/fast_long.json" 2>&1 | grep -E "FAST|\[eager|Traceback|Error:"
echo FAST_DONE
