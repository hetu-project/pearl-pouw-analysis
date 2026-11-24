#!/bin/bash
# Cost probe: CUDA graphs + mining at min_m=16, chat workload.
# Replayed decode steps execute the captured mining kernels with no Python at all, so this
# is the floor for "all host work removed". It is NOT valid mining (wins are never checked).
set -u
cd /workspace/pearl || exit 1
B=/workspace/pearl-pouw-analysis/benchmarks/GPU-mining-impact
M=pearl-ai/Llama-3.1-8B-Instruct-pearl
CFG=miner/vllm-miner/src/vllm_miner/config.yaml
RES=/workspace/res
cleanup() { git checkout -- "$CFG" 2>/dev/null; echo "REVERTED: $(git status --porcelain | wc -l) changes"; }
trap cleanup EXIT

run() {  # run <name> <config> <env...>
    local name=$1 cfg=$2; shift 2
    echo "##### $name"
    env PEARL_GPU_UTIL=0.5 "$@" uv run --no-sync python "$B/serving_hostfix.py" run \
        --pearl /workspace/pearl --config "$cfg" --model "$M" \
        --only chat --out "$RES/graph_${name}.json" 2>&1 | grep -E "FAST: applied|force_graph|\[eager|\[graph|Traceback|Error:|rror" | cut -c1-600
}

run nomine graph PEARL_FORCE_GRAPH=1
sed -i "s/^\( *\)min_m: .*/\1min_m: 16/" "$CFG"; grep -n "min_m" "$CFG"
run mine_stock eager_mine PEARL_FORCE_GRAPH=1
run mine_fast  eager_mine PEARL_FORCE_GRAPH=1 PEARL_FAST=1
run mine_fast2 eager_mine PEARL_FORCE_GRAPH=1 PEARL_FAST=1
echo GRAPH_DONE
