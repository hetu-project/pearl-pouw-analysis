#!/bin/bash
# Repeat runs for the min_m=16 (100% coverage) optimisation result.
#
# Four configurations, chat workload only, two further repeats each:
#   ref    no mining, stock source
#   A      mining, stock source                       (Pearl as shipped, threshold lowered)
#   B      + .view() instead of .to() + weight-hash cache
#   C      + host-side fixes (config memoised, pow target single-transfer)
#
# Everything is reverted at the end and git status is printed.
set -u
cd /workspace/pearl || exit 1

B=/workspace/pearl-pouw-analysis/benchmarks/GPU-mining-impact
M=pearl-ai/Llama-3.1-8B-Instruct-pearl
CFG=miner/vllm-miner/src/vllm_miner/config.yaml
GEMM=miner/vllm-miner/src/vllm_miner/gemm_operators.py
RES=/workspace/res

cleanup() {
    git checkout -- "$CFG" "$GEMM" 2>/dev/null
    echo "REVERTED: $(git status --porcelain | wc -l) changes"
}
trap cleanup EXIT

set_min_m()  { sed -i "s/^\( *\)min_m: .*/\1min_m: $1/" "$CFG"; grep -n "min_m" "$CFG"; }
apply_view() { sed -i 's/A\.to(torch\.uint8)/A.view(torch.uint8)/; s/B\.to(torch\.uint8)/B.view(torch.uint8)/' "$GEMM"; grep -n "view(torch.uint8)" "$GEMM"; }
drop_view()  { git checkout -- "$GEMM"; }

one() {  # one <name> <rep> <config> <env...>
    local name=$1 rep=$2 cfg=$3; shift 3
    local out=$RES/rep${rep}_${name}.json
    echo "##### rep=$rep cfg=$name"
    env "$@" uv run --no-sync python "$B/serving_hostfix.py" run \
        --pearl /workspace/pearl --config "$cfg" --model "$M" \
        --only chat --out "$out" 2>&1 | grep -E "HOSTFIX|BHASH|\[eager|Traceback|Error"
}

for rep in 2 3; do
    git checkout -- "$CFG" "$GEMM"

    one ref "$rep" eager PEARL_HOSTFIX=0 PEARL_GPU_UTIL=0.5

    set_min_m 16
    one A "$rep" eager_mine PEARL_HOSTFIX=0 PEARL_GPU_UTIL=0.5

    apply_view
    one B "$rep" eager_mine PEARL_HOSTFIX=0 PEARL_BHASH=1 PEARL_GPU_UTIL=0.5
    one C "$rep" eager_mine PEARL_HOSTFIX=1 PEARL_BHASH=1 PEARL_GPU_UTIL=0.5

    drop_view
    git checkout -- "$CFG"
done

echo REPEATS_DONE
