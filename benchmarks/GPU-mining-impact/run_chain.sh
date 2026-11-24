#!/bin/bash
# Private regtest chain: one pearld with no peers (loopback only), Pearl's real gateway,
# and vLLM mining against it. Usage: run_chain.sh <tag> <min_m> [chain_mine.py args...]
# The chain keeps growing across invocations (datadir is kept); pass RESET=1 to start fresh.
set -u
TAG=$1; MINM=$2; shift 2
P=/workspace/pearl
B=/workspace/pearl-pouw-analysis/benchmarks/GPU-mining-impact
W=/workspace/chain; mkdir -p $W/logs
CFG=miner/vllm-miner/src/vllm_miner/config.yaml
ADDR=rprl1p50ecun6548wx3per3hz7hp7qsnaqqdxtweyz7s37ty3kumszqscqrdsxrj   # Pearl's test address, regtest prefix
RPCU=chainuser; [ -s $W/rpcpass ] || python3 -c "import secrets;print(secrets.token_hex(16))" > $W/rpcpass; RPCP=$(cat $W/rpcpass); touch $W/prlctl.conf
CTL="$P/bin/prlctl --configfile=$W/prlctl.conf --regtest --notls --rpcserver=127.0.0.1:18334 --rpcuser=$RPCU --rpcpass=$RPCP"
cd $P || exit 1

[ "${RESET:-0}" = 1 ] && rm -rf $W/data
cleanup() {
    [ -n "${GW:-}" ] && kill $GW 2>/dev/null
    [ -n "${ND:-}" ] && kill $ND 2>/dev/null
    sleep 3
    git checkout -- "$CFG" 2>/dev/null
    echo "REVERTED tracked changes: $(git status --porcelain --untracked-files=no | wc -l)"
}
trap cleanup EXIT

# node: regtest, RPC and P2P on loopback only, no DNS seeds, no listening, no outbound peers
$P/bin/pearld --regtest --datadir=$W/data --logdir=$W/logs/node_$TAG \
    --rpcuser=$RPCU --rpcpass=$RPCP --rpclisten=127.0.0.1:18334 --notls \
    --nolisten --nodnsseed --connect=127.0.0.1:1 --miningaddr=$ADDR \
    > $W/logs/pearld_$TAG.out 2>&1 &
ND=$!
for i in $(seq 60); do $CTL getblockcount >/dev/null 2>&1 && break; sleep 1; done
H0=$($CTL getblockcount 2>&1); echo "HEIGHT_BEFORE $H0"
echo "PEERS $($CTL getconnectioncount 2>&1)"

rm -f /tmp/pearlgw.sock
PEARLD_RPC_URL=http://127.0.0.1:18334 PEARLD_RPC_USER=$RPCU PEARLD_RPC_PASSWORD=$RPCP \
PEARLD_MINING_ADDRESS=$ADDR uv run --no-sync pearl-gateway start > $W/logs/gateway_$TAG.log 2>&1 &
GW=$!
for i in $(seq 60); do [ -S /tmp/pearlgw.sock ] && break; sleep 1; done
sleep 3

sed -i "s/^\( *\)min_m: .*/\1min_m: $MINM/" "$CFG"; grep -n "min_m" "$CFG"
uv run --no-sync python $B/chain_mine.py --tag "$TAG" "$@" > $W/logs/miner_$TAG.log 2>&1
echo "miner rc=$?"
grep -E "^CHAIN" $W/logs/miner_$TAG.log
sleep 5
H1=$($CTL getblockcount 2>&1); echo "HEIGHT_AFTER $H1"
echo "BLOCKS_ACCEPTED $((H1 - H0))"
echo "PEERS $($CTL getconnectioncount 2>&1)"
echo "--- miner log counts"
for p in "Block found!" "Received PlainProof" "verified ZK proof" "Failed to verify" "Error" "rejected" "old header" "stale"; do
    echo "  miner   '$p': $(grep -c "$p" $W/logs/miner_$TAG.log)"; done
echo "--- gateway log counts"
for p in "Received PlainProof" "verified ZK proof" "submitblock" "Block submitted" "accepted" "rejected" "Failed" "Error" "old header" "stale"; do
    echo "  gateway '$p': $(grep -ci "$p" $W/logs/gateway_$TAG.log)"; done
echo "--- node log"
grep -hiE "accepted block|rejected block|block .* rejected|PROCESS: " $W/logs/node_$TAG/regtest/*.log 2>/dev/null | tail -5 | cut -c1-220
grep -hicE "rejected" $W/logs/node_$TAG/regtest/*.log 2>/dev/null | head -1
echo "CHAIN_RUN_DONE $TAG"
