#!/usr/bin/env bash
# Run the P01 benchmark against a pinned Pearl checkout (see scripts/setup-pearl.sh).
# Pass "plain" as the first argument to skip ZK proving (seconds instead of ~20 s, ~6 GB RAM).
set -euo pipefail

HERE="$(cd "$(dirname "$0")" && pwd)"
ROOT="$(cd "$HERE/../.." && pwd)"
PEARL_DIR="${PEARL_DIR:-$ROOT/third_party/pearl}"
MODE="${1:-zk}"

if [ ! -d "$PEARL_DIR/zk-pow" ]; then
  echo "Pearl not found at $PEARL_DIR; run scripts/setup-pearl.sh first" >&2
  exit 1
fi

cp "$HERE/p01_any_matrix.rs" "$PEARL_DIR/zk-pow/examples/"
(cd "$PEARL_DIR/zk-pow" && cargo +1.88.0 build --release --example p01_any_matrix)

ARGS=()
[ "$MODE" = "zk" ] && ARGS=(zk)

mkdir -p "$HERE/results"
LOG="$HERE/results/run-$(date +%Y%m%d-%H%M%S).log"
{
  echo "# cmd: p01_any_matrix ${ARGS[*]-}  ($(rustc +1.88.0 --version), pearl @$(git -C "$PEARL_DIR" rev-parse --short HEAD 2>/dev/null || echo unknown), $(uname -sm), $(date))"
  (cd "$PEARL_DIR/zk-pow" && cargo +1.88.0 run -q --release --example p01_any_matrix -- "${ARGS[@]+"${ARGS[@]}"}")
} 2>&1 | tee "$LOG"

echo
echo "Log written to $LOG"
grep -c "ACCEPTED" "$LOG" | sed 's/^/ACCEPTED lines: /'
