#!/usr/bin/env bash
# Fetch Pearl at the pinned commit into third_party/pearl.
# Set PEARL_DIR to reuse an existing checkout instead.
set -euo pipefail

PEARL_REPO="https://github.com/pearl-research-labs/pearl.git"
PEARL_COMMIT="156ef9a8eb9dc7c74fab8c502598b8b53d4be7b1"
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
PEARL_DIR="${PEARL_DIR:-$ROOT/third_party/pearl}"

if [ -d "$PEARL_DIR/.git" ]; then
  echo "Using existing checkout: $PEARL_DIR"
else
  mkdir -p "$PEARL_DIR"
  git -C "$PEARL_DIR" init -q
  git -C "$PEARL_DIR" remote add origin "$PEARL_REPO"
  git -C "$PEARL_DIR" fetch -q --depth 1 origin "$PEARL_COMMIT"
  git -C "$PEARL_DIR" checkout -q FETCH_HEAD
fi

HEAD_COMMIT="$(git -C "$PEARL_DIR" rev-parse HEAD)"
if [ "$HEAD_COMMIT" != "$PEARL_COMMIT" ]; then
  echo "WARNING: $PEARL_DIR is at $HEAD_COMMIT, expected $PEARL_COMMIT" >&2
  echo "Results and line numbers in this repository refer to $PEARL_COMMIT." >&2
fi

if ! rustup run 1.88.0 rustc --version >/dev/null 2>&1; then
  echo "Rust 1.88.0 is required: rustup toolchain install 1.88.0" >&2
  exit 1
fi
echo "Pearl ready at $PEARL_DIR ($HEAD_COMMIT)"
