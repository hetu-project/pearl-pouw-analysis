#!/bin/bash
# Build pearld + prlctl from the Pearl checkout (same commands as Pearl's Taskfile.yml).
# Go toolchain is installed under /workspace/tools (official tarball, sha256 checked against go.dev).
set -euo pipefail
export PATH="$HOME/.cargo/bin:/workspace/tools/go/bin:$PATH"
export GOPATH=/workspace/tools/gopath GOCACHE=/workspace/tools/gocache
export CGO_ENABLED=1 CGO_LDFLAGS_ALLOW=".*zk_pow_ffi.*"
GOV=$(grep -m1 '^go ' /workspace/pearl/go.mod | awk '{print $2}')

mkdir -p /workspace/tools
if [ ! -x /workspace/tools/go/bin/go ]; then
    cd /workspace/tools
    F=go${GOV}.linux-amd64.tar.gz
    curl -fsSL -o "$F" "https://go.dev/dl/$F"
    WANT=$(curl -fsSL "https://go.dev/dl/?mode=json&include=all" | python3 -c "
import json,sys
for r in json.load(sys.stdin):
    for f in r['files']:
        if f['filename']=='$F': print(f['sha256'])")
    GOT=$(sha256sum "$F" | awk '{print $1}')
    echo "go tarball sha256 want=$WANT got=$GOT"
    [ -n "$WANT" ] && [ "$WANT" = "$GOT" ] || { echo "CHECKSUM MISMATCH"; exit 1; }
    tar xzf "$F" && rm "$F"
fi
go version; cargo --version

cd /workspace/pearl/zk-pow
if [ ! -s src/circuit/v2_cache.bin ] || [ ! -s src/v1/v1_cache.bin ]; then
    cargo run --release --no-default-features --bin build_cache src/circuit/v2_cache.bin src/v1/v1_cache.bin 2>&1 | tail -3
fi
echo STEP_CACHE_OK
cd /workspace/pearl/zk-pow/bindings/go && cargo build --release 2>&1 | tail -2
echo STEP_GOBIND_OK
cd /workspace/pearl/xmss && { [ -f libxmss.a ] || make CC=gcc CXX=g++ 2>&1 | tail -2; }
echo STEP_XMSS_OK
cd /workspace/pearl && mkdir -p bin
go build -tags xmss,zkpow -o bin/pearld ./node 2>&1 | tail -5
go build -tags xmss,zkpow -o bin/prlctl ./node/cmd/prlctl 2>&1 | tail -5
ls -la bin/
bin/pearld --version
echo "git status:"; git status --porcelain | head
echo NODE_BUILD_DONE
