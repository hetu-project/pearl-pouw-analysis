#!/usr/bin/env python3
"""
P04 benchmark, part 1: read the public mining parameters of Pearl mainnet blocks.

Each block starts with its certificate (node/wire/certificate.go):
    version(4) | block_hash(32) | public_data_len(4) | public_data | proof_len(4) | proof
and the public data is (zk-pow/src/api/proof.rs, PublicProofParams):
    mining_config(52) | hash_a(32) | hash_b(32) | hash_jackpot(32) | m(4) | n(4) | t_rows(4) | t_cols(4)
with mining_config = k(4) | rank(2) | mma_type(2) | rows_pattern(6) | cols_pattern(6) | moe(32).

Raw blocks come from the public Blockbook instance run by PRL Labs. Only public data is read;
nothing is sent to the network. Only the parsed public fields are kept, not the proofs.

Usage: p04_fetch_certs.py <out.jsonl> <start> <end> <step> [<last_n_consecutive>]
Heights already present in out.jsonl are skipped, so an interrupted run can be resumed.
"""
from __future__ import annotations

import json
import struct
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

BASE = "https://blockbook.pearlresearch.ai/api/v2"
UA = {"User-Agent": "pearl-pouw-analysis/1.0 (+https://github.com/hetu-project/pearl-pouw-analysis)"}


def http_json(url: str, tries: int = 6):
    for i in range(tries):
        try:
            req = urllib.request.Request(url, headers=UA)
            with urllib.request.urlopen(req, timeout=60) as r:
                return json.load(r)
        except urllib.error.HTTPError as e:
            if e.code not in (429, 500, 502, 503, 504):
                raise
            print(f"  HTTP {e.code} on {url}, retrying", file=sys.stderr)
            time.sleep(10 * (i + 1))
        except (urllib.error.URLError, TimeoutError, ConnectionError) as e:
            print(f"  {type(e).__name__} on {url}, retrying", file=sys.stderr)
            time.sleep(3 * (i + 1))
    raise RuntimeError(f"giving up on {url}")


def pattern_from_bytes(x: bytes) -> list[int]:
    """PeriodicPattern::from_bytes + to_list (zk-pow/src/api/proof_utils.rs:105-166)."""
    shape, min_stride = [], 1
    for i in range(3):
        factor, length = 1 + x[2 * i], 1 + x[2 * i + 1]
        stride = factor * min_stride
        shape.append((stride, length))
        min_stride = stride * length
    res = [0]
    for stride, length in shape:
        res = [r + i * stride for i in range(length) for r in res]
    return sorted(res)


def parse_block(raw: bytes) -> dict:
    cert_version, = struct.unpack_from("<I", raw, 0)
    pd_len, = struct.unpack_from("<I", raw, 36)
    pd = raw[40:40 + pd_len]
    proof_len, = struct.unpack_from("<I", raw, 40 + pd_len)
    k, rank, mma = struct.unpack_from("<IHH", pd, 0)
    rows, cols = pattern_from_bytes(pd[8:14]), pattern_from_bytes(pd[14:20])
    e, top_k = struct.unpack_from("<HH", pd, 20)
    m, n, t_rows, t_cols = struct.unpack_from("<IIII", pd, 52 + 96)
    return dict(cert_version=cert_version, public_data_len=pd_len, proof_len=proof_len,
                k=k, r=rank, mma_type=mma, h=len(rows), w=len(cols), rows=rows, cols=cols,
                moe_e=e, moe_top_k=top_k, m=m, n=n, t_rows=t_rows, t_cols=t_cols)


def main():
    out = Path(sys.argv[1])
    start, end, step = int(sys.argv[2]), int(sys.argv[3]), int(sys.argv[4])
    last_n = int(sys.argv[5]) if len(sys.argv) > 5 else 0
    heights = list(range(start, end + 1, step))
    heights += [h for h in range(end - last_n + 1, end + 1) if h not in heights]
    done = set()
    if out.exists():
        done = {json.loads(line)["height"] for line in out.read_text().splitlines() if line.strip()}
    todo = [h for h in sorted(set(heights)) if h not in done]
    print(f"{len(heights)} heights requested, {len(todo)} to fetch", file=sys.stderr)
    with out.open("a") as f:
        for i, h in enumerate(todo):
            meta = http_json(f"{BASE}/block/{h}")
            raw = bytes.fromhex(http_json(f"{BASE}/rawblock/{h}")["hex"])
            rec = dict(height=h, time=meta["time"], hash=meta["hash"], **parse_block(raw))
            f.write(json.dumps(rec, separators=(",", ":")) + "\n")
            f.flush()
            if (i + 1) % 50 == 0:
                print(f"  {i + 1}/{len(todo)} (height {h})", file=sys.stderr)
            time.sleep(0.3)


if __name__ == "__main__":
    main()
