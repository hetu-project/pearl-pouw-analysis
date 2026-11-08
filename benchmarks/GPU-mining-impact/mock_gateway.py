#!/usr/bin/env python3
"""
Run Pearl's gateway with the mock block template from Pearl's own tests, until killed.

Same setup as the pearl_gateway_process fixture in Pearl's miner/conftest.py (via
p05_gpu_check.start_mock_gateway): no node, no network, a fixed template at height 100000.
`vllm serve` with Pearl's plugin connects to it over /tmp/pearlgw.sock.
Usage: python mock_gateway.py --pearl <pearl_dir>
"""
import argparse
import time
from pathlib import Path

from kernel_overhead import load_p05

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--pearl", required=True)
    load_p05().start_mock_gateway(Path(ap.parse_args().pearl))
    print("mock gateway ready", flush=True)
    while True:
        time.sleep(60)
