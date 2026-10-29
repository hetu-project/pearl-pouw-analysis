#!/usr/bin/env python3
"""
P05 benchmark, part 1: what Pearl's own tests and published data can and cannot show about the
accuracy of the useful output.

Checks, all against the Pearl source and the pinned 8B model card:
  A. kernel tests (miner/pearl-gemm/tests/test_pearl_gemm.py): tolerance, and the largest k tested;
  B. the vLLM end-to-end test (miner/vllm-miner/tests/test_vllm_execution.py): every prompt is sent
     alone, and its UTF-8 length bounds its token count (a byte-level BPE token is at least one
     byte). If every prompt is shorter than the plugin's min_m, no forward step reaches the mining
     threshold, so the "mining" runs never execute a noisy GEMM;
  C. with mining switched on, activations of mining layers are quantized to 7 bits whether or not
     a step mines; with mining switched off they are quantized to 8 bits. The end-to-end test's
     "mining" vs "no mining" comparison is therefore 7-bit vanilla GEMM vs 8-bit vanilla GEMM;
  D. the 8B model card publishes no accuracy numbers.

Usage: p05_code_facts.py <pearl_dir> <llama-8b-model-card.md>
"""
from __future__ import annotations

import ast
import json
import re
import sys
from pathlib import Path

KERNEL_TEST = "miner/pearl-gemm/tests/test_pearl_gemm.py"
E2E_TEST = "miner/vllm-miner/tests/test_vllm_execution.py"


def find(pearl: Path, rel: str, pattern: str):
    for i, line in enumerate((pearl / rel).read_text().splitlines()):
        if re.search(pattern, line):
            return i + 1, line.strip()
    return None


def main():
    pearl, card = Path(sys.argv[1]), Path(sys.argv[2])
    failures = 0

    def fact(claim, rel, pattern):
        nonlocal failures
        hit = find(pearl, rel, pattern)
        if hit is None:
            print(f"  [MISSING] {claim}\n            {rel}: no line matches /{pattern}/")
            failures += 1
        else:
            print(f"  - {claim}\n      {rel}:{hit[0]}  {hit[1]}")

    print("A. Kernel tests")
    fact("absolute tolerance 0.1", KERNEL_TEST, r"atol = 1e-1")
    fact("relative tolerance 1%", KERNEL_TEST, r"rtol = 1e-2")
    src = (pearl / KERNEL_TEST).read_text()
    ks = {int(v) for lst in re.findall(r'parametrize\("k",\s*\[([^\]]*)\]', src) for v in re.findall(r"\d+", lst)}
    shapes = re.search(r"SHAPES_MNK = \[(.*?)\n\s*\]", src, re.S)
    ks |= {int(t[2]) for t in re.findall(r"\((\d+),\s*(\d+),\s*(\d+)\)", shapes.group(1))} if shapes else set()
    print(f"  - k values exercised by the kernel tests: {sorted(ks)}; largest k = {max(ks)}")

    print("\nB. vLLM end-to-end test")
    yaml = (pearl / "miner/vllm-miner/src/vllm_miner/config.yaml").read_text()
    min_m = int(re.search(r"min_m:\s*(\d+)", yaml).group(1))
    tree = ast.parse((pearl / E2E_TEST).read_text())
    prompts = {}
    for node in ast.walk(tree):
        if isinstance(node, ast.Call) and getattr(node.func, "id", "") == "ModelTestConfig":
            kw = {k.arg: k.value for k in node.keywords}
            mid = ast.literal_eval(kw["id"])
            for field in ("prompt", "consistency_prompt"):
                prompts[f"{mid}.{field}"] = ast.literal_eval(kw[field])
    for name, text in prompts.items():
        print(f"  - {name}: {len(text.encode())} UTF-8 bytes, so at most {len(text.encode())} tokens")
    longest = max(len(t.encode()) for t in prompts.values())
    fact("each generate() call sends one prompt", E2E_TEST, r"llm_on\.generate\(model_config\.consistency_prompt")
    fact("generation is at most 100 tokens; decode steps then have m = 1", E2E_TEST, r"max_tokens=100")
    ok = longest < min_m
    print(f"  - longest prompt {longest} bytes < plugin min_m {min_m}: {'yes' if ok else 'NO'}"
          " -> no step of the test reaches the mining threshold")
    failures += not ok
    refs = json.loads((pearl / "miner/vllm-miner/tests/reference_outputs.json").read_text())
    for key in sorted(k for k in refs if k.endswith("_mining") and not k.endswith("_no_mining")):
        twin = key[: -len("_mining")] + "_no_mining"
        print(f"  - reference outputs: {key} == {twin}: {refs[key] == refs.get(twin)} ({refs[key]!r})")

    print("\nC. What the mining switch changes")
    fact("mining-layer path quantizes activations to 7 bits before choosing noisy or vanilla GEMM",
         "miner/vllm-miner/src/vllm_miner/vllm_kernels.py", r"x_q, x_s, _ = quant_7bit\(")
    fact("the vanilla (non-mining) branch uses the same 7-bit operands",
         "miner/vllm-miner/src/vllm_miner/vllm_kernels.py", r"return pearl_gemm_vanilla\(")
    fact("with mining switched off, activations are quantized to 8 bits instead",
         "miner/vllm-miner/src/vllm_miner/vllm_kernels.py", r"x_q, x_s, _ = quant_8bit\(")

    print("\nD. Published accuracy data")
    text = card.read_text()
    hits = re.findall(r"(?i)mmlu|gsm8k|humaneval|accuracy|benchmark|lm-eval", text)
    print(f"  - Llama-3.1-8B-Instruct-pearl model card ({card.name}): {len(hits)} mentions of accuracy or benchmarks")

    print()
    if failures:
        print(f"{failures} check(s) failed")
        sys.exit(1)
    print("All checks passed.")


if __name__ == "__main__":
    main()
