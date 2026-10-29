#!/usr/bin/env python3
"""
P05 benchmark, part 2: CPU emulation of pearl-gemm's fp16 denoising, to size the error that mining
adds to the useful GEMM output and to check it against the kernel test tolerance.

Kernel facts emulated (Pearl 156ef9a8, miner/pearl-gemm):
  * noise as in the official test generator (src/pearl_gemm/testing/gemm_tensor_generator.py):
    EAL (m x r), EBR (n x r) uniform integers in [-32, 31]; EAR, EBL (k x r) with two non-zero
    entries per row, +1 and -1;
  * A + EA = A + EAL EARᵀ, B + EB = B + EBR EBLᵀ, noisy product in exact int32;
  * denoise factors stored in fp16 after scaling (csrc/gemm/pearl_gemm_constants.hpp):
    A·EBL x 2^-14, (B+EB)·EAR x 2^-12, EBR x -4, EAL x -1;
  * epilogue (csrc/gemm/collective_epilogue.hpp): D = fp32(noisy product) / 2^12, add the two
    fp16 x fp16 correction products with fp32 accumulation, multiply back by 2^12, apply the
    per-row/column scales and round to bf16.
Assumption: fp16 products are exact in fp32 and accumulation is IEEE fp32 (numpy float32 matmul);
the Hopper accumulation order differs, so the size of the error, not its exact value, is the result.

Two input families: "uniform" = the kernel test's own int7 inputs; "heavy-tailed" = synthetic
activations (Student-t, a few outlier channels) and Gaussian weights quantized to int7, used only
to compare the denoising error with the quantization error in order of magnitude.

Usage: p05_denoise_error.py
"""
import numpy as np

R = 128
K_VALUES = (2048, 4096, 8192, 14336, 16384, 28672)
M = N = 256
ATOL, RTOL = 0.1, 0.01  # miner/pearl-gemm/tests/test_pearl_gemm.py
rng = np.random.default_rng(0)


def bf16(x):
    b = np.asarray(x, dtype=np.float32).view(np.uint32).astype(np.uint64)
    b = (b + 0x7FFF + ((b >> 16) & 1)) & 0xFFFF0000  # round to nearest even
    return b.astype(np.uint32).view(np.float32)


def sparse_pm(k, r):
    E = np.zeros((k, r), dtype=np.int32)
    for i in range(k):
        a, b = rng.choice(r, 2, replace=False)
        E[i, a], E[i, b] = 1, -1
    return E


def quant(x, bits):
    q = 2 ** (bits - 1) - 1
    s = np.abs(x).max(axis=1, keepdims=True) / q
    return np.clip(np.rint(x / s), -q, q).astype(np.int32), s


def run(k, family):
    if family == "uniform":
        Aq = rng.integers(-63, 64, (M, k)).astype(np.int32)
        Bq = rng.integers(-63, 64, (N, k)).astype(np.int32)
        sa, sb = np.ones((M, 1), np.float32), np.ones((N, 1), np.float32)
        X = W = None
    else:
        X = rng.standard_t(3, (M, k)).astype(np.float32)
        X[:, rng.choice(k, 8, replace=False)] *= 20.0
        W = (rng.standard_normal((N, k)) * 0.02).astype(np.float32)
        Aq, sa = quant(X, 7)
        Bq, sb = quant(W, 7)
    EAL = rng.integers(-32, 32, (M, R)).astype(np.int32)
    EBR = rng.integers(-32, 32, (N, R)).astype(np.int32)
    EAR, EBL = sparse_pm(k, R), sparse_pm(k, R)
    ApEA, BpEB = Aq + EAL @ EAR.T, Bq + EBR @ EBL.T
    assert np.abs(ApEA).max() <= 127 and np.abs(BpEB).max() <= 127
    noisy = ApEA.astype(np.int64) @ BpEB.T.astype(np.int64)
    exact = Aq.astype(np.int64) @ Bq.T.astype(np.int64)
    AxEBL16 = ((Aq @ EBL).astype(np.float32) * 2.0**-14).astype(np.float16).astype(np.float32)
    EARx16 = ((BpEB @ EAR).astype(np.float32) * 2.0**-12).astype(np.float16).astype(np.float32)
    D = noisy.astype(np.float32) / np.float32(4096)
    D = D + (-1.0 * EAL.astype(np.float32)) @ EARx16.T
    D = D + AxEBL16 @ (-4.0 * EBR.astype(np.float32)).T
    D = D * np.float32(4096)
    assert np.isfinite(D).all()
    err = D.astype(np.float64) - exact
    mined = bf16(D * sa * sb)
    vanilla = bf16(exact.astype(np.float32) * sa * sb)
    ref = bf16(exact * sa * sb)
    out = dict(err_rms=np.sqrt(np.mean(err**2)), err_max=np.abs(err).max(), err_share=np.mean(err != 0),
               bf16_differ=np.mean(mined != vanilla),
               tol_pass=bool(np.all(np.abs(mined - ref) <= ATOL + RTOL * np.abs(ref))))
    if X is not None:
        Y = X.astype(np.float64) @ W.T.astype(np.float64)
        ry = np.sqrt(np.mean(Y**2))
        A8, s8a = quant(X, 8)
        B8, s8b = quant(W, 8)
        out.update(
            q7=np.sqrt(np.mean(((exact * sa * sb) - Y) ** 2)) / ry,
            q8=np.sqrt(np.mean(((A8.astype(np.int64) @ B8.T) * s8a * s8b - Y) ** 2)) / ry,
            denoise_rel=np.sqrt(np.mean(((D.astype(np.float64) - exact) * sa * sb) ** 2)) / ry,
        )
    return out


def main():
    # numpy's Accelerate-backed float matmul on macOS raises spurious divide/overflow warnings;
    # results are checked to be finite instead (see the assert in run()).
    np.seterr(all="ignore")
    print(f"m = n = {M}, r = {R}; kernel test tolerance atol={ATOL}, rtol={RTOL}\n")
    print("Table 1. Denoising error with the kernel test's own inputs (uniform int7)")
    print(f"  {'k':>6s}  {'error rms':>9s}  {'error max':>9s}  {'elements with error':>19s}  {'bf16 output differs':>19s}  passes test tolerance")
    for k in K_VALUES:
        r = run(k, "uniform")
        print(f"  {k:6d}  {r['err_rms']:9.3g}  {r['err_max']:9.3g}  {r['err_share']:19.1%}  {r['bf16_differ']:19.1%}  {'yes' if r['tol_pass'] else 'NO'}")
    print("  (error in units of the exact integer product A·B; 0 means the denoised result is exact)\n")

    print("Table 2. Denoising error vs quantization error (synthetic heavy-tailed inputs, order of magnitude only)")
    print(f"  {'k':>6s}  {'error rms':>9s}  {'elements with error':>19s}  {'denoise rel. error':>18s}  {'W7A7 quant. rel. error':>22s}  {'W8A8 quant. rel. error':>22s}")
    for k in K_VALUES:
        r = run(k, "heavy-tailed")
        print(f"  {k:6d}  {r['err_rms']:9.3g}  {r['err_share']:19.1%}  {r['denoise_rel']:18.2e}  {r['q7']:22.2e}  {r['q8']:22.2e}")
    print("  (relative errors against an fp64 float GEMM of the unquantized operands)")


if __name__ == "__main__":
    main()
