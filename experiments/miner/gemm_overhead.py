#!/usr/bin/env python3
"""
Per-call operation counts of vllm_miner.gemm_operators.pearl_gemm_noisy (pearl 156ef9a8),
relative to the plain int8 product m*n*k.  ESTIMATE: counts ops/bytes, not time.

Steps (gemm_operators.py line refs):
  quant_7bit          vllm_kernels.py:202     ~ m*k elementwise  (+ Hadamard if block_size>0)
  A.to(uint8) copy    :126                    m*k bytes (copy)
  tensor_hash(A)      :125-130                BLAKE3 over m*k bytes
  B.to(uint8) copy    :134                    n*k bytes (copy of the whole weight shard)
  tensor_hash(B)      :133-138                BLAKE3 over n*k bytes (every call)
  commitment hashes   :143-150                O(1) BLAKE3
  noise_gen           :153-170                (m+n)*r + 4*k*r bytes of BLAKE3-XOF output
  noise_A (in kernel) noisy_gemm              ApEA = A + EAL*EAR   (m*k*r int8 MAC, dense WGMMA)
                                              AxEBL = A * EBL      (m*k*r int8 MAC)
  noise_B (in kernel)                         BpEB = B + EBL*EBR   (n*k*r int8 MAC)
                                              EARxBpEB = EAR*BpEB  (n*k*r int8 MAC)
  main GEMM                                   ApEA x BpEB          (m*n*k int8 MAC)
  inner hash                                  every r along k: XOR-reduce 128-elem fragment
                                              -> m*n*(k/r) 32-bit XORs; m*n/128 BLAKE3 compressions
  denoise epilogue                            -AxEBL*EBR, -EAL*EARxBpEB (2*m*n*r fp16 MAC, fp32 acc)
"""

R = 128


def overhead(m, n, k, r=R, fp16_cost=2.0):
    base = m * n * k
    noise_a = 2 * m * k * r
    noise_b = 2 * n * k * r
    denoise = 2 * m * n * r
    xor_ops = m * n * (k // r)
    hash_bytes = m * k + n * k
    blake_compress = m * n / 128 + hash_bytes / 64
    return dict(
        noiseA=noise_a / base,
        noiseB=noise_b / base,
        denoise_fp16=denoise / base,
        tc_equiv=(noise_a + noise_b + fp16_cost * denoise) / base,
        xor_per_mac=xor_ops / base,
        hash_bytes_per_mac=hash_bytes / base,
        blake_compress=blake_compress,
        extra_B_bytes=n * k * 4,  # copy(write)+copy read for hash + noiseB read + BpEB write  (rough)
    )


SHAPES = {
    "8B gate_up (TP1)": (28672, 4096),
    "8B qkv (TP1)": (6144, 4096),
    "8B o_proj (TP1)": (4096, 4096),
    "70B gate_up (TP1/DP)": (57344, 8192),
    "70B o_proj (TP1/DP)": (8192, 8192),
    "70B gate_up (TP4)": (14336, 8192),
    "70B o_proj (TP4)": (8192, 2048),
}

if __name__ == "__main__":
    print("extra tensor-core work relative to m*n*k (r=128; fp16 MAC counted as 2 int8 MAC on H100/H200)")
    print(f"{'shape':24s} {'m':>6s} {'noiseA=2r/n':>12s} {'noiseB=2r/m':>12s} {'denoise=2r/k':>13s} {'TC-equiv':>9s} {'hashB/MAC':>10s}")
    for name, (n, k) in SHAPES.items():
        for m in (1024, 2048, 4096, 8192):
            o = overhead(m, n, k)
            print(f"{name:24s} {m:6d} {o['noiseA']:12.2%} {o['noiseB']:12.2%} {o['denoise_fp16']:13.2%} {o['tc_equiv']:9.1%} {o['hash_bytes_per_mac']:10.5f}")
    # whole-model B-side memory traffic per mined forward (rough)
    for label, wbytes in (("8B mined int7 weights", 4.70e9), ("70B mined int7 weights", 46.3e9)):
        traffic = 4 * wbytes
        print(f"{label}: ~{wbytes/1e9:.1f} GB int7 weights -> ~{traffic/1e9:.0f} GB extra B-side traffic per mined forward"
              f" (~{traffic/4.8e12*1e3:.0f} ms at 4.8 TB/s H200)")
