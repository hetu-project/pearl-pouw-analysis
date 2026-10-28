"""Compare mining configurations: attempts, bound factors, and expected wins per unit of work
under several cost models. Units: expected wins are in units of T/2^256 (T = target from nbits),
so 'wins per MAC' = 1.0 means win-prob per MAC equals T/2^256.
Cost models are ESTIMATES (see notes 04 section 5), not code facts.
Run: python3 -I configs.py
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from pattern import from_list, attempts_along, size, to_list
from targets import factor_unpenalized, factor_penalized, dot_len

def contiguous(n): return list(range(n))
NODE_ROWS = [0, 8, 64, 72]; NODE_COLS = [0, 1, 8, 9, 32, 33, 40, 41]
MB_ROWS = [0, 8]; MB_COLS = [c for b in range(0, 256, 8) for c in (b, b + 1)]

CONFIGS = [
    # name, r, k, rows, cols, m, n
    ("node CPU default (post #275)", 128, 2048, NODE_ROWS, NODE_COLS, 256, 512),
    ("node CPU default (pre #275)", 32, 1024, NODE_ROWS, NODE_COLS, 256, 512),
    ("miner-base GPU, k=4096", 128, 4096, MB_ROWS, MB_COLS, 4096, 4096),
    ("hw=256 (16x16), r=256,k=4096 (mo-miner-like)", 256, 4096, contiguous(16), contiguous(16), 4096, 4096),
    ("hw=256, r=128, k=4096", 128, 4096, contiguous(16), contiguous(16), 4096, 4096),
    ("hw=32 (2x16), r=128, k=4096", 128, 4096, contiguous(2), contiguous(16), 4096, 4096),
    ("hw=256, r=1024, k=16384", 1024, 16384, contiguous(16), contiguous(16), 4096, 4096),
    ("hw=256, r=128, k=16384", 128, 16384, contiguous(16), contiguous(16), 4096, 4096),
    ("hw=256, r=128, k=65536", 128, 65536, contiguous(16), contiguous(16), 1024, 1024),
    ("hw=256, r=1024, k=65472 (k%r=960)", 1024, 65472, contiguous(16), contiguous(16), 1024, 1024),
    ("tiny matrices: hw=256, r=128, k=4096, m=n=16", 128, 4096, contiguous(16), contiguous(16), 17, 17),
]

def row(name, r, k, rows, cols, m, n):
    rp, cp = from_list(rows), from_list(cols)
    h, w = size(rp), size(cp)
    att = attempts_along(rp, m) * attempts_along(cp, n)
    d = dot_len(k, r)
    Fpre, Fpost = factor_unpenalized(h, w, k, r), factor_penalized(h, w, k, r)
    wins_pre, wins_post = att * Fpre, att * Fpost
    full_macs = m * n * k                       # (a) whole noisy GEMM, all k
    tile_macs = att * h * w * d                 # (b) only lottery tiles, only dot part
    sparse_overhead = 2 * m * n * r + 4 * (m + n) * k   # (c) denoise with sparse E_AR/E_BL (estimate)
    dense_overhead = 2 * m * k * r + 2 * n * k * r + 2 * m * n * r  # (d) dense torch ref (noisy_gemm.py)
    cells = att * h * w * (d // r)              # (e) transcript checkpoints (cuPOW lower-bound model)
    reductions = att * (d // r)                 # (f) tile-level XOR reductions (GPU sync points, per hash tile)
    hash_bytes = (m + n) * k                    # commitment hashing input
    return dict(name=name, r=r, k=k, h=h, w=w, m=m, n=n, att=att,
                Fpre=Fpre, Fpost=Fpost,
                pre_per_tileMAC=wins_pre / tile_macs, post_per_tileMAC=wins_post / tile_macs,
                pre_per_fullMAC=wins_pre / full_macs, post_per_fullMAC=wins_post / full_macs,
                pre_per_useful_sparse=wins_pre / (full_macs + sparse_overhead),
                post_per_useful_sparse=wins_post / (full_macs + sparse_overhead),
                pre_per_useful_dense=wins_pre / (full_macs + dense_overhead),
                post_per_useful_dense=wins_post / (full_macs + dense_overhead),
                pre_per_cell=wins_pre / cells, post_per_cell=wins_post / cells,
                macs_per_reduction=tile_macs / reductions, hash_bytes_per_tileMAC=hash_bytes / tile_macs)

if __name__ == "__main__":
    rows_ = [row(*c) for c in CONFIGS]
    print("| config | r | k | h×w | m×n | attempts | F_pre | F_post | pre/tileMAC | post/tileMAC | pre/fullMAC | post/fullMAC | pre/(MAC+sparse denoise) | post/(MAC+sparse denoise) | pre/(MAC+dense denoise) | post/(MAC+dense denoise) | pre/cell | post/cell | MACs per tile-reduction | hash bytes per tileMAC |")
    print("|" + "---|" * 21)
    for x in rows_:
        if x['r'] < 128:
            for key in list(x):
                if key.startswith('post_'): x[key] = float('nan')
        print(f"| {x['name']} | {x['r']} | {x['k']} | {x['h']}×{x['w']} | {x['m']}×{x['n']} | {x['att']} | {x['Fpre']} | {x['Fpost']} | "
              f"{x['pre_per_tileMAC']:.3f} | {x['post_per_tileMAC']:.3f} | {x['pre_per_fullMAC']:.3f} | {x['post_per_fullMAC']:.3f} | "
              f"{x['pre_per_useful_sparse']:.3f} | {x['post_per_useful_sparse']:.3f} | {x['pre_per_useful_dense']:.3f} | {x['post_per_useful_dense']:.3f} | {x['pre_per_cell']:.0f} | {x['post_per_cell']:.0f} | "
              f"{x['macs_per_reduction']:.0f} | {x['hash_bytes_per_tileMAC']:.4f} |")
