"""Experiment (component 3, items 7 & 9):
 (a) a tiny worked jackpot example h=2, w=4, k=64, r=32 using the exact Rust rules
     (NB: these dims violate sanity_checks.rs (k>=1024, k>=16r, h*w>=32) - illustration only);
 (b) emulation of the GPU TileHashAccumulator schedule (pow_utils.hpp:128-200 +
     collective_mainloop.hpp:270-311) versus helper.rs's schedule for many (bK, R, K),
     showing they write the same transcript word with the same accumulator snapshot.
"""
from pearl_ref import (header_bytes, mining_config_bytes, job_key, commitment_hash, compute_noise,
                       compute_jackpot, jackpot_hash, matrix_root, rotl32, M32, difficulty_bound)

out = []


def log(*a):
    s = " ".join(str(x) for x in a)
    print(s)
    out.append(s)


# ----------------------------------------------------------------------------- (a)
h, w, k, r = 2, 4, 64, 32
m, n = 16, 16
rows_pat, cols_pat = [0, 8], [0, 1, 8, 9]
t_rows, t_cols = 3, 2
A = [[((3 * i + 5 * l) % 128) - 64 for l in range(k)] for i in range(m)]          # m x k, int7
Bt = [[((7 * j + 11 * l + 1) % 128) - 64 for l in range(k)] for j in range(n)]     # n x k (= B^T)
hdr = header_bytes(1, bytes(range(32)), bytes(range(32, 64)), 1_700_000_000, 0x1d00ffff)
cfg = mining_config_bytes(k, r, rows_pat, cols_pat)
jk = job_key(hdr, cfg)
ha, hb = matrix_root(A, jk), matrix_root(Bt, jk)
b_seed, a_seed = commitment_hash(jk, ha, hb, m, n, salted=True)
log("# (a) worked example  h=2 w=4 k=64 r=32  (illustrative dims, not consensus-valid)")
log(f"job_key      = {jk.hex()}")
log(f"hash_a(root) = {ha.hex()}   (keyed blake3 of pad1024(A row-major), key=job_key)")
log(f"hash_b(root) = {hb.hex()}   (keyed blake3 of pad1024(B^T row-major), key=job_key)")
log(f"b_noise_seed = {b_seed.hex()}   (V3 salted)")
log(f"a_noise_seed = {a_seed.hex()}   (= jackpot key)")
rows = [t_rows + x for x in rows_pat]
cols = [t_cols + x for x in cols_pat]
log(f"tile rows of A = {rows}, tile cols of B = {cols}")
na, nb, (e_al, e_ar, e_bl, e_brt) = compute_noise(k, r, b_seed, a_seed, rows, cols)
log(f"E_AR pairs (first,second) for l=0..7: {[(f, s) for f, s, _ in e_ar[:8]]}")
log(f"E_AL[row {rows[0]}][0..8] = {e_al[0][:8]}")
log(f"noise_A[row {rows[0]}][0..8] = {na[0][:8]}   (= E_AL[u][first_l] - E_AL[u][second_l])")
sa = [A[i] for i in rows]
sb = [Bt[j] for j in cols]
trace = []
msg = compute_jackpot(sa, sb, na, nb, k, r, trace)
for t in trace:
    log(f"\n-- reduction at ll={t['ll']} (sum over l in [0,{t['ll']}))  -> word tid={t['tid']}")
    for u, row in enumerate(t['acc']):
        log(f"   acc[{u}] = {row}   as u32 hex = {[hex(v & M32) for v in row]}")
    log(f"   XOR of the {h * w} u32 = {hex(t['xor'])}")
    log(f"   msg[{t['tid']}]: {hex(t['before'])} -> rotl13 ^ xor = {hex(t['after'])}")
log(f"\njackpot_msg (16 x u32) = {[hex(x) for x in msg]}")
jh = jackpot_hash(msg, a_seed)
log(f"hash_jackpot = blake3(msg_le, key=a_noise_seed) = {jh.hex()}")
log(f"as LE uint256 = {int.from_bytes(jh, 'little'):#066x}")
log(f"difficulty factor h*w*(k-k%r) = {h * w * (k - k % r)}")

# What does / does not enter: verify that changing values outside the tile, or reordering
# products inside one r-block, leaves the jackpot unchanged; changing the r-block split does not.
A2 = [row[:] for row in A]
A2[0][0] += 1  # row 0 is not a tile row
log(f"\nchange A[0][0] (row not in tile) -> jackpot_msg unchanged on the same noise: "
    f"{compute_jackpot([A2[i] for i in rows], sb, na, nb, k, r) == msg}  (but hash_a and hence both seeds change)")
# permute l within the first r-block identically for A-row, B-col and noise:
perm = list(range(r))[::-1] + list(range(r, k))
p = lambda v: [v[i] for i in perm]
log(f"reverse l-order inside block [0,32) for all operands -> same jackpot_msg: "
    f"{compute_jackpot([p(x) for x in sa], [p(x) for x in sb], [p(x) for x in na], [p(x) for x in nb], k, r) == msg}")
# moving one l across the block boundary changes the intermediate snapshot
perm2 = list(range(k)); perm2[31], perm2[32] = perm2[32], perm2[31]
p2 = lambda v: [v[i] for i in perm2]
log(f"swap l=31 and l=32 (crosses the r boundary) -> same jackpot_msg: "
    f"{compute_jackpot([p2(x) for x in sa], [p2(x) for x in sb], [p2(x) for x in na], [p2(x) for x in nb], k, r) == msg}")

# word usage for a realistic k/r
log("\n# word usage: number of rotl13^xor folds per msg word")
for kk, rr in [(2048, 128), (4096, 128), (8192, 128), (2048, 32), (65536, 128)]:
    nred = kk // rr
    folds = [len([i for i in range(nred) if i % 16 == t]) for t in range(16)]
    log(f"k={kk:6d} r={rr:4d}: reductions={nred:4d} folds per word={folds}")

# ----------------------------------------------------------------------------- (b)
log("\n# (b) GPU TileHashAccumulator schedule emulation vs helper.rs")


def rust_schedule(K, R):
    # helper.rs:21-33: for ll in (r..=k).step_by(r): snapshot at ll, word (ll/r-1)%16
    return [(ll, (ll // R - 1) % 16) for ll in range(R, K + 1, R)]


def gpu_schedule(K, R, bK, atomK=32):
    k_blocks_per_tile = bK // atomK
    reduce_every_k = R // atomK
    accums = max(1, k_blocks_per_tile // reduce_every_k)
    last_full = K // atomK                     # collective_mainloop.hpp:270-271
    k_tile_count = -(-K // bK)
    transcript_pos = 0                         # m_reduction_count
    count = 0                                  # m_k_block_count
    events = []
    for k_tile in range(k_tile_count):
        base = transcript_pos                  # preload(transcript(m_reduction_count + i))
        for k_block in range(k_blocks_per_tile):
            count += 1                         # accumulate(): ++m_k_block_count
            if count % reduce_every_k == 0 and count <= last_full:
                idx = k_block // reduce_every_k
                events.append((count * atomK, (base + idx) % 16))
        # writeback + advance
        if (k_blocks_per_tile // reduce_every_k > 0) or (count % reduce_every_k == 0):
            transcript_pos = (transcript_pos + accums) % 16
    return events


bad = 0
cases = 0
for K in [1024, 1088, 2048, 4096, 6144, 8192, 65536]:
    for R in [32, 64, 128, 256, 512, 1024]:
        if K < 16 * R or K % 64:
            continue
        for bK in [64, 128, 256]:
            if 16 % max(1, (bK // 32) // (R // 32)):
                continue  # static_assert(MSG_BLOCK_SIZE_U32 % accums_per_tile == 0)
            cases += 1
            if rust_schedule(K, R) != gpu_schedule(K, R, bK):
                bad += 1
                log(f"MISMATCH K={K} R={R} bK={bK}")
log(f"compared {cases} (K,R,bK) combos: mismatches={bad}")
log("(K not a multiple of bK is included, e.g. K=1088 with bK=128/256: GPU loads a zero-padded last tile;"
    " reductions stop at count*32 <= K)")
# show one explicit schedule
log(f"example K=2048 R=128 bK=128: first 5 GPU events (k_prefix, word) = {gpu_schedule(2048, 128, 128)[:5]}")
log(f"example K=2048 R=256 bK=128: first 5 GPU events = {gpu_schedule(2048, 256, 128)[:5]}")
log(f"example K=2048 R=64  bK=128: first 5 GPU events = {gpu_schedule(2048, 64, 128)[:5]}")

with open("exp_jackpot_example.out", "w") as fh:
    fh.write("\n".join(out) + "\n")
