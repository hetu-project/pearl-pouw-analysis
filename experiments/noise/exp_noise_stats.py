"""Experiment (component 2, item 5): statistics of Pearl noise, generated with the exact
Rust rules (pearl_ref.py, cross-checked byte-for-byte against zk-pow in xcheck_diff.out).

Parameters follow mainnet-like settings: r = 128 (PENALTY_BASE_RANK / miner default),
k = 2048 (= 16 r, the smallest k sanity checks allow for r=128; node CPU miner default).
Seeds are arbitrary 32-byte values blake3("exp-seed-<i>"); the noise rules do not depend
on how the seed was produced, only that it is a 32-byte key.

Only describes data; no evaluation.
"""
import collections
import math
import random
import sys

from blake3_py import blake3
from pearl_ref import (SEED_LABEL_A, SEED_LABEL_B, uniform_rows, perm_pairs, noise_rows)

R = 128
K = 2048
M_ROWS = 256           # rows of E_AL sampled for distribution / correlation stats
N_SEEDS_GRAPH = 30     # seeds for the cheap (pairs-only) statistics
P = 2**31 - 1          # prime for rank computations


def seed(i):
    return blake3(f"exp-seed-{i}".encode())


def rank_mod_p(mat):
    m = [[x % P for x in row] for row in mat]
    rank, ncols = 0, len(m[0])
    rows = len(m)
    for c in range(ncols):
        piv = next((i for i in range(rank, rows) if m[i][c]), None)
        if piv is None:
            continue
        m[rank], m[piv] = m[piv], m[rank]
        inv = pow(m[rank][c], P - 2, P)
        prow = [(x * inv) % P for x in m[rank]]
        m[rank] = prow
        for i in range(rows):
            if i != rank and m[i][c]:
                f = m[i][c]
                m[i] = [(a - f * b) % P for a, b in zip(m[i], prow)]
        rank += 1
        if rank == rows:
            break
    return rank


def components(pairs, r):
    parent = list(range(r))

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x
    for f, s, _ in pairs:
        a, b = find(f), find(s)
        if a != b:
            parent[a] = b
    return len({find(i) for i in range(r)})


def pearson(x, y):
    n = len(x)
    mx, my = sum(x) / n, sum(y) / n
    sxy = sum((a - mx) * (b - my) for a, b in zip(x, y))
    sxx = sum((a - mx) ** 2 for a in x)
    syy = sum((b - my) ** 2 for b in y)
    return sxy / math.sqrt(sxx * syy)


out = []


def log(*a):
    s = " ".join(str(x) for x in a)
    print(s)
    out.append(s)
    sys.stdout.flush()


log(f"# Pearl noise statistics  r={R} k={K} rows_sampled={M_ROWS}")

# ------------------------------------------------------------------ 1. pairs (E_AR / E_BL)
log("\n## 1. sparse +-1 pairs (first, second), pooled over", N_SEEDS_GRAPH, "seeds x", K, "columns")
first_c, delta_c, second_c = collections.Counter(), collections.Counter(), collections.Counter()
joint = collections.Counter()
dup_same, dup_swap, comps, max_deg, min_deg = [], [], [], [], []
for i in range(N_SEEDS_GRAPH):
    pairs = perm_pairs(SEED_LABEL_A, seed(i), K, R)
    deg = collections.Counter()
    seen = collections.Counter((f, s) for f, s, _ in pairs)
    dup_same.append(sum(c * (c - 1) // 2 for c in seen.values()))
    dup_swap.append(sum(seen[(f, s)] * seen[(s, f)] for (f, s) in seen if f < s))
    for f, s, u in pairs:
        assert f != s and 0 <= s < R
        first_c[f] += 1
        second_c[s] += 1
        d = f ^ s
        delta_c[d] += 1
        joint[(f // 16, d // 16)] += 1
        deg[f] += 1
        deg[s] += 1
    comps.append(components(pairs, R))
    max_deg.append(max(deg.values()))
    min_deg.append(min(deg[j] for j in range(R)))
tot = N_SEEDS_GRAPH * K
log(f"first: min/max count over {R} values = {min(first_c.values())}/{max(first_c.values())} (uniform expectation {tot / R:.1f})")
log(f"second: min/max count = {min(second_c.values())}/{max(second_c.values())}")
log(f"delta = first^second: support = [{min(delta_c)}, {max(delta_c)}], #values={len(delta_c)}, "
    f"min/max count = {min(delta_c.values())}/{max(delta_c.values())} (expectation {tot / (R - 1):.1f})")
chi = 0.0
fr = collections.Counter(); dr = collections.Counter()
for (a, b), c in joint.items():
    fr[a] += c; dr[b] += c
for a in fr:
    for b in dr:
        e = fr[a] * dr[b] / tot
        chi += (joint[(a, b)] - e) ** 2 / e
log(f"independence chi^2 of (first//16, delta//16) 8x8 table = {chi:.1f} (dof 49)")
log(f"identical columns (same ordered pair) per seed: mean={sum(dup_same) / len(dup_same):.1f} "
    f"(birthday estimate k^2/(2 r (r-1)) = {K * K / (2 * R * (R - 1)):.1f})")
log(f"negated columns (pair (f,s) and (s,f)) per seed: mean={sum(dup_swap) / len(dup_swap):.1f}")
log(f"vertex degree (#columns touching an index) min over seeds={min(min_deg)}, max={max(max_deg)}, "
    f"mean=2k/r={2 * K / R:.0f}")
log(f"connected components of the pair graph on r vertices: {collections.Counter(comps)}")
log(f"=> rank(E_AR) over Q = r - #components (graph incidence matrix) : {collections.Counter(R - c for c in comps)}")

# exact mapping check: delta - 1 = floor((r-1) u / 2^32); first = u mod r
pairs = perm_pairs(SEED_LABEL_A, seed(0), K, R)
lo_hi_same = sum(1 for f, s, u in pairs if (f ^ s) - 1 == ((R - 1) * u) >> 32)
log(f"check delta-1 == floor((r-1)*u/2^32) on seed0: {lo_hi_same}/{K}")

# ------------------------------------------------------------------ 2. dense values and noise values
log("\n## 2. value distributions (seed 0, a-side)")
s0 = seed(0)
cache = {}
e_al = uniform_rows(SEED_LABEL_A, s0, list(range(M_ROWS)), R, cache)
vals = collections.Counter(v for row in e_al for v in row)
log(f"E_AL values: support [{min(vals)}, {max(vals)}], #distinct={len(vals)}, "
    f"min/max count={min(vals.values())}/{max(vals.values())} (expect {M_ROWS * R / 64:.0f} each)")
e_ar = perm_pairs(SEED_LABEL_A, s0, K, R)
e_a = noise_rows(e_al, e_ar)
nv = collections.Counter(v for row in e_a for v in row)
N = M_ROWS * K
log(f"E_A = E_AL*E_AR values: support [{min(nv)}, {max(nv)}], #distinct={len(nv)}")
mean = sum(v * c for v, c in nv.items()) / N
var = sum((v - mean) ** 2 * c for v, c in nv.items()) / N
log(f"  mean={mean:.3f} var={var:.1f} (iid-difference model: 0, 2*(64^2-1)/12={2 * (64 ** 2 - 1) / 12:.1f})")
log("  value : empirical freq  vs triangular (64-|d|)/4096")
tv = 0.0
for d in range(-63, 64):
    emp = nv.get(d, 0) / N
    th = (64 - abs(d)) / 4096
    tv += abs(emp - th)
    if d in (-63, -62, -32, -1, 0, 1, 32, 62, 63):
        log(f"  {d:4d} : {emp:.5f} vs {th:.5f}")
log(f"  total-variation distance to triangular = {tv / 2:.4f}")
log(f"  P(E_A == 0) = {nv.get(0, 0) / N:.5f}")

# ------------------------------------------------------------------ 3. ranks
log("\n## 3. ranks (mod p = 2^31-1, equals rank over Q with high probability)")
k_small = 512
e_ar_small = e_ar[:k_small]
EAR = [[0] * k_small for _ in range(R)]
for l, (f, s, _) in enumerate(e_ar_small):
    EAR[f][l] += 1
    EAR[s][l] -= 1
log(f"column sums of E_AR (all should be 0): {set(sum(EAR[i][l] for i in range(R)) for l in range(k_small))}")
log(f"rank(E_AL) [{M_ROWS}x{R}] = {rank_mod_p(e_al)}")
log(f"rank(E_AR) [{R}x{k_small}] = {rank_mod_p(EAR)}")
EA_small = [row[:k_small] for row in e_a[:192]]
log(f"rank(E_A)  [192x{k_small}] = {rank_mod_p(EA_small)}")
log(f"note: 1^T E_AR = 0 (every column is e_f - e_s), so rows of E_A lie in the ({R}-1)-dim row space of E_AR")

# ------------------------------------------------------------------ 4. correlations
log("\n## 4. correlations (seed 0, E_A with 256 rows x 2048 cols)")
rng = random.Random(1)
cols = list(zip(*e_a))  # column vectors over 256 rows
shared_pos, shared_neg, ident, disjoint = [], [], [], []
for _ in range(4000):
    a, b = rng.sample(range(K), 2)
    fa, sa_, _ = e_ar[a]
    fb, sb_, _ = e_ar[b]
    c = pearson(cols[a], cols[b])
    if (fa, sa_) == (fb, sb_):
        ident.append(c)
    elif fa == fb or sa_ == sb_:
        shared_pos.append(c)
    elif fa == sb_ or sa_ == fb:
        shared_neg.append(c)
    else:
        disjoint.append(c)


def summ(xs):
    if not xs:
        return "n=0"
    mu = sum(xs) / len(xs)
    sd = math.sqrt(sum((x - mu) ** 2 for x in xs) / len(xs))
    return f"n={len(xs)} mean={mu:+.3f} sd={sd:.3f} min={min(xs):+.3f} max={max(xs):+.3f}"


log("column-column Pearson over the 256 rows, 4000 random column pairs:")
log(f"  same index set (identical pair)        : {summ(ident)}")
log(f"  share an index with same sign          : {summ(shared_pos)}")
log(f"  share an index with opposite sign      : {summ(shared_neg)}")
log(f"  disjoint index pairs                   : {summ(disjoint)}")
log(f"  fraction of random pairs sharing an index: {(len(ident) + len(shared_pos) + len(shared_neg)) / 4000:.4f} "
    f"(approx 4/r = {4 / R:.4f})")
# exhaustive count of column pairs sharing an index
inc = collections.defaultdict(list)
for l, (f, s, _) in enumerate(e_ar):
    inc[f].append(l); inc[s].append(l)
share_pairs = sum(len(v) * (len(v) - 1) // 2 for v in inc.values())
log(f"  exhaustive: column pairs sharing >=1 index = {share_pairs} of {K * (K - 1) // 2} "
    f"({share_pairs / (K * (K - 1) / 2):.4f})")
row_c = [pearson(e_a[a], e_a[b]) for a, b in (rng.sample(range(M_ROWS), 2) for _ in range(300))]
log(f"row-row Pearson over 2048 columns, 300 random row pairs: {summ(row_c)} (1/sqrt(k)={1 / math.sqrt(K):.3f}, 1/sqrt(r)={1 / math.sqrt(R):.3f})")
# within one row: lag correlation between neighbouring columns
lag = [pearson(e_a[u][:-1], e_a[u][1:]) for u in range(32)]
log(f"within-row lag-1 correlation (32 rows): {summ(lag)}")

# ------------------------------------------------------------------ 5. B side quick check
log("\n## 5. B side (seed 0): E_B^T rows = E_BR^T[col][f_l] - E_BR^T[col][s_l]")
e_bl = perm_pairs(SEED_LABEL_B, s0, K, R)
e_brt = uniform_rows(SEED_LABEL_B, s0, list(range(64)), R)
e_bt = noise_rows(e_brt, e_bl)
bv = collections.Counter(v for row in e_bt for v in row)
log(f"E_B values support [{min(bv)}, {max(bv)}], P(0)={bv.get(0, 0) / (64 * K):.5f}")
log(f"E_AR pairs vs E_BL pairs identical positions (seed0, same key): "
    f"{sum(1 for x, y in zip(e_ar, e_bl) if x[:2] == y[:2])}/{K} (labels differ: A_tensor vs B_tensor)")

# ------------------------------------------------------------------ 6. A + E range
log("\n## 6. A + E_A range with int7 data")
for lo, hi, label in [(-64, 63, "miner int7 [-64,63] (noisy_gemm.py:156-164)"),
                      (-64, 64, "verifier-accepted [-64,64] (verify.rs:113-118, mine.rs:15-16)")]:
    mn = min(lo + min(nv), lo + max(nv)); mx = max(hi + max(nv), hi + min(nv))
    rng2 = random.Random(7)
    smin, smax = 10**9, -10**9
    for row in e_a[:64]:
        for e in row:
            a = rng2.randint(lo, hi)
            smin = min(smin, a + e); smax = max(smax, a + e)
    worst_lo = min(lo + e for row in e_a for e in row)
    worst_hi = max(hi + e for row in e_a for e in row)
    log(f"{label}: analytic range [{lo - 63}, {hi + 63}], adversarial(A=const extreme) observed "
        f"[{worst_lo}, {worst_hi}], random-A observed [{smin}, {smax}]; int8 = [-128, 127]")

with open("exp_noise_stats.out", "w") as fh:
    fh.write("\n".join(out) + "\n")
