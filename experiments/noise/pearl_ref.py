"""Pure-Python re-implementation of Pearl's seed chain, noise generation and jackpot,
written line-by-line from the Rust reference (commit 156ef9a8):

  zk-pow/src/api/proof_utils.rs   (job_key, commitment_hash, serialization, compute_jackpot_hash)
  zk-pow/src/api/seed.rs          (V3 salted bind_root_a / bind_root_b)
  zk-pow/src/circuit/pearl_noise.rs (get_random_hash, uniform / permutation matrices, noise)
  zk-pow/src/circuit/chip/jackpot/helper.rs (compute_jackpot)
  zk-pow/src/api/sanity_checks.rs (difficulty bound)
"""
from blake3_py import blake3

# ---------------------------------------------------------------- constants (pearl_noise.rs:13-30)
NOISE_RANGE = 128
IDXS_PER_COL = 2
UNIFORM_NOISE_RANGE = NOISE_RANGE // IDXS_PER_COL      # 64
ZERO_POINT_TRANSLATION = UNIFORM_NOISE_RANGE // 2       # 32
RANGE_MASK = UNIFORM_NOISE_RANGE - 1                    # 63
SEED_LABEL_A = b"A_tensor" + b"\x00" * 24
SEED_LABEL_B = b"B_tensor" + b"\x00" * 24
JACKPOT_SIZE = 16
LROT_PER_TILE = 13
M32 = 0xFFFFFFFF

SEED_SALT_A = blake3(b"pearl/cert-v3/noise-seed/A")
SEED_SALT_B = blake3(b"pearl/cert-v3/noise-seed/B")


# ---------------------------------------------------------------- serialization
def header_bytes(version, prev_block, merkle_root, timestamp, nbits):
    # proof_utils.rs:519-527 ; hashes are byte-reversed
    return (version.to_bytes(4, "little") + prev_block[::-1] + merkle_root[::-1]
            + timestamp.to_bytes(4, "little") + nbits.to_bytes(4, "little"))


def pattern_from_list(lst):
    """PeriodicPattern::from_list (proof_utils.rs:177-220) -> list of (stride, length)."""
    p = list(lst)
    shape = []
    while len(p) > 1:
        for period in range(1, len(p)):
            if len(p) % period == 0:
                s = p[period]
                if all(p[i] + s == p[i + period] for i in range(len(p) - period)):
                    shape.append((s, len(p) // period))
                    p = p[:period]
                    break
        else:
            raise ValueError("not periodic")
    shape.reverse()
    per = shape[-1][0] * shape[-1][1] if shape else 1
    while len(shape) < 3:
        shape.append((per, 1))
    return shape


def pattern_to_bytes(shape):
    # proof_utils.rs:143-155
    out, min_stride = b"", 1
    for stride, length in shape:
        out += bytes([stride // min_stride - 1, length - 1])
        min_stride = stride * length
    return out


def pattern_to_list(shape):
    res = [0]
    for stride, length in shape:
        res = [r + i * stride for i in range(length) for r in res]
    return res


def offset_is_valid(shape, offset):
    # proof_utils.rs:223-231
    for stride, length in reversed(shape):
        offset %= stride * length
        if offset >= stride:
            return False
    return True


def mining_config_bytes(common_dim, rank, rows_list, cols_list, moe=None):
    # proof_utils.rs:448-457 ; 4+2+2+6+6+32 = 52 bytes
    trailer = bytearray(32)
    if moe:
        e, top_k = moe
        trailer[0:2] = e.to_bytes(2, "little"); trailer[2:4] = top_k.to_bytes(2, "little")
    return (common_dim.to_bytes(4, "little") + rank.to_bytes(2, "little") + (0).to_bytes(2, "little")
            + pattern_to_bytes(pattern_from_list(rows_list)) + pattern_to_bytes(pattern_from_list(cols_list))
            + bytes(trailer))


def job_key(hdr: bytes, cfg: bytes) -> bytes:
    assert len(hdr) == 76 and len(cfg) == 52
    return blake3(hdr + cfg)          # proof_utils.rs:348-353 (unkeyed, 128 bytes = 2 blocks)


# ---------------------------------------------------------------- commitments
def pad1024(b: bytes) -> bytes:
    return b + b"\x00" * ((-len(b)) % 1024)


def matrix_root(rows, key):
    """keyed BLAKE3 over the row-major int8 bytes, zero padded to a 1024 multiple."""
    raw = bytes((v & 0xFF) for row in rows for v in row)
    return blake3(pad1024(raw), key)


def bind_root(root, dim, salt):
    # seed.rs:41-59
    return blake3(root + dim.to_bytes(4, "little") + b"\x00" * 28, salt)


def commitment_hash(jk, hash_a, hash_b, m, n, salted: bool):
    # proof_utils.rs:362-373 (dense case)
    if salted:
        hash_a = bind_root(hash_a, m, SEED_SALT_A)
        hash_b = bind_root(hash_b, n, SEED_SALT_B)
    b_seed = blake3(jk + hash_b)
    a_seed = blake3(b_seed + hash_a)
    return b_seed, a_seed


# ---------------------------------------------------------------- noise (pearl_noise.rs)
def get_random_hash(index, seed, key, prepend_index):
    msg = bytearray(64)
    msg[prepend_index * 4: prepend_index * 4 + 4] = ((1 + index) & M32).to_bytes(4, "little")
    msg[32:64] = seed
    return blake3(bytes(msg), key)


def uniform_rows(seed, key, row_indices, num_cols, cache=None):
    """generate_uniform_random_matrix (pearl_noise.rs:61-80): value = (byte & 63) - 32."""
    out = []
    for row in row_indices:
        start = row * num_cols
        vals = []
        for blk in range(start // 32, (start + num_cols + 31) // 32):
            h = cache.get(blk) if cache is not None else None
            if h is None:
                h = get_random_hash(blk, seed, key, 0)
                if cache is not None:
                    cache[blk] = h
            for kk, byte in enumerate(h):
                idx = blk * 32 + kk
                if start <= idx < start + num_cols:
                    vals.append((byte & RANGE_MASK) - ZERO_POINT_TRANSLATION)
        out.append(vals)
    return out


def mul_hi_u32(a, b):
    return ((a * b) >> 32) & M32


def perm_pairs(seed, key, k, r):
    """generate_permutation_matrix (pearl_noise.rs:90-116): k pairs (first, second)."""
    res = []
    for i in range((k + 7) // 8):
        h = get_random_hash(i, seed, key, 1)
        for j in range(8):
            if len(res) == k:
                break
            u = int.from_bytes(h[4 * j:4 * j + 4], "little")
            first = u & (r - 1)
            second = first ^ (1 + mul_hi_u32(r - 1, u))
            res.append((first, second, u))
    return res


def noise_rows(e_l_rows, pairs):
    """matvec_sparse_perm: out[l] = row[first_l] - row[second_l]."""
    return [[row[f] - row[s] for (f, s, _u) in pairs] for row in e_l_rows]


def compute_noise(k, r, b_seed, a_seed, a_rows, b_cols):
    e_al = uniform_rows(SEED_LABEL_A, a_seed, a_rows, r)
    e_ar_t = perm_pairs(SEED_LABEL_A, a_seed, k, r)
    e_bl = perm_pairs(SEED_LABEL_B, b_seed, k, r)
    e_br_t = uniform_rows(SEED_LABEL_B, b_seed, b_cols, r)
    return noise_rows(e_al, e_ar_t), noise_rows(e_br_t, e_bl), (e_al, e_ar_t, e_bl, e_br_t)


# ---------------------------------------------------------------- jackpot (helper.rs:9-35)
def rotl32(x, n):
    return ((x << n) | (x >> (32 - n))) & M32


def compute_jackpot(sa, sb, na, nb, k, r, trace=None):
    h, w = len(sa), len(sb)
    acc = [[0] * w for _ in range(h)]
    msg = [0] * JACKPOT_SIZE
    for ll in range(r, k + 1, r):
        for u in range(h):
            for v in range(w):
                for l in range(ll - r, ll):
                    acc[u][v] += (sa[u][l] + na[u][l]) * (sb[v][l] + nb[v][l])
                # Rust: i32 (wrapping in release) ; keep in i32 range for reporting
                acc[u][v] = ((acc[u][v] + 2**31) % 2**32) - 2**31
        x = 0
        for row in acc:
            for val in row:
                x ^= val & M32
        tid = (ll // r - 1) % JACKPOT_SIZE
        before = msg[tid]
        msg[tid] = rotl32(msg[tid], LROT_PER_TILE) ^ x
        if trace is not None:
            trace.append(dict(ll=ll, acc=[row[:] for row in acc], xor=x, tid=tid, before=before, after=msg[tid]))
    return msg


def jackpot_hash(msg, a_seed):
    # proof_utils.rs:1502-1505
    return blake3(b"".join(w.to_bytes(4, "little") for w in msg), a_seed)


# ---------------------------------------------------------------- difficulty (sanity_checks.rs)
def nbits_to_target(nbits):
    exp, mant = nbits >> 24, nbits & 0xFFFFFF
    if mant == 0 or exp == 0:
        return 0
    return mant * 256 ** (exp - 3) if exp >= 3 else mant >> (8 * (3 - exp))


def difficulty_bound(nbits, h, w, k, r):
    f = h * w * (k - k % r)
    t = nbits_to_target(nbits)
    return min(t * f, 2**256 - 1)
