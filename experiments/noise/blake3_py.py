"""Minimal pure-Python BLAKE3 (hash / keyed_hash, arbitrary length, 32-byte output).

Written from the BLAKE3 spec so the experiments need no third-party packages.
Validated in test_vectors.py against the pinned vectors in
pearl/zk-pow/src/api/seed.rs (which were themselves computed with Python `blake3`).
"""

IV = [0x6A09E667, 0xBB67AE85, 0x3C6EF372, 0xA54FF53A,
      0x510E527F, 0x9B05688C, 0x1F83D9AB, 0x5BE0CD19]
MSG_PERMUTATION = [2, 6, 3, 10, 7, 0, 4, 13, 1, 11, 12, 5, 9, 14, 15, 8]
CHUNK_START, CHUNK_END, PARENT, ROOT, KEYED_HASH = 1, 2, 4, 8, 16
BLOCK_LEN, CHUNK_LEN = 64, 1024
M32 = 0xFFFFFFFF


def _rotr(x, n):
    return ((x >> n) | (x << (32 - n))) & M32


def _g(s, a, b, c, d, mx, my):
    s[a] = (s[a] + s[b] + mx) & M32
    s[d] = _rotr(s[d] ^ s[a], 16)
    s[c] = (s[c] + s[d]) & M32
    s[b] = _rotr(s[b] ^ s[c], 12)
    s[a] = (s[a] + s[b] + my) & M32
    s[d] = _rotr(s[d] ^ s[a], 8)
    s[c] = (s[c] + s[d]) & M32
    s[b] = _rotr(s[b] ^ s[c], 7)


def compress(cv, block_words, counter, block_len, flags):
    s = list(cv) + IV[:4] + [counter & M32, (counter >> 32) & M32, block_len, flags]
    m = list(block_words)
    for r in range(7):
        _g(s, 0, 4, 8, 12, m[0], m[1]); _g(s, 1, 5, 9, 13, m[2], m[3])
        _g(s, 2, 6, 10, 14, m[4], m[5]); _g(s, 3, 7, 11, 15, m[6], m[7])
        _g(s, 0, 5, 10, 15, m[8], m[9]); _g(s, 1, 6, 11, 12, m[10], m[11])
        _g(s, 2, 7, 8, 13, m[12], m[13]); _g(s, 3, 4, 9, 14, m[14], m[15])
        if r < 6:
            m = [m[i] for i in MSG_PERMUTATION]
    return [s[i] ^ s[i + 8] for i in range(8)]  # first 8 words = chaining value


def words(b):
    b = b + b"\x00" * (64 - len(b))
    return [int.from_bytes(b[4 * i:4 * i + 4], "little") for i in range(16)]


def cv_bytes(cv):
    return b"".join(w.to_bytes(4, "little") for w in cv)


def _chunk_state(key_words, data, chunk_index, base_flags):
    """Returns (cv_before_last_block, last_block_words, last_len, last_flags)."""
    cv = key_words
    blocks = [data[i:i + BLOCK_LEN] for i in range(0, max(len(data), 1), BLOCK_LEN)] or [b""]
    for i, blk in enumerate(blocks):
        flags = base_flags | (CHUNK_START if i == 0 else 0)
        if i == len(blocks) - 1:
            return cv, words(blk), len(blk), flags | CHUNK_END
        cv = compress(cv, words(blk), chunk_index, BLOCK_LEN, flags)


def blake3(data: bytes, key: bytes | None = None) -> bytes:
    if key is None:
        kw, bf = IV, 0
    else:
        assert len(key) == 32
        kw, bf = [int.from_bytes(key[4 * i:4 * i + 4], "little") for i in range(8)], KEYED_HASH
    nchunks = max(1, (len(data) + CHUNK_LEN - 1) // CHUNK_LEN)
    if nchunks == 1:
        cv, bw, bl, fl = _chunk_state(kw, data, 0, bf)
        return cv_bytes(compress(cv, bw, 0, bl, fl | ROOT))
    # chunk CVs (non-root)
    cvs = []
    for c in range(nchunks):
        cv, bw, bl, fl = _chunk_state(kw, data[c * CHUNK_LEN:(c + 1) * CHUNK_LEN], c, bf)
        cvs.append(compress(cv, bw, c, bl, fl))

    # left-complete binary tree (largest power of two on the left)
    def subtree(lo, hi, is_root):
        n = hi - lo
        if n == 1:
            return cvs[lo]
        left = 1 << ((n - 1).bit_length() - 1)
        l = subtree(lo, lo + left, False)
        r = subtree(lo + left, hi, False)
        return compress(kw, l + r, 0, BLOCK_LEN, bf | PARENT | (ROOT if is_root else 0))

    return cv_bytes(subtree(0, nchunks, True))
