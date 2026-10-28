"""Pure-Python helpers for the Pearl FP8 whitepaper experiments (no numpy).

E4M3FN RNE rounding with saturation (Appendix C.1), BF16 RNE, and the
group-of-32 truncating accumulator of Appendix C.2.
NOTE: BLAKE3 is not installed here; experiments use hashlib.blake2b XOF-like
stand-in only as a uniform byte source (the distributional claims do not
depend on which PRF is used).
"""
import math
import struct
import hashlib
from fractions import Fraction


def fp8_values():
    """All finite non-negative E4M3FN values (codes 0x00..0x7E)."""
    vals = []
    for code in range(0x7F):
        e = (code >> 3) & 0xF
        m = code & 7
        if e == 0:
            v = m / 8 * 2 ** -6
        else:
            v = (1 + m / 8) * 2 ** (e - 7)
        vals.append(v)
    return vals  # 0x7F is NaN; 448 = 0x7E is max


FP8_POS = fp8_values()


def rne_fp8(x: float) -> float:
    """Round-to-nearest-even onto E4M3FN, saturating |x|>464 to 448 (C.1)."""
    s = -1.0 if x < 0 else 1.0
    a = abs(x)
    if a >= 464:  # tie at 464 rounds to 448 (even), above saturates
        return s * 448.0
    # binary search nearest
    lo, hi = 0, len(FP8_POS) - 1
    while hi - lo > 1:
        mid = (lo + hi) // 2
        if FP8_POS[mid] <= a:
            lo = mid
        else:
            hi = mid
    a_lo, a_hi = FP8_POS[lo], FP8_POS[hi]
    if a - a_lo < a_hi - a:
        r = a_lo
    elif a - a_lo > a_hi - a:
        r = a_hi
    else:  # tie -> even code
        r = a_lo if lo % 2 == 0 else a_hi
    return s * r


def rne_bf16(x: float) -> float:
    b = struct.unpack('>I', struct.pack('>f', x))[0]
    lsb = (b >> 16) & 1
    b = (b + 0x7FFF + lsb) & 0xFFFF0000
    return struct.unpack('>f', struct.pack('>I', b))[0]


def stored_exp_fp8(x: float) -> int:
    a = abs(x)
    if a < 2 ** -6:
        return -6
    return math.floor(math.log2(a))


def trunc(x: Fraction, u: Fraction) -> Fraction:
    """Truncate toward zero to a multiple of u."""
    q = x / u
    qi = math.floor(q) if q >= 0 else math.ceil(q)
    return qi * u


def group_accumulate(a, b, c: Fraction, W: int) -> Fraction:
    """Appendix C.2 pseudocode, exact rational arithmetic."""
    prods = [Fraction(x) * Fraction(y) for x, y in zip(a, b)]
    exps = [stored_exp_fp8(x) + stored_exp_fp8(y) for x, y in zip(a, b) if x != 0 and y != 0]
    if c != 0:
        exps.append(math.floor(math.log2(abs(c))) if abs(c) >= 2 ** -126 else -126)
    if not exps:
        return Fraction(0)
    eta = max(exps)
    u = Fraction(2) ** (eta - W)
    S = trunc(c, u) + sum(trunc(p, u) for p in prods)
    if S == 0:
        return Fraction(0)
    e = math.floor(math.log2(abs(S)))
    c = trunc(S, Fraction(2) ** (e - W))
    if W > 23:
        e = math.floor(math.log2(abs(c)))
        c = trunc(c, Fraction(2) ** (e - 23))
    return c


def matmul_cell(arow, brow, W: int):
    c = Fraction(0)
    partials = []
    for g in range(0, len(arow), 32):
        c = group_accumulate(arow[g:g + 32], brow[g:g + 32], c, W)
        partials.append(c)
    return c, partials


def prf_bytes(seed: bytes, msg: bytes, n: int) -> bytes:
    out = b''
    ctr = 0
    while len(out) < n:
        out += hashlib.blake2b(msg + ctr.to_bytes(4, 'little'), key=seed[:64]).digest()
        ctr += 1
    return out[:n]


def sample_line(seed: bytes, X: int, Z: int, l: int, r: int = 32):
    """Appendix C.3 (with blake2b standing in for BLAKE3-XOF)."""
    bs = prf_bytes(seed, bytes([X, Z]) + l.to_bytes(4, 'little'), r)
    xs = [(-1 if (b >> 7) else 1) * (1 + (b & 0x7F)) for b in bs]
    q = math.isqrt(32 * 32 * sum(x * x for x in xs))
    gamma = rne_bf16(rne_bf16(8192.0) / rne_bf16(float(q)))
    return [rne_fp8(rne_bf16(x * gamma)) for x in xs]
