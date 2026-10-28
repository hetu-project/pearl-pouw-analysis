"""Jackpot bound / rank penalty numerics.
Formulas copied from zk-pow/src/api/sanity_checks.rs:148-245 and proof_utils.rs:54-82,496-500.
Run: python3 -I targets.py
"""
from fractions import Fraction
PENALTY_BASE_RANK = 128
U256_MAX = (1 << 256) - 1

def nbits_to_target(nbits):
    e, mant = nbits >> 24, nbits & 0xFFFFFF
    if mant == 0 or e == 0 or mant & 0x800000: return 0
    return mant >> (8 * (3 - e)) if e <= 3 else mant << (8 * (e - 3))

def dot_len(k, r): return k - k % r
def factor_unpenalized(h, w, k, r): return h * w * dot_len(k, r)            # difficulty_adjustment_factor
def factor_penalized(h, w, k, r): return h * w * (dot_len(k, r) // r) * PENALTY_BASE_RANK  # penalized_adjustment_factor

def bound(T, f):  # consensus path saturates (scale), pool path returns None
    return U256_MAX if T > U256_MAX // f else T * f

if __name__ == "__main__":
    T = nbits_to_target(0x1b00ffff)
    print(f"mainnet PowLimitBits 0x1b00ffff -> target = {T:#x} ~ 2^{T.bit_length()-1}.{'' }")
    print(f"  per-MAC win prob at pow limit (pre-fork, any r) = T/2^256 = 2^{(T/2**256).__format__('.3e')}")
    print()
    # Table A: rank sweep, fixed tile h*w and k (k must satisfy 16r<=k<=4r^2, k>=1024, k<=65536)
    h, w = 4, 8
    print("Table A: h*w=32, common k chosen per row; columns are factors (bound = T*factor)")
    print("r | k | dot | F_pre=hw*dot | F_post=hw*(dot/r)*128 | F_post/F_pre | MACs/attempt=hw*dot | pre win/MAC (xT/2^256) | post win/MAC | transcript cells/attempt=hw*dot/r | pre win/cell | post win/cell | valid post?")
    for r in [32, 64, 128, 256, 512, 1024]:
        for k in sorted({max(1024, 16 * r), 16384 if 16*r <= 16384 <= 4*r*r else None} - {None}):
            if not (16 * r <= k <= 4 * r * r and k <= 65536): continue
            d = dot_len(k, r); fpre = factor_unpenalized(h, w, k, r); fpost = factor_penalized(h, w, k, r)
            macs = h * w * d; cells = h * w * d // r
            valid = r >= PENALTY_BASE_RANK
            print(f"{r} | {k} | {d} | {fpre} | {fpost} | {Fraction(fpost, fpre)} | {macs} | {Fraction(fpre, macs)} | "
                  f"{Fraction(fpost, macs) if valid else 'rejected'} | {cells} | {Fraction(fpre, cells)} | "
                  f"{Fraction(fpost, cells) if valid else 'rejected'} | {valid}")
    print()
    # Table B: the k range allowed for each r
    print("Table B: allowed k range per r (k%64==0, k>=1024, 16r<=k<=min(4r^2,2^16))")
    for r in [32, 64, 128, 256, 512, 1024]:
        lo, hi = max(1024, 16 * r), min(4 * r * r, 1 << 16)
        print(f"r={r}: k in [{lo}, {hi}]  chunks k/r in [{lo//r}, {hi//r}]")
    print()
    # Table C: k % r != 0 example
    print("Table C: k not a multiple of r (dot = k - k%r)")
    for k, r in [(2112, 128), (4160, 256), (65472, 1024), (4096, 128)]:
        print(f"k={k} r={r}: dot={dot_len(k,r)} unused tail={k%r} F_pre/hw={dot_len(k,r)} F_post/hw={(dot_len(k,r)//r)*128}")
    print()
    # Overflow / saturation behaviour for pool share targets
    print("Table D: largest target T with T*F <= U256_MAX, i.e. where saturation (consensus path) / None (pool path) begins")
    for F in [32 * 1024, 32 * 2048, 256 * 65536]:
        Tmax = U256_MAX // F
        print(f"F={F} (2^{F.bit_length()-1}): Tmax ~ 2^{Tmax.bit_length()}")
