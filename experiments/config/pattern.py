"""Re-implementation of zk-pow PeriodicPattern (zk-pow/src/api/proof_utils.rs:100-256)
for checking semantics numerically. Pure python, no deps.
Run: python3 -I pattern.py
"""
NUM_DIMS = 3

def from_list(p):
    assert p and all(a < b for a, b in zip(p, p[1:])) and p[0] == 0
    p = list(p); shape = []
    while len(p) > 1:
        for period in range(1, len(p)):
            if len(p) % period == 0:
                s = p[period]
                if all(p[i] + s == p[i + period] for i in range(len(p) - period)):
                    shape.append((s, len(p) // period)); p = p[:period]; break
        else:
            raise ValueError("Pattern is not periodic")
    shape.reverse()
    per = shape[-1][0] * shape[-1][1] if shape else 1
    while len(shape) < NUM_DIMS:
        shape.append((per, 1))
    if len(shape) > NUM_DIMS:
        raise ValueError("needs more than 3 dims")
    assert is_valid(shape), "Constructed pattern is not valid"
    return shape

def to_bytes(shape):
    out, ms = [], 1
    for s, l in shape:
        out += [s // ms - 1, l - 1]; ms = s * l
    return bytes(out)

def from_bytes(b):
    assert len(b) == 6
    shape, ms, done = [], 1, False
    for i in range(3):
        f, l = 1 + b[2*i], 1 + b[2*i+1]
        if l == 1 or done:
            if not (f == 1 and l == 1): raise ValueError("non-canonical")
            done = True
        elif f <= 1 and ms != 1:
            raise ValueError("A single stride must not be broken")
        if not ms <= (1 << 24) // (f * l): raise ValueError("period > 2^24")
        s = f * ms; shape.append((s, l)); ms = s * l
    if to_bytes(shape) != b: raise ValueError("roundtrip")
    return shape

def is_valid(shape):
    try: return from_bytes(to_bytes(shape)) == shape
    except Exception: return False

def to_list(shape):
    res = [0]
    for s, l in shape:
        res = [r + i * s for i in range(l) for r in res]
    return res

def offset_is_valid(shape, o):
    for s, l in reversed(shape):
        o %= s * l
        if o >= s: return False
    return True

def size(shape):
    n = 1
    for _, l in shape: n *= l
    return n

def period(shape):
    s, l = shape[-1]; return s * l

def tiling_check(shape):
    P = period(shape); cover = [0] * P
    offs = [o for o in range(P) if offset_is_valid(shape, o)]
    for o in offs:
        for i in to_list(shape):
            cover[(o + i) % P] += 1
    return len(offs), all(c == 1 for c in cover)

def attempts_along(shape, m):
    """number of valid offsets t with t + max(pattern) < m (sanity_checks.rs:57)."""
    mx = max(to_list(shape))
    return sum(1 for t in range(0, max(0, m - mx)) if offset_is_valid(shape, t))

if __name__ == "__main__":
    examples = {
        "node rows [0,8,64,72]": [0, 8, 64, 72],
        "node cols [0,1,8,9,32,33,40,41]": [0, 1, 8, 9, 32, 33, 40, 41],
        "miner-base rows [0,8]": [0, 8],
        "miner-base cols (64 elems)": [c for b in range(0, 256, 8) for c in (b, b + 1)],
        "contiguous 16": list(range(16)),
    }
    for name, lst in examples.items():
        sh = from_list(lst)
        n_off, ok = tiling_check(sh)
        print(f"{name}: shape={sh} bytes={to_bytes(sh).hex()} size={size(sh)} period={period(sh)} "
              f"valid_offsets_per_period={n_off} exact_partition={ok}")
    sh = from_list([0, 8, 64, 72])
    print("valid row offsets in [0,128):", [o for o in range(128) if offset_is_valid(sh, o)])
    print("attempts along m=256 for rows [0,8,64,72]:", attempts_along(sh, 256))
    sh = from_list([0, 1, 8, 9, 32, 33, 40, 41])
    print("attempts along n=512 for cols [0,1,8,9,32,33,40,41]:", attempts_along(sh, 512))
    # non-canonical / rejected encodings
    for b in [bytes([0,1,0,1,0,0]), bytes([7,1,0,0,3,1]), bytes([0,0,0,0,0,0])]:
        try: print(b.hex(), "->", from_bytes(b))
        except Exception as e: print(b.hex(), "-> rejected:", e)
