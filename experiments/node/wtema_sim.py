#!/usr/bin/env python3
"""
Pearl WTEMA difficulty simulation (component 6, node/consensus).

Faithfully reproduces node/blockchain/difficulty.go:calcNextRequiredDifficulty
  new_target = old_target + (t - T) * old_target / halfLife   (big.Int, Go truncating division)
  clamp to powLimit, BigToCompact (truncating 23-bit mantissa)
  t = lastNode.Timestamp - parent.Timestamp  (one-block lag: the target for block n+1
      uses the solve-time of block n)
plus the timestamp rules:
  ts_n >= ts_{n-1} + 1        (MinTimestampDeltaSeconds, validate.go:26 / :690)
  ts_n <= wallclock + 5 min   (MaxTimeOffsetMinutes = 5, params.go:362)

Hashrate model: H = hashrate measured in "genesis-difficulty units", i.e. at H=1 and
target=powLimit the expected solve time is exactly T. Expected solve time
  = T * (powLimit/target) / H.
Mining is memoryless (exponential solve times).

Usage: python3 -I wtema_sim.py   (pure stdlib)
"""
import math
import random

T = 3 * 60 + 14            # 194 s, params.go:357
HL = 168 * 3600            # 604800 s, params.go:358
POW_LIMIT = (1 << 208) - 1  # params.go:29
POW_LIMIT_BITS = 0x1b00ffff


# ---------- exact compact encoding (node/blockchain/internal/workmath) ----------
def compact_to_big(c):
    mant = c & 0x007fffff
    neg = c & 0x00800000
    exp = c >> 24
    if exp <= 3:
        n = mant >> (8 * (3 - exp))
    else:
        n = mant << (8 * (exp - 3))
    return -n if neg else n


def big_to_compact(n):
    if n == 0:
        return 0
    exp = (n.bit_length() + 7) // 8
    if exp <= 3:
        mant = n << (8 * (3 - exp))
    else:
        mant = n >> (8 * (exp - 3))
    if mant & 0x00800000:
        mant >>= 8
        exp += 1
    return (exp << 24) | mant


def go_div(a, b):
    """Go big.Int.Div is Euclidean; for positive divisor it equals floor for a>=0,
    and for negative a it rounds toward -inf when remainder != 0 (Euclidean: r>=0).
    big.Int.Div implements Euclidean division (q = floor(a/b) for b>0)."""
    return a // b  # b > 0 => Euclidean == floor


def next_bits(prev_bits, t):
    old = compact_to_big(prev_bits)
    adj = go_div((t - T) * old, HL)
    new = old + adj
    if new > POW_LIMIT:
        new = POW_LIMIT
    if new <= 0:
        new = 1
    return big_to_compact(new)


def rel_diff(bits):
    return POW_LIMIT / compact_to_big(bits)


# ---------- generic simulator ----------
def simulate(hashrate_fn, start_bits, n_max=None, wall_max=None, seed=1,
             start_t_prev=T, record_every=1, stop_fn=None):
    """hashrate_fn(wall_seconds) -> H. Returns list of (height, wall, ts, bits, solve_t)."""
    rng = random.Random(seed)
    bits = start_bits
    wall = 0.0
    ts = 0
    t_prev = start_t_prev  # solve time of the "previous" block, used for first retarget
    out = []
    h = 0
    while True:
        h += 1
        # target for this block derives from the previous block's solve time
        if h > 1:
            bits = next_bits(bits, t_prev)
        H = hashrate_fn(wall)
        mean = T * rel_diff(bits) / H
        found = wall + rng.expovariate(1.0 / mean)
        ts_new = max(ts + 1, int(found))
        # cannot be more than 5 min ahead of wall clock: wait
        if ts_new > found + 300:
            found = ts_new - 300
        solve = ts_new - ts
        wall, ts, t_prev = found, ts_new, solve
        if h % record_every == 0:
            out.append((h, wall, ts, bits, solve))
        if n_max and h >= n_max:
            break
        if wall_max and wall >= wall_max:
            break
        if stop_fn and stop_fn(h, wall, bits):
            break
    return out


def equilibrium_bits(H):
    tgt = int(POW_LIMIT / H)
    return big_to_compact(tgt)


# ---------- analytic (continuous) logistic model ----------
# x = target/target_eq = H/D_rel; per block d ln(target) = (t-T)/HL, t = T/x
# => dx/dwall = x(1-x)/HL  (logistic, wall-clock time constant HL)
def logistic_x(x0, w):
    return 1.0 / (1.0 + (1.0 / x0 - 1.0) * math.exp(-w / HL))


def logistic_blocks(x0, w):
    # blocks = (1/T) * integral_0^w x dw = (HL/T) ln(1 + x0 (e^{w/HL} - 1))
    return (HL / T) * math.log(1 + x0 * (math.exp(w / HL) - 1))


def wall_to_reach(x0, x_target):
    # solve logistic for w
    # 1/x - 1 = (1/x0 - 1) e^{-w/HL}
    return -HL * math.log((1 / x_target - 1) / (1 / x0 - 1))


def rolling_mean(vals, k):
    out = []
    s = 0
    for i, v in enumerate(vals):
        s += v
        if i >= k:
            s -= vals[i - k]
        out.append(s / min(i + 1, k))
    return out


DAY = 86400


def scenario_drop():
    print("=" * 78)
    print("Scenario (a): sudden hashrate drop from equilibrium")
    print("=" * 78)
    H0 = 1e6
    b0 = equilibrium_bits(H0)
    for frac in (0.5, 0.1):
        x0 = frac  # after the drop, target is too hard: x = H/D = frac
        print(f"\n-- hashrate drops to {frac*100:.0f}% (i.e. -{(1-frac)*100:.0f}%) --")
        print(f"   initial expected block time = {T/frac:.0f} s ({T/frac/60:.1f} min)")
        for xt in (0.8, 0.9, 0.95, 0.99):
            w = wall_to_reach(x0, xt)
            n = logistic_blocks(x0, w)
            print(f"   analytic: block time within {100*(1/xt-1):5.1f}% of T (x={xt}): "
                  f"{w/DAY:6.1f} days, {n:7.0f} blocks")
        # Monte Carlo, exact integer arithmetic
        reach = {0.9: [], 0.99: []}
        for seed in range(20):
            res = simulate(lambda w: H0 * frac, b0, wall_max=120 * DAY, seed=seed)
            # expected block time from difficulty (removes sampling noise)
            for xt in reach:
                for (h, wall, ts, bits, s) in res:
                    if (H0 * frac) / rel_diff(bits) >= xt:
                        reach[xt].append((h, wall))
                        break
        for xt, lst in reach.items():
            hs = sorted(a for a, _ in lst)
            ws = sorted(b for _, b in lst)
            print(f"   MonteCarlo(20 runs) x>={xt}: median {hs[len(hs)//2]:.0f} blocks, "
                  f"{ws[len(ws)//2]/DAY:.1f} days (range {ws[0]/DAY:.1f}-{ws[-1]/DAY:.1f} d)")
        # blocks lost vs schedule during first 30 days
        w30 = 30 * DAY
        print(f"   in first 30 days: analytic {logistic_blocks(x0, w30):.0f} blocks "
              f"vs schedule {w30/T:.0f}")


def scenario_startup():
    print()
    print("=" * 78)
    print("Scenario (b): launch at genesis difficulty (bits=0x1b00ffff = powLimit)")
    print("=" * 78)
    # public data: ~40000 blocks in 5.9 days, ~13 s/block in the first week
    target_blocks, target_days = 40000, 5.9
    w = target_days * DAY
    # Fit constant hashrate multiple x0 (analytic, ignoring 1-s floor)
    lo, hi = 1.0, 1e9
    for _ in range(200):
        mid = math.sqrt(lo * hi)
        if logistic_blocks(mid, w) < target_blocks:
            lo = mid
        else:
            hi = mid
    x0_fit = lo
    print(f"\nAnalytic fit (constant hashrate from t=0): H/H_genesis = {x0_fit:,.0f}")
    print(f"   implied block time at day 5.9: {T/logistic_x(x0_fit, w):.0f} s")
    print(f"   min-difficulty-rise bound: each block can raise difficulty by at most "
          f"factor 1/(1-(T-1)/HL) = {1/(1-(T-1)/HL):.6f}")
    print(f"   => catching up a factor X takes >= ln(X)*HL/(T-1) = {HL/(T-1):.0f}*ln(X) blocks")
    print(f"      e.g. X=1e5 -> >= {math.log(1e5)*HL/(T-1):,.0f} blocks")

    print("\nMonte Carlo (exact integer WTEMA + 1s/5min timestamp rules):")
    print(f"{'H/H_gen':>10} {'blocks@5.9d':>12} {'avg s/blk wk1':>14} {'blocktime@5.9d':>15} "
          f"{'blocks@30d':>11} {'days to x<=1.1':>15}")
    rows = []
    for x0 in (1e2, 1e3, 1e4, 3e4, 1e5, 3e5, 1e6):
        res = simulate(lambda _w, x0=x0: x0, POW_LIMIT_BITS, wall_max=60 * DAY, seed=7)
        b59 = max(h for (h, wall, *_r) in res if wall <= 5.9 * DAY)
        b7 = max(h for (h, wall, *_r) in res if wall <= 7 * DAY)
        b30 = max(h for (h, wall, *_r) in res if wall <= 30 * DAY)
        bt = None
        for (h, wall, ts, bits, s) in res:
            if wall >= 5.9 * DAY:
                bt = T * rel_diff(bits) / x0
                break
        dcatch = None
        for (h, wall, ts, bits, s) in res:
            if x0 / rel_diff(bits) <= 1.1:
                dcatch = wall / DAY
                break
        rows.append((x0, b59, 7 * DAY / b7, bt, b30, dcatch))
        print(f"{x0:>10.0e} {b59:>12,} {7*DAY/b7:>14.1f} {bt:>15.0f} {b30:>11,} "
              f"{(f'{dcatch:.1f}' if dcatch else '>60'):>15}")

    # Growth ramp: hashrate grows exponentially from H_start to H_end over D days
    print("\nRamp scenarios (hashrate grows geometrically, then flat):")
    print(f"{'H_start':>8} {'H_end':>8} {'ramp d':>7} {'blocks@5.9d':>12} {'blocks@30d':>11} "
          f"{'blocks@87d':>11}")
    for hs, he, rd in ((1e2, 3e5, 3), (1e3, 1e6, 7), (1e4, 1e6, 14), (1e3, 3e6, 30), (1e4, 1e7, 60)):
        def hf(wl, hs=hs, he=he, rd=rd):
            f = min(wl / (rd * DAY), 1.0)
            return hs * (he / hs) ** f
        res = simulate(hf, POW_LIMIT_BITS, wall_max=88 * DAY, seed=3, record_every=1)
        def at(d):
            return max(h for (h, wall, *_r) in res if wall <= d * DAY)
        print(f"{hs:>8.0e} {he:>8.0e} {rd:>7} {at(5.9):>12,} {at(30):>11,} {at(87):>11,}")
    return x0_fit


def scenario_quantization():
    print()
    print("=" * 78)
    print("Quantization check: compact (23-bit, truncating) vs exact at equilibrium")
    print("=" * 78)
    for H in (1e3, 1.3e3, 3e5, 7.7e5):
        b0 = equilibrium_bits(H)
        mant = b0 & 0x7fffff
        res = simulate(lambda _w, H=H: H, b0, n_max=60000, seed=11)
        tail = res[20000:]
        mean_solve = sum(r[4] for r in tail) / len(tail)
        mean_exp = sum(T * rel_diff(r[3]) / H for r in tail) / len(tail)
        print(f"  H={H:>9.3g} start mantissa=0x{mant:06x}: mean timestamp delta "
              f"{mean_solve:6.1f} s, mean expected solve {mean_exp:6.1f} s (T={T})")


if __name__ == "__main__":
    scenario_drop()
    scenario_startup()
    scenario_quantization()
