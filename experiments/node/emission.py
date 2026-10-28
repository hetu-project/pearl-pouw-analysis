#!/usr/bin/env python3
"""
Pearl emission checks (node/blockchain/validate.go:160 CalcBlockSubsidy).

  K = (4*365*24*3600) / TargetTimePerBlockSeconds   (integer division)
  subsidy(h) = floor( S*K / ((h+K)*(h-1+K)) ),  h>=1 ; subsidy(0)=0
  S = 2_100_000_000 * 1e8 grains

Telescoping: S*K/((h+K)(h-1+K)) = S*K*(1/(h-1+K) - 1/(h+K))
  => sum_{h=1..H} = S*K*(1/K - 1/(H+K)) = S*H/(H+K)   (before flooring)

Usage: python3 -I emission.py
"""
from fractions import Fraction

GRAIN = 10**8
S = 2_100_000_000 * GRAIN
T = 194
K = (4 * 365 * 24 * 60 * 60) // T


def subsidy(h):
    if h == 0:
        return 0
    return (S * K) // ((h + K) * (h - 1 + K))


def cum_closed(H):
    return Fraction(S * H, H + K)


def prl(g):
    return float(g) / GRAIN


print(f"K = {K}  (4y*365d*86400 / {T} = {4*365*86400/T:.3f})")
print(f"subsidy(1)      = {prl(subsidy(1)):.4f} PRL  (= S/(K+1))")
print(f"subsidy(40000)  = {prl(subsidy(40000)):.4f} PRL")
print(f"subsidy(K)      = {prl(subsidy(K)):.4f} PRL")

# exact check of the telescoping identity vs floored sum
checkpoints = [1, 10, 1000, 40000, 91630, 110000, K, 2 * K, 5_000_000]
acc = 0
h = 0
print("\nH            floor-sum (PRL)          S*H/(H+K) (PRL)     diff (grains)   diff bound H")
for H in checkpoints:
    while h < H:
        h += 1
        acc += subsidy(h)
    closed = cum_closed(H)
    print(f"{H:<10} {prl(acc):>22.8f} {float(closed)/GRAIN:>22.8f} {float(closed-acc):>14.1f} {H:>10}")

print("\nFraction of S issued at height H: H/(H+K)")
for years in (1, 2, 4, 8, 12, 20, 36, 100):
    H = years * 365 * 86400 // T
    print(f"  {years:>3} y on schedule (H={H:>9,}): {H/(H+K)*100:6.2f}%")

# Early over-issuance estimates
DAY = 86400
cases = [
    ("public data: 40000 blocks at 5.9 days", 40000, 5.9 * DAY),
    ("MC startup (H/Hgen=3e5) 39634 @5.9d", 39634, 5.9 * DAY),
    ("MC ramp 1e3->1e6/7d: 55902 @30d", 55902, 30 * DAY),
    ("git: DenseOnly 91630 ~ 2026-07-23 ~10:00Z (87.0 d)", 91630, 87.04 * DAY),
    ("git: SaltedSeed 99000 active by 2026-08-12 11:10Z (<=107.1 d)", 99000, 107.09 * DAY),
]
print("\nOver-issuance vs on-schedule chain (height_sched = wall/T):")
print(f"{'case':<62} {'issued':>12} {'sched':>12} {'excess':>12} {'excess%S':>9} {'h ahead':>8}")
for name, H, w in cases:
    Hs = int(w // T)
    a = float(cum_closed(H)) / GRAIN
    b = float(cum_closed(Hs)) / GRAIN
    print(f"{name:<62} {a/1e6:>10.2f}M {b/1e6:>10.2f}M {(a-b)/1e6:>10.2f}M "
          f"{(a-b)/2.1e9*100:>8.2f}% {H-Hs:>8,}")
print("\nNote: the 'excess' is a time-shift of the fixed per-height schedule, not extra"
      " total supply: the cap S and per-height subsidy are unchanged; coins are"
      " issued earlier in wall-clock time.")
