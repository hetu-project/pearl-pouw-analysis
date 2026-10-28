"""Experiments for notes/08-fp8-protocol.md. Run: python3 -I experiments.py
(pure Python; takes ~1-2 min)."""
import math
import random
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from fractions import Fraction
from fp8lib import rne_fp8, rne_bf16, sample_line, matmul_cell, FP8_POS

random.seed(1)
r, c = 32, 256.0


def header(t):
    print('\n' + '=' * 70 + '\n' + t + '\n' + '=' * 70)


# ---------------------------------------------------------------- E1
header('E1  SampleLine: norm ~256, entry distribution, min-entropy estimate')
seed = b'\x11' * 32
lines = [sample_line(seed, 0, 0, l) for l in range(4000)]
norms = [math.sqrt(sum(v * v for v in L)) for L in lines]
print('mean line norm %.2f  min %.2f  max %.2f' % (sum(norms) / len(norms), min(norms), max(norms)))
from collections import Counter
cnt = Counter(v for L in lines for v in L)
tot = sum(cnt.values())
pmax = max(cnt.values()) / tot
H = -sum(n / tot * math.log2(n / tot) for n in cnt.values())
print('distinct FP8 values per entry: %d, Shannon H/entry=%.2f bits, min-entropy/entry=%.2f bits'
      % (len(cnt), H, -math.log2(pmax)))
print('naive line min-entropy (32 x per-entry, ignores norm coupling) = %.1f bits'
      % (32 * -math.log2(pmax)))

# ---------------------------------------------------------------- E2
header('E2  N = E F^T entry rms vs c^2/sqrt(r); check beta = alpha*l2*delta*sqrt(r)/c^2')
k = 1024
E = [sample_line(seed, 0, 0, l) for l in range(8)]
F = [sample_line(seed, 0, 1, u) for u in range(k)]
Ns = [[sum(e[t] * f[t] for t in range(r)) for f in F] for e in E]
rms = math.sqrt(sum(v * v for row in Ns for v in row) / (len(Ns) * k))
peak = max(abs(v) for row in Ns for v in row)
print('rms(N)=%.1f   c^2/sqrt(r)=%.1f   peak|N|=%.1f  (Cauchy-Schwarz bound c^2=%.0f)'
      % (rms, c * c / math.sqrt(r), peak, c * c))
for delta in (0.5, 1.0):
    ratios = []
    for _ in range(20):
        X = [random.gauss(0, 1) for _ in range(k)]
        l2 = math.sqrt(sum(x * x for x in X) / k)
        linf = max(abs(x) for x in X)
        alpha = 448 / (linf + delta * math.sqrt(r) * l2)
        beta = alpha * l2 * delta * math.sqrt(r) / c ** 2
        N = Ns[_ % 8]
        noise_norm = math.sqrt(sum((beta * v) ** 2 for v in N))
        sig_norm = math.sqrt(sum((alpha * x) ** 2 for x in X))
        ratios.append(noise_norm / sig_norm)
    print('delta=%.1f: ||beta N||/||alpha X|| mean %.3f (target %.1f), min %.3f max %.3f'
          % (delta, sum(ratios) / len(ratios), delta, min(ratios), max(ratios)))
    # how far below Q=448 the peak actually sits
    print('   sigma (quantized units) = delta*alpha*l2 for gaussian row ~ %.1f'
          % (delta * 448 / (math.sqrt(2 * math.log(k)) * 1.1 + delta * math.sqrt(r))))

# ---------------------------------------------------------------- E3
header('E3  FP8 rounding-error variance constant (whitepaper: ~2^-10.5 * (x^2+sigma^2))')
for x_over_sigma in (0.0, 0.5, 1.0, 2.0, 4.0):
    sig = 20.0
    x = x_over_sigma * sig
    s2 = t2 = 0.0
    n = 40000
    for _ in range(n):
        S = x + random.gauss(0, sig)
        T = rne_fp8(S) - S
        s2 += S * S
        t2 += T * T
    print('x/sigma=%.1f  E[T^2]/E[S^2] = 2^%.2f' % (x_over_sigma, math.log2(t2 / s2)))

# ---------------------------------------------------------------- E4
header('E4  "B = 1 -> Q(B+E_B F_B^T)_ij = B_ij in ~10% of entries" (Sec 5.2)')
# Row of all ones: l2 = linf = 1. Compare quantized noised value with quantized clean value.
for delta in (0.5, 1.0):
    alpha = rne_bf16(448 / rne_bf16(rne_bf16(delta * math.sqrt(r)) * 1 + 1))
    beta = alpha * 1 * delta * math.sqrt(r) / c ** 2
    clean_q = rne_fp8(alpha * 1)
    same = 0
    tot = 0
    for row in Ns:
        for v in row:
            tot += 1
            if rne_fp8(max(-448, min(448, alpha + beta * v))) == clean_q:
                same += 1
    print('delta=%.1f alpha=%.2f Q(alpha)=%.0f  fraction unchanged = %.3f' % (delta, alpha, clean_q, same / tot))

# ---------------------------------------------------------------- E5
header('E5  Lambda(x,sigma) fixed-point log2 (Appendix C.5): one-sided error < 0.016')


def Lam(x, s):
    e = max(math.floor(math.log2(abs(x))) if x != 0 else -10 ** 9, math.floor(math.log2(s)))
    mx, ms = abs(x) / 2 ** e, s / 2 ** e
    kappa = (math.floor(2 ** 30 * mx * mx) + math.floor(2 ** 30 * ms * ms)) // 2 ** 17
    return (2 * e - 13) + math.floor(64 * math.log2(kappa)) / 64


worst_lo, worst_hi = 0.0, -1.0
for _ in range(200000):
    x = random.choice([0.0, 1.0]) * random.uniform(-1, 1) * 2 ** random.uniform(-20, 20)
    s = 2 ** random.uniform(-20, 20)
    true = math.log2(x * x + s * s)
    d = true - Lam(x, s)
    worst_lo = max(worst_lo, d)
    worst_hi = max(worst_hi, -d)
print('max(true - Lambda) = %.5f  (bound 0.016);  max(Lambda - true) = %.2e (should be <= ~0)'
      % (worst_lo, worst_hi))

# ---------------------------------------------------------------- E6
header('E6  Appendix C.2 accumulator: small summands vanish below the grid 2^(eta-W)')
for W in (13, 25):
    a = [448.0] + [2 ** -6 * 0.125] * 31  # one huge product + tiny ones
    b = [448.0] + [1.0] * 31
    c_out, _ = matmul_cell(a, b, W)
    exact = sum(Fraction(x) * Fraction(y) for x, y in zip(a, b))
    print('W=%d  exact=%s  device=%s  lost=%s' % (W, float(exact), float(c_out), float(exact - c_out)))
# Unnormalised stored exponent example from C.2: 1.5*1.5 keeps exponent 0
a = [1.5] * 32
b = [1.5] * 32
print('32 x (1.5*1.5): device(W=13)=%s exact=%s' % (float(matmul_cell(a, b, 13)[0]), 32 * 2.25))
# random tile: how often does the H100-like truncation differ from exact FP32 sum
diff = 0
for _ in range(200):
    a = [rne_fp8(random.gauss(0, 30)) for _ in range(128)]
    b = [rne_fp8(random.gauss(0, 30)) for _ in range(128)]
    d13 = matmul_cell(a, b, 13)[0]
    d25 = matmul_cell(a, b, 25)[0]
    ex = sum(Fraction(x) * Fraction(y) for x, y in zip(a, b))
    diff += (d13 != d25)
print('random k=128 cells where W=13 and W=25 results differ: %d / 200' % diff)

# ---------------------------------------------------------------- E7
header('E7  Liveness: an all-dense Gaussian row has how many idle entries?')
for delta in (0.5, 1.0):
    idle = tot = 0
    for _ in range(50):
        X = [random.gauss(0, 1) for _ in range(1024)]
        l2 = math.sqrt(sum(x * x for x in X) / 1024)
        idle += sum(1 for x in X if abs(x) >= 8 * delta * l2)
        tot += 1024
    print('delta=%.1f: idle fraction = %.5f  (eps_idle = %.5f; threshold |x| >= %.0f*l2)'
          % (delta, idle / tot, 1 / 64, 8 * delta))
# heavy-tailed rows (e.g., outlier channels)
for df in (3, 2):
    idle = tot = 0
    for _ in range(50):
        X = [random.gauss(0, 1) / math.sqrt(random.gammavariate(df / 2, 2 / df)) for _ in range(1024)]
        l2 = math.sqrt(sum(x * x for x in X) / 1024)
        idle += sum(1 for x in X if abs(x) >= 8 * 0.5 * l2)
        tot += 1024
    print('student-t df=%d, delta=0.5: idle fraction = %.5f' % (df, idle / tot))
