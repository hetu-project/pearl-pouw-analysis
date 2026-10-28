# 纯 Python（环境无 numpy）
# 1) 验证论文 Algorithm 6.3/6.4 去噪恒等式 C' - C'' = AB (在 F_q 上)，并验证分块 transcript 最后一项 = C'
# 2) 验证 Pearl 代码式噪声结构: N_A = E_AL * E_AR, E_AR 每个 l 恰有一个 +1 和一个 -1 -> rank <= r-1
import random
random.seed(0)
q = 65521
def rand(a, b): return [[random.randrange(q) for _ in range(b)] for _ in range(a)]
def mm(X, Y):
    return [[sum(X[i][t]*Y[t][j] for t in range(len(Y))) % q for j in range(len(Y[0]))] for i in range(len(X))]
def add(X, Y): return [[(x+y) % q for x, y in zip(a, b)] for a, b in zip(X, Y)]
def sub(X, Y): return [[(x-y) % q for x, y in zip(a, b)] for a, b in zip(X, Y)]
n, r = 24, 4
A, B = rand(n, n), rand(n, n)
EL, ER, FL, FR = rand(n, r), rand(r, n), rand(n, r), rand(r, n)
E, F = mm(EL, ER), mm(FL, FR)
Cp = mm(add(A, E), add(B, F))
Cpp = add(mm(mm(A, FL), FR), mm(EL, mm(ER, add(B, mm(FL, FR)))))   # Alg 6.4 step 2 的结合顺序
print("Alg6.4: C'-C'' == AB :", sub(Cp, Cpp) == mm(A, B))
Ap, Bp = add(A, E), add(B, F)
C = [[0]*n for _ in range(n)]
for l in range(n // r):
    blkA = [row[l*r:(l+1)*r] for row in Ap]; blkB = Bp[l*r:(l+1)*r]
    C = add(C, mm(blkA, blkB))
print("block transcript final == C' :", C == Cp)

def rank_mod(M, p=1000003):
    M = [[x % p for x in row] for row in M]; rk = 0; cols = len(M[0])
    for c in range(cols):
        piv = next((i for i in range(rk, len(M)) if M[i][c]), None)
        if piv is None: continue
        M[rk], M[piv] = M[piv], M[rk]; inv = pow(M[rk][c], p-2, p)
        M[rk] = [x*inv % p for x in M[rk]]
        for i in range(len(M)):
            if i != rk and M[i][c]:
                f = M[i][c]; M[i] = [(x - f*y) % p for x, y in zip(M[i], M[rk])]
        rk += 1
    return rk
k, r2, h = 512, 16, 40
EAL = [[random.randrange(64) - 32 for _ in range(r2)] for _ in range(h)]
first = [random.getrandbits(32) for _ in range(k)]
pairs = []
for u in first:
    a = u & (r2-1); b = a ^ (1 + ((r2-1)*u >> 32)); assert a != b and b < r2; pairs.append((a, b))
NA = [[EAL[i][a] - EAL[i][b] for (a, b) in pairs] for i in range(h)]
vals = [x for row in NA for x in row]
print("N_A entry range:", min(vals), max(vals), "; rank(N_A) =", rank_mod(NA), "with r =", r2)
print("distinct (first,second) pairs among k=%d: %d (upper bound r(r-1)=%d)" % (k, len(set(pairs)), r2*(r2-1)))
