# 组件 1：Pearl / cuPOW 的数学构造

> 资料：论文 arXiv:2504.09971v4（2025-11-13 版），文本 `papers/cupow_2504.09971.txt`；Pearl 源码 `third_party/pearl` @ `156ef9a8`；FP8 白皮书 `papers/pearl_whitepaper_fp8.txt`。
> 验算脚本：`experiments/math/verify_denoise.py`（纯 Python，见 §2.4、§7.2）。
>
> 标注约定：
> - **【论文】** 论文原文的说法（附章节/编号；引用为意译，个别短语保留英文原词）。
> - **【代码】** 代码事实（附 `文件:行号`，路径相对仓库根 `third_party/pearl`）。
> - **【推导】** 我自己的推导或理解。
> - **【不确定】** 没有完全核实的内容。
>
> 版本说明：任务说明里的作者是 "Komargodski, Schen, Weinstein"，但本地 v4 文本署名只有 Ilan Komargodski 和 Omri Weinstein（txt:3）。任务提到的 "Remark 5.7" 在 v4 中**不存在**：第 5 节只有 Remark 5.2–5.5 和 Claim 5.6。下文的开放问题按 v4 实际编号整理（§6.3）。

---

## 0. 一句话概括

论文的核心思路是：把**低秩随机噪声**加到矿工自选的矩阵 A、B 上，然后对**按块（tile）计算 A'B' 的中间过程（transcript）**做哈希抽签，而不是对最终输出抽签。
- 低秩：honest 矿工去噪只需 O(n²r)，相对 n³ 可以忽略，所以有用性开销是 1+o(1)。
- 哈希 transcript 而不是输出：防止"低秩输出可以走捷径"，例如 A=B=0 时 E·F 只需 O(n²r) 就能算出。
- 安全性没有证明，依赖一个新假设：低秩随机矩阵乘法的全部中间值，算起来不比通用算法快（Assumption 6.4，摘要里称作 direct-product 猜想）。

Pearl 的代码实现保留了"低秩噪声 + 分块中间值 + 哈希"的骨架，但在下面这些地方都不一样：域、噪声分布、transcript 的粒度与压缩方式、ticket 定义、验证范围、种子派生顺序。

---

## 1. 论文的形式化模型

### 1.1 计算模型与参与方
- **【论文 §4】** 工作在随机预言机模型（ROM）中，O: {0,1}* → {0,1}^λ。复杂度把 **RO 查询次数**和**域运算次数**分开计数，**一次 RO 查询算一个单位**。实际部署时可启发式地换成 SHA-256 / BLAKE3。
- **【论文 §4、§6】** 底层是有限域 F_q，任务是 n×n 矩阵乘法 A·B，A, B ∈ F_q^{n×n}。脚注 4 说可推广到长方形矩阵。
- **【论文 §5】** 参与方是 prover P 和 verifier V。任务 f: {0,1}^n → {0,1}^n 有一个规范算法 Alg，其最坏情况时间为 t(n) = max_x t(n,x)。原文写作 "max_x f(n,x)"，是笔误。安全参数 λ 满足 n ≫ λ。协议是 public-coin，可以用 Fiat–Shamir 变成非交互。
- **【论文 §5】** 种子 σ ∈ {0,1}^λ 用来保证工作是"新鲜的"，即在 σ 公开之后才做，防止预计算。可选的 Keygen/GenPP 被省略。

### 1.2 语法
- Solve(σ, x) → (y, π)：概率算法。
- Verify(σ, π) → {0,1}：确定性算法。

### 1.3 Definition 5.1（ε-difficulty、α-overhead PoUW）
**【论文 Def 5.1】**，原文意译：
1. **Efficiency**：对任意 σ、x，Solve(σ,x) 运行时间为 t(n)·(1+α(n))；对 Solve 的任何输出 π，Verify(σ,π) 运行时间为 c·t(n)，其中常数 c ∈ (0,1)。
2. **Completeness**：对一切 λ, n, σ, x，有 Pr[f(x)=y] = 1 且 Pr[Verify(σ,π)=1] ≥ ε(n)。概率取自 Solve 的随机性和 RO。
3. **Hardness**：存在常数 C>0，使得对任何运行时间 t' ≤ t(n)、允许多项式时间预处理的 Solve*，
   ```
   Pr[ Verify(σ, π) = 1 ] < max( C · t'·ε(n) / t , 2^{-λ} )
   ```
   概率取自均匀 σ、π ← Solve*(σ) 和 RO。原文写 "running in time t′ = t(n) with t′(n) ≤ t(n)"，排版有误，按上式理解。

**【推导】** 关于这三条的理解：
- Hardness 说的是"单位时间成功概率"不能显著超过 honest 的 ε/t，允许常数因子 C。所以它本身容许常数倍优势；"optimal security" 要求 C≈1，但这一点只出现在摘要和 §1.1 的猜想表述中，定义本身没有要求。
- α(n) 是相对开销。Remark 5.4 说有意义的区间是 α = o(1)。原文写成 "α(t(n)) ≪ t(n)…α(n)=o(t(n))"，记号混乱，按语义应理解为 α(n) = o(1)。
- Definition 5.1 的 Completeness 量词是 "for every x"，即 A、B 由矿工任意选。Hardness 中 Solve* 只拿到 σ，自己挑 x。所以"矿工自选输入"被内建在定义里。

### 1.4 Remark 与平凡构造
- **Remark 5.2（可调难度）**：可以固定 ε 改 n，也可以固定 n 改 ε。最终协议两者都可调。
- **Remark 5.3**：要求 honest 输出精确正确（概率 1）。论文提到可放宽为近似。
- **Remark 5.5**：短证明和快速验证都可以用 SNARG 通用地实现，只在中签时计算，成本被摊薄。
- **§5.1 + Claim 5.6（平凡 PoUW）**：先正常计算 f，再独立做 T = c·t(n) 次 hashcash，得到 α = c、ε = 1−e^{−c}。缺点是：恶意者跳过 f 即可快 (1+1/c) 倍；要把这个优势压到亚线性，就要 c = ω(1)，这又让 honest 的开销变成 (1+c) 倍。这说明 "O(1) 开销的 PoUW 是平凡的"（摘要），非平凡的目标是 α = o(1)。

### 1.5 区块链版本（附录 B）
- **Definition B.3**：honest 挖矿事件流应构成速率 ρ 的 Poisson 过程；(T, 𝒯, α)-对抗过程的速率 ≤ ρ。论文**只陈述**这个定义并"猜想"其 PoW 满足（§5 正文：We conjecture...）。对 Bitcoin backbone 的安全性，论文说证明可沿用 [GKP20]，正文没有给出证明。

---

## 2. 构造的三步演进

记 A' = A+E，B' = B+F，C' = A'B'。

### 2.1 第一步：满秩均匀噪声（§2 概述 + §6.4 Algorithm 6.3）
**【论文】**
```
Encode(σ, A, B):  σ ↦ (E, F) ∈ F_q^{n×n} 均匀；   A' = A+E,  B' = B+F
Solve:            C' = A'·B'；  z = O(C')；  π = (A, B, z)
Decode(σ, C'):    C'' = A·F + E·(B+F)；  C = C' − C''
Verify:           由 σ, A, B 重算 z，检查 z < 2^{λ − log(1/ε)}
```
- 这里 r = 1，transcript 就是输出 C'。因为 E、F 均匀，C' 的分布与 A、B 无关，所以 z 低于阈值的概率恰为 ε。
- **Assumption 6.3（原文意译）**：不存在能在 o(n^ω) 时间内计算两个随机 n×n 矩阵乘积的算法。
- **问题**：Decode 需要额外两次满规模乘法，honest 总成本约为 3·t(n)，而恶意者跳过 Decode 只需 1·t(n)，所以 α ≥ 2（原文说 "at least three times the work"）。结论是：这是 PoW，但没有用处。

### 2.2 第二步：低秩噪声（§2 概述 + §6.5 Algorithm 6.4，前半）
**【论文 Alg 6.4】**
```
Encode_r(σ, A, B):  σ ↦ E_L, F_L ∈ F_q^{n×r},  E_R, F_R ∈ F_q^{r×n}
                    E = E_L·E_R,  F = F_L·F_R          （2 次 n×r×n 乘法）
                    A' = A+E,  B' = B+F
Decode_r(σ, C'):    C'' = (A·F_L)·F_R + E_L·(E_R·(B + F_L·F_R))   （5 次"一边是 r"的乘法）
                    C = C' − C''
```
- 去噪成本是 O(n²r)，相对 n³ 可以忽略。
- 但如果仍然对输出 C' 做哈希，**就不安全了**（§2 原文）。恶意者取 A = B = 0，则 C' = E·F = E_L(E_R F_L)F_R，可在 O(n²r) 内算出，远快于 honest 的 n³。

### 2.3 第三步：哈希计算过程（transcript）（§2 概述 + Alg 6.2 + Alg 6.4）
**【论文】** 用分块算法 MatMul_r 计算 A'B'，把全部中间块
```
C'^{(ℓ)}_{i,j} = C'^{(ℓ-1)}_{i,j} + A'_{i,ℓ} · B'_{ℓ,j},   i, j, ℓ ∈ [n/r]
```
记作 tr。然后令 z = O(tr)（§2 版本），或者让 π 包含整个 z（Alg 6.2 版本）。
- 对于 A=B=0：输出 EF 仍然便宜，但中间值 C'^{(ℓ)}_{i,j} = Σ_{t≤ℓ} E_{i,t}F_{t,j} 是"部分和"，不能由低秩结构直接给出，后文 §5 详述。论文由此主张："multiplying the all-zeroes matrices … is as hard as any other product"（§1.1）。

**Algorithm 6.2（PoUW 模板，原文）**：
```
Solve(σ,A,B): A',B' = Encode_r(σ,A,B); C' = MatMul_r(A',B'), z = {C^{(k)}_{i,j}}; C = Decode_r(σ,C'); 输出 (C, π=(A,B,z))
Verify(σ,π):  解析 π=(A,B,z)，从 σ,A,B 重算正确的 z，当且仅当 z 正确时输出 1
```
**【推导】** Alg 6.2 的 Verify 只检查 z 是否正确，**没有出现"低于阈值"这一步**。阈值检查只出现在 §2 的概述版（z = O(tr) < 2^{λ−log(1/ε)}）。正式版本里 ε 如何出现并不明确。可以理解为 Alg 6.2 + Def 6.2 的 ε 刻画"猜中整个 transcript"的概率，而实际的抽签概率 ε 来自 §2 概述版的 O(tr) 阈值。两种写法并存，论文没有统一。

### 2.4 去噪恒等式的逐步验证（【推导】）
论文 §6.4/§6.5 原文写作 `C′ − C′′ = (A+E)(B+F) − A·F + E·(B+F) = AB`，**漏了括号**，应为 `(A+E)(B+F) − (A·F + E·(B+F))`。逐步展开如下：
```
C'  = (A+E)(B+F)
    = A·B + A·F + E·B + E·F                        （分配律；矩阵乘法不可交换，左右顺序保持不变）
C'' = A·F + E·(B+F)
    = A·F + E·B + E·F                              （分配律）
C' − C'' = A·B + (A·F − A·F) + (E·B − E·B) + (E·F − E·F) = A·B   ∎
```
再看 Alg 6.4 中的结合顺序（结合律）：
```
(A·F_L)·F_R               = A·(F_L·F_R)          = A·F
E_L·(E_R·(B + F_L·F_R))   = (E_L·E_R)·(B + F)    = E·(B+F)
```
各项成本（naive 乘法，按 MAC 计）：
- A·F_L：n×n·n×r = n²r
- (A F_L)·F_R：n×r·r×n = n²r
- F_L·F_R：n²r
- E_R·(B+F)：r×n·n×n = n²r
- E_L·(…)：n×r·r×n = n²r

合计 5n²r。原文说 "invoking MatMul 5 times"，与此一致。Encode 中的 E_L E_R 和 F_L F_R 另有 2n²r，其中 F_L F_R 可以与 Decode 共用。

**数值验算**：`experiments/math/verify_denoise.py` 在 F_65521 上取 n=24、r=4，验证了 Alg 6.4 的 C' − C'' = AB，以及分块 transcript 最后一项等于 C'。两项输出都是 True。

---

## 3. "开销 1+o(1)" 的精确推导

### 3.1 论文给出的结论
**【论文 §6.5 "Efficiency of PoUW"】**
```
α(n) = O(n² + n²·r) / t_MatMul(n)
```
在不使用快速矩阵乘法时，α = o(1)。

### 3.2 逐项计算量（【推导】，n×n 方阵，naive 基线 t_MatMul = n³）
| 步骤 | 计算量 |
|---|---|
| RO 展开得到 E_L, E_R, F_L, F_R | 4nr 个域元素；论文模型中按 RO 查询计 |
| Encode：E = E_L E_R，F = F_L F_R | 2n²r |
| A' = A+E，B' = B+F | 2n² |
| MatMul_r(A', B')，产生 (n/r)³ 个 r×r 中间块 | (n/r)³·r³ = n³ |
| 写出或哈希 transcript：(n/r)³·r² = **n³/r** 个域元素 | Θ(n³/r)（若按字节计哈希成本） |
| Decode：5 次 rank-r 乘法 + 减法 | 5n²r + n² |

所以
```
T_Solve = n³ + n³/r + 7n²r + O(n²)
α(n)    = 1/r + 7r/n + O(1/n)
```
- 只有当 r → ∞ 且 r = o(n) 时，α = o(1)。
- 取 r = Θ(√n) 时，α 最小，为 Θ(n^{-1/2})。
- 论文例子 r = n^{0.3}：α = Θ(n^{-0.3})。

**与论文的差别**：论文的 α 公式**没有计入 n³/r 的 transcript 哈希项**。§2 原文说 "step 3 requires hashing a string of length (n/r)³ which takes time o(n)"，但 transcript 实际含 (n/r)³ 个块、每块 r² 个元素，共 n³/r 个元素。在论文的计数模型里，"一次 RO 查询算一单位"，所以这一项被记为 O(1)。按真实哈希成本，它贡献 1/r 的相对开销。这一项仍然是 o(1)，所以结论"1+o(1)"不变，只是 o(1) 的速率是 max(1/r, r/n)，而不是 r/n。【推导，论文未明说】

### 3.3 参数 r 与 n 的关系、论文的具体例子
- **【论文 §6.1】** 几个特例：r = n 时只有 1 个中间值，就是输出；r = n/2 时有 8 个中间值；r = 1 时有 n³ 个标量中间值。
- **【论文 §6.1】** 用快速乘法计算全部中间值：
  - 方阵 FMM：O(nmk / r^{3−ω})。
  - 长方 FMM：固定 ℓ 时所有 C^{(ℓ)}_{i,j} 构成"列条 × 行条"的外积，即 n×r 乘 r×n。总时间为 (n/r)·n^{ω_r}。论文写作 n^{ω_r+1}/r，其中 ω_r 是 n×r 乘 r×n 的指数。
  - 当前纪录：对 k ≤ 0.321334 有 ω_k = 2 [Gal24, VXXZ24]。所以 r = n^{0.3} 时全部中间值可在 n^{2.7+o(1)} 内算出。
- **【论文 §6.1 结尾】** 实践中最快的仍是直接算法，其他算法常数大、需要大量内存或有内在串行性。
- 论文前后引用的 ω 值不一致：§2 写 ω < 2.371339 [ADV+24]，§6.1 写 ω = 2.371552 [VXXZ24]。这不影响论证。

### 3.4 一个结构性张力（【推导】）
honest 用 naive 算法，成本 t_MatMul = n³。Assumption 6.4 给出的下界是 n^{ω_r+1}/r，在 r = n^{0.3} 时等于 n^{2.7}。论文 §6.5 "Efficiency" 段写 "there is no algorithm that runs in o(t_MatMul(n))"，混用了 t_MatMul 和 t_r。

严格按假设，攻击者若有实用的长方 FMM，最多可以比 honest naive 快 r 倍。论文的处理方式是把 t_r(n) 定义为"当前最优算法"，并在 §6.1 声明实践中 naive 最优。也就是说，安全性是相对于"实用算法"而言，而不是相对于渐近最优算法。

---

## 4. MatMul_r 与 transcript 的精确定义

### 4.1 Algorithm 6.1（原文）
```
MatMul_r(A, B):  A ∈ F_q^{n×k}, B ∈ F_q^{k×m}, r | n, k, m
 1. 初始化全 0 矩阵 C^{(0)}            （原文写 C^{(0)} ∈ [−q,q]^{n×m}，记号不严谨）
 2. 把 A, B, C 划分成 r×r 块，X_{i,j} 表示第 (i,j) 块
 3. 对每个 i∈[n/r], j∈[m/r], ℓ∈[k/r]：  C^{(ℓ)}_{i,j} := C^{(ℓ−1)}_{i,j} + A_{i,ℓ}·B_{ℓ,j}
    （A_{i,ℓ}·B_{ℓ,j} 可用任意经典算法计算）
 4. 输出 C^{(n)}                       （应为 C^{(k/r)}）
```

### 4.2 Definition 6.1（Transcript）
```
Tr(MatMul_r, A, B) = { C^{(ℓ)}_{i,j} }_{i∈[n/r], j∈[m/r], ℓ∈[k/r]}
```
原文上标写成 (k)、下标范围写 ℓ，属于笔误。

- 共 nmk/r³ 个中间矩阵，每个 r×r，合计 **nmk/r 个域元素**，即 n³/r·log q 比特。
- transcript 是**累积部分和**（running sums），不是每步的增量积 A_{i,ℓ}B_{ℓ,j}。【推导】两者可互相线性推出：增量 = C^{(ℓ)} − C^{(ℓ−1)}，所以信息量相同。

### 4.3 Definition 6.2（transcript-unpredictable encoding）
- **Completeness**：对一切 A, B, r，Pr_σ[Decode_r(σ, A'B') = AB] = 1。
- **ε-transcript unpredictability**（原文意译）：存在常数 C，对所有 A, B, r，以及任何运行时间 t 的 "iterative numerical algorithm" 𝒜，𝒜(A', B') 输出的 {Z^{(k)}_{i,j}} 与真实 transcript **完全相等**的概率至多为 C·t(n)/t_MatMul(n) · ε(n)。概率取自 σ 和 𝒜 的随机性。
- **【推导】** 有两点值得注意：
  1. 这里的"成功"是猜中**全部** transcript。部分猜中不算成功；配合 ROM，部分猜中对 O(tr) 也没有帮助。
  2. 定义只对 "iterative numerical algorithm" 量化，论文没有给出这个术语的定义。

### 4.4 Remark 2.1（内存）以及 Pearl 实际采用的变体
- **【论文 Remark 2.1】** 存储完整 tr 需要大量内存。替代方案：**对 (n/r)³ 个中间矩阵分别哈希，每一个都作为一次独立的抽签机会**。这样尝试次数与矩阵规模成正比，天然支持不同尺寸。论文认为同样的硬度论证适用：每个中间矩阵是两个"边际均匀"矩阵之积，在 fine-grained 意义上不可预测。
- **Remark 2.2**：verifier 开销大，用 SNARK 委托验证；需要隐私时用 zkSNARK。
- **Remark 2.3**：不能直接拿 SNARK 计算本身当难度来源，因为每次尝试都要生成 SNARK，效率太差。cuPOW 只在中签时生成 SNARK。
- Pearl 的 ticket 设计介于"整个 transcript 一次哈希"和 Remark 2.1 的"每个中间块单独哈希"之间，见 §7.3。

---

## 5. 安全性论证

### 5.1 假设原文与含义
| 假设 | 原文（意译） | 含义 | 为什么需要 |
|---|---|---|---|
| **Assumption 6.3**（§6.4） | 没有算法能在 o(n^ω) 时间计算两个随机 n×n 矩阵之积 | 随机实例上矩阵乘法没有平均情况捷径 | 支撑第一步（满秩噪声）方案的 hardness |
| **Assumption 6.4**（§6.5） | 对两个随机 n×n、秩为 r 的矩阵相乘，没有算法能在 o(n^{ω_r+1}/r) 时间算出全部中间值 | 低秩随机矩阵的 MatMul_r transcript 不能比通用算法更快地算出 | 支撑 Alg 6.4 的 transcript unpredictability，从而支撑 PoUW hardness |
| **"direct-product conjecture"**（摘要、§1.1、§3） | "a hardness assumption of a certain direct-product problem of correlated instances (essentially random low-rank linear equations)" | 把每个 r×r 中间块看作一个子问题：子问题之间相关（共享秩 r 结构），但同时解出所有子问题的代价仍是单个代价之和，而且成功概率不会因为相关性而显著提高 | Assumption 6.4 只说"全部算出要多少时间"，PoUW 还需要"在时间 t' < t 内成功概率 ≲ t'/t·ε"这种直积形式 |
| **Conjecture A.1**（附录 A） | 对 √n-秩满的 A, B，在 Alg 1.1（随机 Hadamard 旋转）中 r = √n、d = O(1)，即使允许 O(n³) 预处理，计算全部 n³/r³ 个部分和的摊还代价仍为 Ω(r³)（word-RAM，w = O(log n)） | 附录 A 自抵消噪声方案的硬度前提 | "This is the analogue of Assumption 6.4" |

**【推导】** 关于"direct-product 猜想"：论文正文**没有把它写成带编号的形式化陈述**。正文中唯一带编号的相关假设是 Assumption 6.4，它是一个"时间下界"命题，不含概率或成功率。摘要和 §1.1 所说的"在直积猜想下，任何运行时间相当的对手都无法获得非平凡优势"，对应的是 Def 6.2 中的 ε-transcript unpredictability。**从 Assumption 6.4 到 Def 6.2 的推导论文没有给出。** §6.3 "Hardness" 段直接写 "follows directly from the ε-transcript unpredictability property"，也就是把 Def 6.2 当作前提，而不是把它证出来。

### 5.2 有无归约、有无证明
**【论文 + 推导】**
- **已证明的**：
  - Alg 6.4 的 completeness（代数恒等式）。
  - Lemma 6.5：在 ROM 中 E 是（统计近似）均匀的秩 r 矩阵，任意 r×r 子块边际（近似）均匀。
  - Claim 6.6：随机 n×d 矩阵满秩的概率为 1 − O(q^{−(n−d+1)})。
- **没有归约、没有证明的**：Assumption 6.3、6.4、direct-product 形式，以及"PoUW hardness 由 transcript unpredictability 推出"。§6.3 的论证是直接引用 Def 6.2。
- **论文给出的支持论据**：
  1. **边际均匀性**（§2、Lemma 6.5）：每个 A'_{i,ℓ}、B'_{ℓ,j} 在边际上均匀，所以单个中间块等价于一次 r×r 随机矩阵乘法，在 Assumption 6.3 下不能加速。
  2. **相关性恢复 ≈ 解线性方程组**（§2）：各中间块之间相关，因为 E、F 秩为 r；但要利用这种相关性，基本上等于解一个线性方程组，论文猜想这和做矩阵乘法一样难。
  3. **利用低秩结构的算法并不更快**（§6.5）：把 A = A_L A_R、B = B_L B_R，计算部分和 T_z = Σ_{i≤z} A_R B_L（按 r 分块），再算 A_L·T_z·B_R。每个 z 需要 n^{ω_r}，共 n/r 个 z，合计 O(n^{ω_r+1}/r)，与通用算法相同。
  4. **输出规模下界**（§6.5）：写出全部中间值需要 Θ(n³/r) 个元素。r 足够小时 ω_r = 2，通用算法的时间 n^{ω_r+1}/r = n³/r 正好等于这个平凡下界。论文因此说 "we can see the above assumption is true"。【推导】这一点只说明在 r ≤ n^{0.32} 的区间里 Assumption 6.4 作为"时间下界"是**无条件成立**的，因为输出规模本身就是下界。但这个下界是 n³/r，比 honest 的 n³ 小 r 倍。所以它对"恶意者不能比 honest naive 快"并没有直接帮助，见 §3.4。

### 5.3 Lemma 6.5 的细节
**【论文】** 在 ROM 下，E = E_L E_R 的分布与 𝓔_{r,n}（全体秩 r 的 n×n 矩阵）上的均匀分布统计接近；进而以高概率每个 r×r 子矩阵边际均匀。

**【推导】** 原证明有两处小问题：
- 原文说 "row space of E 是 row space of E_L"，应为 E_R；"column space of E 是 column space of E_R"，应为 E_L。
- 某个 r×r 子块 E_{i,ℓ} = (E_L)_i·(E_R)_ℓ 是两个独立均匀 r×r 矩阵之积。它与均匀分布并不完全相同，奇异的概率约为两倍，但统计距离为 O(1/q)，在 q 很大时可以忽略。

另外，**Lemma 只覆盖 E 本身**。A' = A+E 的子块是"固定矩阵 + 近似均匀块"，因此也近似均匀。但**多个块的联合分布**依赖 A 的结构，这正是假设要处理的部分。

### 5.4 为什么 A=B=0 时仍然困难（论文的论证 + 【推导】）
- A = B = 0 时，A' = E、B' = F 是随机秩 r 矩阵。**Assumption 6.4 本身就是针对这种情形陈述的**（"two random n×n rank-r matrices"）。所以 A=B=0 是这个假设的"规范实例"，论文据此说"multiplying all-zeroes … is as hard as any other product"（§1.1）。
- 输出 EF 可以在 O(n²r) 内算出，但 transcript 是 Σ_{t≤ℓ} E_{i,t}F_{t,j}。每个块边际均匀，相关性由秩 r 结构决定。最好的已知方法就是 §5.2 第 3 点的 A_L T_z B_R，它不比通用算法快。
- **【推导】** 对于一般的 A, B，Assumption 6.4 并**没有**覆盖"A + 随机秩 r"这种分布，因为 A 由对手选择。论文的隐含论证是每个块边际均匀 "no matter how A, B are chosen"（§3、附录 A 开头）。如果 A 的秩为 s，A' 的秩 ≤ s + r，相当于 rank-(s+r) 实例，这可以归入同类假设。但对于高秩、结构化的 A（例如稀疏或有特殊代数结构），没有单独的论证。

### 5.5 附录 A：自抵消噪声
**【论文】** 令 A' = AR、B' = R^T B，其中 R = H_n·D 是 Hadamard 矩阵乘以随机块对角 ±1 矩阵，且 RR^T = I。这样输出自动等于 AB，不需要去噪；旋转成本 O(n² log n)。
- 缺点：A = B = 0 时没有用处，需要高秩输入。论文建议加一个随机置换单位阵来保证。
- 依赖 Conjecture A.1，安全性更"微妙"。
- 块大小 d = r 时总成本 O(n²(r + log n))，不比 Alg 6.4 快；所以论文猜想 d = O(1) 也足够。
- 留待以后研究。

---

## 6. 验证者与开放问题

### 6.1 论文中的验证算法
- **Alg 6.2 Verify**：从 σ, A, B **重算**整个 z，检查是否相等（§2 概述版还要检查阈值）。成本等于 honest 的 Encode + MatMul_r，约 t(n)。
- **【推导】** 这与 Def 5.1 要求的 "Verify 时间 c·t(n)，c ∈ (0,1)" **字面上冲突**：重算的成本约为 1·t(n)。论文在 §2 说 "We did not put any non-trivial restriction on the running time of Verify"，又用 Remark 2.2 的 SNARK 方案作为降低验证成本的手段。
- **Remark 2.2**：中签后由 prover 生成 SNARK，证明"verifier 会接受"，于是 verifier 只需验证 SNARK。需要隐私时用 zkSNARK。因为只在中签时生成，成本被摊薄（Remark 2.3、5.5）。
- 论文没有给出"只抽查部分 transcript"的验证方式。

### 6.2 验证成本小结
| 方案 | 验证成本 |
|---|---|
| 朴素（Alg 6.2） | ≈ n³（或 t_r(n)）+ 哈希 |
| SNARK（Remark 2.2） | SNARK 验证成本（通常 polylog 或常数），加上读取公共输入 |

### 6.3 论文列出的开放问题（v4 实际编号）
- **Problem 3.1**：除 MatMul 外，哪些任务 f 存在 1+o(1) 开销的 PoUW？能否刻画？候选包括线性方程组、图搜索/路由、模式匹配、DNA 序列等。
- **Problem 3.2**：MatMul 是否有比 Alg 6.4 更好的 PoUW，无论是开销（O(n²r)）还是假设，理想情况两者兼得？论文指出噪声只需满足两点：(i) 加入和去除都是 o(n³)；(ii) 无论 A, B 怎么选，tile 的边际都不可预测。附录 A 是一个候选。
- **Problem 3.3**：能否基于更标准的假设（例如通用程序混淆）构造可证安全的 PoUW？
- **附录 A**：自抵消噪声方案的理论和实践探索（Conjecture A.1）。
- **§5 + 附录 B**：更强的安全定义，包括多实例摊还和流式挑战下的优势。论文只是猜想自己的 PoW 满足 Bitcoin PoW 的所有性质。
- **附录 C**：Unpredictable Encoding Scheme（UES）作为独立原语，与 randomized encoding 的关系。
- **§2 Remark 2.1**：内存问题（逐块哈希的变体）。
- （Remark 5.7 在 v4 中不存在。）

---

## 7. 论文构造与 Pearl 代码的对应和差异

### 7.1 代码中的端到端流程（【代码】）
1. **任务定义**：m×k 的 A 乘以 k×n 的 B，B 按 B^T 的行存储。元素被限制在 [−64, 64]，即 "Int7xInt7ToInt32"。
   - `zk-pow/src/api/proof.rs:23-25`（MMAType 只有 `Int7xInt7ToInt32 = 0`）
   - `zk-pow/src/api/verify.rs:113-118`（plain 验证中检查 [−64,64]）
   - `zk-pow/src/circuit/chip/range_table.rs:19`（电路中的 IRange7P1 范围表，−64..=64）
2. **承诺与种子**：
   - job_key = BLAKE3(IncompleteBlockHeader ‖ MiningConfiguration)，见 `zk-pow/src/api/proof_utils.rs:348-353`。header 含 prev_block、merkle_root、timestamp、nbits（`zk-pow/src/api/proof.rs:10-16`）。
   - hash_a、hash_b 是 A、B^T 以 job_key 为 key 的 BLAKE3 树根（注释见 `zk-pow/src/api/proof.rs:90-98`）。
   - 种子派生（`zk-pow/src/api/proof_utils.rs:362-373`）：
     ```
     b_noise_seed = BLAKE3(job_key ‖ hash_b)
     a_noise_seed = BLAKE3(b_noise_seed ‖ hash_a)
     ```
     V3 证书会先对根加盐，绑定 m、n（`zk-pow/src/api/seed.rs:20-25, 51-58`）。
3. **噪声**：E_AL（h×r 稠密）、E_AR（r×k 稀疏）、E_BL（k×r 稀疏）、E_BR（r×w 稠密），见 `zk-pow/src/circuit/pearl_noise.rs:142-145`。verifier 只为被选中的 h 行、w 列生成噪声（`pearl_noise.rs:118-160`）。电路中噪声作为预处理列（NOISE_PACKED_PREP）输入（`zk-pow/src/circuit/pearl_air.rs:17`）。
4. **Ticket 计算**：`zk-pow/src/circuit/chip/jackpot/helper.rs:9-35`。
5. **Ticket 哈希**：J = BLAKE3(64 字节 msg, key = a_noise_seed)，见 `zk-pow/src/api/proof_utils.rs:1502-1505`。
6. **难度判定**：J 按小端 uint256 解释，要求 J ≤ target(nbits)·h·w·dot_len（`zk-pow/src/api/sanity_checks.rs:148-157, 183-192, 229-231`）。rank-penalty 软分叉之后，还要求 r ≥ 128 且 J ≤ target·h·w·(dot_len/r)·128（`sanity_checks.rs:13, 164-180, 194-196`；提交 51a6e11b）。
7. **GPU 端**：int8 WGMMA，int32 累加（`miner/pearl-gemm/csrc/gemm/kernel_traits.hpp:24`）。每个线程每隔 R/32 个 k-block 对自己的累加器 fragment 做 XOR 归约，再做 rotl13-XOR 写入 16 个 u32 槽位（`miner/pearl-gemm/csrc/gemm/pow_utils.hpp:128-200`；调用处在 `collective_mainloop.hpp:274-311`）。tile 结束后做 keyed BLAKE3 压缩并与 target 比较（`pow_utils.hpp:205-234`；`pearl_gemm_kernel.h:225-231, 262-271`）。
8. **去噪（GPU，honest 输出）**：
   ```
   C = C' − (A·E_BL)·E_BR − E_AL·(E_AR·(B+N_B))
   ```
   用 fp16 因子在 tensor core 上做 MMA、fp32 累加，带 2^{12}/2^{14} 缩放（`collective_epilogue.hpp:361-457`，其中 422-428 和 437-443 是两次去噪 GEMM；`pearl_gemm_constants.hpp:4-16`；`pearl_gemm_host.h:25` 中 `ElementDenoise = half_t`）。结合顺序与论文 Alg 6.4 完全一致。
9. **默认参数**：noise_rank = 128；GEMM tile 128×256×128；rows_pattern = [0, 8]（h=2）；cols_pattern 共 64 列（w=64，即 h·w = 128）；见 `miner/miner-base/src/miner_base/settings.py:9-26`。这个模式就是 WGMMA 下单个线程持有的累加器寄存器在 128×256 tile 中的坐标【推导：与 pow_utils 的"每线程 fragment 做 XOR"相对应】。
10. **参数约束**（`zk-pow/src/api/sanity_checks.rs:28-58`）：
    - r 为 2 的幂，32 ≤ r ≤ 1024
    - k ≤ 2^16，64 | k，16r ≤ k ≤ 4r²，k ≥ 1024
    - h, w 为偶数（TILE_H = 2），32 ≤ h·w ≤ 256
    - m, n ≤ 2^24
    - (h+w)·dot_len ≤ 2^22

### 7.2 噪声生成的精确形式（【代码】`zk-pow/src/circuit/pearl_noise.rs`）
- 常量：NOISE_RANGE = 128，IDXS_PER_COL = 2，因此稠密取值为 (byte & 63) − 32 ∈ [−32, 31]（行 13-17、74）。
- 随机源：`get_random_hash(index, label, key=noise_seed, prepend_index)` = keyed BLAKE3(64 字节消息：在第 prepend_index 个 i32 槽写入 1+index，后 32 字节是标签 "A_tensor"/"B_tensor")，见行 29-30、46-57。参数名 `seed` 实际是标签，`key` 才是噪声种子。
- 稠密矩阵 E_AL / E_BR^T：把行 row_idx 映射到全局字节流偏移 row_idx·r，可以随机访问（行 61-80）。
- 稀疏矩阵 E_AR^T / E_BL（k×r）：每 4 字节 u 生成一对下标（行 90-116）：
  ```
  first  = u & (r−1)
  second = first XOR (1 + mulhi(r−1, u))
  ```
  因为 1 + mulhi ∈ [1, r−1]，所以 second ≠ first 且 second < r。对应行有 +1 在 first、−1 在 second。GPU 端相同的公式见 `noise_generation_kernel.h:396-410`。
- 因此（行 35-43、149、153）：
  ```
  N_A[u, l] = E_AL[u, first_A(l)] − E_AL[u, second_A(l)]
  N_B[l, v] = E_BR[first_B(l), v] − E_BR[second_B(l), v]
  ```
- **【推导 + 数值验证】**
  - E_AR^T 的每一行都形如 e_a − e_b，元素和为 0，所以它的行空间落在 {x : Σx = 0} 中，维数为 r−1。因此 **rank(N_A) ≤ r−1**。脚本在 r=16 时得到 rank = 15。
  - N_A 的元素是两个独立 U[−32,31] 之差，服从 [−63,63] 上的三角分布。
  - N_A 的每一列只可能是 r(r−1) 种"列差"之一。在 k ≤ 4r² 的约束下，列重复必然出现（k > r(r−1) 时由抽屉原理），大 k 时也很常见。
  - A' = A + N_A ∈ [−127, 127]，正好是 int8。部分和的绝对值 ≤ 127²·2^16 ≈ 1.06·10^9 < 2^31，int32 不会溢出。

### 7.3 Ticket 与 "transcript 哈希" 的精确形式（【代码】`zk-pow/src/circuit/chip/jackpot/helper.rs:9-35`）
设 I_A（h 行）、I_B（w 列）是由周期模式加偏移 t_rows、t_cols 给出的非连续下标集（`proof_utils.rs:265-283`）。
```
for ℓ = 1 .. K/r:                          （K = dot_len = k − k mod r，见 proof_utils.rs:496-500）
    Z_ℓ[u,v] = Σ_{l < ℓr} A'[u,l]·B'[l,v]    （u∈I_A, v∈I_B，累积部分和，行 21-29）
    x_ℓ      = XOR_{u,v} (u32) Z_ℓ[u,v]      （行 30）
    s        = (ℓ−1) mod 16
    msg[s]   = rotl(msg[s], 13) XOR x_ℓ      （行 31-32；常量见 circuit/pearl_program.rs:21-25）
J = BLAKE3_keyed(msg 的 64 字节小端序列, key = a_noise_seed)
```
k ≥ 16r 的约束保证 K/r ≥ 16，即 16 个槽位都至少被写入一次。【推导】

### 7.4 逐条差异及其对论文论证适用性的影响
以下只描述"论文的证明或论证是否覆盖这个实现"，不评价好坏。前提是：论文本身对 hardness 只有假设、没有证明。所以这里的"覆盖"指"论文的假设或论证能否直接套用"。

| # | 论文 | Pearl 代码 | 论文论证是否覆盖 |
|---|---|---|---|
| D1 域/环 | F_q，q 大；元素在 F_q 中均匀 | 整数环 Z，A, B ∈ [−64,64]，A', B' ∈ int8，int32 累加不溢出（`proof.rs:23-25`，`verify.rs:113-118`，`kernel_traits.hpp:24`） | **不直接覆盖**。Lemma 6.5 和 Claim 6.6 都是有限域上的均匀性陈述，而这里元素来自很小的有界字母表，不是均匀分布。Assumption 6.3/6.4 讨论的是 F_q 上"随机矩阵"的代数复杂度，对小整数、低熵输入（可能存在查表或位级技巧）没有讨论。 |
| D2 噪声因子分布 | E_L, E_R, F_L, F_R 都在 F_q 上均匀，E 是均匀秩 r 矩阵 | 一侧稠密 U[−32,31]，另一侧是"每行恰一个 +1 一个 −1"的稀疏差分矩阵；秩 ≤ r−1；N 的列来自 r(r−1) 种模式，会重复；元素服从三角分布（`pearl_noise.rs:61-116, 142-153`） | **不覆盖**。Lemma 6.5（均匀秩 r）不成立，子块边际均匀性也不成立。论文"边际均匀则单块不可加速"的支持论据不能直接套用。稀疏设计的效果是：加噪变成 O(1)/元素，A·E_BL 和 E_AR·B' 变成 O(mk)/O(nk)（【推导】）。 |
| D3 种子派生 | E, F = O(σ, A, B) 同时派生；σ 均匀且对手事先不知 | σ ≈ job_key = H(header ‖ config)；F 的种子只依赖 (job_key, hash_b)，E 的种子依赖 (F 种子, hash_a)，有先后顺序（`proof_utils.rs:362-373`）；A、B 以 Merkle 根形式承诺 | **部分覆盖**。如果把 BLAKE3 建模为 RO，"种子在承诺之后才确定"这一点保留了下来。但对手可以**在看到 B 侧噪声之后再选 A**，论文模型里没有这种自适应顺序（FP8 白皮书 §4.6 明确承认了这一点）。σ 是 header 而不是均匀随机串，header 本身由矿工部分可控（timestamp、merkle_root），可以改变 σ 来"重新抽"噪声；论文 Def 5.1 中 σ 是均匀采样的。 |
| D4 transcript 粒度 | r×r 输出块 × r 长内积步，(n/r)³ 个中间块；r 同时是噪声秩和块边长 | 输出 tile 是 h×w（默认 2×64，下标非连续），内积步长为 r（同时是噪声秩），每个 ticket 有 K/r 个快照（`helper.rs:21-29`） | **不直接覆盖**。Assumption 6.4 是针对 r×r×r 的分块陈述的。代码中 tile 形状与 r 解耦，每步处理的是 h×r 的 A' 块和 r×w 的 B' 块。h = 2 远小于 r（128），这时"单块等价于一次 r×r 随机乘法"的论据形式不同。 |
| D5 transcript 压缩 | z = O(tr)，RO 作用于**完整** transcript（n³/r 个元素） | 每步把 h·w 个 int32 XOR 成 1 个 u32，再按 ℓ mod 16 做 rotl13-XOR 折叠进 16 个 u32，最后一次 64 字节 keyed BLAKE3 压缩（`helper.rs:30-32`，`proof_utils.rs:1502-1505`，`pow_utils.hpp:27-34, 80-115, 167-182, 205-234`） | **不覆盖**。论文的 ROM 论证要求 RO 输入包含全部中间值，才能推出"要么算出整个 transcript，要么只能盲猜"。代码中 RO 的输入是 transcript 经过 **GF(2)-线性压缩**后的 512 比特（XOR 和循环移位对比特表示都是线性的），大量不同的 transcript 会映射到同一条消息。对手只需预测这 512 比特，例如每步只需要 h·w 个值的 XOR，而不需要每个值本身。论文没有分析这种中间压缩。 |
| D6 ticket 与难度 | 主定义：一个 (σ, A, B) 对应一次抽签，ε 固定；Remark 2.1：每个中间块各自是一张 ticket | 每个 h×w 输出 tile（完整 K 长度、全部 K/r 个快照）是一张 ticket；中签阈值按 h·w·K 缩放（分叉后按 h·w·(K/r)·128）（`sanity_checks.rs:183-196`） | **介于两者之间**。它最接近 Remark 2.1（中签概率 ∝ 工作量），但 ticket 单位是"tile × 全部 k"而不是单个 r×r 中间块。论文对 Remark 2.1 只给出了一句非形式的硬度说明。rank-penalty 把单张 ticket 的"工作量"记作 h·w·K·128/r，这偏离了论文 MAC 计数的直接口径；其依据在提交 51a6e11b 的说明里（"the noise rank divides that work"），论文中没有对应内容。 |
| D7 验证范围 | Verify 重算整个 z（或通过 SNARK 证明"会接受"） | 只验证**单个 tile**：用 Merkle 打开 h 行 A、w 列 B 的 strip（长度 k），由 verifier 生成这些行列的噪声，重算这一张 ticket；可以是 plain 方式（`verify.rs:101-134`），也可以是 STARK 方式（`verify.rs:22-94`，噪声作为预处理列） | **语义变了**。verifier 不检查矿工是否算了其余部分，也不检查 C 是否正确。所以共识层的安全性只依赖"单张 ticket 的不可预测性"，即 Remark 2.1 那种变体，而不是 Def 6.2 中"整个 transcript"的版本。用 zk-STARK 隐藏 strip 对应 Remark 2.2 的 zkSNARK 思路。 |
| D8 去噪的精确性 | Decode 精确，Pr[C = AB] = 1（Def 5.1、Remark 5.3） | GPU 去噪用 fp16 因子 + fp32 累加 + 2^{12}/2^{14} 缩放，结果是浮点数（`collective_epilogue.hpp:369-457`，`pearl_gemm_constants.hpp:4-16`） | 论文的 completeness（精确）**不适用**【不确定：是否在全部取值范围内都恰好等于整数 AB，未验证】。这不影响共识，因为 verifier 不检查 C；它影响的是"useful"一侧的严格含义（参见 Remark 5.3 的近似放宽）。 |
| D9 参数 | 渐近：r = n^{0.3} 等，方阵 | r ∈ [32,1024]（分叉后 ≥ 128）、16r ≤ k ≤ 4r²、k ≤ 2^16、h·w ≤ 256、长方形 m×k×n（`sanity_checks.rs:28-58`） | 论文是渐近陈述，不给具体安全参数。代码中开销的主项约为 2·m·n·r（去噪，fp16）/ (m·n·k)（int8）≈ 2r/k 量级【推导；没有考虑 fp16 和 int8 吞吐差异】，受 k ≥ 16r 约束时 ≤ 1/8。 |
| D10 transcript 尾部 | r 整除 n | K = k − (k mod r)；内积最后 k mod r 列不进入 ticket（`proof_utils.rs:496-500`，`helper.rs:21`） | 论文假设 r \| k，这一点没有对应。 |

### 7.5 GPU 实现与电路实现的对应（【代码 + 推导】）
- **电路侧**：`helper.rs:30` 中，tile 的 h·w 个 int32 值按 u32 做 XOR。
- **GPU 侧**：`pow_utils.hpp:177` 的 `xor_reduction(tensor)` 对**该线程持有的全部累加寄存器**做 XOR（tensor 即 `tCrC`，见 `collective_mainloop.hpp:304`）。`rotl_xor<13>`（`pow_utils.hpp:179-180`）对应电路的 `rotate_left(13) ^`。
- **槽位**：GPU 在每个 k_tile 内用 `k_block / ReduceEveryK` 加上循环偏移 `m_reduction_count` 确定槽位（`pow_utils.hpp:178, 186-198`）。电路用 (ℓ−1) mod 16。
- 所以一张 GPU ticket = 一个线程的 fragment = 电路中的一个 (I_A, I_B) 周期模式。默认 rows [0,8] × 64 列，正好是 WGMMA 的线程寄存器布局【推导】。
- 每个 CTA tile 结束时，每个线程独立做一次 BLAKE3 和目标比较（`pearl_gemm_kernel.h:262-271`）。所以整个 GEMM 中 ticket 的数量 ≈ mn/(hw)。
- 【不确定】对于 R > bK 等边界情形，GPU 与电路的槽位映射是否逐比特一致，我没有在本笔记中逐比特核对，只确认了结构上一致。

---

## 8. FP8 白皮书相对论文的变化（简要，详见另一份研究）
**【白皮书】**
- **矩阵乘积形式**（§2）：C̃ = Q(A+N_A)·Q(B+N_B)^T。其中 Q 是 FP8（E4M3FN）的逐元素量化；N 是秩 r 的噪声；去噪为 Ĉ = C̃ − A N_B^T − N_A B^T − N_A N_B^T（近似）。
- **核心变化**：量化是非线性的，Q(A+N_A) ≠ Q(A) + Q(N_A)，所以 Q(N_A) 的秩可能远高于 r。白皮书主张这会**堵住输出捷径**，因此 ticket **直接哈希输出 tile**，不再哈希 transcript（§2 "Relation to the transcript-unpredictability assumption"）。
- **Assumption 1**（非形式，quantized-subspace hardness）：对每个合法的 k，存在 r ≪ k，使得以高概率（对随机 r 维子空间 U_A, U_B）、在平均意义上，计算 Q(N_A)Q(N_B)^T 并不比计算一般随机矩阵的 Q(A)Q(B)^T 实质更容易，两者都约需 mnk 次 MAC。具体取 r = 32、k ≤ 2^16。白皮书明确说它比较的是计算难度，而不是分布。A = B = 0 正好"隔离"出这一前提。
- **Ticket**（§3、§4.3）：Ticket = H_jackpot(Extract(C̃_{I_A,I_B}); noise_seed_A)。Extract 把 tile 分成 16 个子 tile，每个子 tile 用 `w ← rotl13(w·c + bits(x)) mod 2^32` 折叠成一个 32 位字，得到 64 字节。与 int 版不同的是，它对 FP32 比特模式做带奇数乘法常数的混合，而 int 版是纯 XOR 归约。
- **验证**（§4）：只重算一个 tile。依赖 Badash 等人对 GPU 浮点运算的逐比特复现。难度按 |I_A|·|I_B|·k 缩放。参数为 r = 32、1024 ≤ k ≤ 2^16、|I_A|·|I_B| ∈ [256, 2048]。
- **新增 Jackpot policy**（§5）：entry liveness、noise floor、tamed products、unpredictable summands 四项逐 tile 的准入检查，用来应对矿工在 A、B 中植入能"穿过加噪和量化"的结构（§5.2）。这些检查在论文中没有对应，是对"任意 A, B"这一缺口的工程化补丁【推导】。
- **种子顺序**（§4.6）：B 先承诺，A 可依赖第一个挑战，并承认 grinding 和跨承诺共享工作需要由"批量/grinding 安全定义"来处理。
- **【代码】** 在 156ef9a8 中，共识代码仍只有 `Int7xInt7ToInt32`（`proof.rs:23-25`），没有找到 FP8 共识实现（`zk-pow/src` 中 grep 不到 e4m3/fp8）。

---

## 9. 疑点（留待问题阶段分析）
1. **压缩前置**：ticket 的 RO 输入是 transcript 经过 GF(2)-线性压缩（逐步 XOR、rotl13 折叠到 16 个槽）后的 512 比特。论文的 ROM 论证要求哈希完整 transcript，这一层压缩是否引入"只需预测 XOR 值"的捷径，尚未分析。
2. **噪声秩与熵**：N_A 的秩 ≤ r−1，列只来自 r(r−1) 种差分模式。在 k 接近 4r² 时大量列重复或互为相反数，这对计算部分和的捷径有什么影响，尚未分析。
3. **A=B=0 时的增量捷径**：Z_ℓ = E_AL[I_A]·T_ℓ·E_BR[:,I_B]，其中 T_ℓ = Σ_{l<ℓr} E_AR[:,l]E_BL[l,:]，每步只增加 O(r) 个 ±1 的稀疏外积。增量维护 u_ℓ = E_AL[I_A]·T_ℓ 的成本约 O(h·r)/步，再乘 E_BR 约 O(h·w·r)/步。这与 honest 的每步 h·w·r 是同阶的，常数和可利用的结构需要实测和细化。
4. **h 很小（默认 h=2）**：每张 ticket 只涉及 A' 的 2 行。基于"r×r 块是随机 r×r 乘法"的论据形式不再适用，单张 ticket 的低秩分解捷径成本需要重新估计。
5. **自适应选择 A**：A 可以在看到 b_noise_seed（即 E_BL、E_BR）之后再选，论文的同时派生模型没有覆盖这一点。
6. **σ 可被研磨**：σ 来自矿工可调的 header（timestamp、merkle_root），也可以通过换 config 来调。可以对噪声种子做研磨，寻找结构较好的稀疏下标（例如更多重复列）。
7. **rank-penalty 的口径**：提交 51a6e11b 说"rank 越大单次尝试越便宜"，把工作量按 128/r 折算。这说明开发者认为单张 ticket 的实际成本随 r 变化。其来源是捷径、XOR 归约频率还是别的因素，需要查证。另外 r = 128 时公式与原口径一致，但当 r ∈ {256, …, 1024} 时 k ≤ 4r² 的上界也随之放宽。
8. **Assumption 6.4 只给出 n³/r 量级的下界**，而 honest 用 naive 算法成本是 n³。理论上，实用的长方 FMM（或者 int8 上的其他技巧）能把差距拉大到最多 r 倍，论文只用"实践中 naive 最优"来回应。
9. **Def 6.2 / direct-product 形式没有形式化陈述和证明**，"Assumption 6.4 ⇒ PoUW hardness"这一步在论文中是直接断言的。
10. **Lemma 6.5 的小瑕疵**：行/列空间写反了；"子块边际均匀"实际上是近似的。在 Z 上、小字母表的情形下，这个引理完全不适用。
11. **尾部未进入哈希**：K = k − (k mod r)，最后 k mod r 列不进入 ticket，但仍然被承诺、也在 GEMM 中被计算。
12. **单 tile 验证**：verifier 不检查矿工是否真的计算了整个乘积，"useful"在共识上不可验证。论文的 Solve/Verify 框架同样不验证 C，但论文的 Verify 要检查整个 transcript。
13. **GPU 去噪精度**：使用 fp16 去噪因子（数值很大的 E_AR·(B+N_B) 被缩放到 fp16）。它与精确 AB 的误差界没有核实，这影响 Remark 5.3 意义上的"正确性"。
14. **白皮书 Extract 改用 `w·c + bits(x)` 乘法混合**，而 int 版是纯 XOR。这可能说明设计者已经意识到纯 XOR 折叠的线性问题，需要对照 int 版的历史决策。
15. **论文效率公式漏掉了 transcript 哈希的 Θ(n³/r) 项**（§2 中 "o(n)" 的说法），以及 Alg 6.2 的 Verify 字面上不满足 Def 5.1 的 "c<1"。这些都是论文自身的表述问题，需要确认是否在更新版本中修正。
