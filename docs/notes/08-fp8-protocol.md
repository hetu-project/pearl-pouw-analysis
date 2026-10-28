# 组件 8：FP8 新协议（Pearl Floating Point Scheme）

> 研究阶段笔记：只描述协议是什么、怎么工作、为什么这样设计。疑点只在最后一节列出，不展开。
>
> 标注约定：
> - **【原文】**：白皮书 *Pearl Floating Point Scheme Specification*（Pearl Research Team, 2026-09），`papers/pearl_whitepaper_fp8.txt`，用 §节号 / (式号) 引用；KW 原论文 `papers/cupow_2504.09971.txt` 记为 [KW]。
> - **【代码事实】**：Pearl 仓库 `third_party/pearl`。FP8 实现**只在远程分支 `origin/fp8-scheme`**（head `1280a21c`，2026-09-28）上，写作 `fp8:路径:行号`；`master`（`156ef9a8`）上的写作 `master:路径:行号`。
> - **【推导】**：我自己的推导、验算或理解。
> - **【实验】**：`experiments/fp8/experiments.py` 的结果，完整输出见 `experiments/fp8/experiments_output.txt`，工具函数在 `experiments/fp8/fp8lib.py`。环境里没有 numpy 和 blake3，所以用纯 Python 写，PRF 用 blake2b 代替 BLAKE3。这里只验证分布性质，与具体 PRF 无关。
> - **【网络】**：带链接。

---

## 0. 一页摘要

| 项 | 内容 |
|---|---|
| 工作单元 | GPU 上 E4M3FN（FP8）× E4M3FN 的矩阵乘，FP32 累加 |
| 彩票对象 | 输出 tile `C̃_{IA,IB}` 本身，经 Extract 折叠成 64 字节后做 BLAKE3。**不再哈希计算 transcript** |
| 关键思想 | 先加低秩噪声，再做非线性量化 Q，低秩捷径就失效了（`Q(A+N_A) ≠ Q(A)+Q(N_A)`）；去噪仍然便宜（§2） |
| 安全假设 | Assumption 1：量化子空间困难性（§2），外加 jackpot policy 四项统计检查（§5） |
| 验证 | 验证者在 CPU 上逐比特重算一个 tile。依据是 Hawkeye（Badash–Boneh–Komargodski–Srivastava）对 GPU 累加行为的刻画，因此必须设置 **Device 白名单**（Hopper / Blackwell） |
| 落地 | 代码在 `fp8-scheme` 分支上已基本完整（Rust 验证器、ZK STARK、Go 节点 V4 证书、vLLM 矿工），**未合入 master**。主网和测试网都没有设置 `Fp8ForkHeight`，PIP-3 仍是 Draft，没有激活高度 |

---

## 1. 协议总体

### 1.1 核心公式【原文 §2】

- 有用乘积：`A ∈ R^{m×k}`, `B ∈ R^{n×k}`, `C := A Bᵀ`（两个操作数都按行存储，`C_ij = ⟨A_i, B_j⟩`，§3）。
- 噪声：由种子派生的秩 r 噪声 `N_A ∈ R^{m×k}`、`N_B ∈ R^{n×k}`。每个噪声矩阵都在对应输入承诺**之后**才变得可知（§2）。
- 彩票乘积 (1)：
  `C̃ := Q(A + N_A) · Q(B + N_B)ᵀ`
- 去噪（"cheap correction"，§2）：
  `Ĉ := C̃ − A N_Bᵀ − N_A Bᵀ − N_A N_Bᵀ`
  不量化时 `Ĉ = C`。量化后的配置使 `Ĉ` 对 `C` 的近似精度与基线 `Q(A)Q(B)ᵀ` 相当。r 很小，所以减去的三项都便宜。
- 实际实现里噪声是带行缩放的：`X̃_t = RNE_FP8(α_t X_t + β_t N_t)`（App. C.4）。所以去噪时还要除掉每行的 α 缩放（§6 "per-row descaling"）。

### 1.2 与 int7 方案的根本区别：量化非线性 → 直接哈希输出【原文 §2 "Relation to…"；推导】

- KW / int7 的处境：精确算术下，敌手取 `A = B = 0` 时，加噪乘积就是 `N_A N_Bᵀ = (E_A F_Aᵀ)(E_B F_Bᵀ)ᵀ`，用低秩分解 `O((m+n)kr + mnr)` 就能算出来，**输出本身有捷径**。所以 KW 不能哈希输出，只能哈希计算的 **transcript**：所有 `(n/r)^3` 个中间 r×r 块的部分和，见 [KW] 第 300–330 行与 Assumption 6.4（[KW] 第 848 行）。Pearl 的 int7 实现对应为"每 r 个公共维度做一次 inner hash"，见 `master:zk-pow/src/api/proof.rs:65` 中 rank 的注释 "length of inner product per inner hash invocation"。
- FP8 方案：Q 逐元素作用、非线性，`Q(N_A)` 的秩远高于 r（§2）。白皮书据此假设：即使在 A=B=0 这种"最省事"的情况下，`Q(N_A)Q(N_B)ᵀ` 也没有捷径。于是可以**直接哈希输出 tile**。这样做的好处：
  1. 矿工只需要标准的 FP8 GEMM 内核，不必在 GEMM 内部插入每 r 步的 transcript 哈希（§6 step 4："the same kernel the job would run without the protocol"）；
  2. 验证只需重算一个 tile 的最终值。
- 代价【推导】：
  1. 安全性不能再依赖 KW 的"精确算术 + 直积猜想"框架，改为依赖 FP8 量化器的具体性质（Assumption 1）和 GPU 累加的具体行为。因此 Quant 与 Device 都要白名单化（§2 末段："This premise depends on the data type and quantizer, so the protocol whitelists both"）。
  2. 敌手控制 A、B 后可以尝试"植入"能扛过加噪和量化的结构，这需要第 5 节的统计检查来兜底（§2 末段）。

### 1.3 Assumption 1（量化子空间困难性）【原文 §2】

原文（意译要点；完整原文见 txt 第 127–138 行）：
> 对每个允许的 k，存在 r ≪ k，使得以下成立：对随机 r 维子空间 `U_A, U_B ⊂ R^k`，绝大多数情况下，若 `N_A, N_B` 的行分别从 `U_A, U_B` 中随机抽取，那么**平均而言**，计算 `Q(N_A)Q(N_B)ᵀ` 并不比计算 `Q(A)Q(B)ᵀ`（A、B 为一般随机行）实质上更容易。两边用同一个协议批准的 Q，计算量都约为 mnk 次 MAC。

- 参数：`r = 32`，适用于 `k ≤ 2^16`（§2）。
- 白皮书特别说明："The assumption compares computational difficulty, not the two input distributions"，即它比较的是计算难度，不是两个输入分布是否相同。
- 含义【推导】：
  - 这是**平均情形**的、**非正式**的计算假设，没有给出归约。
  - 它把"零输入"情形（`C̃_0 = Q(N_A)Q(N_B)ᵀ`）孤立出来，作为最干净的困难性前提。
  - 对任意（敌手选的）A、B，Assumption 1 本身**不覆盖**。白皮书把这部分交给 §5 的 jackpot policy（"A miner controls A and B and can try to plant structure… Section 5 specifies the policy checks"）。

### 1.4 与 KW 直积猜想的关系【原文 §2；KW；推导】

| | KW（[KW] §2、Assumption 6.4、Conjecture A.1） | Pearl FP8 Assumption 1 |
|---|---|---|
| 算术模型 | 精确算术，随机预言机模型 | 具体 FP8 格式 + 白名单量化器 + 设备累加语义 |
| 困难对象 | 两个随机秩 r 矩阵乘积的全部中间值（transcript），"direct-product problem of correlated instances"（[KW] 第 143 行） | 量化后低秩噪声矩阵乘积的**输出** |
| A=B=0 | 在直积猜想下"与任何乘积一样难"（[KW] 第 144–145 行） | 正是 Assumption 1 的设定 |
| 任意 A,B | 由猜想本身覆盖（"multiplying the all-zeroes matrices… is as hard as any other product"） | 不被假设覆盖，依赖 §5 policy |

白皮书称 Assumption 1 为 transcript-unpredictability 假设的"floating-point analogue"（§2）。【推导】两者结构相同，都是"低秩噪声 + 某种破坏捷径的机制"。KW 用"必须给出所有中间值"来破坏捷径，FP8 用"量化非线性"来破坏捷径。

---

## 2. 公共参数、状态窗口、哈希与种子链

### 2.1 公共参数【原文 §4.2，式 (2)】

`p_B := (n, k, r, Quant, Device, hash_id_B, P_B, e)`, `p_A := (m, hash_id_A, P_A)`

- `r`：噪声秩，必须为 32（(5)）。
- `Quant`：量化方案枚举。白皮书只定义了一种：FP10 承诺类型（C.6）+ noisy quantization（C.4）。
- `Device`：GPU 架构枚举，取自共识白名单。
- `hash_id_X`：哈希方式标识，控制 BLAKE3 内部的 chunk 大小。【代码事实】可选 chunk 长度为 128/256/512/1024，见 `fp8:zk-pow/src/api/fp8/public_params.rs:53-79`。
- `P_A, P_B`：矿工选择的**周期性划分**，决定 tile 形状，以及 Extract 如何把 tile 切成 16 个子块（§4.2、§4.3）。
- `e`：MoE 专家数，普通矩阵乘时为 0。
- 【推导】公共维度的参数（k、r、Quant、Device）都放在 `p_B` 里，因此在 `noise_seed_B` 生成之前就已固定。这些参数无法根据噪声再挑选。
- 【代码事实】`Device` 只有 `H100 = 0`、`B200 = 1` 两个取值（`fp8:zk-pow/src/api/fp8/public_params.rs:144-147`），未知值直接拒绝（同文件 `TryFrom<u8>`，约 200–209 行）。`CommonParams::check` 要求 `r == 32`、`32 | k`、`1024 ≤ k ≤ 2^16`（同文件 218–226 行）。

### 2.2 状态窗口 D = 3【原文 §3 "Blockchain state…"、§4.6 "State window"】

- 头部按相对深度编号：`σ0 = σ̂`（提议头），`σ1` 为父（当前 tip），`σ2` 为祖父。
- B 的承诺可以绑定到窗口内任意一个 `σ_d`（`d < D`），A 的承诺**必须**绑定 `σ̂`。
- 论证（§4.6）：
  - 这至多给 B 侧带来 D 倍的预处理机会；
  - 对有用矿工来说，tip 变化后还有最多 D−1 个区块间隔可以更换矩阵，不必立刻重做；
  - A 的承诺绑定提议头本身。
- 【代码事实】V4 证书最多携带 2 个祖先头（`fp8:node/wire/certificate_v4.go:16`，`MaxCertificateV4AncestorHeaders = 2`），即 σ0、σ1、σ2，对应 D=3。祖先关系的验证在 Rust 中进行（`fp8:node/zkpow/verify.go:117-172`）。

### 2.3 域分离哈希【原文 §3】

- `Subkey(q, ℓ) := BLAKE3(ℓ; key = q)`
- `H_ℓ(x; q) := BLAKE3(x; key = Subkey(q, ℓ))`
- 没有父密钥时：`H_ℓ(x) := BLAKE3(x; key = BLAKE3(ℓ))`。保留域分离接口，但不提供密钥不可预测性。
- 实际标签包含协议版本（§3）。【代码事实】标签前缀是 `pearl/v4/FP8/`，共有 `key-A`、`key-B`、`seed-A`、`seed-B`、`zk-public`、`noise-line`、`jackpot` 七个（`fp8:zk-pow/src/api/fp8/transcript.rs:58-77`）。消息是定宽字段的直接拼接，无标签、无长度前缀（同文件 29–30 行）。

### 2.4 种子链与挑战顺序【原文 §4.5 step 2–3，式 (9)(10)；§4.6】

```
key_B        := H_"key-B"(σ_d)                         ← 任意窗口内祖先
key_A        := H_"key-A"(σ̂)                           ← 提议头
(H_B, B_IB)  := VerifyOpen_B(π_B; key_B, p_B, I_B)     ← Merkle 根以 key_B 为哈希密钥
(H_A, A_IA)  := VerifyOpen_A(π_A; key_A, p_A, I_A)
noise_seed_B := H_"seed-B"(H_B, key_B, p_B)
noise_seed_A := H_"seed-A"(H_A, noise_seed_B, key_A, p_A)
J            := H_"jackpot"(z; noise_seed_A)
```
【代码事实】与上式逐项一致，见 `fp8:zk-pow/src/api/fp8/transcript.rs:90-106,167-188`。MoE 情形在 `H_A` 之后插入 `H_R, H_O`（同文件 174–183 行）。

白皮书的论证（§4.6）：
- **Challenge order**：矿工先固定 B 和 `p_B`，才拿到 `noise_seed_B`；之后可以选 A，但必须在拿到 `noise_seed_A` 之前固定 A 和 `p_A`。所以 **A 可以依赖第一个挑战，但不能依赖第二个挑战**（Fiat–Shamir 顺序）。
- **把 p_X 并入种子**：维度、噪声秩、Quant、Device、哈希 id、划分都与噪声挑战绑定。原文明确承认："These bindings do not prevent transcript grinding or shared work across commitments and depths. The batched and grinding security definitions must account for those choices."
- **Hash domains**：`key-A/B`、`seed-A/B` 区分操作数角色和 transcript 阶段。采样器用种子派生的密钥加上类型化地址 `(X, Z, ℓ)` 作为消息，因此可以随机访问，且与抽取顺序无关。种子已经绑定了 k 和 r，所以地址里不重复它们。`noise-line` 与 `jackpot` 从同一个种子派生出不同的密钥。
- 【推导】Merkle 树的哈希以 `key_X` 为密钥（`key_X` 来自区块头）。因此**承诺本身**也无法在区块头出现前预计算，这比只用种子绑定更强：A 的整棵树必须在 σ̂ 已知后才能计算。
- 【代码事实，与白皮书 §4.5 不一致之处】白皮书 §4.5 step 4 写的是 `F_X[u,:] := SampleLine(s_X, X, F, u)`，按字面理解 `F_A` 由 `noise_seed_A` 派生。但白皮书 §5.2 又说 "F_A, E_B, or F_B are derived from B's commitment alone"。代码采用后者：`F_A` 和 `F_B` 都以 `noise_seed_B` 为密钥，只是地址里的 Side 不同（`fp8:zk-pow/src/api/fp8/noise.rs:25-29,163-170`）。这个改动来自提交 `5dcecff3`（2026-09-23，"Key F_A from seedB…"），动机是让"peel"的中间项成为每个作业的常量，便于矿工预处理。

---

## 3. 验证者完整流程（§4.1 / §4.4 / §4.5）

### 3.1 证明内容（§4.4）
证明携带：所选祖先头 `σ_d`、`p_A, p_B`、tile 基址 `t_A, t_B`、开口 `π_A, π_B`（干净的 A/B 条带的叶子和 Merkle 兄弟哈希）。
**不携带**：根 `H_A, H_B`、噪声种子、噪声因子、加噪条带、乘积 tile、extractor 输出、jackpot 哈希。这些全部由验证者重算。

### 3.2 Step 1：CheckPublic，式 (4)–(8)
- (4) `CheckPublic(Σ, σ_d, p_A, p_B, t_A, t_B, π_A, π_B)`：检查 σ̂ 的父连接、σ_d 属于 Σ、其他链字段，对 p_A、p_B 做健全性检查，并要求 e=0。
- (5) `1 ≤ m,n ≤ 2^24`，`1024 ≤ k ≤ 2^16`，`r = 32`，`32 | k`。
- (6) `I_A ⊆ {0..m−1}`，`I_B ⊆ {0..n−1}`，其中 `I_X = Rows(P_X, t_X)`（(3)）。
- (7) Extract 输出的字数组形状 `b_A × b_B` 满足 `b_A·b_B = 16`（64 字节），`|I_A| ≥ 4`，`|I_B| ≥ 16`，`256 ≤ |I_A||I_B| ≤ 2048`。
- (8) `k(|I_A| + |I_B|) ≤ 2^22`（性能约束）。
- §4.6 的解释：上界限制验证工作量，下界保证 Extract 有足够的熵输入。`k ≥ 1024` 和 `|I_B| ≥ 16` 还用于抵抗噪声种子 grinding（§5.2）。
- 【代码事实】`check_lottery_layout` 实现了 (7)，另外要求每个子块元素数在 `[16, 256]` 之间（`fp8:zk-pow/src/api/layout.rs:25-75`）。(8) 的实现见 `fp8:zk-pow/src/api/fp8/public_params.rs:936-946`。

### 3.3 Step 2：Merkle 打开，式 (9)
`VerifyOpen_X` 检查以下几点：
- `π_X` **恰好**打开 `I_X`；
- 使用唯一的最小叶子与兄弟集合，防止填充证明，并在算术开始前约束输入规模（§4.6 "Openings"）；
- 打开的值按 Quant 指定的编码（FP10）返回；
- 拒绝 NaN/Inf，接受次正规数。

§4.6 指出："An opening authenticates only the disclosed rows. It does not prove knowledge of the undisclosed matrix entries"，即打开只认证公开的那几行，不证明矿工知道未公开的矩阵元素。

### 3.4 Step 3：噪声种子，式 (10)（见 §2.4）

### 3.5 Step 4：SampleLine → NoisyQuantize → MatMul_Device → Extract → J
- `E_X[ℓ,:] := SampleLine(s_X, X, E, ℓ)`（ℓ ∈ I_X），`F_X[u,:] := SampleLine(s_X, X, F, u)`（0 ≤ u < k）。得到 `E_X ∈ FP8^{|I_X|×r}`、`F_X ∈ FP8^{k×r}`，每个条带只在打开的行上展开秩 r 的噪声 `E_X F_Xᵀ`。
- `Ã = NoisyQuantize_{p_A}(A_IA, E_A, F_A)`，`B̃` 同理（App. C.4）。
- `T := MatMul_Device(Ã, B̃)`（App. C.2，与矿工所用内核**逐比特一致**）。
- `z := Extract(T, P_A, P_B)`：把 tile 划成 16 个子块，每个子块折叠成一个 32 位字（§4.3）：
  `w ← rotl13(w·c + bits(x)) mod 2^32`，其中 c 为奇常数，`bits(x)` 为 FP32 位模式。
  【代码事实】`c = 0x9E3779B1`（黄金比例常数），先乘加再循环左移 13 位，16 个字按小端序拼成 64 字节，恰好是一个 BLAKE3 块（`fp8:zk-pow/src/api/fp8/utils.rs:540-569`）。PDF 抽取的文字里括号位置不清楚，这里以代码为准。
- `J := H_"jackpot"(z; noise_seed_A)`。
- 所有中间结果必须有限，出现 NaN/±Inf 即拒绝。

### 3.6 Step 5：policy 与目标，式 (11)(12)
- (11) `JackpotPolicy(I_A, A_IA, I_B, B_IB; σ̂, σ_d, p_A, p_B, H_A, H_B) = approve`（§5.1）
- (12) `uint256_LE(J) ≤ τ · |I_A| · |I_B| · k`
- §4.6 "Difficulty and eligibility"：目标值按 `T_tile = |I_A||I_B|k` 缩放，使每个 tile 的期望中奖次数正比于它的直接 MAC 成本。**policy 决定 tile 是否有资格参与，target 决定有资格的 tile 的中奖率**，两者分工不同。
- 【代码事实】`check_jackpot_difficulty`：`bound = target × (h·w·k)`，溢出时饱和到 `U256::MAX`（`fp8:zk-pow/src/api/proof_utils.rs:123-138`）。plain 验证的总入口是 `fp8:zk-pow/src/api/verify.rs:56-88`：打开 → 噪声 → NoisyQuantize → policy（同时得到 extract 消息）→ ticket → 难度检查。

---

## 4. 附录 C：逐比特算术

### 4.1 E4M3FN 编码【原文 C.2、C.1】
- 1 位符号、4 位指数（偏置 7）、3 位尾数。非零有限码值为：
  `x = s·m·2^-6`（e=0，次正规数，m ∈ {0,1/8,…,7/8}）；`x = s·(1+m)·2^(e−7)`（e>0）。
  存储指数 `ε(x) = −6`（e=0）或 `e−7`。
- 最大有限值 448（码 0x7E）。"FN" 表示没有 Inf，NaN 为 `S.1111.111`。【推导】最小次正规数是 2^-9。
- 降型转换 RNE 到 FP8：`|x| > 464` 饱和为 448；`|x| = 464` 的平局按 ties-to-even 舍到 448（C.1）。BF16 溢出则直接拒绝。
- 逐元素运算（C.1）：Mul/Div 先在 FP32 中按 IEEE 754-2019 计算再降型到 BF16；FMA_BF16 一次舍入**直接到 BF16**；Min/Max 不舍入，平局返回第一个操作数（包括 +0/−0）。这些运算与 Device 无关。

### 4.2 累加器精度与按 32 一组截断累加【原文 C.2】
- 乘积精确计算。
- 累加器内部尾数位数随架构而变：**Hopper W=13**（14 位有效数字），**Blackwell W=25**（26 位）；FP32 为 23 位。指数范围 [−126, 127]。
- 每 32 个乘积为一组：用组内（乘积 + 累加器）的最大存储指数对齐，每项向零截断到 `2^(η−W)` 的倍数，精确求和，再向零截断到内部精度。若 W > 23（Blackwell），输出时再向零截断到 FP32。
- 乘积**不做规格化**，例如 1.5×1.5=2.25 保持指数 0+0=0。次正规输入的存储指数为 −6。
- 伪代码（原文）：
```
products:     p_i ← a_i b_i                       (exact)
exponent:     η ← max(ε(c), max_i{ε(a_i)+ε(b_i)})
summation:    S ← Trunc(c, 2^(η−W)) + Σ_i Trunc(p_i, 2^(η−W))   (exact)
zero canon.:  if S = 0: return +0
rounding:     c ← Trunc(S, 2^(⌊log2|S|⌋−W))
output:       if W > 23: c ← Trunc(c, 2^(⌊log2|c|⌋−23))
```
- 【实验 E6】按上述伪代码实现（有理数精确算术）。一个乘积为 448×448、其余为极小乘积时，W=13 丢失 0.0605，W=25 丢失 0.0137，说明低于网格的小被加数会被截掉。随机 k=128 的单元中，W=13 与 W=25 的结果在 197/200 个上不同，说明 Device 参数决定比特级结果。
- 【代码事实，与 C.2 的差异】
  1. B200：`matmul_fp8_windowed` 用 26 位窗口按 32+1 项一组链式累加，与 C.2 一致（`fp8:zk-pow/src/api/fp8/utils.rs:12-17,147-298,329-347`）。
  2. H100：代码**不是** C.2 描述的单一链式累加。它的做法是每 128 个乘积（4 个 rank-32 WGMMA atom）开一个新的本地累加器（初值 +0，14 位窗口），窗口结束时用**一次 FP32 RNE 加法**并入全局总和，即 "promotion every 4 WGMMAs"（同文件 19–20、444–549 行；提交 `0b18ef42`，2026-09-27）。白皮书 C.2 没有描述这种两级结构。

### 4.3 Noise-line sampler【原文 C.3】
- `noise_key_X := Subkey(noise_seed_X, "noise-line")`。
- `(b_0..b_{r−1}) := BLAKE3-XOF((X,Z,ℓ); key=noise_key_X)[0:r]`。
- 每字节拆成 `b = 128s + m`，`x_u = (−1)^s(1+m)`，在 ±{1..128} 上均匀分布，不含 0，也没有取模偏差。
- `q := isqrt(32²·Σx_u²)`，`γ := Div_BF16(8192, RNE_BF16(q))`；`SampleLine_u := RNE_FP8(Mul_BF16(x_u, γ))`。范数约为 8192/32 = 256 =: c。
- 【代码事实】地址为 `side|factor|line(u32 LE)` 补零到 64 字节（一个 BLAKE3 块）（`fp8:zk-pow/src/api/fp8/noise.rs:90-141`）。
- 【实验 E1】4000 条线的平均范数为 255.8（范围 248.8–263.5）；每个元素约有 120 个不同的 FP8 取值，Shannon 熵约 5.95 bit/元素，最小熵约 4.51 bit/元素。按 32 个元素独立估算，一条线的最小熵约 144 bit，与 §5.2 中 "SampleLine has about 150 bits of min-entropy" 的量级一致。

### 4.4 Noisy quantization，α、β 推导【原文 C.4；推导；实验】
目标（每行 t）：
1. `α_t X_t + β_t N_t` 的每个元素都落在 [−Q, Q] 内，Q=448；
2. `‖β_t N_t‖_2 = δ‖α_t X_t‖_2`，噪声与信号的范数比固定为 δ。

**β 的来历（自行验算）**：
- 记 `ℓ2(X_t)` 为**均方根**（RMS），所以 `‖α_t X_t‖_2 = α_t √k · ℓ2(X_t)`。
- `N_tu = ⟨E_t, F_u⟩`，两条线的范数都约为 c，在 r 维中方向近似随机且独立。对固定的 e 和随机方向的 f，有 `E[(e·f)²] = ‖e‖²‖f‖²/r = c⁴/r`，所以 `rms(N_t) ≈ c²/√r`，`‖N_t‖_2 ≈ √k · c²/√r`。
- 代入条件 2：`β_t · √k · c²/√r = δ · α_t · √k · ℓ2` ⟹ **`β_t = α_t · ℓ2(X_t) · δ · √r / c²`**，与原文一致。
- 条件 1：由 Cauchy–Schwarz 得 `|N_tu| ≤ c²`，所以 `|β_t N_tu| ≤ α_t ℓ2 δ √r`。又 `|α_t X_tu| ≤ α_t ℓ∞`，取等号即得 **`α_t = Q / (ℓ∞(X_t) + δ√r · ℓ2(X_t))`**（信号峰值加噪声峰值）。
- 【实验 E2】实测 `rms(N) = 11664`，理论值 `c²/√r = 11585`；在 8192 个元素上实测峰值为 43686，低于 C–S 上界 65536。高斯行上，δ=0.5 时 `‖βN‖/‖αX‖` 的均值为 0.502（范围 0.489–0.525），δ=1 时为 1.005。原文的两条说明与此相符："(2) holds only approximately and with high probability"；clamp 只处理舍入导致的越界。
- 每行噪声尺度（量化单位）：`σ_t = δ · α_t · ℓ2(X_t)`（§5.1）。【实验 E2】k=1024 的高斯行，σ 在 δ=0.5 时约 32，δ=1 时约 46，远高于 σ_min = 1。

逐比特计算（C.4，原文）：
```
dr := RNE_BF16(δ√r),  er := RNE_BF16(δ√r / c²)
ℓ̂2 := max(ℓ2, 2^-32), ℓ̂∞ := max(ℓ∞, 2^-32)
z_t := dr·ℓ̂2 + ℓ̂∞            (FMA_BF16)
α_t := Q / z_t
β_t := (α_t · ℓ̂2) · er
N   := RNE_BF16(MatMul_Device(E, F))       ← E@F 也用设备的 FP8 MMA
y_tu := α_t·X_tu + β_t·N_tu               (FMA_BF16，一次舍入)
X̃_tu := RNE_FP8(clamp(y_tu, −Q, Q))
```
- 固定参数（原文）：`Q = 448`，`δ = 1/2`，`c = 256`。
- 【代码事实】`derive_row_scales` 和 `noisy_quantize` 与上面逐行对应（`fp8:zk-pow/src/api/fp8/quantization.rs:135-196`）。**δ 在代码中与设备相关**：`lg2_delta` 在 H100 上为 0（δ=1），在 B200 上为 −1（δ=1/2）（`fp8:zk-pow/src/api/fp8/public_params.rs:152-166`），同样来自提交 `0b18ef42`（"device-dependent delta (0.5 Blackwell, 1.0 Hopper)"）。白皮书写的是固定的 δ=1/2。

### 4.5 FP10 承诺格式【原文 C.6】
- 每个操作数承诺为两个矩阵：`V ∈ {−128..127}^{n×k}`（int8），以及 `Σ ∈ BF16^{n×k/8}`（每 8 个元素共享一个块尺度）。元素值为 `V_tu · Σ_{t,⌊u/8⌋}`，平均每元素 8 + 16/8 = 10 bit，因此叫 "FP10"。
- 两个矩阵各自按行建 Merkle 树，聚合根 `H_X` 是 `R_V`、`R_Σ` 和两者形状的哈希。Σ 中出现 NaN/Inf 即拒绝。
- 解码：`X_tu := RNE_BF16(V_tu·Σ)`。乘积最多 15 个有效位，在 FP32 中精确，只经过一次舍入。
- 行范数直接由 V、Σ 计算：`S_tb = Σ V²`、`A_tb = max|V|`。按块顺序用 FP32 累加 `s_{t,b+1} = s_{t,b} + S_tb·Σ²_tb`，然后 `ℓ2 = Grid(RNE_BF16(√(s_t/k)))`，`ℓ∞ = RNE_BF16(max_b A_tb·Σ_tb)`。Grid 把 BF16 最低 2 位尾数舍掉（向最近、平局向上）。
- 预量化（把真实矩阵编码成 FP10）不在协议规定范围内（§6 step 1）。
- 【代码事实，与 C.6 的差异】代码中的 `exact_norms` **不是**按块顺序的 FP32 累加。它用整数帧做精确求和（`Wl2 = 27 − ⌈log2 k⌉` 位精度），开方时只舍入一次（`fp8:zk-pow/src/api/fp8/prequant.rs:24-27,36-50,63-123`）。提交 `33bfaaeb` 让矿工端对齐到"verifier integer-based L2 formula"，随后 `b2053745` 又在矿工端恢复了 fp32 公式"for GPU compatibility"。`L2_ROUNDED_BITS = 2` 与 Grid 一致。

### 4.6 不可预测性 Λ 的定点近似与误差界【原文 C.5】
- 目标：判定 `v_iju < ulp(M_ij)²`，其中 `v_iju = (Ā²_iu + σ_A²)(B̄²_ju + σ_B²)/2^21`。
- `Λ(x,σ) ≈ log2(x² + σ²)`：
  1. 取 `e = max(⌊log2|x|⌋, ⌊log2σ⌋)`，令 `x = 2^e m_x`、`σ = 2^e m_σ`，则 `m_x² + m_σ² ∈ [1, 8)`；
  2. `κ := ⌊(⌊2^30 m_x²⌋ + ⌊2^30 m_σ²⌋) / 2^17⌋ ∈ [2^13, 2^16)`，即 13 位小数；
  3. `Λ := 2e − 13 + ⌊64·log2 κ⌋/64`，即 6 位小数。
- 误差：只有向下舍入（floor），所以 Λ ≤ 真值。κ 截断的损失小于 `log2(1 + 2^-13(1 + 2^-16)) < 2^-12`，最后一次 floor 的损失小于 2^-6，合计 `< 2^-6 + 2^-12 < 0.016`。
- `Λ_iju := Λ(A) + Λ(B) − 21`，满足 `Λ_iju ≤ log2 v_iju < Λ_iju + 0.032`。因此 `Ŝ ⊇ S`，多计入的只可能是 `ulp² ≤ v < 2^0.032·ulp²` 的边缘项，即策略**稍严**。
- 【实验 E5】随机 20 万组 (x, σ) 中，`max(真值 − Λ) = 0.01577 < 0.016`，`Λ − 真值` 的最大值为 −1.4e−6，始终 ≤ 0，单侧误差界成立。
- 【代码事实】代码用 16 位规格化有效数字和查表 `LOG_BOUNDS`（`⌈2^(15+i/64)⌉`）实现 `G(κ) = ⌊64 log2 κ⌋`，并加偏置 `SCORE_BIAS = 508` 使分数为正。注释给出的误差为"< 1.1 个 2^-6 单位"（`fp8:zk-pow/src/circuit/fp8/unpredictability.rs:27-63,105-178,223-301`）。

---

## 5. 第 5 节 Jackpot policy（四项检查）

### 5.1 记号【原文 §5.1】
- `Ā, B̄`：每行规格化到量化单位后的矩阵（即 `αX`）。
- `δ := 1/2`，`σ_i^A := δ·ℓ2(Ā_i)`，`σ_j^B := δ·ℓ2(B̄_j)`。噪声的逐元素边际分布约为 `N(0, σ²)`。
- `Ã_iu = Q(Ā_iu + N_iu^A)`，`B̃` 同理。
- 重放幅度：`M_ij := max( max_u |Ã_iu B̃_ju|, max_{1≤t≤G} |c_ij,t| )`，`G = k/32`，`c_ij,t` 为第 t 组之后的 FP32 累加器值。
- 不可预测性 (13)：`v_iju := (Ā²_iu + σ_A²)(B̄²_ju + σ_B²)/2^21`。
- 常数：`τ_idle = 8`，`ε_idle = 1/64`，`σ_min = 1`，`τ_tame = 256`，`ε_tame = 1/64`，`ε_pred = 1/16`。
- "All checks… phrased as if arithmetic is happening in the real numbers, but the verifier implementation is authoritative"：各检查按实数算术表述，但以验证器实现为准。

### 5.2 四项检查：公式、常数、针对的攻击、为什么有效

| 检查 | 精确条件 | 针对的攻击（§5.2） |
|---|---|---|
| ① Entry liveness | `D_A := {(i,u): |Ā_iu| ≥ τ_idle σ_i^A}`，要求 `|D_A| ≤ ε_idle·|I_A|·k`；B 侧同理 | **No-noise entries**：元素远大于本行噪声，量化把噪声吸收掉 |
| ② Noise floor | `σ_i^A ≥ σ_min` 对所有 i ∈ I_A 成立；B 侧同理 | **Low local entropy**：行内有一个尖峰，规格化后其余元素（包括噪声）被压向 0 |
| ③ Tamed products | `U := {(i,j): ufp_Device(M_ij) > τ_tame·√k·σ_i^A σ_j^B}`，要求 `|U| ≤ ε_tame|I_A||I_B|`，其中 `ufp(x) = ulp(x)·2^W` 为不超过 |x| 的最大 2 的幂 | **Coherent dot products**：A 的行与 B 的行对齐，点积集中 |
| ④ Unpredictable summands | `S := {(i,j,u): v_iju < ulp_Device(M_ij)²}`，要求 `|S| ≤ ε_pred·k·|I_A||I_B|`（实现中用 Λ 下界计算 Ŝ ⊇ S） | **Unpredictability / 截断攻击**：把乘法安排成让累加截断吃掉噪声 |

另外两类攻击由验证器本身处理，不靠 policy：**Degenerate arithmetic**（NaN/Inf：NaN 传播，Inf 吸收后续一切加数，tile 变得可以只从少数输入预测）通过全程禁止非有限值来处理；**Noise seed grinding** 通过参数下界和熵来处理（见下文）。

**我对每项检查为什么有效的理解【推导】：**

- **① Entry liveness**
  - 若 `|Ā_iu| ≥ 8σ`，噪声标准差不超过元素的 1/8。FP8 只有 3 位尾数，相对间距为 1/8 到 1/16，所以量化结果只会落在相邻的几个码上，**不需要知道种子**就能大致预测 `Ã_iu`。
  - 如果这类元素很多，敌手可以在种子揭晓前预计算 tile 的大部分（例如预先算出可预测部分的乘积，只对剩余部分做小规模计算）。
  - 限制这类"死元素"不超过整个条带的 1/64，意味着能预先算出的部分最多是一个小常数比例。
  - 这项检查只依赖干净数据和公共常数（代码注释里叫 "x-only"），**对所有噪声抽取结果都一样**，因此 grinding 种子帮不上忙。
  - 【实验 E7】δ=0.5 时阈值为 4·ℓ2：高斯行的死元素比例约为 1e−4，自由度 2–3 的 Student-t 重尾行约为 0.6–0.7%，都低于 1/64≈1.56%。
  - 原文还提到，不做检查也能仅凭 δ 给出这类元素比例的先验上界，但这个上界较松（§5.2）。
- **② Noise floor**
  - α 由 `ℓ∞ + δ√r·ℓ2` 决定。一个大尖峰会把 α 压得很小，导致整行 `α·ℓ2`（即 σ）很小，量化后噪声只剩极少几个 FP8 小值（例如 0、±最小次正规数）。
  - 这样的操作数字母表很小，可以用专用电路（低位宽乘法器或查表）实现，比通用 FP8 MAC 便宜。这会破坏"每次 MAC 的成本等于工作模型的计价"这一前提。
  - 要求 σ ≥ 1 个量化单位，保证每个噪声元素都有若干比特的熵，字母表不会塌缩。
- **③ Tamed products**
  - 累加网格的间距由单元里出现过的最大幅度 M 决定。如果 A 行与 B 行高度相关，`|C̃_ij|` 约为 `k·|Ā||B̄|`，远大于噪声的自然尺度 `√k·σ_Aσ_B`。网格因此变粗，噪声贡献落在被保留的高位**之下**，单元的值主要由信号决定，可以预测。
  - 用 τ=256 的倍数界定"异常相干"，并限制这样的单元不超过 1/64。原文："τ_tame is chosen so that typical AI workloads easily fit this bound"。
  - 它还防止整组被加数一起被舍掉（§7 "Precision shortcuts" 中 "keeps cells out of the magnitude regime where whole groups of summands round away at once"）。
- **④ Unpredictable summands**
  - 设 `S^X = X̄ + N`（量化前），`T^X = X̃ − S^X`（量化误差），则 `C̃_ij = Σ_u (S^A + T^A)(S^B + T^B)`。
  - 若敌手取低秩的 A、B，则 `S^A`、`S^B` 的秩不超过 rank(A)+r。展开后，凡是含 S 因子的项都有低秩捷径，**唯一昂贵的项是 (15) `Σ_u T^A_iu T^B_ju`**，即两个量化误差的乘积。
  - 原文给出 T 的方差约为 `2^-10.5·(x² + σ²)`。【实验 E3】在 x/σ ∈ {0, 0.5, 1, 2, 4} 下实测 `E[T²]/E[S²] ≈ 2^-10.46 ~ 2^-10.53`，与原文吻合。因此 `v_iju ≈ Var(T^A)·Var(T^B)`，正是 (15) 中第 u 项的方差。
  - 如果 `v < ulp(M)²`，这一项的标准差小于累加网格间距，大概率被截断丢掉，矿工可以直接跳过不算。
  - 检查④要求这样"可跳过"的项不超过 1/16，从而保证**昂贵部分 (15) 中绝大多数项确实影响最终比特**。这样，"低秩信号 + 廉价修正 + 跳过被截断的误差积"这条路线的节省最多是 ε_pred 比例。
- **Noise seed grinding（参数层面的防线）**
  - `F_A, E_B, F_B` 只依赖 B 的承诺，因此要求其中"可利用结构"稀有到 grinding 不划算。`E_A` 还依赖 A 的承诺，所以阈值可以放宽，这也是检查对两侧不对称的原因。
  - 原文例子 1：若 grinding 使同一个 I_B 中的两行 E_B 完全相同，节省比例为 `|I_B|/(|I_B|−1) < 1.1`。而 SampleLine 有约 150 bit 最小熵，grinding 的代价太高。
  - 原文例子 2：B=1 时，约 10% 的元素量化后保持不变；k ≥ 1024 使得在整个 tile 上 grinding 出明显更高的比例不现实。
  - 【实验 E4】按"量化后等于不加噪时的量化码"计算，我复现出 δ=0.5 时约 5.5%、δ=1 时约 3.6%，与原文的"roughly 10%"同一量级但偏低，可能是原文对"不变"的定义不同。列入疑点。

### 5.3 【代码事实】实现中的 policy 已与白皮书 §5.1 不同
- 首个实现（提交 `6a8868ac` "new scheme"，2026-09-09）与白皮书**完全一致**：四项检查，常数 `τ_tame=256, ε_tame=1/64, ε_pred=1/16`，W=25（`git show 6a8868ac:zk-pow/src/api/fp8/jackpot_policy.rs`，第 19–40、60–79 行）。
- 当前分支 head（提交 `0b18ef42` 之后）只有**三项**检查（`fp8:zk-pow/src/api/fp8/jackpot_policy.rs:19-37,131-165`）：
  1. Entry liveness，常数不变（`τ_idle=8, ε_idle=1/64`，69–77 行）。判据化简为 `|X_iu| ≥ τ_idle·δ·ℓ2_i`，与 α 无关（172–185 行）。
  2. Noise floor：`σ_min = 1`。
  3. 合并的 "M/Z census"，**取代了 Tamed products 和原来的 Unpredictable summands**：阈值为 `T_iju = max(ulp_Device(Z_ij,s), 32·ulp_23(M_ij))²/32`，其中 Z 是 u 所在窗口的幅度。H100 的窗口是 128 项；B200 上 Z=M。预算为 **`⌊k·h·w/20⌋`（即 1/20，不是 1/16）**（`fp8:zk-pow/src/circuit/fp8/unpredictability.rs:1-101,330-353`）。
- 【推导】在 B200 上 `max(2^(e−25), 2^(e−18)) = 2^(e−18)`，所以 `T = 2^(2e−41)`，而白皮书 (14) 是 `ulp_25(M)² = 2^(2e−50)`。新阈值大 2^9 倍，更多被加数被判为"可跳过"，同时预算从 1/16 收紧到 1/20，整体更严。注释说明，"32·ulp_23(M)" 这一支在功能上承担了原来 tamed-products 对高幅度单元的约束：测试 `plain_verifier_rejects_coherent_fixture_under_new_census` 用历史上的"相干"夹具验证新 census 也会拒绝它（`fp8:zk-pow/src/api/verify.rs:104-111`）。
- 设计原则（代码注释，`jackpot_policy.rs:6-10,35-36`）：
  - 优先使用只依赖干净数据的检查，使 grinding 无效；
  - 密度类违规按**整个 tile** 计数，使诚实矿工的计数集中在均值附近，不会因尾部波动而失败；
  - 结果是二元判定，不给按程度的 credit，避免激励矿工把数据"整形"成刚好不相干。

---

## 6. 第 6 节 有用矿工流程（简述，供对照）【原文 §6】
1. **Prequantize**：把 A、B 编码为 FP10（编码方法不强制，C.6 给出推荐）。
2. **Commit**：用最近的区块头作为密钥哈希 B，得到 `noise_seed_B`；再在提议头下承诺 A，得到 `noise_seed_A`。
3. **Noise & quantize**：成本 `O((m+n)kr)` 次 MAC，外加一次遍历。
4. **Multiply**：在完整的加噪操作数上运行一次 `MatMul_Device`，成本 mnk，用的是"the same kernel the job would run without the protocol"。
5. **Scan**：按 P_A、P_B 切 tile，逐个计算 Extract 和 J 并与目标比较，期望中奖数正比于总 MAC 数。
6. **Open on a win**：只为中奖且通过 policy 的 tile 组装证明。

去噪：信号–噪声交叉项含秩 r 因子，成本 `O((mn+mk+nk)r)`，外加每行的去缩放。无法去掉的是量化误差项 (15)，它与"直接把 A、B 量化到 FP8"引入的误差同阶，因此有用结果具有 FP8 级精度。总开销为 `1 + o(1)`（m, n, k ≫ r 时）。B 固定时，其哈希和加噪只需每隔几次 tip 变化刷新一次。

---

## 7. 第 7 节 已知的挖矿加速（原文要点）
原文说明本节的目的不是规定缓解措施，而是记录设计时考虑过哪些加速类别（"record which high-level speedup classes were in scope"）。

1. **Input crafting**：不需要乘积的矿工可以为速度而不是为用途选择操作数，例如低秩矩阵、少数尖峰主导的行、噪声会被量化掉的行、行高度相关的操作数对。防线是 §5.1 的 jackpot policy，它强制要求保证不可预测性的统计性质。
2. **Precision shortcuts**：设备算术（C.2）会截断低于累加网格的被加数，矿工可以跳过已知会被舍掉的被加数，节省这些 MAC。不可预测被加数检查把可跳过的比例限制在 ε_pred；tamed-products 检查让单元不进入"整组被加数一起被舍掉"的幅度区间。
3. **Seed and commitment grinding**：反复重试承诺，寻找有利的噪声（§5.2 中因子的可利用结构，或其他有利 transcript）。每次放弃都要付出一次承诺加种子派生的成本，由常规 Fiat–Shamir grinding 论证约束。
4. **Work reuse**：试图从一次乘法中取出多张彩票。方法包括把加噪输出作为新输入重新提交（链式实例），或在一批作业之间摊销共享的操作数、噪声或预处理。只有当公开一个加噪 tile 会让对应干净 tile 的彩票更容易预测时，这类攻击才可能成功。

（第 5 类 "Better kernels and hardware" 原文明确**不算攻击**：更快的逐比特一致 `MatMul_Device` 实现或更好的芯片带来的优势，诚实矿工同样可以获得，工作模型对所有人按同一单位计价。）

---

## 8. 附录 A（MoE）与附录 B（ZK）要点

### 8.1 附录 A：Mixture-of-experts verifier【原文】
- e ≥ 1 个专家，每个专家的右操作数为 `B^(w) ∈ R^{η×k}`，`η = n/e`。`R[w]` 是路由到专家 w 的 A 行的有序列表，`s_w = |R[w]|`，`A^(w)_i = A_{R[w][i]}`，`C^(w) = A^(w)(B^(w))ᵀ`。一个 token 可以出现在多个列表中，也可以一个都不在。
- 编码：把所有 `B^(w)` 纵向堆叠成一个 B 统一承诺和加噪，`H_B` 覆盖全部专家，e 写入 `p_B`。
- `p_A` 扩展为 `(m, hash_id_A, P_A, hash_id_R, hash_id_O)`。`O_i = Σ_{w≤i} s_w` 为累计计数；`R_flat = R[0]‖…‖R[e−1]`；在 key_A 下用 hash_id_R 和 hash_id_O 分别哈希，得到 `H_R`、`H_O`。
- 证明额外携带：中奖专家 w、完整的 O、覆盖完整 R[w] 的开口 `π_R`。w、t_A、t_B 都在两个种子生成**之后**才选定。
- 验证改动：
  - `1 ≤ e ≤ 1024`，`e | n`，`0 ≤ w < e`；O 单调，`|R[w]| = O_w − O_{w−1}`，R[w] 在 [0, m) 内严格递增，其余路由列表不做验证。
  - 坐标为专家局部：`U_A = Rows(P_A,t_A) ⊆ [0,s_w)`，`I_A = R[w][U_A]`；`U_B ⊆ [0,η)`，`I_B = wη + U_B`。划分基址和 extraction 布局检查用 U，全局行界检查用 I，tile 形状界不变。
  - `noise_seed_A := H_"seed-A"(H_A, H_R, H_O, noise_seed_B, key_A, p_A)`。
  - E 线用**全局行号**寻址，F 线只依赖 k。因此噪声是"globally canonical"的：A 只需加噪一次，即可在各专家之间复用。
  - 其余部分（NoisyQuantize、MatMul、Extract、policy、目标）不变。
- 【代码事实】MoE 支持见提交 `459fe0d1`（2026-09-28），`transcript.rs:174-183`。但 vLLM 矿工 README 写明 "MoE expert weights are not mined yet"（`git show origin/fp8-scheme:miner/vllm-miner/README.md`，第 44 行附近）。

### 8.2 附录 B：Zero-knowledge verifier【原文】
- 外层验证器加上 ZK 电路共同执行 §4 的接受规则（MoE 时包括附录 A），**所有明文条件都不变**，附录 B 只规定公开哪些内容。
- 公开输入：`X_ZK = (p_ZK, σ̂, σ_d, p_B, p_A, H_B, H_A, t_B, t_A, J)`；MoE 再追加 `(H_R, H_O, w, O_{w−1}, O_w, O_{e−1}, I_A)`。
- 私有见证：`A_IA, B_IB`、完整开口 π_A 和 π_B（MoE 还包括 O、R[w]、π_R），以及加噪条带、乘积 tile、extractor 输出、policy 中间量。
- 差异：
  1. `H_pub := H_"zk-public"(X_ZK)` 在 ZK Fiat–Shamir 的第一个挑战之前被吸收；
  2. 外层验证器负责认证 σ_d、所有公开输入检查、**所有协议层 Fiat–Shamir 派生、噪声计算**以及目标检查 (12)。电路负责所有依赖私有数据的检查，并约束公开的 J 等于由私有数据算出的 jackpot 哈希；
  3. MoE 隐藏开口的参数化方式。
- 【代码事实】实现为 Starky 多表 batch STARK：`blake3_stark`、`input_quant_stark`、`scale_stark`、`matmul_b200_stark` / `matmul_h100`、`xor_fold_stark`、`luts`（`fp8:zk-pow/src/circuit/fp8/`）。外面套两级 plonky2 递归包装（第二级 `zero_knowledge: true`），按 Device 各有一个通用验证器，验证器缓存按 device 字节索引（`fp8:zk-pow/src/api/fp8/zk.rs:1-77`）。验证器侧从公开种子重算 `(E@F)` 的 bf16 噪声码，与附录 B "outer verifier… computes the noise" 一致（同文件 30–44 行）。

---

## 9. Hawkeye 与设备白名单

### 9.1 Hawkeye 论文【网络：https://arxiv.org/abs/2603.20421 ，HTML v2：https://arxiv.org/html/2603.20421v2】
- 题目：*Hawkeye: Reproducing GPU-Level Non-Determinism*。作者 Erez Badash, Dan Boneh, Ilan Komargodski, Megha Srivastava。v1 2026-03-20，v2 2026-05-15，MLSys 2026。
- 解决的问题：浮点加法不满足结合律，GPU Tensor Core 的舍入方向、次正规数处理、累加顺序与分组都是未公开的硬件行为。第三方因此无法在 CPU 上逐比特复现 GPU 上的矩阵乘结果，审计 ML 训练和推理也就无从谈起。
- 方法：用内联 PTX 直接调用 MMA 指令写定制 CUDA 内核，在 16×16 tile 上做针对性测试，归纳出舍入方向、次正规数处理、累加顺序与内部精度。然后编码成软件模拟器，与硬件输出逐比特比对。代码在 github.com/badasherez/gpu-simulator。
- 结论：覆盖 Ampere、Hopper、Ada Lovelace 三代架构，以及 FP16、BF16、FP8 (E4M3) 三种精度。随机 16×16 矩阵乘测试实现 100% 逐比特复现（正文给出 100,000 次）。在 Apple M4 Pro CPU 上复现一个 4096×4096 的 Hopper FP8 乘法耗时约 40.6 s。作者认为审计只需复现一次，这个速度可以接受。论文列出的局限：不覆盖其他厂商；卷积、attention 等高层算子需要额外逆向；不讨论 cuBLAS 的内核选择、split-K 等库层面因素。
- 【网络/核对】从 arXiv 摘要和 HTML 页面看，论文**没有覆盖 Blackwell**。白皮书 §4.6 称 [5] "characterize these behaviors across NVIDIA architectures and precisions, including FP8"。代码中 B200 的参数注释引用的是内部文档 "B200.md"（`fp8:zk-pow/src/api/fp8/utils.rs:14-16`），不是 Hawkeye。

### 9.2 FP8 方案为什么依赖它【原文 §1、§4.2、§4.6 "Arithmetic and extraction"、App. C；推导】
- 彩票是 tile 输出比特的哈希，验证者必须在 CPU 上（或在 ZK 电路中）得到**与矿工 GPU 完全相同**的 FP32 比特。差 1 ulp，哈希就完全不同。
- 原文："E4M3FN operands alone do not determine a bit-exact result on all devices. Reproducing a GPU matmul exactly requires pinning down rounding direction, subnormal handling, and accumulation order." 意思是只给 E4M3FN 操作数还不能在所有设备上确定逐比特结果，必须把舍入方向、次正规数处理和累加顺序都固定下来。
- 因此：
  - **Quant** 固定承诺类型、量化类型、量化算法和 BF16 舍入；
  - **Device** 固定其余算术，即 MMA 的分组大小（32）、内部精度 W（13 或 25）、截断方式，以及代码中 H100 的 128 项提升结构；
  - Device 写入 `p_B`，在噪声挑战之前就固定。
- 设备白名单：Device 是共识允许列表中的枚举（§4.2）。【代码事实】目前只有 H100(0) 和 B200(1)（`public_params.rs:144-209`）。消费级 Blackwell（SM120：RTX 50、RTX PRO 6000）在矿工端有专门内核（提交 `85216510`），但没有单独的 Device 值。【推导】它应该复用 B200 的算术语义，前提是 SM120 的 FP8 MMA 累加行为与 SM100 相同，代码里没有看到对此的独立论证。分支上的 vLLM 矿工只支持 SM100/SM120（README 第 3–5、33–36 行）。H100 只在验证器和 STARK 中支持。
- 【网络】Kryptex 根据 PIP-3 和 9 月 15 日 AMA 整理：只有 Hopper 和 Blackwell 两类设备；创始人确认 RTX 5090 和 RTX PRO 6000 Blackwell 在分叉后仍可挖矿；Ada（RTX 40/L40S）、Ampere（RTX 30/A100）及 AMD 均没有 profile（https://pool.kryptex.com/de/articles/pearl-fp8-fork-en ，2026-10-07 更新）。

---

## 10. 落地状态

### 10.1 仓库搜索结果【代码事实】
- `git branch -a`：存在远程分支 **`origin/fp8-scheme`**。
  - 从 `master` 的 `5b09d844`（2026-09-16）分叉，**领先 master 21 个提交**，落后 3 个。
  - 首个提交是 `6a8868ac` "new scheme"（2026-09-09，vaktibabat），最新是 `1280a21c`（2026-09-28）。
  - `git branch -a --contains 6a8868ac` 只返回 `origin/fp8-scheme` 和 `origin/reduce-comments`，**master 不包含**，也没有任何 tag 包含。
- `git log --all -i --grep='fp8\|e4m3\|floating'` 的关键提交：

| 提交 | 日期 | 内容 |
|---|---|---|
| `6a8868ac` | 09-09 | new scheme（首个实现，policy 与白皮书一致） |
| `01c149e7` | 09-13 | V4 证书携带完整祖先头 |
| `5dcecff3` | 09-23 | F_A 改由 seedB 派生 |
| `d6755d9c` / `f7fe16c8` | — | V4 证书尺寸限制 |
| `0b18ef42` | 09-27 | H100 支持；M/Z census 取代原 policy；δ 与设备相关 |
| `33bfaaeb` / `b2053745` | 09-27 | L2 公式对齐 |
| `4c7177be` | 09-27 | 移除 CPU 参考矿工和 GPU 模拟 |
| `85216510` | 09-28 | 消费级 Blackwell 支持 |
| `459fe0d1` | 09-28 | FP8 MoE |
| `50dafd3e` / `1280a21c` | 09-28 | B200 上的 vLLM 端到端测试 |

- 分支内容：
  - Rust 验证器 `zk-pow/src/api/fp8/*`（13 个文件）；
  - FP8 STARK `zk-pow/src/circuit/fp8/*`；
  - Go 节点 `node/wire/certificate_v4.go`、`node/zkpow/verify.go`（V4 分派）、`node/blockchain/fp8_fork_test.go`；
  - 矿工 `miner/pearl-gemm`（CuTe DSL 内核，含 SM120）和 `miner/vllm-miner`。
- 硬分叉开关：`Fp8ForkHeight`，为 0 时禁用（`fp8:node/chaincfg/params.go:283-289,335-337`）。分支上**只有 RegressionNet 和 SimNet 设为 1**（同文件 496、796 行）；**MainNet（358 行起）和各 TestNet 均未设置**，即禁用。
- `master` 上没有 FP8 共识代码：`zk-pow/src/api/` 下没有 fp8 目录，唯一的 MMA 类型是 `Int7xInt7ToInt32`（`master:zk-pow/src/api/proof.rs:19-24`）。master 中 fp8 只出现在 vLLM 矿工里，用于**非挖矿层**（例如 "int7 gate/up + fp8-block down" 的 MoE 层），见 `master:miner/vllm-miner/src/vllm_miner/vllm_config.py:106-163`。
- 结论：**FP8 协议已在功能分支上基本完整实现（含 ZK 与节点分叉开关），但没有合入 master，没有发布 tag，主网和测试网都未激活。**

### 10.2 官方与外部上线信息【网络】
- PIP-3 "FP PoUW Certificates"：Status **Draft**，Created 2026-07-30，Deployment 写的是 "TBD"，设备为 Hopper 和 Blackwell（https://github.com/pearl-research-labs/pips/blob/main/PIPS/pip-0003.md ）。注意 PIP-3 草稿仍写 "certificate version 3"、"e.g., r = 16"、"pre-quantization format TBD"，**落后于**白皮书（r=32、FP10）和代码（V4 证书）。
- CryptoBriefing：Pearl 于 **2026-09-14** 发布 FP8 浮点方案，FP4 方案在讨论中，没有激活计划（https://cryptobriefing.com/pearl-blockchain-ai-inference-mining/ ）。
- Kryptex（2026-10-07 更新）：PIP-3 为 Draft，**没有确定日期或区块高度**。创始人在 9-15 AMA 表示只有在对实现安全性有信心时才会升级，"如果仍有疑虑就不会分叉"；旧节点不接受新证书，需要全网升级（https://pool.kryptex.com/de/articles/pearl-fp8-fork-en ）。
- Datawallet（西语版）称 10-02 检查主网节点参数时没有 FP8 激活高度（https://www.datawallet.com/es/cripto/pearl-explained ，来自搜索摘要，未逐页核对）。
- 没有找到 Pearl Research Labs 官方博客或 X 上的 FP8 上线公告。

---

## 11. 与 int7 方案的对照表

> int7 一侧只列与本组件对照所需的要点，细节以 int7 组件笔记为准。int7 代码引用均为 `master`。

| 维度 | int7 方案（当前主网） | FP8 方案（白皮书 / fp8-scheme 分支） |
|---|---|---|
| 输入格式 | 整数矩阵，MMA 类型 `Int7xInt7ToInt32`（`master:zk-pow/src/api/proof.rs:19-24`） | FP10 承诺（int8 值加每 8 个元素一个 BF16 尺度，C.6），解码为 BF16，加噪后量化为 E4M3FN，FP32 累加 |
| 噪声 | 低秩 `E·F`，秩 r 为 2 的幂，范围 32–1024；另有 rank-penalty 软分叉，基准秩 128（`master:zk-pow/src/api/sanity_checks.rs:12-30`） | 秩 r=32 固定；每条线 32 字节 XOF，按 ±[1,128] 取值并规范化到范数 256 后转 FP8；每行按 α、β 缩放，噪声与信号比为 δ（白皮书 1/2；代码中 H100 为 1，B200 为 1/2） |
| 量化 | 无（整数精确算术） | 加噪**之后**做非线性 RNE 量化到 FP8，这是去掉低秩捷径的关键 |
| 哈希对象 | 计算 **transcript**：沿公共维度每 r 个做一次 inner hash（KW 方案的实例化） | **输出 tile 本身**：16 个子块经 rotl13 乘加折叠成 64 字节，再做一次 keyed BLAKE3 |
| 有用输出 | 精确：`C = C' − (AF + E(B+F))` | 近似：去噪后为 FP8 级精度（量化误差项 (15) 无法消除） |
| 验证 | 重算 tile 的整数 transcript，做哈希比较（明文或 ZK） | 逐比特重放所选设备的 FP8 MMA（分组 32、W=13 或 25、向零截断；H100 还有 128 项 FP32 提升），通过 jackpot policy 后比较哈希与按 `|I_A||I_B|k` 缩放的目标；ZK 版用 Starky 多表加 plonky2 两级包装 |
| 矿工自由度 | 选 A、B、r（带 rank penalty）、划分、哈希 id | 选 A、B（受 policy 约束）、划分 P_A/P_B、哈希 id、Device、状态窗口深度 d（D=3）、中奖后的 tile 基址；r 固定为 32 |
| 安全假设 | KW 的 transcript 不可预测性和直积猜想（精确算术，ROM）；任意 A,B 在猜想下与 0 矩阵同样难 | Assumption 1（量化子空间困难性，非正式，平均情形，只针对 A=B=0），外加 jackpot policy 的统计检查（启发式），外加 Hawkeye 式设备行为刻画的正确性 |
| 硬件要求 | 任意能高效做 int8 MMA 的硬件；逐比特结果与设备无关（整数算术精确） | **白名单设备**：H100（Hopper）与 B200（Blackwell）；矿工内核支持 SM100/SM120；Ada、Ampere、AMD 均无 profile |
| 设备相关性 | 无 | Device 是公共参数，写入 p_B；同一输入在不同 Device 下 tile 比特不同（实验 E6） |

---

## 12. 疑点（留待问题阶段分析）

只列不展开：

1. 统计检查（liveness、noise floor、census）能否被**构造输入**绕过：例如让每一项都恰好低于阈值，或用不触发检查的低秩/相干结构获得超过 ε 比例的加速。
2. Assumption 1 是非正式的平均情形假设，没有归约；它只覆盖 A=B=0，对敌手选择的 A、B 完全依赖启发式 policy。
3. 逐比特复现导致**设备白名单**：Ada、Ampere、AMD 被排除，硬件来源集中，新架构上线需要共识升级，存在中心化与治理风险。
4. Hawkeye 论文只覆盖 Ampere、Hopper、Lovelace，**没有 Blackwell**。B200 的参数来自内部 "B200.md"；SM120（消费级 Blackwell）没有独立 Device 值，复用 B200 语义的依据不明。
5. 量化对有用结果**精度**的影响：去噪后误差项 (15) 是否真与直接 FP8 量化同阶？在 δ=1/2（H100 上 δ=1）时，噪声占据 FP8 动态范围（α 中 `δ√r·ℓ2` 一项），可能挤压信号的有效精度。
6. 白皮书与代码不一致：policy 从四项变为三项（tamed products 被并入 M/Z census），ε_pred 从 1/16 变为 1/20，阈值形式改变，δ 改为与设备相关，H100 的累加结构（128 项 FP32 提升）不同于 C.2，F_A 改由 seedB 派生（§4.5 与 §5.2 自相矛盾），L2 范数计算方式不同于 C.6。哪个版本才是"规范"？
7. §5.2 "B=1 时约 10% 元素量化后不变"：我的复现约为 5.5%（δ=1/2）和 3.6%（δ=1），定义或数值待核。
8. 状态窗口 D=3 给 B 侧 D 倍预处理和 grinding 机会，再加上 `p_X` 中 Device、hash_id、划分等矿工可选参数，原文承认这些绑定"不阻止 transcript grinding 或跨承诺/深度的共享工作"，批量与 grinding 的安全定义尚未给出。
9. Work reuse（链式实例、跨作业摊销）只给了"仅当……才成功"的定性论述，没有形式化论证。
10. 中奖后才选择 tile 基址（以及 MoE 中的专家 w）：矿工在一次乘法后扫描全部 tile，这是设计意图，但与 policy 结合时是否存在"挑选通过 policy 的 tile"的选择偏差。
11. PIP-3 草稿（cert v3、r=16 示例、预量化 TBD）与白皮书和代码（V4、r=32、FP10）不同步，规范的权威来源不清楚。
12. Extract 的折叠是 32 位乘加循环（非密码学），在 rotl13 之后接 BLAKE3：64 字节压缩的抗碰撞或抗构造性是否足够，尤其是矿工可控的子块排列 P_A/P_B。
13. ZK 版本把"噪声计算"放在外层验证器，电路只检查依赖私有数据的部分，这一边界的完备性需要核查（例如 policy 中间量全部是私有的）。
