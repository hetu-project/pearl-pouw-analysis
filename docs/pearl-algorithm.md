# Pearl 挖矿与验证算法（int7 方案）

> 版本：`pearl-research-labs/pearl` @ `156ef9a8`（主网当前规则：证书 V3、秩惩罚已生效）
> 推导与实验依据见 `docs/notes/02-03-noise-seed-jackpot.md`（种子、噪声、彩票）与 `docs/notes/04-mining-config-targets.md`（配置与门槛）。其中的 Python 复刻（`experiments/noise/pearl_ref.py`）已与 Pearl 自身代码逐字节核对一致。

## 1. 参数表

### 矩阵与尺寸

| 符号 | 含义 | 取值 / 约束 | 节点默认 | GPU 矿工典型值 |
|---|---|---|---|---|
| **A** | 左矩阵，m 行 × k 列。AI 推理中是"用户输入变成的数字"（激活值） | 每个元素为整数，范围 [−64, 64] | 随机数 | 推理激活值（7 bit 量化） |
| **B** | 右矩阵，k 行 × n 列。AI 推理中是"模型权重" | 同上 | 随机数 | 模型权重（7 bit 量化） |
| **m** | A 的行数 | ≤ 2²⁴ | 256 | 本步处理的 token 数（≥1024 才挖矿） |
| **n** | B 的列数 | ≤ 2²⁴ | 512 | 该层的输出维度 |
| **k** | 公共维度（A 的列数 = B 的行数），即每个输出元素的乘加次数 | 64 的倍数；1024 ≤ k ≤ 65536；16r ≤ k ≤ 4r² | 2048 | 该层的输入维度，如 4096 |

### 噪声

| 符号 | 含义 | 取值 / 约束 | 默认 |
|---|---|---|---|
| **r** | 噪声的秩：噪声矩阵由两个"瘦"矩阵相乘得到，r 是中间维度；同时也是彩票"每隔多少步记一次"的步长 | 2 的幂，32–1024；秩惩罚分叉后要求 **r ≥ 128** | 128 |
| **E_AL** | A 侧噪声的稠密因子，m × r | 每个元素均匀分布在 [−32, 31] | — |
| **E_AR** | A 侧噪声的稀疏因子，r × k | 每列只有一个 +1 和一个 −1，其余为 0 | — |
| **E_A** | A 侧噪声 = E_AL × E_AR，m × k | 每个元素在 [−63, 63] | — |
| **E_BL、E_BR** | B 侧噪声的两个因子（E_BL 为 k × r 稀疏，E_BR 为 r × n 稠密） | 同上 | — |
| **E_B** | B 侧噪声 = E_BL × E_BR，k × n | 同上 | — |

### 小块（彩票）

| 符号 | 含义 | 取值 / 约束 | 节点默认 | GPU 矿工 |
|---|---|---|---|---|
| **h、w** | 一张彩票对应输出矩阵中 h 行 × w 列的小块 | 均为偶数；32 ≤ h·w ≤ 256 | 4 × 8 | 2 × 64 |
| **rows_pattern、cols_pattern** | 小块取哪几行、哪几列的"模板"（可不连续） | 写在挖矿配置中 | 行 [0,8,64,72]，列 [0,1,8,9,32,33,40,41] | 行 [0,8]，列 [0,1,8,9,…,249] |
| **t_rows、t_cols** | 模板的偏移量，决定是哪一块 | 矿工中奖后给出 | — | — |
| **K** | 参与彩票的公共维度长度 = k − (k mod r) | — | 2048 | — |
| **msg[0..15]** | 彩票"记录"：16 个 32 位整数，共 64 字节 | — | — | — |

### 区块与难度

| 符号 | 含义 |
|---|---|
| **header** | 76 字节的不完整区块头：版本(4) ‖ 上一块哈希(32) ‖ 交易 Merkle 根(32) ‖ 时间戳(4) ‖ 难度 nbits(4) |
| **config** | 52 字节的挖矿配置：k(4) ‖ r(2) ‖ 运算类型(2，固定为 0) ‖ 行模板(6) ‖ 列模板(6) ‖ MoE 参数(32，非 MoE 时为 0) |
| **nbits / target** | 压缩格式的难度；展开后得到 256 位目标值 target |
| **job_key** | 本次挖矿任务的密钥，把区块头和配置绑定在一起 |
| **hash_a、hash_b** | A、Bᵀ 的 Merkle 根（承诺） |
| **b_seed、a_seed** | B 侧、A 侧噪声的种子 |

## 2. 算法一：挖矿（矿工）

```
输入：header（76 字节）、config（52 字节）、矩阵 A (m×k)、B (k×n)
输出：中奖时的证明 PlainProof

// ── 第 1 步：绑定任务 ──
job_key = BLAKE3( header ‖ config )                      // 128 字节 → 32 字节

// ── 第 2 步：承诺矩阵（先封存，之后不能再改）──
hash_b = BLAKE3_keyed( Bᵀ 按行展开，补零到 1024 的倍数, key = job_key )
hash_a = BLAKE3_keyed( A  按行展开，补零到 1024 的倍数, key = job_key )

// ── 第 3 步：种子链（V3 加盐版）──
bound_b = BLAKE3_keyed( hash_b ‖ n(4 字节) ‖ 0×28, key = BLAKE3("pearl/cert-v3/noise-seed/B") )
bound_a = BLAKE3_keyed( hash_a ‖ m(4 字节) ‖ 0×28, key = BLAKE3("pearl/cert-v3/noise-seed/A") )
b_seed  = BLAKE3( job_key ‖ bound_b )                    // 先确定 B 侧噪声
a_seed  = BLAKE3( b_seed  ‖ bound_a )                    // 再确定 A 侧噪声

// ── 第 4 步：生成噪声（以 A 侧为例；B 侧对称，用 b_seed 和标签 "B_tensor"）──
用 a_seed 作密钥的 BLAKE3 生成伪随机字节（标签 "A_tensor"）：
  E_AL[u][i] = (随机字节 & 63) − 32                      // u < m, i < r，范围 [−32, 31]
  对每个 l < k，取一个随机 32 位数 U_l：
    first_l  = U_l mod r
    second_l = first_l XOR (1 + ⌊(r−1)·U_l / 2³²⌋)       // 保证 second_l ≠ first_l
  E_A[u][l] = E_AL[u][first_l] − E_AL[u][second_l]       // 范围 [−63, 63]

// ── 第 5 步：加噪后做矩阵乘法，同时计算彩票 ──
A' = A + E_A,  B' = B + E_B                              // 元素范围在 int8 内
对每个 h×w 小块（行 = rows_pattern + t_rows，列 = cols_pattern + t_cols）：
  acc[h][w] = 0                                          // 32 位整数累加器
  msg[0..15] = 0
  for ll = r, 2r, 3r, …, K:                              // 每走 r 步记一次
      对小块中每个 (u, v)：
          acc[u][v] += Σ_{l = ll−r}^{ll−1} A'[u][l] × B'[l][v]
      x   = acc 中全部 h·w 个值做 XOR                    // 压缩成一个 32 位数
      tid = (ll/r − 1) mod 16
      msg[tid] = rotl13(msg[tid]) XOR x                  // 循环左移 13 位后再 XOR
  hash_jackpot = BLAKE3_keyed( msg（64 字节）, key = a_seed )

  // ── 第 6 步：判断是否中奖 ──
  if  hash_jackpot（按小端读作 256 位整数） ≤ target × h × w × ⌊K/r⌋ × 128:
      输出 PlainProof = { m, n, k, r,
                          小块用到的 h 行 A（原始值，未加噪）+ Merkle 路径,
                          小块用到的 w 列 B（原始值，未加噪）+ Merkle 路径 }

// ── 第 7 步（只有有用矿工才做）：去噪，得到真正的 A×B ──
A×B = A'×B' − A×E_B − E_A×B'                            // 低秩，计算量小
```

**中奖门槛的含义**：`target × h × w × ⌊K/r⌋ × 128` 等于"难度目标 × 这张彩票的计算量"。小块越大、k 越长，单张彩票越难算，门槛也相应放宽，所以每次乘加的中奖概率固定，与小块形状无关。秩惩罚分叉（主网高度 96251）之前的公式是 `target × h × w × K`；分叉后当 r > 128 时，门槛按 128/r 收紧。

## 3. 算法二：验证（节点、矿池）

```
输入：header、config、PlainProof
输出：通过 / 拒绝

① 解析证明：用 Merkle 路径重算 hash_a、hash_b，确认这 h 行、w 列属于矿工承诺的矩阵
② 检查参数：r、k、h·w、m、n 是否满足约束（见参数表）
③ 检查数值：这 h 行、w 列的每个元素都在 [−64, 64]       ← 对矩阵内容唯一的检查
④ 用同样的方法重新计算 job_key、b_seed、a_seed 和噪声（只算这 h 行、w 列需要的部分）
⑤ 用这 h 行、w 列重算这一个小块的彩票 msg 和 hash_jackpot
⑥ 检查 hash_jackpot ≤ target × h × w × ⌊K/r⌋ × 128
```

对应代码：`zk-pow/src/api/verify.rs:101-134`（①在 `:108`，②在 `:111`，③在 `:114-118`，④在 `:124`，⑤在 `:127`，⑥在 `:130-131`）。

上链时使用零知识证明代替 PlainProof：链上只公开 164 字节（配置、hash_a、hash_b、hash_jackpot、m、n、t_rows、t_cols），电路证明 ①③⑤ 成立，④⑥ 由节点直接计算。共识层另外检查证书版本、是否禁用 MoE 和秩惩罚（`node/blockchain/validate.go:529-560`）。

**验证者的计算量**：重算 h·w·K 次乘加，矿工做 m·n·k 次。节点默认配置下为 1/4096；按大模型典型尺寸（m=n=4096）约为百万分之八。
