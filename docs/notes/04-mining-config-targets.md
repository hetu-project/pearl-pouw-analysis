# 组件 4：挖矿配置、中奖门槛与参数空间

> 源码基线：`third_party/pearl` commit `156ef9a8`。下文路径均相对该仓库根目录，行号以该 commit 为准。
> 计算脚本：`experiments/config/`
> - `pattern.py`：PeriodicPattern 编解码、合法偏移、平铺检查的 Python 复刻
> - `targets.py`：nbits→target，未惩罚与秩惩罚的放大因子，各 r 的对比表
> - `configs.py`：多种配置的"单位工作量期望份额"对比（输出见 `configs_out.md`）
>
> 标注约定：**【代码】** 代码事实（附 文件:行号）；**【历史】** git 历史事实（附 commit）；**【计算】** 脚本计算结果；**【理解】** 我的推断或解释，未经代码直接证实。

---

## 0. 一页摘要

- 一张"彩票"（一次尝试）对应噪声矩阵乘积 (A+E_A)(B+E_B) 的一个 h×w 输出小块：h 行由 `rows_pattern + t_rows` 给出，w 列由 `cols_pattern + t_cols` 给出。沿公共维 k 每走 r 步，就把这 h·w 个**累加和**做一次 XOR 归约，再旋转异或进 16 个 u32 槽；最后对这 64 字节做一次 keyed blake3，得到 `hash_jackpot`。
- 中奖条件（共识，未惩罚）：`LE256(hash_jackpot) ≤ target(nbits) · h·w·(k − k%r)`。
- 秩惩罚软分叉（#275，`51a6e11b`；主网从高度 96251 生效）额外要求 `r ≥ 128`，且 `LE256(hash_jackpot) ≤ target · h·w·⌊k/r⌋·128`，相当于把门槛再乘以 128/r。
- chainwork 只看 header 的 `Bits`（`CalcWork`），与挖矿配置无关。

---

## 1. MiningConfiguration 与 PeriodicPattern

### 1.1 字段（【代码】`zk-pow/src/api/proof.rs:53-72`）

| 字段 | 类型 | 线上字节 | 语义 |
|---|---|---|---|
| `common_dim` | u32 | 4 | 公共维 k。注释写的是 "common dimension of the matmul, k"（proof.rs:64） |
| `rank` | u16 | 2 | 噪声秩 r。注释原话："Denotes length of inner product per inner hash invocation"（proof.rs:65） |
| `mma_type` | `MMAType`(u16) | 2 | 目前只有 `Int7xInt7ToInt32 = 0`，见 proof.rs:18-25；`from_bytes` 遇到其他值直接 bail（proof_utils.rs:470-473） |
| `rows_pattern` | PeriodicPattern | 6 | A 的行模式。h = `rows_pattern.size()`（proof_utils.rs:321-323） |
| `cols_pattern` | PeriodicPattern | 6 | B 的列模式（以 Bᵗ 的行看）。w = `cols_pattern.size()`（proof_utils.rs:324-326） |
| `moe` | `Option<MoEConfig{e:u16, top_k:u16}>` | 32（尾段） | 尾段布局为 `e(2) | top_k(2) | 28 字节 0`。`e==0` 表示普通任务，此时要求 `top_k==0`，其余字节必须全 0（proof_utils.rs:477-483）；`e>0` 表示 GROUPED_GEMM（MoE）任务 |

**字节编码（【代码】proof_utils.rs:428-494）**：所有字段小端，共 52 字节（`SERIALIZED_SIZE = 52`，proof_utils.rs:435）。

```
[0..4)  common_dim  u32 LE
[4..6)  rank        u16 LE
[6..8)  mma_type    u16 LE (=0)
[8..14) rows_pattern 6B
[14..20) cols_pattern 6B
[20..52) trailer: e u16 | top_k u16 | 28×0x00
```

`from_bytes` 最后会做 roundtrip 检查（proof_utils.rs:492），所以每个配置只有一种合法编码。
注释写道："IncompleteBlockHeader (76) + MiningConfiguration (52) = 128 bytes = 2 blake3 blocks"（proof_utils.rs:434）。
`job_key = blake3(header76 ‖ config52)`（proof_utils.rs:348-353），由此 k、r、模式和 MoE 字段都绑定进 job_key，进而绑定进噪声种子（commitment_hash，proof_utils.rs:362-372）。

**PublicProofParams 线上布局（【代码】proof.rs:76-83；proof_utils.rs:1141-1234）**：非 MoE 共 164 字节。

```
[0..52) mining_config | [52..84) hash_a | [84..116) hash_b | [116..148) hash_jackpot
[148..152) m | [152..156) n | [156..160) t_rows | [160..164) t_cols   (u32 LE)
MoE 尾段: expert_idx u16 | routing_offsets e×u32 | hash_routing 32 | outer_count u8 | outer_indices u32×cnt
```

`mining_config_and_jackpot_from_wire_bytes` 只解析前缀中的 config 和 jackpot（proof_utils.rs:1197-1206），供 `check_rank_penalty` 的 FFI 使用。

### 1.2 PeriodicPattern 精确语义

**【代码】** `shape: [(stride, length); 3]`，表示的集合是 `{a·s0 + b·s1 + c·s2 : a<l0, b<l1, c<l2}`（proof.rs:27-34；`to_list` 在 proof_utils.rs:158-170）。

**6 字节编码**（每维 2 字节：`factor−1, length−1`），见 `from_bytes`（proof_utils.rs:105-140）和 `to_bytes`（143-155）：
- 记 `span_{-1}=1`，`stride_i = factor_i · span_{i-1}`，`span_i = stride_i · length_i`。因为 factor ≥ 1，每一维的步长不小于前一维的跨度，各维嵌套且不重叠，所以元素互不相同。
- 规范性要求：
  1. 长度为 1 的维只能出现在末尾，且必须编码为 `00 00`（proof_utils.rs:121-123）。
  2. 第 i≥1 个非平凡维的 factor 必须 ≥2，否则报 "A single stride must not be broken"（124-126），因为 factor=1 会与前一维合并。
  3. 周期 `span ≤ 2^24`（128-131）。
  4. 每维 length ≤ 256，factor ≤ 256（各占 1 字节）。
- `from_list` 自内向外寻找最小周期来分解（177-220），必须以 0 开头、严格递增，最多 3 维，不足的维补 `(period,1)`。

**参与彩票的行列集合（【代码】）**：
- 非 MoE：行 = `rows_pattern.to_list() + t_rows`（`a_inner_indices`，proof_utils.rs:265-267），列 = `cols_pattern.to_list() + t_cols`（275-284）。
- MoE：A 的行改用 `outer_indices`（268-274）；B 的列再加上 `expert_idx·n`（279-280）。
- `t_rows`/`t_cols` 的注释为 "the minimum element of the pattern"（proof.rs:101-102）。

**偏移合法性（【代码】）**：`offset_is_valid`（proof_utils.rs:223-231）从最外维向内逐维检查 `t mod span_i < stride_i`。
- 这项检查只在**解析阶段**执行：`from_wire_bytes` 中 proof_utils.rs:1304-1311，`list_to_pattern` 中 `zk-pow/src/ffi/plain_proof.rs:352`。
- `public_params_sanity_check` 只检查 `t_rows + max < m` 和 `t_cols + max < n`（sanity_checks.rs:57-58）。

**【计算】**（`pattern.py`）合法偏移把模式平移后，在一个周期内恰好不重不漏地覆盖全部下标（exact_partition=True）。也就是说，一次矩阵乘的输出被划分成互不相交的彩票块。

| 模式 | shape | 6B 编码 | size | 周期 | 每周期合法偏移数 |
|---|---|---|---|---|---|
| 节点行 [0,8,64,72] | (8,2),(64,2),(128,1) | 070103010000 | 4 | 128 | 32 |
| 节点列 [0,1,8,9,32,33,40,41] | (1,2),(8,2),(32,2) | 000103010101 | 8 | 64 | 8 |
| miner-base 行 [0,8] | (8,2),(16,1),(16,1) | 070100000000 | 2 | 16 | 8 |
| miner-base 列（64 元素） | (1,2),(8,32),(256,1) | 0001031f0000 | 64 | 256 | 4 |

- 节点行的编码与 #275 之前 Go 硬编码的字节一致（`51a6e11b` 删除的 `defaultMiningConfigV1`：`0x07,0x01,0x03,0x01,0x00,0x00` / `0x00,0x01,0x03,0x01,0x01,0x01`）。
- 节点默认 m=256、n=512 时，合法尝试数为 64×64 = 4096 = (256/4)·(512/8)。

### 1.3 一次尝试算什么（中奖消息）

**【代码】** `zk-pow/src/circuit/chip/jackpot/helper.rs:9-35`：

```
for ll in (r..=k).step_by(r):            # 只覆盖 ⌊k/r⌋·r = dot 个 l，k%r 尾部不进入彩票
    for u<h, v<w: J[u][v] += Σ_{l∈[ll-r,ll)} (a[u][l]+NA[u][l])·(b[v][l]+NB[v][l])   # J 不清零：累计前缀和
    x = XOR_{u,v} (u32)J[u][v]
    tid = (ll/r − 1) % 16
    msg[tid] = rotl(msg[tid], 13) ^ x
hash_jackpot = blake3(msg as 64B LE, key = a_noise_seed)
```

- `compute_jackpot_hash` 在 proof_utils.rs:1502-1505；`verify_plain_proof` 以 `compiled.a_noise_seed()` 为 key（verify.rs:130）。
- `JACKPOT_SIZE=16`、`LROT_PER_TILE=13` 定义在 `zk-pow/src/circuit/pearl_program.rs:23-25`。
- 噪声是低秩的：`E_A = E_AL·E_AR`，其中 `E_AL` 是 m×r 均匀阵，值域 [-32,31]；`E_AR` 每列恰有一个 +1 和一个 −1（`pearl_noise.rs:59-116,141-153`）。所以 `NA[u][l] = E_AL[u][p1(l)] − E_AL[u][p2(l)]`，取值在 [-63,63]。

**【理解】** 一张彩票的计算量：
- h·w·dot 次 int7×int7→int32 乘加；
- (dot/r) 次对 h·w 个值的 XOR 归约；
- 1 次 blake3 压缩。

`sanity_checks.rs:187-192` 对放大因子的注释是 "in proportion to the work one attempt costs: the hashed tile size times the dot product length"。

---

## 2. 参数约束全表

### 2.1 `public_params_sanity_check`（【代码】`zk-pow/src/api/sanity_checks.rs:15-144`）

其中 `TILE_D=16`、`TILE_H=2`（circuit/pearl_program.rs:21-22），`DWORD_SIZE=8`。

| # | 约束 | 位置 | 代码给出的理由 |
|---|---|---|---|
| 1 | r 是 2 的幂，且 32 ≤ r ≤ 1024 | :28-31 | "Currently, noise generation supports only these ranks: powers of 2 from 2^5 to 2^10."（:27） |
| 2 | r % 16 == 0 | :32 | 无（已被 #1 蕴含） |
| 3 | k ≤ 2^16 | :33 | 无 |
| 4 | k % 64 == 0 | :34 | 无 |
| 5 | k ≤ 4r² | :35 | 无注释 |
| 6 | k ≥ 16r | :36 | 无注释。`scale` 的文档说 sanity check 通过要求 k ≥ 16r 来排除零放大因子（:204-206） |
| 7 | k ≥ 1024 | :37 | "required for collision resistance since matrices are padded to a multiple 1024" |
| 8 | h、w 都是 2 的倍数 | :38-41 | 电路的小块高度 TILE_H |
| 9 | h·w ≥ 32 | :42 | 无 |
| 10 | h·w ≤ 256 | :43 | 无。电路编译处也有同样检查（pearl_program.rs:264）。sanity 层的这一条由 `77fef91e`（#173）加入，v1 版本没有（`zk-pow/src/v1/api/sanity_checks.rs:39`） |
| 11 | dot % 8 == 0 | :44-47 | DWORD 对齐 |
| 12 | m ≤ 2^24，n ≤ 2^24 | :48-49 | 无 |
| 13 | (h+w)·dot ≤ 2^22 | :50-54 | "Worker input supported up to 4 MiB" |
| 14 | t_rows + max(rows_pattern) < m，t_cols + max(cols_pattern) < n | :55-58 | 彩票块不得越界 |
| 15 | config.moe 与 public.moe 要么同时存在，要么同时不存在 | :59-63 | — |
| 16 | MoE：e>0；top_k>0；e ≤ 1024；routing_offsets 长度为 e；top_k < e；expert_idx < e；m·top_k 不溢出 u32；最后一个 offset = m·top_k；offsets 单调；每个专家 ≤ m；n·e ≤ 2^24；inner 下标落在本专家区间内；outer_indices 严格递增、个数与 h 相同、每个 < m | :65-141 | 各条均有错误信息 |

### 2.2 其他位置的约束（【代码】）

- **解析阶段**：
  - PeriodicPattern 的规范性和周期 ≤ 2^24（proof_utils.rs:105-140）；
  - mma_type==0；尾段保留字节全 0（proof_utils.rs:470-483）；
  - 偏移合法，且 t_rows<m、t_cols<n（proof_utils.rs:1304-1313）；
  - 线上长度为 164，或位于 MoE 区间内（1163-1165）；outer_indices ≤ 128 个（1152、1273-1278）。
- **PlainProof 路径**：m、n、k 必须能放进 u32，noise_rank 必须能放进 u16（`ffi/plain_proof.rs:555-559`，#299 `4a0c24bc`）；矩阵元素必须在 [-64,64] 内（verify.rs:113-118）。
- **电路 / STARK**：
  - k % 16、h % 2、w % 2、h·w ≤ 256（pearl_program.rs:261-264）；
  - STARK 行数 < 2^(2·13)；`stark_degree_bits ≤ 19`；`degree_bits + rate_bits[0] ≤ 20`（`circuit/pearl_circuit.rs:150-163`）；
  - 行数约为 (h/2)(w/2)(k/r)·max(r/16, 5) 加上 blake 行数（pearl_program.rs:46-71）。
- **秩惩罚规则**（分叉后）：r ≥ 128，且 `penalized_adjustment_factor > 0`（sanity_checks.rs:166-174）。
- **矿工侧**：
  - `MinerSettings.noise_rank ≥ PENALTY_BASE_RANK`（`miner/miner-base/src/miner_base/settings.py:40-48`）；
  - `MiningJob.adjust_target` 同样要求 r ≥ 128（`miner/pearl-gateway/src/pearl_gateway/comm/dataclasses.py:212-225`）；
  - 官方 GPU 内核只实现了 R ∈ {64, 128}（`miner/pearl-gemm/csrc/gemm/pearl_gemm_api.cpp:158-163`，其他值触发 TORCH_CHECK 失败）。

### 2.3 维度上限与派生量（【计算】`targets.py` Table B）

- m、n ≤ 2^24（MoE 下 n·e ≤ 2^24）；k ∈ [1024, 65536] 且是 64 的倍数；h·w ∈ [32, 256]，h、w 为偶数。
- 各 r 下 k 的允许区间（16r ≤ k ≤ min(4r², 2^16)）：

| r | k 范围 | 块数 k/r |
|---|---|---|
| 32 | [1024, 4096] | 32–128 |
| 64 | [1024, 16384] | 16–256 |
| 128 | [2048, 65536] | 16–512 |
| 256 | [4096, 65536] | 16–256 |
| 512 | [8192, 65536] | 16–128 |
| 1024 | [16384, 65536] | 16–64 |

- (h+w)·dot ≤ 4 MiB 在 h·w=256 时的影响：16×16 不受限；2×128 时 dot ≤ 32263，即 k 实际上限约 32256。
- 放大因子 F = h·w·dot 的范围：[32·1024, 256·65536] = [2^15, 2^24]。

**【理解】** 对两条无注释约束的推测：
- k ≥ 16r 保证至少有 16 个 r 块，使 16 个 jackpot 槽都至少被写一次（JACKPOT_SIZE=16）。它同时把去噪开销（约 2r/k，见 §4.3）限制在 12.5% 以内。
- k ≤ 4r² 等价于 r ≥ √k/2，即噪声秩不能相对 k 过小。

两者都没有代码注释佐证。

---

## 3. 中奖门槛

### 3.1 nbits → target（【代码】`proof_utils.rs:54-82`）

`target = mantissa · 256^(exponent−3)`，其中 `exponent = nbits>>24`，`mantissa = nbits & 0xFFFFFF`。mantissa 或 exponent 为 0、或 mantissa 的符号位 0x800000 为 1 时返回 0。
Go 侧 `CompactToBig` 的语义与之相同。

**【计算】** 主网 `PowLimitBits = 0x1b00ffff`（`node/chaincfg/params.go:355`）对应 target = 0xffff·2^192 ≈ 2^208。也就是说 target/2^256 ≤ 2^-48。

### 3.2 `check_jackpot_against_nbits`（【代码】sanity_checks.rs:148-157, 182-231）

```
nbits  = nbits_override.unwrap_or(header.nbits)
F      = h·w·(k − k%r)                         # difficulty_adjustment_factor = tile_size × dot_product_length
bound  = target(nbits)·F，若溢出则饱和为 U256::MAX    # scale()
接受 ⇔ U256::from_little_endian(hash_jackpot) ≤ bound   (含等号)
```

- tile_size = rows_pattern.size()·cols_pattern.size()（:183-185）；dot_product_length = k − k%r（proof_utils.rs:496-500）。
- 调用点有两个：
  - ZK 验证：`prepare_verification`（verify.rs:48-54），在 plonky2 验证之前执行；
  - 明文验证：`verify_plain_proof`（verify.rs:101-134），先重算 jackpot，再做检查。

**【理解】** 按 h·w·dot 缩放的含义如下：
- 一次尝试的工作量约为 W = h·w·dot 次乘加，中奖概率 p ≈ target·W/2^256。
- 因此**每次乘加的中奖概率恒为 target/2^256**，与 h、w、k 怎么选无关。
- 不同形状的彩票按"乘加数"归一化，使 target 可以直接解读为"每次乘加的中奖率"。
- 在 #275 之前，r 只通过 k%r 出现在 F 中；k 是 r 的倍数时，F 与 r 无关。

**【历史】** #275 之前，矿工侧 gateway 用 `hash_tile_h·hash_tile_w·rounded_common_dim` 缩放 target。其注释原话是 "We reduce difficulty for larger hash tiles (as they represent more work) and for larger rank (as it's the k dimension of the hash tile)"（`51a6e11b^:miner/pearl-gateway/src/pearl_gateway/comm/dataclasses.py:221-233`）。

---

## 4. 秩惩罚（rank penalty）

### 4.1 公式（【代码】sanity_checks.rs:12-13, 159-245）

- `PENALTY_BASE_RANK = 128`（:13），注释为 "Minimum accepted noise rank and normalization point for the rank penalty"。

**`check_rank_penalty(config, hash_jackpot, nbits)`**：
1. 要求 `r ≥ 128`；
2. 计算 `F' = penalized_adjustment_factor = h·w·(dot / r)·128`（:194-196），要求 F' > 0；
3. 要求 `LE256(hash_jackpot) ≤ scale(target(nbits), F')`。这里与共识路径一样**饱和**到 U256::MAX。

注释说明该函数本身不判断分叉高度："Deciding *when* the rule applies is the caller's job ... `nbits` is ... a block header's nbits for consensus, or a pool's share target."（:159-163）

**`penalized_target_bound(target, config)`**：返回 `target·F'`；若 F'=0 或乘积溢出则返回 `None`，**不饱和**（:233-245）。注释说明理由："a miner scaling a share target must be told the target is unusable rather than handed U256::MAX, which every hash satisfies."（:236-238）

**关系**：dot = ⌊k/r⌋·r，所以 F' = F·128/r。r=128 时惩罚中性；r=256 时门槛减半；r=1024 时为 1/8。单测在 :277-298 中验证了这两点。

**共识接入（【代码】`node/blockchain/validate.go:529-559`）**：
- `CheckCertificateRules` 在 `params.IsRankPenaltyForkActive(height)` 为真、且未设置 `BFNoPoWCheck` 时，调用 `zkpow.CheckRankPenalty(header.Bits, cert.PublicDataBytes())`。
- `CheckRankPenalty` 是 Go 包装（`node/zkpow/verify.go:190-217`），调用 FFI `check_rank_penalty`（`zk-pow/bindings/go/src/verify.rs:181-236`），后者只解析 config 和 jackpot 前缀。
- 分叉高度（`node/chaincfg/params.go`）：主网 96251（:369）、testnet 36761（:576）、testnet2 80627（:672）、regtest 1（:465）；0 表示禁用（`IsRankPenaltyForkActive`）。
- `blockchain.New` 拒绝把分叉高度设在 MoEForkHeight 之前，因为只有 V2+ 证书携带噪声秩（`51a6e11b` 提交说明，chaincfg/params.go:272-273 注释；检查在 node/blockchain/chain.go:2225-2232）。
- 未惩罚的门槛仍在 `VerifyCertificate`（无上下文、不依赖高度）中检查；惩罚门槛在带上下文的规则里检查。r ≥ 128 时 F' ≤ F，所以分叉后起约束作用的是惩罚门槛。

### 4.2 引入提交与漏洞描述（【历史】）

- 提交 `51a6e11b`，标题 "feat(consensus): rank-penalty softfork (#275)"，作者 Or Aharonee，日期 2026-08-05，版本 1.3.0。提交说明摘录：

  > "A proof's jackpot bound is scaled by the work one attempt costs, and the noise rank divides that work: mining at a larger rank buys a proportionally easier bound for the same effort. From RankPenaltyForkHeight on, the rank must be at least 128 and the jackpot must meet a bound scaled by 128/rank, cancelling the advantage. The rule only tightens validity, so pre-fork nodes accept every post-fork block."

- 同一提交还做了以下改动：
  - 节点矿工改用 Rust 的 `default_mining_config`，秩从 32 提到 128；
  - Python gateway 改为通过 `penalized_target_bound` 缩放 share target；
  - `CheckCertificateVersion` 改名为 `CheckCertificateRules`；
  - 用 `NetBehaviorFlags` 统一 SimNet 跳过 PoW 的逻辑。
- 网页上的 PR #275（github.com/pearl-research-labs/pearl/pull/275，WebFetch 获取）描述与提交说明基本一致，另外提到：
  - testnet 激活高度最初是 36757、testnet2 最初是 80638，第二个提交才改成现值；
  - 链接到 issue #332：testnet 节点在 36760 处卡住，因为拒绝了 36761 号块。
  - PR 页面**没有**更具体的漏洞推导或数值。
- 仓库里没有其他文档解释 "noise rank divides that work" 这句话的含义（docs/ 只有 MoE 和 salted-seed 的升级指南）。

### 4.3 数值：单位工作量的中奖概率（【计算】`targets.py` Table A，h·w=32）

单位：`× target/2^256`。"cell" 指转录检查点（一次尝试产生 h·w·dot/r 个被归约的累加值）。

| r | k | F_pre=h·w·dot | F_post=h·w·(dot/r)·128 | F_post/F_pre | 修复前 每MAC | 修复后 每MAC | 修复前 每cell | 修复后 每cell |
|---|---|---|---|---|---|---|---|---|
| 32 | 1024 | 32768 | (131072) | (4) | 1 | **拒绝**(r<128) | 32 | 拒绝 |
| 64 | 1024 | 32768 | (65536) | (2) | 1 | **拒绝** | 64 | 拒绝 |
| 128 | 2048 | 65536 | 65536 | 1 | 1 | 1 | 128 | 128 |
| 256 | 4096 | 131072 | 65536 | 1/2 | 1 | 1/2 | 256 | 128 |
| 512 | 8192 | 262144 | 65536 | 1/4 | 1 | 1/4 | 512 | 128 |
| 1024 | 16384 | 524288 | 65536 | 1/8 | 1 | 1/8 | 1024 | 128 |

（对 r ≥ 128 取其他合法 k，例如 k=16384，比例完全相同，见脚本输出。）

**【理解】** 从表中可以读出：
- 按"每次乘加"计：修复前所有 r 相同；修复后 r>128 的单位乘加中奖率按 128/r 打折。
- 按"每个转录检查点"计：修复前中奖率正比于 r，大 r 占便宜；修复后恒为 128，与 r 无关。
- 若把惩罚公式套用到 r<128，会反过来给出 2 倍、4 倍的奖励。所以规则对 r<128 直接拒绝，而不是用公式处理。这也是 `PENALTY_BASE_RANK` 同时作为"下限"和"归一点"的原因。

**为什么修复前大 r 占便宜（【理解】）**

代码把一次尝试的价值按 h·w·dot（乘加数）计。但修复说明认为一次尝试"真实"的工作量按 r 缩小，即与 h·w·dot/r 成正比。下面分三个角度看：

1. **噪声生成成本**：基本与 r 无关，大 r 只是略贵。
   - E_AL（m×r）和 E_BR（r×n）由 blake3 产生，每个哈希 32 字节，共 (m+n)·r/32 次哈希；
   - E_AR、E_BL 是 k 对 ±1 下标，共 2k/8 次哈希，与 r 无关；
   - 加噪本身每个元素是 1 次减法（`matvec_sparse_perm`，pearl_noise.rs:35-43），与 r 无关。
   - 结论：大 r 不会因噪声生成而便宜。

2. **去噪成本**：随 r 增长，但纯 PoW 矿工可以完全跳过。
   - C = C' − (A·E_BL)·E_BR − E_AL·(E_AR·(B+E_B))。用稀疏实现约需 2·m·n·r 次乘加加 O((m+n)k) 次加法，相对主乘法的开销约为 2r/k。
   - 参考 Python 实现用稠密 matmul，约 2mkr + 2nkr + 2mnr（`miner/miner-base/src/miner_base/noisy_gemm.py` 中 `noise_A`/`noise_B`）。
   - 对"有用矿工"来说，大 r 更贵。例如 configs.py 中修复前 r=1024、k=16384 的稀疏模型为 0.887，而 r=128、k=16384 为 0.983。
   - 对"只为出块"的矿工来说，他们不需要 C，去噪是 0 成本。
   - 结论：去噪成本**惩罚**有用矿工的大 r，却约束不到纯 PoW 矿工。

3. **每次尝试的工作量**：这是修复说明指向的地方。
   - 彩票不是对输出 C 做哈希，而是对**计算转录**做哈希：每 r 步一个检查点，共 h·w·dot/r 个值。这是 cuPOW 的设计，见 `pearl-research/papers/cupow_2504.09971.txt:288-343`。
   - 原因在于：矿工可以取 A=B=0，此时噪声乘积 E_A·E_B 是秩 r 的矩阵，输出本身可以 O(n²r) 算出，必须依靠"转录不可预测性"。
   - cuPOW 的假设 6.4 只承诺：计算所有中间值不能快于 O(n^{ω_r+1}/r)（论文 :845-858）。论文还指出，写出全部中间值的位数 Θ(n³/r) 本身就是下界，并说明对较小的 r，借助快速矩形乘法，n×r×n 可以做到 n^{2+o(1)}。
   - 因此协议**能担保的**每张彩票工作量正比于检查点数 h·w·dot/r，而不是 h·w·dot。修复前按 h·w·dot 付"门槛"，在大 r 下超付了 r/128 倍（以 r=128 为基准）。修复后按 h·w·(dot/r)·128 付，正好把"每个检查点的中奖率"固定为常数（见上表"每cell"两列）。
   - 实际硬件层面也有一个与 r 成反比的开销：官方 GPU 内核每 r 步要做一次 `warpgroup_wait<0>()`，再对累加寄存器做 XOR 归约（`miner/pearl-gemm/csrc/gemm/pow_utils.hpp:166-178`，`reduce_every_k = R/MMAAtom_K`，见 collective_mainloop.hpp:275-279）。r 越大，同步点越少，张量核流水线越接近峰值吞吐，而修复前门槛不变。外部挖矿软件（MoneroOcean mo-miner）在修复前使用 K=4096 / rank=256，见 §6.4。

   我**没有**找到一个具体的经典算法，能让 A=B=0 时单张彩票的乘加数低于 h·w·dot：按 E_AL·M_z·E_BR 递推，每块成本仍是 h·r·w。所以"大 r 占便宜"的依据是：(a) 理论上的转录下界与快速矩形乘法；(b) 实际实现中每 r 一次的归约和同步开销。修复说明本身没有给出推导，以上是我的解释。

---

## 5. 矿工可选的其他自由度

**【代码】** 矿工可以自由选择的量：
- k：`common_dim`，即 A 的列数（gateway 中 `common_dim=self.A.shape[1]`，dataclasses.py:162-171）；
- r；
- 模式，即 h、w 和形状；
- m、n；
- A、B 的内容，范围 [-64,64]。

这些都在挖矿前通过 job_key/commitment 承诺下来。每个已承诺的矩阵对，尝试次数为 `attempts = #合法t_rows · #合法t_cols`。

各自由度对成本和门槛的影响（【理解】，配合下表）：

| 自由度 | 每次尝试成本 | 门槛 F（修复后） | 每乘加中奖率（修复后） | 其他影响 |
|---|---|---|---|---|
| k（r 固定） | ∝ dot | ∝ dot | 不变 | k%r 尾部要计算（如果算整块乘法）但不计入彩票；大 k 摊薄去噪开销（2r/k）和承诺哈希开销 |
| h·w | ∝ h·w | ∝ h·w | 不变 | 每份矩阵的尝试次数 ∝ 1/(h·w)；单个块越大，块级归约事件越少（每次归约对应 h·w·r 次乘加） |
| 模式形状（步长或连续） | 不变 | 不变 | 不变 | 只影响内存访问和内核布局；GPU 要求模式最大值 ≤ matmul tile（matmul_config.py:29-37） |
| m、n | 不影响单次成本 | 不变 | 不变 | 决定尝试数和固定开销（哈希 (m+n)k 字节、噪声生成）的摊薄；边缘不足一个周期的行列不产生彩票 |
| r | 理论检查点数 ∝ 1/r；实际乘加不变 | ∝ 128/r | ∝ 128/r（r≥128） | 去噪开销 ∝ r |

**【计算】`configs.py` 输出**（完整版见 `experiments/config/configs_out.md`）。数值是期望中奖次数除以工作量，单位为 target/2^256。

列名说明：
- "tileMAC" 只计彩票块乘加，即 attempts·h·w·dot；
- "fullMAC" 指 m·n·k；
- "+稀疏/稠密去噪" 加上 §4.3 中的去噪估算；
- "cell" 指转录检查点。

| 配置 | r | k | h×w | m×n | 尝试数 | 修复前/tileMAC | 修复后/tileMAC | 修复后/fullMAC | 修复前/(MAC+稀疏去噪) | 修复后/(MAC+稀疏去噪) | 修复后/(MAC+稠密去噪) | 修复后/cell | 每次块归约的MAC |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 节点 CPU 默认（#275后） | 128 | 2048 | 4×8 | 256×512 | 4096 | 1.000 | 1.000 | 1.000 | 0.871 | 0.871 | 0.381 | 128 | 4096 |
| 节点 CPU 默认（#275前） | 32 | 1024 | 4×8 | 256×512 | 4096 | 1.000 | 拒绝 | 拒绝 | 0.921 | 拒绝 | 拒绝 | 拒绝 | 1024 |
| miner-base GPU，k=4096 | 128 | 4096 | 2×64 | 4096² | 131072 | 1.000 | 1.000 | 1.000 | 0.939 | 0.939 | 0.842 | 128 | 16384 |
| 16×16，r=256，k=4096（类 mo-miner） | 256 | 4096 | 16×16 | 4096² | 65536 | 1.000 | 0.500 | 0.500 | 0.887 | 0.444 | 0.364 | 128 | 65536 |
| 16×16，r=128，k=4096 | 128 | 4096 | 16×16 | 4096² | 65536 | 1.000 | 1.000 | 1.000 | 0.939 | 0.939 | 0.842 | 128 | 32768 |
| 2×16，r=128，k=4096 | 128 | 4096 | 2×16 | 4096² | 524288 | 1.000 | 1.000 | 1.000 | 0.939 | 0.939 | 0.842 | 128 | 4096 |
| 16×16，r=1024，k=16384 | 1024 | 16384 | 16×16 | 4096² | 65536 | 1.000 | 0.125 | 0.125 | 0.887 | 0.111 | 0.059 | 128 | 262144 |
| 16×16，r=128，k=16384 | 128 | 16384 | 16×16 | 4096² | 65536 | 1.000 | 1.000 | 1.000 | 0.983 | 0.983 | 0.877 | 128 | 32768 |
| 16×16，r=128，k=65536 | 128 | 65536 | 16×16 | 1024² | 4096 | 1.000 | 1.000 | 1.000 | 0.988 | 0.988 | 0.665 | 128 | 32768 |
| 16×16，r=1024，k=65472（k%r=960） | 1024 | 65472 | 16×16 | 1024² | 4096 | 1.000 | 0.125 | 0.123 | 0.948 | 0.119 | 0.024 | 128 | 262144 |
| 极小矩阵 m=n=17 | 128 | 4096 | 16×16 | 17×17 | 1 | 1.000 | 1.000 | 0.886 | 0.578 | 0.578 | 0.028 | 128 | 32768 |

数据要点（只列现象）：
- 只计彩票块乘加时，修复后 r=128 的所有配置都等于 1.000，r>128 等于 128/r。
- 计入整块乘法、去噪、矩阵大小时，各配置之间出现差异，范围约 0.028 到 0.988。
- 不同 h·w 下，"每次块归约对应的 MAC 数"相差 8 倍（4096 对比 32768），h·w 不受任何惩罚。
- 去噪开销模型是估算，不是代码事实。真实 GPU 内核中加噪、去噪的具体实现成本没有测量。

---

## 6. 份额难度（矿池）与区块难度

### 6.1 现有接口（【代码】）

| 接口 | 位置 | 份额门槛 | 是否含秩惩罚 |
|---|---|---|---|
| `verify_plain_proof(header, pp, nbits_override, seed)` | zk-pow/src/api/verify.rs:101-134 | `check_jackpot_against_nbits(nbits_override)`，F = h·w·dot | **否** |
| Go FFI `verify_plain_proof_ffi(header, pp, len, cert_version, nbits_override, err)` | zk-pow/bindings/go/src/plain.rs:16-69 | `nbits_override==0` 表示用 header 的 nbits | **否** |
| Go FFI `verify_zk_proof_v2_with_nbits` / `_v3_with_nbits` | bindings/go/src/verify.rs:139-179 | `prepare_verification` → `check_jackpot_against_nbits(Some)`（api/verify.rs:48-54） | **否** |
| Go `VerifyZKProofFFI(header, cert, nbitsOverride *uint32)` | node/zkpow/verify.go:113-188 | 同上 | **否** |
| Go `CheckRankPenalty(bits, publicData)` / FFI `check_rank_penalty` | node/zkpow/verify.go:190-217；bindings verify.rs:181-236 | `bits` 可以传 share 的 nbits（注释 :190-192） | **是**；参数为线上 public_data |
| Python `verify_plain_proof_v2/v3(header, pp, nbits_override=None)` | py-pearl-mining/src/lib.rs:197-226 | 同 verify_plain_proof | **否** |
| Python `penalized_target_bound(target, config)`、`PENALTY_BASE_RANK` | py-pearl-mining/src/lib.rs:228-252, 521-522 | 返回 target·F'，不可表示时返回 None | 用于矿工侧缩放，不是校验 |

其他相关事实：
- 节点内**没有**任何调用方传入非空的 `nbitsOverride`（verify.go:106,110 都传 nil）。
- 仓库内没有矿池实现（proxy/ 只有 Caddy RPC 缓存）。
- `VerifyCertificateWithNbits` 曾在 `4849abc7` 中恢复，但只存在于分支 `origin/restore-verifycertificatewithnbits`，不在 main 上。
- **Python 示例**（`py-pearl-mining/examples/v1_v2_gateway_example.py:84-96`）中，`validate_share` 只调用 `verify_plain_proof_for_cert_version(..., nbits_override=share_nbits)`。它没有秩惩罚，没有分叉高度判断，也没有 dense-only 判断。
- **矿工侧**：gateway 的 `MiningJob.adjust_target` 用 `penalized_target_bound(self.target, mining_config)` 生成 GPU 的 pow_target（dataclasses.py:212-225；调用在 `miner/vllm-miner/src/vllm_miner/gemm_operators.py:116`，MoE 版在 moe_gemm_operators.py:141-142）。所以官方矿工自己筛选候选时已经用了惩罚门槛。
- **历史**：
  - `74379c25`（#161，2026-06-08）为 `verify_plain_proof` 增加 `nbits_override`，标题 "for pool share difficulty"；
  - `9d1a2951`（#165，2026-07-10）增加 Go FFI `verify_plain_proof_ffi`，用于矿池份额验证。
  - 两者都早于 #275，之后没有为明文路径补充秩惩罚。

### 6.2 份额门槛与区块门槛的关系（【理解】）

- 同一个 header（包含区块 nbits）下，份额和区块的验证对象完全相同，只是比较的 target 不同：T_share ≥ T_block。所以份额中奖集合包含区块中奖集合。
- 分叉后，区块在共识层的有效条件是 `hash ≤ T_block·F` **且** `hash ≤ T_block·F'`。份额如果只按 `T_share·F` 检查，r>128 的份额会被高估 r/128 倍。
- 矿池需要自己补充的检查：
  1. **秩惩罚**：分叉高度之后，份额应满足 r ≥ 128 且 `hash ≤ T_share·F'`。
     - 明文路径不会返回 hash_jackpot，所以 Python/Go 矿池不能对 PlainProof 直接调用 `check_rank_penalty`。
     - 可行的做法有三种：只接受 r==128（此时 F'=F，惩罚中性）；对 r>128 用更难的 share nbits，即 T_share·128/r；或者生成 ZK 证书后对 public_data 调用 `CheckRankPenalty`（代价很高）。
     - Python 的 PlainProof 通过 get_all 暴露了 `noise_rank` 和 `k`（ffi/plain_proof.rs:70-84），可以用来计算缩放。
  2. **分叉高度和证书版本**：`verify_plain_proof_ffi` 只通过 `check_cert_version_eligible` 判断 MoE 需要 ≥V2（ffi/plain_proof.rs:195-204），不知道 dense-only 和 salted-seed 的分叉高度。
  3. **header 的来源**：验证只把证明绑定到调用方给出的 header，header 是否就是矿池下发的任务，要由矿池自己保证。
  4. **输入大小上限**：plain.rs:23-26 明确提示 `pp_len` 无上界，"Suggested cap: ~8 MiB"。
  5. **重复份额和重放**：库里没有相关逻辑。
  6. **非常容易的份额 target**：在 F 下 T > U256_MAX/F（F=2^24 时约 2^232；F=2^15 时约 2^241，见 targets.py Table D），共识路径的 `scale` 会饱和为 U256::MAX，任意 hash 都通过（sanity_checks.rs:207-219）。`penalized_target_bound` 在这种情况下返回 None，`check_rank_penalty` 则同样饱和。

### 6.3 hash 比较语义

所有路径都用 `U256::from_little_endian(hash_jackpot) ≤ bound`（含等号）。GPU 内核的 `check_pow_target` 按 u32 从高位字到低位字比较，与之一致（pow_utils.hpp:205-233）。

### 6.4 MoneroOcean mo-miner PR #5（WebFetch 可以访问）

- 标题 "Fix PearlHash V3 seeds and rank-penalized share targets"，作者 friskens，2026-10-07 关闭。维护者说明已以不同形式并入 `cc870ff`，随 0.9.0 发布。
- PR 描述的两个问题：
  1. V3 证书任务用了旧的噪声种子；
  2. "Pools enforcing the rank penalty received candidates selected against an unpenalized target"。
- PR 的修复：
  - 新增每个矿池可配的 `pearlhash_rank_penalty` 选项，默认 false；描述中提到在 LuckyPool 上测试过；
  - 门槛按 `256 * floor(K/rank) * 128` 计，并说明在 K=4096、rank=256 时是旧门槛的一半。
- 实测数据：一个被拒的 V3 份额，"passed the old bound but failed the rank-penalized bound"，这与矿池的报错一致。说明至少有矿池对份额实施了秩惩罚。
- 并入的 `cc870ff` 中（`helper/hash.js`）：
  - scale = `16n*16n*BigInt(effectiveK/rank*128)`，即 h=w=16；
  - rank 限制在 [128, 1024]；
  - 注释为 "V3 normalizes the target to rank 128. The miner reports rank-128-equivalent GEMM MACs"。
- 该 PR 引用的 pearl 提交 `3fe22676` 是当前 HEAD 的祖先。

---

## 7. 默认配置及其理由

**节点 CPU 挖矿**（regtest、测试用）：
- 【代码】`zk-pow/bindings/go/src/mine.rs:15-26, 172-181`：`DEFAULT_RANK = MIN_NOISE_RANK = 128`；`DEFAULT_COMMON_DIM = 16·128 = 2048`；行 [0,8,64,72]（h=4）；列 [0,1,8,9,32,33,40,41]（w=8）；h·w = 32。
- Go 侧：`DefaultM=256`、`DefaultN=512`、`DefaultNBits=0x1E01FFFF`（node/zkpow/miner.go:26-28）。
- 注释给出的理由：
  - "Sitting on the rank-penalty floor keeps mined blocks valid once that rule activates, without paying the penalty a larger rank would."（mine.rs:15-18）
  - "Smallest common dimension the sanity checks allow for DEFAULT_RANK (k >= 16 * r)"（:21）
- 【历史】#275 之前，Go 硬编码的配置是 k=1024、r=32、相同的模式（`51a6e11b` 的 diff，删除的 `defaultMiningConfigV1`）。
- `zk-pow/src/ffi/mine.rs:19-110`：`try_mine_one` 生成随机矩阵，用**未惩罚**的 `extract_difficulty_bound` 判断中奖。在 r=128、k%r=0 时这与惩罚门槛相同。

**GPU 矿工**（`miner/miner-base/src/miner_base/settings.py`）：
- `noise_rank=128`、`noise_range=128`、`idxs_per_col=2`；
- matmul tile 128×256×128；
- 行 [0,8]（h=2），列为 64 个元素 {0,1,8,9,…,248,249}（w=64），h·w=128，注释为 "Hash tile pattern for the 128x256 tile"；
- k 取实际模型矩阵的 A.shape[1]（`gpu_matmul_config.py:22-41`）。
- 【历史】`noise_rank=128` 从首个提交 `960681d4` 起就是这个值。#275 增加了 `≥ PENALTY_BASE_RANK` 的校验器（settings.py:40-48）。官方 GPU 内核只支持 R=64 或 128（pearl_gemm_api.cpp:158-163）。

**白皮书（FP8 方案，另一协议版本）**写的是 "the protocol uses r = 32 for k ≤ 2^16"（`papers/pearl_whitepaper_fp8.txt:137,309`）。它与当前整数实现中 r ≥ 128 的规则不同，仅作备注。

---

## 8. 难度目标在共识层的用法

- **【代码】chainwork 与配置无关**：
  - `CalcWork(bits) = 2^256 / (CompactToBig(bits)+1)`（`node/blockchain/internal/workmath/difficulty.go:141-152`；包装在 `node/blockchain/difficulty.go:69-71`）；
  - 用于 `blockindex.go:118`（workSum）和 `chain.go:1273, 2070-2071`（选链）；
  - 只依赖 header 的 `Bits`，不读取 k、r、h、w。
- **难度调整**：`calcNextRequiredDifficulty` 使用 WTEMA，同样只看 bits 和时间戳（difficulty.go:73-）。
  - 主网 `TargetTimePerBlock` = 3 分 14 秒，`WTEMAHalfLife` = 1 周（params.go:357-358）；
  - PowLimit = 2^208−1（params.go:29）。
- **`checkProofOfWork`**（validate.go:309-339）：检查 target 在 (0, powLimit] 区间内，然后 `VerifyCertificate`（内部做未惩罚的 jackpot 检查和 ZK 验证）。秩惩罚在 `CheckCertificateRules` 中按高度执行（validate.go:549-556）。
- **【理解】** 修复前，任何配置下"每次乘加的中奖率"都是 T/2^256，所以每个区块的期望乘加数约为 2^256/T，约等于 CalcWork。也就是说 chainwork 大致等于"期望彩票乘加数"。
  修复后，期望乘加数变成 (2^256/T)·(r/128)·(k/dot)，chainwork 对应的是"rank-128 等价乘加数"。mo-miner 的注释 "rank-128-equivalent GEMM MACs" 用的就是这一口径。
  区块难度和份额门槛共用同一套放大因子 F 或 F'，只是 target 不同（见 §6.2）。

---

## 9. 疑点（留待问题阶段分析）

1. 除了 r 之外，其他自由度是否也有类似"秩漏洞"的优势配置？例如：
   - 大 h·w 每 MAC 的块级归约和同步更少，而门槛没有对 h·w 归一化；
   - 极小或极大的 m、n、k 摊薄固定开销；
   - k%r 尾部不计入彩票，可以不算；
   - A=B=0 或低秩 A、B 下是否存在低于 h·w·dot 的具体算法。
2. 明文份额验证路径（`verify_plain_proof` / `verify_plain_proof_ffi` / Python `verify_plain_proof_v*` 和官方示例 `validate_share`）都不含秩惩罚，也不返回 hash_jackpot。矿池可能漏掉惩罚，r>128 的份额会被高估 r/128 倍。
3. `check_rank_penalty` 用饱和的 `scale`，传入很容易的 share nbits 时门槛等于 U256::MAX；而 `penalized_target_bound` 返回 None。两条路径行为不一致。
4. `public_params_sanity_check` 先调用 `dot_product_length()`（sanity_checks.rs:24），后检查 rank（:28）。`MiningConfiguration::from_bytes` 不拒绝 rank=0，此时会出现 `k % 0` 的 panic 路径（FFI 层有 catch_panic）。
5. 偏移合法性（`offset_is_valid`）只在解析层检查，sanity check 不重复检查。其他构造 PublicProofParams 的入口是否都经过解析层？
6. k ≥ 16r、k ≤ 4r²、h·w ∈ [32,256]、k ≥ 1024 等约束缺少设计理由的注释，它们与安全假设的关系不明。
7. 惩罚以 r=128 为基准，等价于假设"r=128 时每乘加的成本与转录下界相符"。这个归一点的依据，以及与白皮书 r=32 的出入，需要核实。
8. 分叉激活前后：testnet 在 36760/36761 卡链（issue #332）；激活前已出的大 r 区块的累计影响；SimNet 和模板通过 BFNoPoWCheck 跳过规则的边界。
9. 节点 CPU 矿工仍用未惩罚的 `extract_difficulty_bound` 筛选（ffi/mine.rs:109）。目前因 r=128 而无影响，但如果改用其他 r 会产出共识拒绝的块。
10. MoE（GROUPED_GEMM）配置在 dense-only 分叉后被禁止，但明文份额路径不检查这一点。MoE 下 h 用的是 outer_indices 的个数，惩罚公式的适用性未单独验证。
