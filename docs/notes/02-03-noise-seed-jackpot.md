# 组件 2（噪声生成与种子链）与组件 3（彩票 / jackpot 计算）技术笔记

- 研究对象：Pearl 源码 `third_party/pearl`，commit `156ef9a8`（2026-09-29）。
- 以主网当前方案为准：int7 数据（`MMAType::Int7xInt7ToInt32`）+ 证书 V3（salted 种子）。主网 `SaltedSeedForkHeight = 99000`、`RankPenaltyForkHeight = 96251`（`node/chaincfg/params.go:369-371`）。V1、V2 的差异在各节末尾说明。
- 另有一个 `origin/fp8-scheme` 分支（例如 `4c7177be`、`0b18ef42`）不在 HEAD 中，本笔记不涉及。
- 标记约定：
  - **【代码】**：直接读代码得到的事实，带 `文件:行号`（路径相对 `third_party/pearl`）。
  - **【实验】**：`experiments/noise/` 下脚本的输出。
  - **【理解】**：我的解读或推导，不是代码原文。
- 实验文件：

| 文件 | 内容 |
|---|---|
| `blake3_py.py` | 纯 Python 实现的 BLAKE3（环境里没有 numpy，也没有 blake3 包） |
| `pearl_ref.py` | 按 Rust 代码逐行复刻：序列化、job_key、种子链、噪声、jackpot、难度 |
| `test_vectors.py` → `test_vectors.out` | 对照仓库内固定的测试向量（`seed.rs`），全部 OK |
| `rust_xcheck/`（Rust，以 path 依赖调用 zk-pow 本体）→ `rust_xcheck.out`；`xcheck_py.py` → `xcheck_py.out`；`xcheck_diff.out` | 同一输入下 Python 复刻与 Pearl 自身 Rust 实现**逐字节一致**（`IDENTICAL`），覆盖 job_key、两个种子（Legacy 与 Salted）、噪声、jackpot 消息、jackpot 哈希 |
| `exp_noise_stats.py` → `exp_noise_stats.out` | 噪声统计（第 5 节） |
| `exp_jackpot_example.py` → `exp_jackpot_example.out` | h=2、w=4、k=64、r=32 手算示例，以及 GPU 调度模拟（第 7、9 节） |
| `config_bytes.out` | 两种默认 MiningConfiguration 的字节编码 |

> 构建 Rust 交叉校验时用的是 `cargo +1.88.0 build --offline`，目标目录放在 scratch 下。`git status` 确认 Pearl 仓库没有被改动。

---

# 组件 2：噪声生成与种子链

## 1. 任务头与挖矿配置的字节布局，以及 job_key

### 1.1 IncompleteBlockHeader（76 字节）

【代码】结构定义见 `zk-pow/src/api/proof.rs:10-16`；序列化见 `zk-pow/src/api/proof_utils.rs:515-527`（`SERIALIZED_SIZE = 76`，`:506`）。

| 偏移 | 长度 | 字段 | 编码 |
|---|---|---|---|
| 0 | 4 | version | u32 LE |
| 4 | 32 | prev_block | 32 字节，**逆序**写入（`proof_utils.rs:522`） |
| 36 | 32 | merkle_root | 32 字节，**逆序**写入（`:523`） |
| 68 | 4 | timestamp | u32 LE |
| 72 | 4 | nbits | u32 LE（Bitcoin compact target） |

【理解】这里不包含 nonce，也不包含 proof commitment。区块头中的 ProofCommitment 是另一个字段（`node/zkpow/verify.go:126-130`），在证书校验时才比对。

### 1.2 MiningConfiguration（52 字节）

【代码】结构定义见 `proof.rs:63-72`；序列化见 `proof_utils.rs:448-457`，反序列化见 `:459-494`（要求逐字节往返一致）。

| 偏移 | 长度 | 字段 | 编码 |
|---|---|---|---|
| 0 | 4 | common_dim (k) | u32 LE |
| 4 | 2 | rank (r，噪声秩) | u16 LE |
| 6 | 2 | mma_type | u16 LE，只允许 0（Int7xInt7ToInt32），见 `:467-471` |
| 8 | 6 | rows_pattern | PeriodicPattern 编码（见下） |
| 14 | 6 | cols_pattern | PeriodicPattern 编码 |
| 20 | 32 | trailer | `e(u16) ‖ top_k(u16) ‖ 0^28`；e=0 表示非 MoE；保留字节必须为 0（`:438-446`、`:478-483`） |

【代码】PeriodicPattern 是最多三维的等差网格 `{a·s0 + b·s1 + c·s2}`（`proof.rs:27-34`）。它编码为 3 组 `(factor−1, length−1)` 字节，满足 `stride_i = factor_i · min_stride`，其中 `min_stride` 等于前一维的 `stride·length`（`proof_utils.rs:105-155`）。周期上限为 `2^24`（`:127-130`）。

【实验】`config_bytes.out` 给出两个默认配置的编码：
- 矿工 GPU 默认：rows `[0,8]` 编码为 `07 01 00 00 00 00`；cols `[0,1,8,9,…,248,249]`（64 个）编码为 `00 01 03 1f 00 00`。
- 节点 CPU 默认：rows `[0,8,64,72]` 编码为 `07 01 03 01 00 00`；cols `[0,1,8,9,32,33,40,41]` 编码为 `00 01 03 01 01 01`。

### 1.3 job_key

【代码】`job_key = BLAKE3(header_bytes(76) ‖ mining_config_bytes(52))`，无 key，共 128 字节，即两个 BLAKE3 块（`proof_utils.rs:348-353`，注释在 `:434`）。miner 端也用同样的定义（`miner/miner-base/src/miner_base/commitment_hash.py:36-37`）；节点 CPU 挖矿见 `zk-pow/src/ffi/mine.rs:431-436`。

【理解】job_key 绑定了 k、r、h·w 的形状和 MoE 参数。它**不包含** m、n、t_rows、t_cols。V3 中 m、n 通过 salt 进入种子，见第 3 节。

### 1.4 证书公开数据布局（参考）

【代码】`mining_config(52) | hash_a(32) | hash_b(32) | hash_jackpot(32) | m(4) | n(4) | t_rows(4) | t_cols(4)`，MoE 字段附在其后（`proof.rs:76-83`，`proof_utils.rs:1334-1343`）。解析时要求 t_rows、t_cols 是对应 pattern 的合法偏移，即位于分块划分的起点（`proof_utils.rs:1304-1311`）。

---

## 2. 矩阵承诺：A 与 Bᵀ 的 BLAKE3 Merkle 树

### 2.1 Rust / Python 参考实现

【代码】
- **被哈希的数据**：A 按行主序展开为 int8 字节（m×k）。B 取**转置**后按行主序展开（n×k，即 B 的列条带连续存放），目的是能打开 B 的列（`commitment_hash.py:52-53`）。两者都**零填充到 1024 的倍数**（`pearl-blake3/src/merkle.rs:15-27`；`miner_base/matrix_merkle_tree.py:44-47`；`ffi/mine.rs:46-47`）。
- **key**：`hash_a = BLAKE3_keyed(pad1024(A), key=job_key)`，`hash_b = BLAKE3_keyed(pad1024(Bᵀ), key=job_key)`（`proof.rs:91-92`；`ffi/mine.rs:451-452`）。
- **树的结构**：叶子是 1024 字节的 BLAKE3 chunk，其 CV 按标准 BLAKE3 chunk 规则计算（keyed，chunk counter = 叶序号，见 `pearl-blake3/src/hasher.rs:80-87, 144-180`）。父节点逐层两两合并，奇数个时直接把最后一个上提（`hasher.rs:130-142`）；最后两节点合并时带 ROOT 标志（`merkle.rs:66-74`）；只有一个 chunk 时直接整体哈希（`merkle.rs:54-61`）。
  - 【理解】“自底向上两两合并、奇数上提”与 BLAKE3 规定的左完全树形状相同。因此 Merkle 根**就是**对填充后数据做的标准 keyed BLAKE3 哈希。仓库测试 `merkle.rs:451-475` 对此做了断言。
- **行与叶子的对应**：第 i 行占据字节 `[i·k, (i+1)·k)`，所需叶子范围是 `⌊i·k/1024⌋ … ⌊((i+1)·k−1)/1024⌋`（`merkle.rs:160-176`）。因为 `k ≥ 1024` 且 `k % 64 == 0`（`sanity_checks.rs:34-37`），一行至少跨一个 chunk；当 k 不是 1024 的倍数时，chunk 会跨越行边界。
- **叶子数校验**：plain proof 解析时要求 `total_leaves == padded_chunk_len(m·k)/1024`。这条校验在 V3 之后加入（commit `10a0ee18`，见 `zk-pow/src/ffi/plain_proof.rs:423-450`）。

### 2.2 GPU 端（`miner/pearl-gemm/csrc/tensor_hash/`）

【代码】
- 入口函数是 `run_tensor_hash(data, key, out, roots)`（`tensor_hash_api.hpp:49-122`）。它对 `data.numel()` 字节做哈希，调用方传入**未填充**的张量（vllm：`gemm_operators.py:125-139`，此处 B 已经是 n×k 布局）。
- 第一阶段 `merkle_tree_roots_kernel`：每个线程处理一个 1024 字节 chunk，每个 CTA 内部构建子树。最后一个不完整的 chunk 由 `load_partial_chunk` 从全局内存读取，**超出数据部分补零**（`merkle_tree_roots_kernel.hpp:592-623`）。注释说明这样做是为了和 Python 参考实现“零填充到完整 1024 字节”一致，并且部分 chunk 也计为一个叶子（`:492-504`）。叶子数是 2 的幂时用完美树，否则用 BLAKE3 的树形（`:506-519`）。
- 第二阶段 `ComputeBlakeMTKernel` 合并各 CTA 的根，第三阶段 `ReduceRootsKernel` 做最后的归约。只有一个 CTA 时，第一阶段直接加 ROOT 标志（`tensor_hash_host.hpp:79-90`）。key 通过 `__constant__ c_key` 传入（`tensor_hash_constants.cuh:5`；`tensor_hash_host.hpp:33-36`）。
- 一致性：`miner/pearl-gemm/tests/test_tensor_hash.py:24-86` 在大量形状上对比了 CUDA 结果与 `MatrixMerkleTree.tensor_hash`（Python 端会先填充），覆盖 256 B、不足 64 B 的尾块、跨 CTA 等情况。

【理解】两端都等价于 `BLAKE3_keyed(pad1024(bytes), job_key)`。GPU 端并没有真的填充张量，而是在加载时补零。不过 1024 字节的 chunk 不论是否补零，其 BLAKE3 CV 计算方式相同（block_len 和标志位都一样），所以两种做法结果一致。本次实验没有在 GPU 上运行，一致性依据是仓库测试和代码阅读。

---

## 3. 种子链：b_noise_seed 与 a_noise_seed

### 3.1 当前代码（V3 = Salted）

【代码】`proof_utils.rs:362-373` 与 `seed.rs:10-59`：

```
(hash_a', hash_b') = Salted ? (bind_root_a(hash_a, m), bind_root_b(hash_b, n)) : (hash_a, hash_b)
hash_activations   = MoE ? BLAKE3(hash_a' ‖ BLAKE3(hash_routing ‖ BLAKE3_keyed(pad1024(offsets_le), job_key)))
                         : hash_a'
b_noise_seed = BLAKE3(job_key ‖ hash_b')             (64 字节，无 key)
a_noise_seed = BLAKE3(b_noise_seed ‖ hash_activations) (64 字节，无 key)

bind_root_a(h, m) = BLAKE3_keyed(h ‖ LE32(m) ‖ 0^28, key = SEED_SALT_A)
bind_root_b(h, n) = BLAKE3_keyed(h ‖ LE32(n) ‖ 0^28, key = SEED_SALT_B)
SEED_SALT_A = BLAKE3("pearl/cert-v3/noise-seed/A")   (常量硬编码，seed.rs:30-33；测试 :67-70 会重新推导)
SEED_SALT_B = BLAKE3("pearl/cert-v3/noise-seed/B")   (seed.rs:36-39)
```

- 每一步都恰好是一个 64 字节的 BLAKE3 块。
- MoE 中，salt 加在 routing 折叠**之前**（`proof_utils.rs:357-361`；`compute_hash_activations` 见 `:34-45`）；n 是单个 expert 的列数。
- 用途：
  - `commitment_hash = (b_noise_seed, a_noise_seed)`（`proof_utils.rs:659`）。
  - `a_noise_seed` 用作 jackpot 哈希的 key（`zk-pow/src/api/verify.rs:74, 130`；`pearl_program.rs:159-161`），也是 ZK 公开输入 `commitment_hash`。
- 证书版本到派生方式的映射：V1/V2 → Legacy，V3 → Salted（`zk-pow/src/ffi/plain_proof.rs:172-177`）。
- GPU 端：`CommitmentHashFromMerkleRootsKernel`（`commitment_hash_from_merkle_roots_kernel.hpp:75-136`）按同样顺序计算：先 salt（`bind_root`，`:168-183`；salt 常量是 LE u32 字，`:8-15`），再做 MoE 折叠（`:99-114`），最后走链 `hash_pair(key, B)`、`hash_pair(seedB, A)`（`:116-125`）。`hash_pair` 是单块、无 key 的压缩，即 `BLAKE3(left‖right)`（`:151-163`）。V3 时由调用方传入 `salted_dims=(m,n)`（`vllm_miner/gemm_operators.py:143-150`）。
- Python 端：`commitment_hash.py:80-108`。

【实验】`test_vectors.out`：
- 纯 Python 实现复现了 `seed.rs:72-109` 中固定的 Legacy 和 Salted 向量（`job_key=0x11…`，`hash_a=0xAA…`，`hash_b=0xBB…`，m=192，n=320），也复现了两个 salt 常量及其 CUDA u32 形式。
- `xcheck_diff.out`：对任意 header 和 config，Python 与 zk-pow 实现得到相同的 job_key、b_seed、a_seed（Legacy 和 Salted 都一致）。

### 3.2 与旧版的差异

【代码】
- V1：`b = BLAKE3(job_key‖hash_b)`，`a = BLAKE3(b‖hash_a)`（`zk-pow/src/v1/api/proof_utils.rs:256-260`）。V1 的 MiningConfiguration 尾部 32 字节是全零保留字段（`v1/api/proof.rs:49`，`v1/api/proof_utils.rs:371-372`），没有 MoE。
- V2：在 V1 基础上增加 MoE（`hash_activations` 替代 hash_a），仍用 Legacy 链。
- V3：只在链的入口对两个根加 salt（绑定 m 或 n，并用不同的 A/B 域分离 key）。链本身、电路和线格式都不变（`docs/salted-seed-fork-upgrade-guide.md:15-17`）。

### 3.3 引入 salted seed 的提交及动机

- 提交 `fc5ca65a`“feat(consensus): salted noise-seed hardfork (#280)”，2026-08-11。之后有 `fadd42af`（主网高度从 98900 推迟到 99000）和 `10a0ee18`（把 Merkle 叶子数钉死到声明的维度）。
- 动机（据提交信息意译）：
  1. 旧链直接使用原始 Merkle 根，没有任何东西把根与承诺时的矩阵维度绑定；A 与 B 的根进入链时也没有区分。
  2. 新规则用带 key 的 BLAKE3 先为每个根加 salt：每一侧用各自的域分离 salt，同时承诺维度，每条消息恰好一个 64 字节块。
  3. 这只改变明文路径的种子派生，ZK 电路、电路缓存和线格式都不动。
  4. 由于验证者派生种子的方式变了，这是硬分叉，所以作为新证书版本 V3 发布。V3 的 ProofCommitment 前缀为 3，起到域分离作用，可以防止同一证明被换标签跨版本复用。
  5. 种子派生参与挖矿本身，所以矿工必须升级。
- `10a0ee18` 的补充动机（意译）：V3 让声明的 m、n 成为共识关键值，但此前 `parse_proof` 不检查 Merkle 树的叶子数是否与维度相符；因此增加检查，防止同一棵已承诺的树被按多种维度解读。该提交同时说明：共识路径只验证 ZK 证明，不经过 `parse_proof`。

---

## 4. 噪声矩阵构造（`zk-pow/src/circuit/pearl_noise.rs`）

### 4.1 常量

【代码】`NOISE_RANGE=128`，`IDXS_PER_COL=2`，`UNIFORM_NOISE_RANGE=64`，`ZERO_POINT_TRANSLATION=32`，`RANGE_MASK=63`（`:13-17`）。标签 `SEED_LABEL_A = "A_tensor"‖0^24`，`SEED_LABEL_B = "B_tensor"‖0^24`（`:19-30`）。

### 4.2 随机块

【代码】`get_random_hash(index, seed, key, prepend_index)`（`:46-57`）：

```
msg[0..32]  = 全 0，只有第 prepend_index 个 i32 槽写入 LE32(1 + index)
msg[32..64] = seed（即上面的标签）
H = BLAKE3_keyed(msg, key)          // 单块 keyed 哈希，32 字节输出
```

- 稠密矩阵（E_AL、E_BRᵀ）使用 `prepend_index=0`，即 `msg[0..4]`。
- 稀疏矩阵（E_AR、E_BL）使用 `prepend_index=1`，即 `msg[4..8]`。
- key 是对应一侧的种子：A 侧用 `a_noise_seed`，B 侧用 `b_noise_seed`（`:142-145`）。

### 4.3 四个矩阵的维度与取值

| 矩阵 | 逻辑维度 | 代码中的表示 | 取值 | key / 标签 |
|---|---|---|---|---|
| E_AL | m × r | `generate_uniform_random_matrix`，只生成需要的行（`:61-80`） | [−32, 31] 均匀 | a_seed / "A_tensor" |
| E_AR | r × k | k 个 `[first, second]` 对，即 E_ARᵀ（`:90-116`） | 每列一个 +1、一个 −1，其余为 0 | a_seed / "A_tensor" |
| E_BL | k × r | k 个对（每行一个 +1、一个 −1） | 同上 | b_seed / "B_tensor" |
| E_BR | r × n | 生成需要的 E_BRᵀ 行，即 B 的列 | [−32, 31] 均匀 | b_seed / "B_tensor" |

- 稠密值的映射（`:65-75`）：对行号 `row`（A 的绝对行号，或 B 的绝对列号；MoE 下为全局列号），令 `idx = row·r + c`，`blk = ⌊idx/32⌋`，`pos = idx mod 32`，则
  `E[row][c] = (H_dense(blk)[pos] & 63) − 32`，其中 `H_dense(blk) = get_random_hash(blk, label, seed, 0)`。
  也就是说，矩阵按行主序展平，每 32 个元素对应一次哈希，每个元素用一个字节的低 6 位。
- ±1 对的精确定义（`:94-112`）：对 l ∈ [0,k)，令 `i = ⌊l/8⌋`，`j = l mod 8`，`H = get_random_hash(i, label, seed, 1)`，`U_l = LE_u32(H[4j..4j+4])`，则

  ```
  first_l  = U_l & (r − 1)                                   = U_l mod r
  second_l = first_l XOR (1 + mul_hi_u32(r − 1, U_l))        = first_l XOR (1 + ⌊(r−1)·U_l / 2^32⌋)
  ```

  - 因为 `1 + ⌊(r−1)U/2^32⌋ ∈ [1, r−1]`，且 r 是 2 的幂，所以 `second ≠ first` 且 `second < r`。GPU 注释也给出了同样的论证（`noise_generation_kernel.h:397-402`）。
  - E_AR 的第 l 列是 `e_{first_l} − e_{second_l}`；E_BL 的第 l 行是 `e_{first'_l}ᵀ − e_{second'_l}ᵀ`。
- 参数约束（`sanity_checks.rs:28-37`）：r 是 2 的幂且属于 [32, 1024]，`r % 16 == 0`，`k ≤ 2^16`，`k % 64 == 0`，`16r ≤ k ≤ 4r²`，`k ≥ 1024`。秩惩罚分叉后还要求 r ≥ 128（`sanity_checks.rs:164-166`）。

### 4.4 噪声元素的显式公式

【代码】`matvec_sparse_perm`（`:35-43`）以及 `:149, :153`：

```
noise_A[u][l] = E_AL[u][first_l]  − E_AL[u][second_l]          (u = A 的行号，l ∈ [0,k))
noise_B[l][v] = E_BR[first'_l][v] − E_BR[second'_l][v]         (v = B 的列号；代码中存成 noise.b[v][l])
```

- 取值范围：两个 [−32,31] 的数相减，结果在 [−63, 63]。
- 因为 `E_A = E_AL·E_AR`，所以 `(E_AL·E_AR)[u][l] = Σ_i E_AL[u][i]·E_AR[i][l]`，而 E_AR 的第 l 列只有两个非零元，于是上式就是这两项之差。

### 4.5 GPU 端是否一致

【代码】`noise_generation_kernel.h`：
- 每个线程做一次单块 keyed 压缩，消息后 32 字节是 `"A_tensor"/"B_tensor"`（`:717-718`）。
- `data[0]`（稠密）或 `data[1]`（稀疏）写入 `thread_coord = bid·NumThreads + tid + 1`，即全局 32 字节块序号 + 1（`:381-390, :732-735`）。
- 稀疏下标：`r0 = u & (R−1)`，`r1 = r0 ^ (1 + __umulhi(R−1, u))`（`:404-408`）。
- 稠密取值：`((int8)byte + 128) % 64 − 32`（`:554-561`）。

【理解】对 `byte ∈ [0,255]`，`(int8(byte) + 128) mod 64 = byte mod 64`，所以 GPU 与 Rust 的 `(byte & 63) − 32` 相同。

【实验】本次没有 GPU。Python 参考复刻与 zk-pow 的 `compute_noise` 在 `xcheck_diff.out` 中逐值一致。miner-base 中还有一个独立的 CPU 实现 `noise_generation.py:84-187`，规则相同。

---

## 5. 噪声统计实验（只描述数据）

脚本：`experiments/noise/exp_noise_stats.py`，输出：`exp_noise_stats.out`。参数为 r=128、k=2048（r=128 时允许的最小 k，也是节点 CPU 的默认值）。下面全部是**【实验】**。

1. **±1 对**（30 个种子 × 2048 列）：
   - first 在 128 个值上的计数范围是 415–548（期望 480）；second 是 425–533。
   - `δ = first ⊕ second` 的支撑为 [1,127]，计数范围 434–526。
   - (first//16, δ//16) 的 8×8 列联表独立性 χ² = 32.9（自由度 49）。
   - 在 2048/2048 个样本上验证了 `δ − 1 = ⌊(r−1)·U/2^32⌋`。
   - 每个种子平均约有 **127.1 对完全相同的列**（同一有序对；生日估计值为 129），以及约 **127.5 对互为相反数的列**（`(f,s)` 与 `(s,f)`）。
   - 每个下标被列触及的次数（度）在 14–54 之间，均值 2k/r = 32。
   - 把对看成 r 个顶点上的图，30/30 个种子都是连通图（1 个连通分量）。
2. **数值分布**（种子 0，256 行）：
   - E_AL 在 [−32,31] 上，64 个值的计数为 462–562（期望 512）。
   - E_A 支撑为 [−63,63]，共 127 个值；均值 −0.040，方差 680.0（若两项独立同分布，方差为 682.5）。
   - 与三角分布 `(64−|d|)/4096` 的总变差距离为 0.0070；`P(E_A=0)=0.0155`；两端 ±63 的频率约 0.00025。
3. **秩**（模 2^31−1）：
   - E_AR 每一列的列和都是 0。
   - `rank(E_AL)[256×128] = 128`，`rank(E_AR)[128×512] = 127`，`rank(E_A)[192×512] = 127`。
   - 【理解】由构造可知 `1ᵀ·E_AR = 0`，所以 `rank(E_AR) ≤ r−1`；实测正好是 r−1，对应图连通的情形（rank 等于 r 减连通分量数）。因此 E_A 的秩上限是 r−1 = 127，而不是 r。
4. **相关性**（种子 0，E_A 256 行 × 2048 列）：
   - 随机抽取 4000 对列，在 256 行上计算 Pearson 相关：
     - 共享一个下标且符号相同：均值 +0.501（sd 0.038）
     - 共享一个下标且符号相反：均值 −0.504（sd 0.035）
     - 不共享下标：均值 −0.001（sd 0.063 ≈ 1/√256）
   - 穷举统计：共享至少一个下标的列对占 3.12%（约等于 4/r）。
   - 行与行之间的相关（在 2048 列上计算，300 对）：均值 +0.001，sd 0.090（≈ 1/√r，而不是 1/√k）。
   - 同一行内相邻列的 lag-1 相关：均值 +0.005，sd 0.021。
   - B 侧取值同样在 [−63,63]，`P(0)=0.0162`。用同一 key 时，E_AR 与 E_BL 的对只在 1/2048 个位置上相同（两者的标签不同）。
5. **A+E 是否落在 int8 内**：

| 数据范围 | 解析范围 | 构造的极端情形实测 | 随机 A 实测 |
|---|---|---|---|
| 矿工 int7 [−64,63]（`miner_base/noisy_gemm.py:156-164`，默认 noise_range=128） | [−127, 126] | [−127, 126] | [−125, 126] |
| 验证者接受的 [−64,64]（`verify.rs:113-118`、电路表 `IRANGE7P1 = −64..=64`（`pearl_layout.rs:14`），节点 CPU 挖矿用 [−64,64]（`ffi/mine.rs:15-16`）） | [−127, 127] | [−127, 127] | [−126, 126] |

   两种情况都在 int8 的 [−128, 127] 之内。

---

## 6. 挑战顺序：每一步矿工已知什么、能选什么

【代码】由种子链（第 3 节）可以读出依赖关系：

| 步骤 | 计算 | 依赖 | 矿工此时已知 / 可选 |
|---|---|---|---|
| 0 | job_key | header（prev_block、merkle_root、timestamp、nbits、version）+ config（k、r、rows/cols pattern、MoE） | 可以选：merkle_root（交易集合）、timestamp、配置参数（受 sanity 约束；rank 受惩罚规则约束） |
| 1 | hash_b，然后 b_noise_seed | job_key、Bᵀ 字节、n（V3） | 可以选 B，以及 n |
| 2 | 得到 b_seed 后 | — | **E_BL 与 E_BR 完全确定**，因而 E_B 和 B+E_B 也确定，而 A 尚未承诺 |
| 3 | hash_a，然后 a_noise_seed | b_seed、A 字节、m（V3）、（MoE：routing 与 offsets） | 可以在已知 E_B 的情况下选择 A |
| 4 | E_AL、E_AR，然后 E_A | a_seed | 任何对 A 的改动都会改变 a_seed，进而改变整个 E_A 以及 jackpot key |
| 5 | 计算 (A+E_A)(B+E_B) 的 transcript，再算 jackpot | a_seed 作为 key | t_rows 和 t_cols **不进入任何种子**；任何满足 pattern 合法偏移的 tile 都可以作为“尝试”，事后选出中奖的那个 |

【代码】实际的 GPU 矿工在同一次调用里先后计算 hash_A、hash_B，再算两个种子（`vllm_miner/gemm_operators.py:118-150`）。B 侧的去噪项依赖 E_AR（`:177` 注释）。V3 下 m、n 被 salt 绑定进种子；V1/V2 中 m、n 不进入种子（`v1/api/proof_utils.rs:256-260`）。

【理解】论文的模型（`papers/cupow_2504.09971.txt:310-323`）中，种子写作 O(σ, A, B)，没有规定 A、B 之间的承诺顺序。Pearl 的实现是一条单向链 job_key → B → A：B 侧噪声只依赖 B，A 侧噪声和最终 key 同时依赖 A 和 B。

---

# 组件 3：彩票（jackpot）计算

## 7. 精确算法：Rust 参考实现与 GPU 实现对照

### 7.1 Rust 参考（`zk-pow/src/circuit/chip/jackpot/helper.rs:9-35`）

```
输入：tile 行 s_a[u]（u<h，长度 k）、tile 列 s_b[v]（v<w，即 Bᵀ 的行）、噪声 noise.a[u]、noise.b[v]
acc[h][w] : i32 = 0;  msg[16] : u32 = 0
for ll in r, 2r, …, ⌊k/r⌋·r:                                  // helper.rs:21
    for u, v: for l in [ll−r, ll):
        acc[u][v] += (s_a[u][l] + noise.a[u][l]) · (s_b[v][l] + noise.b[v][l])   // :25-26，i32
    x   = XOR_{u,v} (acc[u][v] as u32)                         // :30，按行主序 fold；XOR 与顺序无关
    tid = (ll/r − 1) mod 16                                     // :31
    msg[tid] = rotl32(msg[tid], 13) XOR x                       // :32，LROT_PER_TILE=13（pearl_program.rs:25）
return msg
hash_jackpot = BLAKE3_keyed(LE(msg[0]) ‖ … ‖ LE(msg[15]), key = a_noise_seed)   // proof_utils.rs:1501-1505；verify.rs:130
```

数学表达：

- 记 tile 行集合 `U = {t_rows + p | p ∈ rows_pattern}`（|U| = h），列集合 `V = {t_cols + q | q ∈ cols_pattern}`（|V| = w），`A' = A + E_A`，`B' = B + E_B`。
- 记 `J = ⌊k/r⌋`。对 j = 1..J：
  - `C^{(j)}_{uv} = Σ_{l < j·r} A'_{ul} B'_{lv}`（int32 前缀和）
  - `x_j = ⊕_{u∈U, v∈V} (C^{(j)}_{uv} mod 2^32)`
- 对每个字 t ∈ [0,16)，令 `Q_t = #{j ≤ J : (j−1) mod 16 = t}`，则
  `W_t = ⊕_{q=0}^{Q_t−1} rotl^{13·(Q_t−1−q)}(x_{t+1+16q})`。
- `H = BLAKE3_keyed(W_0‖…‖W_15, a_seed)`。

【代码】`k ≥ 16r`（`sanity_checks.rs:36`）保证 `J ≥ 16`，所以 16 个字都至少被写入一次。

【理解】int32 不会溢出：`|A'|,|B'| ≤ 127`，`k ≤ 2^16`，所以 `|C| ≤ 127²·2^16 ≈ 1.06e9 < 2^31`。

### 7.2 GPU（`miner/pearl-gemm/csrc/gemm/`）

【代码】
- **调用时机**：
  - 每个 work tile（bM×bN，默认 128×256）开始时，transcript（16 个 u32 寄存器）清零（`pearl_gemm_kernel.h:227-231`）。
  - mainloop 对每个 k_tile（bK=128）先 `preload`，再对 tile 内每个 k_block（宽度为 MMA atom K=32，int8 wgmma 64×N×32）执行 `gemm(...)`，紧接着调用 `hash_accumulator.accumulate(tCrC, k_block)`；k_tile 结束时 `writeback`（`collective_mainloop.hpp:273-311`）。
  - 整个 K 循环结束后，在 epilogue（去噪、store）之后调用 `check_pow_target(transcript, pow_target, pow_key)`；命中时写入 host signal（`pearl_gemm_kernel.h:261-275`）。
  - mainloop 的输入是**加噪后**的 ApEA、BpEB（`pearl_gemm_host.h:69-70`），所以 tCrC 是 (A+E_A)(B+E_B) 的 int32 累加器，在去噪之前就被哈希。
- **TileHashAccumulator**（`pow_utils.hpp:128-200`）：
  - `ReduceEveryK = R/32`，`accums_per_tile = max(1, KBlocksPerTile/ReduceEveryK)`，且必须整除 16（`:130-134`）。
  - `accumulate`：`++count`；当 `count % ReduceEveryK == 0` 且 `count ≤ last_full_k_block (= K/32)` 时，执行 `warpgroup_wait<0>()`，再 `hash = xor_reduction(tCrC)`，然后 `transcript[k_block/ReduceEveryK] = rotl_xor<13>(transcript[...], hash)`（`:166-182`；`last_full_k_block` 定义在 `collective_mainloop.hpp:270-271`）。
  - `writeback` 之后，`m_reduction_count` 前进 `accums_per_tile`（mod 16）（`:186-199`）。
- **xor_reduction**：用 `lop3.b32 0x96`（三输入 XOR）做树形归约，作用对象是**本线程的全部累加器寄存器**（`pow_utils.hpp:19-24, 37-115`）。
- **rotl_xor**：先 `shf.l.wrap.b32 x,x,13`（等于 rotl 13），再与 y 做 XOR（`:27-34`）。
- **check_pow_target**：把 pow_key 的 8 个字作为 CV，对 transcript 做单块压缩，参数为 `COMPRESS_PARAMS_SINGLE_BLOCK_KEYED`，即 counter=0、block_len=64、`KEYED_HASH|CHUNK_START|CHUNK_END|ROOT`（`blake3/blake3.cuh:155-158`），结果等于 `BLAKE3_keyed(64 字节, key)`。然后按 u32 字从高位（索引 7）到低位逐字比较 `hash ≤ target`，即按小端 uint256 比较（`pow_utils.hpp:204-234`）。
- pow_key 来自 `commitment_hash_A_tensor`，即 a_noise_seed（`gemm_operators.py:209`）。pow_target 来自 `adjust_target`，即带秩惩罚的上界（`pearl_gateway/comm/dataclasses.py:212-225`；`gemm_operators.py:116, 185`）。

### 7.3 两端是否完全一致

- 【理解】逐项对照如下：
  - 累加值都是 int32 前缀和，且对加噪后的操作数计算。
  - 快照时刻相同：都在 r 的整数倍处，且不超过 k。GPU 条件是 `count·32 ≤ K`，K 为 64 的倍数，等价于 Rust 的 `ll ≤ k`。
  - XOR 与元素顺序无关，所以寄存器顺序不影响结果。
  - 字下标都是 `(j−1) mod 16`，旋转都是左旋 13 位。
  - 最终哈希都是 `BLAKE3_keyed(msg_le, a_seed)`，比较方式都是小端 uint256 `≤`。
- 【实验】`exp_jackpot_example.out`（b）按 `pow_utils.hpp` 与 `collective_mainloop.hpp` 的逻辑模拟 GPU 调度（preload / accumulate / writeback / m_reduction_count），在 78 组 (K ∈ {1024, 1088, 2048, 4096, 6144, 8192, 65536}，R ∈ 32…1024，bK ∈ {64,128,256}) 上，与 helper.rs 的 `(快照前缀长度, 字下标)` 序列**完全一致**（0 处不匹配）。覆盖了 R<bK、R=bK、R>bK 以及 K 不是 bK 倍数的情况。
- 【理解】我没有发现计算语义上的差异。GPU 对整个 K 做 MMA，k%r 的尾部会进入输出但不会再被快照，这与 Rust 只取到 `⌊k/r⌋·r` 是一致的。本结论基于代码阅读和调度模拟，没有在 GPU 上实测。

### 7.4 小块（h×w）与寄存器片段 / 行列模式的映射

- 【代码】矿工默认值：`tile_size_m=128, tile_size_n=256, tile_size_k=128, noise_rank=128`，`rows_pattern=[0,8]`，`cols_pattern=[0,1,8,9,…,248,249]`（64 个），注释说明这是 128×256 tile 的哈希 tile 模式（`miner/miner-base/src/miner_base/settings.py:9-26`）。每个 warpgroup 负责 64×bN，bM=128 时有 2 个 warpgroup、256 个 MMA 线程（`kernel_traits.hpp:50-53, 74-81`）。
- 【代码】命中后，`write_host_signal_header` 把本线程累加器片段的 `(row, col)` 坐标（来自 `partition_C(identity)`）写给主机（`pow_utils.hpp:253-294`）。主机去重后，加上 tile 原点，得到 A 的行与 B 的列（`pearl_gemm/helpers.py:72-82`）。所以 **jackpot tile 就是单个线程的寄存器片段**。
- 【理解】依据 CuTe 的 SM90 GMMA C 布局：warp w、lane λ 的线程持有：
  - 行 `{16w + ⌊λ/4⌋, 16w + ⌊λ/4⌋ + 8}`（第二个 warpgroup 再加 64）；
  - 列 `{2(λ mod 4) + j + 8c | j∈{0,1}, c<bN/8}`。

  bN=256 时即为 2 行 × 64 列 = 128 个元素，正好对应 `rows_pattern=[0,8]`、`cols_pattern=[0,1,8,9,…]`，偏移为 `t_rows = tile_m + 16w + ⌊λ/4⌋`（+64），`t_cols = tile_n + 2(λ mod 4)`。256 个线程的片段恰好划分整个 128×256 输出 tile，所以每个输出元素恰好属于一次“尝试”。
- 两种模式的差异：

| | 矿工 GPU 默认 | 节点 CPU 挖矿（`zk-pow/bindings/go/src/mine.rs:16-26, 172-181`） |
|---|---|---|
| rows_pattern | [0,8]（h=2，周期 16） | [0,8,64,72]（h=4，周期 128） |
| cols_pattern | [0,1,8,9,…,249]（w=64，周期 256） | [0,1,8,9,32,33,40,41]（w=8，周期 64） |
| h·w | 128 | 32（`sanity_checks.rs:42` 的下限） |
| r | 128（settings；须 ≥ 128） | `MIN_NOISE_RANK=128`（`bindings/go/src/common.rs:28`） |
| k | 由模型层决定（vLLM 中 k ≥ 1024 才走 noisy GEMM） | `16·r = 2048` |
| tile 枚举 | 每个 GEMM tile 的每个线程一个 | `threads_partition` 枚举所有合法偏移，形成不相交的划分（`ffi/mine.rs:485-497`） |
| 数据范围 | int7 [−64,63] | 随机 [−64,64]（`ffi/mine.rs:15-16, 33-40`） |
| 实现 | CUDA，在 MMA 循环中折叠 | 先整体加噪，再逐 tile 执行与 helper.rs 相同的循环（`ffi/mine.rs:87-117`） |

- 【代码】`miner_base/noisy_gemm.py:285-423` 中的 Python 参考（`InnerHasher`，`inner_hash.py:23-65`）使用**连续的 h×w 子块**，而不是 pattern。它与 GPU 和共识规则都不一致。`origin/fp8-scheme` 分支的 `4c7177be` 已将其删除。

### 7.5 V1/V2

【代码】`zk-pow/src/v1/circuit/chip/jackpot/helper.rs` 与当前版本的差别只在 import 路径；`LROT_PER_TILE`、`JACKPOT_SIZE`、jackpot key（`v1/api/verify.rs:115`）都相同。

---

## 8. 中奖条件与难度缩放

【代码】
- `check_jackpot_against_nbits`：`U256::from_little_endian(hash_jackpot) ≤ extract_difficulty_bound(nbits, config)`（`sanity_checks.rs:148-157`）。
  - `bound = target(nbits) · h·w·(k − k%r)`；乘法溢出时饱和到 `U256::MAX`（`:183-219, 229-231`）。
  - `target(nbits) = mantissa · 256^(exp−3)`；mantissa 带符号位或为 0 时取 0（`proof_utils.rs:54-82`）。
- 秩惩罚（主网高度 ≥ 96251，仅当不是 BFNoPoWCheck 时生效）：额外要求 `r ≥ 128` 且 `hash ≤ target · h·w·⌊(k−k%r)/r⌋·128`（`sanity_checks.rs:164-196`；`node/blockchain/validate.go:549-555`）。r=128 时两个界相等（测试见 `sanity_checks.rs:277-284`）。
  - 动机（意译 `51a6e11b` 的提交信息）：jackpot 的界按“单次尝试的工作量”缩放，而噪声秩会除掉这份工作量，因此更大的 rank 能以相同代价换到更宽的界；惩罚项用于抵消这一点。
- V1 的界相同（`v1/api/sanity_checks.rs` 中的 `extract_difficulty_bound`），没有秩惩罚。

【理解】
- 一次“尝试”是一个 h×w tile 在 `k − k%r` 长度上的点积，也就是 `h·w·(k−k%r)` 次 int8 乘加。这些乘加作用于加噪后的操作数，并且就是有用 GEMM 中这部分输出本来就要做的计算。
  - 矿工 GPU 默认：128·k 次乘加（k=4096 时为 524,288）。
  - 节点 CPU：32·2048 = 65,536 次乘加。
- 单次成功概率约为 `(target·F)/2^256`，其中 `F = h·w·(k−k%r)`。因此期望乘加次数约为 `2^256/target`，与 h、w、k 的选择无关。
- 一个 m×n×k 的 GEMM 提供 `m·n/(h·w)` 次尝试，总因子之和约为 `m·n·k`。

---

## 9. 彩票依赖了哪些计算中间值

【代码】结合 7.1 节可知：

- **进入彩票的**：tile 内 h·w 个输出元素在每个 r 的整数倍处的 int32 **前缀和**快照 `C^{(j)}`（j = 1..⌊k/r⌋），按 u32 位模式参与。每个快照先在 tile 内 XOR 成 1 个 u32，再按 j 循环写入 16 个字。
- **通过种子间接进入的**：A、B 的全部字节（Merkle 根）、m、n（V3）、header、config。这些决定噪声和 key。
- **不进入彩票的**：
  - 每个 r 块内部的部分和，以及单个乘积 `A'_{ul}B'_{lv}`；
  - 同一快照内各元素的单独取值（只保留 XOR）；
  - k%r 的尾部；
  - tile 以外的输出元素（对这一次尝试而言）；
  - 去噪后的真实输出 C = A·B；
  - E_A、E_B 本身（只通过 A'、B' 起作用）。

【实验】`exp_jackpot_example.out`（a）：h=2、w=4、k=64、r=32，m=n=16，t_rows=3、t_cols=2，V3 salted。**这组维度不满足 sanity check，仅作演示。**
- tile 行为 {3,11}，列为 {2,3,10,11}；E_AR 的前 8 个对为 `(3,16),(28,5),…`。
- `noise_A[3][0..8] = [8,15,−14,−1,39,−34,35,33]`，等于 `E_AL[3][first] − E_AL[3][second]`。
- ll=32：acc = `[[3141,2217,−16,2936],[2760,440,2503,6960]]`，XOR 得 `0xffffe9e3`，写入 msg[0]。
- ll=64：acc = `[[−4161,−5742,2275,1173],[−6343,7026,3125,1340]]`，XOR 得 `0xffffff19`，写入 msg[1]。其余 14 个字为 0（这个示例里 k/r=2 < 16）。
- `hash_jackpot = 69c51704…bfcb`；难度因子为 512。
- 敏感性检查：
  - 在噪声固定时修改 tile 外的 `A[0][0]`，jackpot 消息不变（但 hash_a 和种子会变）。
  - 在 [0,32) 内同时重排所有操作数的 l 顺序，消息不变。
  - 交换跨越 r 边界的 l=31 和 l=32，消息改变。
- 每个字的折叠次数：k/r = 16 时每字 1 次；k=8192、r=128 时 4 次；k=65536、r=128 时 32 次；k=2048、r=32 时 4 次。

---

## 10. 设计原因：为什么用 XOR、rotl13、16 个字

| 问题 | 找到的说明 |
|---|---|
| 为什么 rotl | 【代码】Python 参考中只有一句注释：“Rotation prevents cancellation when same hash appears multiple times”（`miner_base/noisy_gemm.py:41-42`）。`pow_utils.hpp:14-15` 只说这是用于混合的旋转量，并要求与 Python 一致（`noisy_gemm.py:23-24`）。 |
| 为什么是 13 | 未找到说明（代码注释、git 提交、docs、论文中均无）。 |
| 为什么用 XOR | 未找到选择理由。【代码】`pow_utils.hpp:17-24, 77-79` 只说明用 lop3 三输入 XOR 树“效率最高”，这是实现层面的说明；`jackpot.rs:1-24` 描述的是电路中 XOR 折叠的实现，并非选择理由。论文中没有 XOR 相关内容（全文检索 xor 无命中），论文 Remark 2.1 提出“对每个中间矩阵分别哈希作为一次尝试”（`cupow_2504.09971.txt:358-368`），但没有给出具体的压缩方式。 |
| 为什么是 16 个字 | 【代码】`noisy_gemm.py:16` 注释说明 transcript 为 64 字节 = 16 个 u32，与 `blake3::MSG_BLOCK_SIZE_U32` 一致，即正好一个 BLAKE3 消息块，最终只需一次压缩。`pow_utils.hpp:133-134` 要求 `accums_per_tile` 整除 16。除“一个 BLAKE3 块”之外，未找到其他理由。 |
| 相关提交 | 这些文件都在初始导入提交 `960681d4`（2026-03-17）中一次引入；`git log` 中没有专门讨论 XOR、rotl、transcript 设计的提交。 |

---

## 疑点（留待问题阶段分析）

只列出，不展开，不下结论。

1. XOR 折叠的计算成本：例如 x_j 的最低位等于 tile 内元素和的奇偶性，而元素和等于（行和）与（列和）的内积。这类性质能否让某些位、乃至整个 x_j 用远少于 h·w·r 的乘加算出？
2. 噪声结构：E_AR 的秩是 r−1 而不是 r；E_A 的每一列都是 E_AL 两列之差；约 3.1% 的列对相关系数为 ±0.5；k=2048 时约有 127 对完全相同的列和 127 对互为相反数的列；E_A 的取值服从三角分布而不是均匀分布。这些是否足以保证“tile 边际分布不可预测”？
3. first 与 second 来自同一个 u32（first 取低位，δ 取高位）；每个 r 块中 E_A 的列只取自 r(r−1) 种组合。
4. A 在已知 b_seed（从而已知完整的 E_B）之后才选择；t_rows、t_cols 不进入种子，可以事后选择中奖 tile。
5. 同一个 a_noise_seed 既用作噪声 BLAKE3 的 key（消息带 "A_tensor" 标签），又用作 jackpot 哈希的 key（消息是任意 64 字节 transcript），两者之间没有显式的域分离。
6. 每个字最多可折叠 k/(16r) 次。r=32、k=2^16 时为 128 次，超过 rotl13 的周期 32：同一旋转相位会重复出现。
7. 验证者接受的数据范围是 [−64,64]（129 个值），而矿工 int7 是 [−64,63]；节点 CPU 挖矿用 [−64,64]。
8. ZK 共识路径中，Merkle 树的叶子数（即 m·k）是否被钉死。`10a0ee18` 的检查只在 `parse_proof`（明文路径）里，而 V3 让 m、n 成为共识关键值。
9. `miner_base/noisy_gemm.py` 的 Python 参考使用连续子块而不是 pattern，与 GPU 和共识规则不一致（属于遗留代码）。
10. k%r 的尾部只参与有用输出而不进入彩票。在秩惩罚之前，rank 与“单次尝试工作量”之间的关系见 `51a6e11b`。
