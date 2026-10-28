# 组件 5：验证与零知识证明（Pearl @ commit 156ef9a8）

> 资料：`third_party/pearl`（只读）。编译/实验在副本 `third_party/pearl`。
> 标注约定：**【代码】**＝代码事实（附 `文件:行号`，路径相对仓库根）；**【实测】**＝本机实测；**【理解】**＝我的解读/推断。
> 本机：Apple M4 / 16GB / 无 NVIDIA GPU。默认 toolchain 1.87 无法编译（依赖要求 ≥1.88），改用本机已装的 `1.88.0` 工具链编译成功。

---

## 0. 一页速览

【理解】Pearl 有两种"证明"：

| | 原始证明 PlainProof | ZK 证书（ZKProof） |
|---|---|---|
| 谁产生 | 矿工（GPU 挖矿时命中 jackpot 即输出） | 网关/矿池/节点把 PlainProof 喂给 plonky2 证明器 |
| 内容 | 被采样的 h 行 A、w 列 Bᵀ 的**完整明文行**（以 1KiB Merkle 叶子形式）+ Merkle 兄弟节点 | 公共数据 public_data（164 B 起）+ 约 59.5 KB 的 plonky2 证明 |
| 验证者 | 矿池（份额校验）、网关 debug | 全节点共识（`ProcessBlock` 中） |
| 验证方式 | 明文重算：Merkle 根 + 噪声 + h×w×k 小矩阵乘 + blake3 jackpot | STARK（Starky）→ 递归 Plonky2 ×2，最后一层开启 ZK |
| 隐私 | 暴露矩阵行内容 | 不暴露矩阵内容（最外层 ZK） |

---

## 1. 原始证明 PlainProof 的结构与大小

### 1.1 数据结构【代码】

`PlainProof`（`zk-pow/src/ffi/plain_proof.rs:73-84`）：

| 字段 | 类型 | 含义 |
|---|---|---|
| `m` | usize | A 的行数（MoE 下为 token 数） |
| `n` | usize | B 的列数（MoE 下为每专家列数 n_e） |
| `k` | usize | 公共维（common_dim） |
| `noise_rank` | usize | 噪声秩 r |
| `a` | `MatrixMerkleProof` | A（行主序 int8）的多叶 Merkle 证明 + 被采样行号 |
| `bt` | `MatrixMerkleProof` | Bᵀ（即 B 的列，行主序存储）的多叶 Merkle 证明 + 被采样列号 |
| `moe` | `Option<MoEProofParams>` | MoE 专属，dense 时为 None |

`MatrixMerkleProof`（`plain_proof.rs:22-25`）＝ `{ proof: MerkleProof, row_indices: Vec<usize> }`。
`MerkleProof`（`pearl-blake3/src/merkle.rs:189-196`）＝ `{ leaf_data: Vec<[u8;1024]>, leaf_indices: Vec<usize>, total_leaves: usize, root: [u8;32], siblings: Vec<[u8;32]> }`。叶子是 BLAKE3 的 1024 字节 chunk；`compute_leaf_indices_from_rows`（`merkle.rs:160-174`）把每个采样行覆盖到的所有 chunk 都列为叶子，所以**被采样行的整行（k 字节）都以明文出现**。

`MoEProofParams`（`plain_proof.rs:89-106`）：`e, top_k, expert_idx, routing_end_offsets: Vec<u32>, inner_a_rows: Vec<usize>, routing_proof: MerkleProof`。

序列化：bincode、fixint 编码（`deserialize_compat`，`plain_proof.rs:375-386`），并兼容"缺最后一个 Option 标签字节"的旧 V1 格式。

### 1.2 PublicProofParams / PrivateProofParams【代码】

`PublicProofParams`（`zk-pow/src/api/proof.rs:85-105`）：

| 字段 | 线上大小 | 说明 |
|---|---|---|
| `block_header: IncompleteBlockHeader` | 不上线（76 B，验证者自带） | version(4)+prev_block(32)+merkle_root(32)+timestamp(4)+nbits(4)，`proof.rs:10-16`；序列化时两个哈希字节反转 `proof_utils.rs:519-527` |
| `seed_derivation` | 不上线 | 由证书版本决定（Legacy/Salted），`seed.rs:11-17` |
| `mining_config: MiningConfiguration` | 52 B | common_dim(4)+rank(2)+mma_type(2)+rows_pattern(6)+cols_pattern(6)+trailer(32: e(2)|top_k(2)|28×0)，`proof.rs:63-72`、`proof_utils.rs:448-457` |
| `hash_a` | 32 | A 的 BLAKE3 keyed(job_key) Merkle 根 |
| `hash_b` | 32 | Bᵀ 的 Merkle 根 |
| `hash_jackpot` | 32 | blake3(jackpot_msg, key=a_noise_seed)，按小端 U256 与难度比较 |
| `m, n, t_rows, t_cols` | 4×4 | t_rows/t_cols 为 jackpot 行/列模式的起始偏移 |
| `moe: Option<MoEParams>` | 变长 | expert_idx(2)+routing_offsets(4·e)+hash_routing(32)+outer_count(1)+outer_indices(4·cnt)，`proof.rs:111-118`、`proof_utils.rs:1361-1370` |

线上 public_data：dense 恰好 **164 B**（`proof_utils.rs:1143`）；MoE 为 199+4e+4·cnt，上限 `MAX_WIRE_SIZE = 199 + 128·4 + 1024·4 = 4807 B`（`proof_utils.rs:1147-1160`）。

`PrivateProofParams`（`proof.rs:158-164`）：`s_a`（h 条长度 L 的 i8 strip）、`s_b`（w 条）、`s_routing`（64B 路由 strip）、`external_msgs`（Merkle 中与 strip 同 chunk 的其余 64B 消息块）、`external_cvs`（Merkle 兄弟 CV）。其中 strip 长度 **L = dot_product_length = k − k mod r**（`proof_utils.rs:496-500`）。PlainProof 通过 `parse_proof` 转换为这两者（`plain_proof.rs:550-638`）。

`ZKProof`（`proof.rs:135-140`）：`pow_bits[3] | rate_bits[3] | zeta[16] | plonky2_proof`，前导 22 B（`proof_utils.rs:1417, 1448-1470`）。

### 1.3 字节数估算【代码+实测】

bincode 布局可推出（dense）：

```
size ≈ 32 (m,n,k,r)
     + Σ_{X∈{A,Bᵀ}} [ 8 + 1032·leaves_X  + 8 + 8·leaves_X + 8 + 32 + 8 + 32·siblings_X + 8 + 8·rows_X ] + 1
leaves_X ≈ rows_X · ceil(k/1024)   （行与 chunk 对齐时）
```
（每个 1024B 叶子额外 8B 长度前缀来自 `serde_chunk_vec`。）

主项 ≈ (h+w)·k 字节。

| 配置 | h,w | k | 预测主项 | **实测 bincode 大小** |
|---|---|---|---|---|
| 节点默认（`bindings/go/src/mine.rs:16-26`，`node/zkpow/miner.go:25-29`）r=128,k=2048,m=256,n=512 | 4,8 | 2048 | 12·2048≈24.6 KB | **26,193 B**（A: 8 叶/17 兄弟；B: 16 叶/13 兄弟）|
| GPU 矿工默认图样（`miner/miner-base/src/miner_base/settings.py:20-26`，h=2,w=64,h·w=128）r=128,k=4096,m=n=256 | 2,64 | 4096 | 66·4096≈270 KB | **277,633 B**（A: 8 叶/10 兄弟；B: 256 叶/64 兄弟）|

两例都与公式逐字节吻合【实测】（`experiments/verify/exp_node.log`、`exp_gpu4096.log`，产物 `experiments/verify/artifacts/*_plain_proof.bincode`）。
上界【理解】：sanity check 要求 (h+w)·L ≤ 4 MiB（`sanity_checks.rs:25,50-54`），所以叶子明文不超过约 4 MiB+对齐冗余；FFI 文档建议矿池在调用前把输入卡在 ~8 MiB（`bindings/go/src/plain.rs:23-26`）。

---

## 2. 原始证明验证 `verify_plain_proof` 逐步流程

入口：`zk-pow/src/api/verify.rs:101-134`。按执行顺序：

1. **`parse_proof`**（`plain_proof.rs:550-638`）
   1. m/n/k → u32、noise_rank → u16 无损窄化，否则报错（:556-559）；MoE 的 e/top_k → u16（:561-567）。
   2. `check_declared_tree_sizes`：A 树 `total_leaves == padded(m·k)/1024`，Bᵀ 树 `== padded(total_b_cols·k)/1024`，MoE 路由树 `== padded(m·top_k·4)/1024`（:434-471）。注释说明目的是 V3 盐化种子把 m/n 纳入后防止"一棵树多种维度解释"。
   3. A 的每个 row_index < m（:573-575）。
   4. `moe_inner_indices`（:475-547）：dense 直接返回；MoE 下检查 expert_idx<e、offsets 长度=e、B 列号落在本专家区间、`bt.row_indices.len() < n`、e ≤ 1024 等。
   5. `list_to_pattern` 把行/列号列表还原为 `PeriodicPattern`+偏移（:340-357，严格递增、可周期分解、偏移合法）。
   6. 组装 `PublicProofParams`，**hash_a/hash_b 直接取 Merkle 证明里声明的 root**，hash_jackpot 先填 0xFF（:581-600）。
   7. `public.compile()` 生成 BLAKE3 程序（`BlakeProgram::new`，`chip/blake3/program.rs:183-267`）及辅助消息/CV 位置。
   8. 从 Merkle 证明中抽取 strip（只取每行前 L 字节，`plain_proof.rs:283-293`）、辅助消息、辅助 CV（若 CV 不在兄弟节点里则递归从叶子重算，:295-337）；MoE 时抽取路由 strip（:254-281）。
   9. **`evaluate_blake` 重算 hash_a/hash_b 并与 root 比较**（:627-631；`program.rs:269-331`）——这一步即 Merkle 成员证明。
   10. MoE：`verify_moe_routing` 重算路由树根、并逐个核对 routing[expert][inner] == outer_index（:633-635, 643-685）。
2. **`public_params.sanity_check()`**（`verify.rs:111` → `sanity_checks.rs:15-144`）：r ∈ {32..1024 的 2 的幂}且被 16 整除；64 | k；1024 ≤ k ≤ 2¹⁶；16r ≤ k ≤ 4r²；h,w 为 2 的倍数；32 ≤ h·w ≤ 256；L 被 8 整除；m,n ≤ 2²⁴；(h+w)·L ≤ 4 MiB；t_rows+max(rows_pattern) < m、同理列；MoE 一致性（offset 单调、最后一个 = m·top_k、top_k<e 等）。
3. **矩阵元素范围**：所有 strip 值 ∈ [−64, 64]（`verify.rs:114-118`，注释称与电路 IRANGE7P1 一致）。
4. `CompiledPublicParams::from`（`verify.rs:121`）：计算 job_key = blake3(header(76)‖config(52))（`proof_utils.rs:348-353`）与 commitment_hash=(b_noise_seed, a_noise_seed)（`proof_utils.rs:362-372`；V3 先盐化根 `seed.rs:20-59`）。
5. **`compute_noise`**（`verify.rs:124` → `pearl_noise.rs:118-173`）：只为 h 个被采样 A 行、w 个 B 列生成噪声行（低秩 E_L·E_R，E_R 为每行 ±1 的稀疏置换）。
6. **`compute_jackpot`**（`verify.rs:127` → `chip/jackpot/helper.rs:9-35`）：对 (A+E_A)、(B+E_B) 的 h×w 小块做 L 长度的内积，每 r 步把整块 XOR 折叠进 16 个 u32 的 jackpot_msg（循环左移 13）。
7. **jackpot 哈希**：`hash_jackpot = blake3(jackpot_msg, key=a_noise_seed)`（`verify.rs:130`；`proof_utils.rs:1502-1505`）。
8. **难度**：`U256_le(hash_jackpot) ≤ target(nbits_override 或 header.nbits) × (h·w·L)`（`verify.rs:131` → `sanity_checks.rs:148-157, 190-192, 229-231`，溢出时饱和到 U256::MAX）。

**未做的**【代码】：rank-penalty 规则（`check_rank_penalty`，`sanity_checks.rs:164-180`）不在此函数内；FRI 参数/STARK 规模上限不涉及；证书版本资格在 FFI 层（`check_cert_version_eligible`）而非此函数内。

### 2.1 验证者重算量 vs 矿工工作量

【代码】验证者乘加：`compute_jackpot` 三重循环 = **h·w·L** 次 MAC（helper.rs:21-29）。附加开销：噪声 (h+w)·k 次减法 + O((h·r + 2k)/32) 次 blake3（`pearl_noise.rs:142-153`），Merkle 重算 ≈ (h+w)·k 字节 blake3 + 兄弟合并。
【代码】矿工（参考实现 `ffi/mine.rs:87-107`）对整个 m×n 输出、长度 k 做乘加 = **m·n·k**（GPU 实现同理）。

```
比例 ρ = (h·w·L) / (m·n·k) ≈ h·w / (m·n) = 1 / (每次 GEMM 中的 jackpot tile 数)
```

| 场景 | h·w | m×n | ρ |
|---|---|---|---|
| 节点默认（4×8, 256×512, k=2048） | 32 | 131,072 | 2.44×10⁻⁴（=1/4096）|
| 本实验 GPU 图样（2×64, 256×256, k=4096） | 128 | 65,536 | 1.95×10⁻³（=1/512）|
| GPU 图样 + 典型 LLM GEMM（m=n=4096） | 128 | 16.8 M | 7.6×10⁻⁶ |

【实测】`verify_plain_proof` 平均耗时：节点默认 **0.376 ms**；GPU 图样 k=4096 **1.62 ms**（20 次平均，含 parse/Merkle/噪声/乘加/哈希）。

---

## 3. 零知识证明：`pearl_circuit.rs` 结构

### 3.1 三层结构【代码】（`zk-pow/src/circuit/pearl_circuit.rs:1-76` 注释 + 实现）

| 层 | 证明系统 | 哈希/配置 | 作用 |
|---|---|---|---|
| Layer 0 | Starky STARK（Goldilocks，二次扩域 D=2） | Poseidon（`InnerC=PoseidonGoldilocksConfig`，:243）；`stark_config` :255-268 | 证明：strip 是 int7、strip 代入 BLAKE3 Merkle 得 HASH_A/HASH_B、带噪矩阵乘+折叠得 JACKPOT_MSG、blake3(JACKPOT_MSG, key=COMMITMENT_HASH)=HASH_JACKPOT |
| Layer 1 | Plonky2 递归电路 | Poseidon；`build_recursion_config(rate1,pow1,1,zk=false)` | 在电路内验证 Layer0 STARK（`verify_stark_proof_circuit`，:329-337）；暴露 STARK 公共输入 + hash_public_data + zeta + 预处理列在 zeta/gζ 的取值（:298-322） |
| Layer 2 | Plonky2 递归电路，**zero_knowledge=true** | Blake3（`OuterC=Blake3GoldilocksConfig`，:244） | 验证 Layer1 证明；再暴露 Layer1 的全部 PI + Layer1 的 constants_sigmas_cap 与 circuit_digest（:383-401）；最终输出 compact proof（:592-593） |

`circuit_utils.rs:122-141` 的递归配置：135 wires、routed 37/34、2 constants、`max_quotient_degree_factor=8`、cap_height=5、FRI arity bits `ConstantArityBits(3,7)`；STARK 用 `ConstantArityBits(4,7)`（:263）。

### 3.2 STARK 迹（Layer 0）

`pearl_columns` 布局（`circuit/pearl_layout.rs:8-85`）——主要列组：

| 列组 | 宽度 | 作用 |
|---|---|---|
| URANGE8/URANGE13/IRANGE7P1/IRANGE8 表+频次 | 各 2 | 范围查表（0..255、0..8191、−64..64、−128..127），`range_table.rs`（表列步长 0/1 约束） |
| I8U8_TABLE/AUX/FREQ | 3 | i8→u8 转换表，`i8u8.rs` |
| CONTROL_PREP + 解包出的各控制位 + JACKPOT_IDX(8) + MAT_ID_LIMBS(2)+MAT_ID | ~37 | 每行行为控制；`control_and_matid_packed.rs` 把预处理列重新打包核对，保证各控制位为布尔、MAT_ID 一致 |
| OUTER_INDICES_PACKED_PREP + OUTER_INDEX_FIRST/SECOND(2+2) | 5 | MoE 路由：要求路由叶子中的 u32 等于 outer index |
| STARK_ROW_IDX | 1 | 0,1,2,… 单调计数（`monotonic_increment.rs`），作 RAM 索引 |
| MAT_UNPACK(8)、UINT8_DATA(8)、NOISE_PACKED_PREP、NOISE_UNPACK(8)、NOISED_PACKED(2)、MAT_FREQ | 28 | **Input chip**（`input.rs` 注释 1-10）：一行加载一个 8 字节 dword；同一份数据一边经 I8U8 查表成为 BLAKE3 消息字节，一边与噪声相加成为 NOISED_PACKED 供矩阵乘——保证"被哈希的数据"与"被相乘的数据"相同 |
| BLAKE3_MSG_BUFFER(16)、CV_OR_TWEAK_PREP、CV_IN(8)、BLAKE3_MSG(16)、BLAKE3_CV(8)、**BLAKE3_ROUND(1056)**、CV_OUT(8)、CV_OUT_FREQ | ~1115 | **Blake3 chip**（`blake3.rs:1-21`）：每次压缩 8 行（7 轮 + 1 输出行），消息缓冲每行左移一个 dword；CV 来自 JOB_KEY / COMMITMENT_HASH 公共输入或经 logup 从先前行的 CV_OUT 读取 |
| AB_ID_PREP、AB_ID_LIMBS(4)、A_ID、B_ID、A_NOISED(8)/UNPACK(32)、B_NOISED(8)/UNPACK(32)、CUMSUM_TILE(4) | ~91 | **Matmul chip**（`matmul.rs:1-10`，`matmul/constraints.rs:24-110`）：按 A_ID/B_ID 通过 RAM-logup 读取 2×16 的带噪 tile，CUMSUM_TILE(2×2) 累加内积，支持 reset/update 控制 |
| CUMSUM_BUFFER(4)、JACKPOT_MSG(16)、BIT_REG(32) | 52 | **Jackpot chip**（`jackpot.rs:1-24`，`jackpot/constraints.rs`）：LOAD/XOR/SHIFT3/STORE 操作把 tile 结果 XOR 折叠进 16 个 u32，带旋转补偿，对应 `compute_jackpot` |

常量：TILE_D=16、TILE_H=2、JACKPOT_SIZE=16、MIN_STARK_LEN=2¹³、LROT_PER_TILE=13（`pearl_program.rs:21-25`）。行数 = max(blake 指令数×8 + 8, matmul/jackpot 行数)，补到 2 的幂且 ≥ 8192（`pearl_program.rs:46-77`）。约束次数 3（`pearl_stark.rs:214-216`）。

查找（logup）列表（`pearl_stark.rs:128-212`）：UINT8_DATA∈u8；MAT_ID/AB_ID/OUTER 各 limb ∈ u13；**MAT_UNPACK、NOISE_UNPACK ∈ [−64,64]**；A/B_NOISED_UNPACK ∈ i8；I8U8 转换（仅 IS_MSG_MAT 行）；A/B tile 读与 NOISED_PACKED@MAT_ID 的 RAM 一致性（仅 IS_UPDATE_CUMSUM 行）；CV_IN@CV_OR_TWEAK_PREP 与 CV_OUT@STARK_ROW_IDX 的一致性。

顶层约束汇总（`pearl_air.rs:47-121`）：依次调用各 chip，最后在 IS_HASH_A/B/JACKPOT/ROUTING 行把 CV_OUT 约束为对应公共输入。

### 3.3 公共输入【代码】

- STARK 公共输入 `pearl_public`（`pearl_layout.rs:87-96`）：JOB_KEY(8)、COMMITMENT_HASH(8，即 a_noise_seed)、HASH_A(8)、HASH_B(8)、HASH_ROUTING(8)、HASH_JACKPOT(8)，每个 u32 一个 Goldilocks 元素（共 48）。
- Layer1/Layer2 PI 顺序（`pearl_circuit.rs:641-662`）：上述 48 个 → hash_public_data(4) → zeta(2) → 预处理列 evals@ζ(2×5) → evals@gζ(2×5) → [Layer2 额外] Layer1 constants_sigmas_cap(4·2⁵) + circuit_digest(4)。
- **验证者自己计算全部 PI**（`verify.rs:68-94`）：job_key、commitment_hash 从 header+public_data 推出；hash_a/b/jackpot/routing 来自 public_data；hash_public_data = blake3("V2"‖header(76)‖public_data‖pow_bits(3)‖rate_bits(3)) 约化为 4 个 Goldilocks 元素（`proof_utils.rs:1393-1412`、:85-98）；预处理列由验证者重新生成并在 zeta 处求值（`pearl_circuit.rs:631-634`）。证明本身只提供 zeta 与 plonky2 证明字节。

### 3.4 预处理列【代码】（`pearl_stark.rs:228-236`；生成 `pearl_preprocess.rs:100-269`）

| 列 | 内容 |
|---|---|
| CONTROL_PREP | 22 个静态控制位（reset/update cumsum、use_job_key、use_commitment_hash、is_hash_a/b/routing/jackpot、is_cv_in、is_new_blake、is_last_round、3 个消息源位、is_first/second_outer、load/xor/shift3、store 位、dump）+ JACKPOT_IDX 多路选择位 + MAT_ID（:178-216） |
| OUTER_INDICES_PACKED_PREP | MoE：该行路由消息里两个 u32 槽对应的 outer index 打包值（:58-83, 168-174） |
| NOISE_PACKED_PREP | **噪声**：该行 dword 对应的 8 个噪声值以 129 为基打包（:218-225）——即噪声矩阵由验证者从 commitment_hash 推出、作为"程序"的一部分 |
| CV_OR_TWEAK_PREP | 读 CV 的源行号，或 BLAKE3 tweak（counter/flags/block_len）（:227-238） |
| AB_ID_PREP | 本行矩阵乘读取的 A、B tile 的 ID（:240-249） |

【代码/理解】绑定方式：Starky 修改版把在线列与预处理列放进**同一个 Merkle 承诺**（`plonky2/starky/src/prover.rs:87`），并在承诺前把 `hash_public_data` 吸收进 Fiat-Shamir（`prover.rs:103-104`）；Layer1 电路把 STARK 在 ζ、gζ 处对预处理列的开值连到 PI（`starky/src/recursive_verifier.rs:44-69`），验证者用明文列求值后填入 PI。即"预处理列正确性"靠一次随机点检查 + FRI 低度性保证，而非验证者持有预处理承诺。

### 3.5 zeta、pow_bits、rate_bits、SECURITY_BITS【代码】

- **zeta**：STARK 的 DEEP/OOD 挑战点（二次扩域元素，16 B），由 `prove_and_get_zeta` 返回（`pearl_circuit.rs:472-480`），写入 proof_data 前导；Layer1 中把 PI zeta 与电路内 Fiat-Shamir 推出的 zeta `connect`（:340）。验证者额外要求 zeta 不在基域（`verify.rs:76-79`），并用它在本地计算预处理列开值。
- **pow_bits / rate_bits**：三阶段各自的 FRI grinding 位数与码率（blowup=2^rate）。白名单（`pearl_circuit.rs:129-133`）：Stage0 ∈ {(1,18),(2,18)}，Stage1 = (3,18)，Stage2 = (7,22)。证明者默认 pow=[18,18,22]，rate=[1,3,7]（degree_bits ≥15）否则 [2,3,7]（`prove.rs:52-57`）。它们由证明自带，并进入 hash_public_data（因此被 Fiat-Shamir 绑定）。
- **查询轮数** = ⌈(SECURITY_BITS − pow)/rate⌉（`circuit_utils.rs:117-119`）：Stage0 rate1→102、rate2→51；Stage1→34；Stage2→14。
- **SECURITY_BITS = 120**（`pearl_circuit.rs:119-125`）：注释称目标 ~110 位，因 2025 年对 FRI/Reed-Solomon 邻近界的新攻击论文而上调到 120。【理解】这是"猜想型"安全度（rate·queries + pow ≈ 120）。
- 其他上限（`pearl_circuit.rs:145-165`）：expected_num_rows < 2²⁶（RAM 查找索引用两个 13 位 limb）；stark_degree_bits ≤ 19；degree_bits + rate0 ≤ 20。
- ZK 仅在 Layer2 开启（`compile_circuits` :272-273）；注释列出依据 eprint 2024/1037 的四项措施，并声明未做"挑战与查找值不相交"，泄露可忽略（:63-72）。

### 3.6 证明大小【实测】

| 来源 | public_data | proof_data（含 22 B 前导） | plonky2 部分 |
|---|---|---|---|
| fixture `zk-pow/fixures/v2_stark_proof.bin`（dense，k=1216,r=64,8×8） | 164 | 59,560 | 59,538；pow=[18,18,22] rate=[2,3,7] |
| fixture `v2_stark_proof_moe.bin`（e=4） | 231 | 59,560 | 59,538 |
| 本机生成：节点默认（degree_bits=13） | 164 | 59,560 | 59,538 |
| 本机生成：GPU 图样 k=4096（degree_bits=16） | 164 | 59,560 | 59,538 |

【理解】最终大小只由 Layer2 电路决定，与 STARK 规模无关（Layer1 证明在两例中为 162,438 / 171,078 B，但被递归压平）。上限 `MAX_ZK_PROOF_SIZE = MaxZKProofSize = 60000`（`bindings/go/src/common.rs:25`、`node/wire/certificate.go:72`、gateway `zk_certificate.py:59`），余量约 440 B。两次对同一 PlainProof 生成的 proof_data 不同（ZK 随机化），public_data 相同【实测】。

---

## 4. 证书版本、PublicData、ProofCommitment

### 4.1 V1/V2/V3【代码】

| 版本 | 线格式（MsgCertificate: Version(4) 之后） | 证明内容 | 噪声种子 | 验证实现 |
|---|---|---|---|---|
| V1 | BlockHash(32)+PublicData(固定164)+ProofLen(4)+ProofData（`wire/certificate_v1.go:61-101`） | 仅 dense | Legacy | 独立旧代码 `zk-pow/src/v1`（`verify_v1`，`v1/mod.rs:27-52`），独立 V1 电路缓存；hash_public_data 域分隔符 "V1"（`v1/api/proof_utils.rs:970-988`）；mining config 尾部为 reserved |
| V2 | BlockHash(32)+PublicDataLen(4)+PublicData(≤4807)+ProofLen(4)+ProofData（`certificate_v2.go:79-128`） | dense + MoE | Legacy | `verify_zk_proof_v2` → `verify_block_cached_circuits_only`，"V2" 域分隔 |
| V3 | 与 V2 完全相同（内嵌 CertificateV2，`certificate_v3.go:13-24`） | dense + MoE | **Salted**：先 `blake3(root‖dim‖0…, key=SALT_A/B)` 绑定 m/n（`seed.rs:20-59`） | `verify_zk_proof_v3`，同 V2 电路与缓存；ProofCommitment 中版本号写 3 |

版本→种子映射唯一处：`plain_proof.rs:172-177`。按高度强制版本：`chaincfg/params.go:330-338`（主网 MoEFork=71935、DenseOnly=91630（软分叉，禁 MoE）、RankPenalty=96251、SaltedSeed=99000，`params.go:364-371`）。`CertificateV2.IsMoE()` 以 PublicDataLen>164 判断（`certificate_v2.go:44-46`）。

### 4.2 ProofCommitment 覆盖范围【代码】

- `ProofCommitment = SHA256d(CertVersion_LE(4) ‖ PublicData)`（`certificate.go:95-97`、`certificate_v1.go:54-59`、`certificate_v2.go:66-77`、`certificate_v3.go:22-24`；gateway 同式 `zk_certificate.py:150-157`）。
- 区块头 108 B = Version+PrevBlock+MerkleRoot+Timestamp+Bits+**ProofCommitment**（`wire/blockheader.go:17, 21-41`），`BlockHash = SHA256d(header)`（:44-48）。
- **覆盖**：证书版本、整个 public_data（mining config、hash_a/b/jackpot、m/n/t、MoE 尾部）。
- **不覆盖**：ProofData（pow_bits/rate_bits/zeta/plonky2 证明字节）不进入区块哈希；证书中的 BlockHash 字段只是冗余副本，验证时要求等于头哈希（`node/zkpow/verify.go:59-63, 119-124`）。
- 【理解】proof_data 中 pow/rate 通过 hash_public_data 进入 STARK Fiat-Shamir，但不进入区块哈希；同一区块头可配多个不同但都有效的 proof_data（实测两次证明字节不同）。

---

## 5. 原始验证 vs ZK 验证：逐项对照

| # | 检查 | 原始 `verify_plain_proof` | ZK `verify_block*` |
|---|---|---|---|
| 1 | 输入来源 | 维度、行号、Merkle 树全部来自 PlainProof | 维度/配置/哈希来自 public_data（`from_wire_bytes`，`proof_utils.rs:1216-1331`） |
| 2 | MiningConfiguration 规范性 | 由行号列表重建（`list_to_pattern`）；mma_type 固定为 0 | `from_bytes`：mma_type=0、trailer 28 字节为 0、e=0⇒top_k=0、模式规范、round-trip（`proof_utils.rs:459-494`、:105-140） |
| 3 | 维度窄化 | 显式 u32/u16 检查（`plain_proof.rs:556-567`） | 线格式天然 u32 |
| 4 | Merkle 树大小 ↔ 维度 | **显式** `check_declared_tree_sizes` | 隐式：BLAKE3 程序按 padded(m·k) 决定根位置与 ROOT 标志（`program.rs:191-221, 363-364`） |
| 5 | public_params_sanity_check | 有（`verify.rs:111`） | 有（`verify.rs:53`） |
| 6 | t_rows/t_cols 偏移合法且 < m/n | `list_to_pattern` + sanity | `from_wire_bytes` :1304-1313 + sanity |
| 7 | 矩阵元素 ∈ [−64,64] | 显式循环（`verify.rs:114-118`） | IRANGE7P1 查表约束 MAT_UNPACK（`pearl_stark.rs:147-152`） |
| 8 | Merkle 成员（strip→hash_a/b） | 明文 `evaluate_blake`（`plain_proof.rs:627-631`） | Blake3 chip + HASH_A/B PI 约束（`pearl_air.rs:91-103`） |
| 9 | 噪声 | 明文 `compute_noise` | 验证者生成 NOISE_PACKED_PREP，ζ 点检查 |
| 10 | 带噪矩阵乘 + 折叠 | 明文 `compute_jackpot` | Matmul + Jackpot chip |
| 11 | jackpot 哈希 | 验证者**自己算** | 电路证明 blake3(msg,key=COMMITMENT_HASH)=HASH_JACKPOT，值取自 public_data |
| 12 | 难度（含 nbits_override） | `check_jackpot_against_nbits`（`verify.rs:131`） | 同函数（`verify.rs:54`） |
| 13 | MoE 路由 | `moe_inner_indices` + `verify_moe_routing` | HASH_ROUTING PI + OUTER_INDICES_PACKED_PREP + 外部索引约束；`from_wire_bytes` 长度/上限检查 |
| 14 | FRI 参数白名单、degree_bits≤19、行数<2²⁶、degree+rate0≤20 | **无** | 有（`pearl_circuit.rs:145-185`） |
| 15 | zeta ∉ 基域 | 无（不适用） | 有（`verify.rs:76-79`） |
| 16 | header+public_data+pow/rate 绑定进 Fiat-Shamir | 无（不适用） | 有（hash_public_data） |
| 17 | 证书版本资格（MoE 需 ≥V2） | 在 FFI/Python 层（`bindings/go/src/plain.rs:48-51`、py `generate/verify_*_for_cert_version`），不在函数本体 | Go 共识按高度 `CheckCertificateRules`（`validate.go:535-547`） |
| 18 | rank-penalty | 无 | Rust 验证函数内也无；由 Go `CheckCertificateRules` 单独调用 `check_rank_penalty`（`validate.go:549-556`） |
| 19 | PublicData/proof 大小上限 | 无（文档要求调用方 ~8 MiB 截断） | FFI：proof ≤60000、public_data 长度合法（`bindings/go/src/verify.rs:43-58`）；wire 层同（`certificate_v2.go:109-122`） |

一方有、另一方没有：
- **仅原始验证有**：#3 显式窄化、#4 显式树大小绑定（ZK 为隐式）、`moe_inner_indices` 中 `bt.row_indices.len() < n` 等若干 MoE 结构检查（ZK 无同名检查，是否等价未核实）。
- **仅 ZK 有**：#2 配置的严格规范化（trailer 零、round-trip），#14 STARK 规模与 FRI 参数上限，#15 zeta，#16 Fiat-Shamir 绑定，#19 大小上限。
- **两者都不在 Rust 验证函数内**：rank-penalty（#18）、证书版本↔高度（#17）。

---

## 6. 证明由谁、在哪生成；耗时与内存

### 6.1 生成方【代码】

1. **矿工（GPU）** 只产生 PlainProof 并提交给网关（矿工侧 `miner_base/block_submission.py` 等）。
2. **pearl-gateway**：`SubmissionService.submit_plain_proof` 先 `check_cert_version_eligible`（`miner/pearl-gateway/src/pearl_gateway/submission_service.py:46-53`），再 `ProofGenerator.generate_block`（`proof_generator.py:20-57`）：按模板要求的 cert_version 调用 `pearl_mining.generate_proof_for_cert_version`（py-pearl-mining `src/lib.rs:445-456` → `generate_proof_impl` :83，`zk_prove_plain_proof(..., sanity_check=true, ...)`），debug 模式下再 `verify_proof_for_cert_version` 自检；最后 `ZKCertificate.from_pearl_header` 计算并写入 header.proof_commitment（`zk_certificate.py:132-148`）。
3. **矿池**（Go 链接）：`prove_plain_proof_ffi`（`bindings/go/src/plain.rs:83-134`，sanity_check=false，`common.rs:123-141`）。
4. **节点内置 CPU 挖矿**（测试/regtest）：`zkpow.Mine` → Rust `mine`（`bindings/go/src/mine.rs`）一步完成挖矿+证明。
5. Python 侧电路缓存为空起步（`py-pearl-mining/src/lib.rs:73-75`），首次证明需构建电路；`warmup_prove`（`prove.rs:83-112`）可预热。

### 6.2 耗时/内存【实测】（Apple M4，release，`experiments/verify/exp_*.log`）

| 配置 | STARK 行数 / degree_bits | 首次证明（含建电路） | 第二次证明（缓存） | 其中 STARK / L1 / L2 | 进程峰值（`/usr/bin/time -l`） |
|---|---|---|---|---|---|
| 节点默认 4×8,k=2048 | 3,496 → 2¹³ | 6.98 s | 5.48 s | 1.46 / 1.34 / 2.65 s | max RSS 3.70 GB；peak footprint 5.18 GB |
| GPU 图样 2×64,k=4096 | 36,488 → 2¹⁶ | 12.82 s | 14.12 s | 7.37 / 3.48 / 3.05 s | max RSS 4.55 GB；peak footprint 8.97 GB |

（峰值为整个进程：两次证明 + 一个证明者缓存 + 一个验证者缓存。）ZK 验证：新缓存（需编译验证电路）1.36 s / 1.88 s；**仅用缓存验证平均 ~5 ms**（5.09 ms / 4.83 ms）。fixture 测试 `test_proof_fixture`（含编译验证电路）1.09 s、RSS ~1.0 GB。

---

## 7. 节点侧调用

### 7.1 cgo 接口【代码】（`node/zkpow/verify.go`，build tag `zkpow`；无 tag 时 `zkpow_stub.go` 一律返回错误）

- 链接静态库 `zk-pow/bindings/go/target/release/libzk_pow_ffi.a`，头文件 `zk_pow_ffi.h` 由 cbindgen 生成（仓库中不存在，构建时生成）（`verify.go:9-17`）。
- `VerifyCertificate`（:41-52）按类型分派 V3/V2/V1。
- V1（:58-99）：检查证书 Hash==头哈希、头 ProofCommitment==证书计算值、ProofData 非空 → `C.verify_zk_proof_v1`。
- V2/V3 `VerifyZKProofFFI`（:114-188）：同样三项前置检查 + public_data 非空；public_data 复制进 C 结构的定长数组（4807 B），proof 用 `runtime.Pinner` 固定后传指针；按版本调用 `verify_zk_proof_v2/_v3[_with_nbits]`；返回 0=接受、1=拒绝、2=系统错误。
- `blockHeaderToC`（:225-237）：把 PrevBlock/MerkleRoot 从线序反转为显示序，timestamp 取 `Unix()` 截为 u32。
- `CheckRankPenalty`（:193-217）→ Rust `check_rank_penalty`（`bindings/go/src/verify.rs:192-236`），只解析 config 与 hash_jackpot。

### 7.2 Rust 侧与全局锁【代码】（`zk-pow/bindings/go/src/`）

- `verify_zk_proof_inner`（`verify.rs:26-93`）：catch_unwind 包裹；空指针→2；proof 空/超 60000→1；public_data_len 不合法→1；`ZKProof::deserialize`；**`acquire_cache()`** 后调用 `verify_block_cached_circuits_only`（不编译电路，缓存未命中即失败）。
- 全局缓存（`common.rs:38-59`）：`lazy_static` 的 `Mutex<CircuitCache>`，初始化自 `embedded_cache::CACHE_DATA`；中毒时 `into_inner` 恢复。V1 另有一把锁和缓存。验证与证明（`zk_prove`，:123-141）共用同一把 V2 锁——【理解】所有 V2/V3 验证串行化，且节点内置挖矿证明期间会阻塞验证。
- 嵌入式验证者缓存：feature `embedded_cache` 时 `include_bytes!("v2_cache.bin")`（`circuit/embedded_cache.rs:5-10`；Go 绑定强制开启 `bindings/go/Cargo.toml:13`）。该文件不入库（`zk-pow/.gitignore:2`），由 `Taskfile.yml:13-16` 运行 `build_cache` 生成：`fill_verifier_cache` 预编译 degree_bits 13..19 × Stage0{(1,18),(2,18)}（过滤 deg+rate≤20）× Stage1 × Stage2 共 13 组（`pearl_circuit.rs:429-457`）。

### 7.3 在区块处理流程中的位置【代码】

`ProcessBlock`（`blockchain/process.go:154-256`，全程持 `chainLock`）：
1. 重复/孤块检查（:164-177）
2. `checkBlockSanity`（:180 → `validate.go:415-500`）：交易数、vsize、coinbase、Merkle 根、重复交易……**最后**调用 `CheckBlockHeaderSanity`（:498-499，注释"证书验证计算密集，放最后"）：证书存在、≤65000 B、版本在允许集合、时间戳精度/不超前 → `checkProofOfWork`（`validate.go:309-339`）：target>0 且 ≤ powLimit、证书非空 → **`zkpow.VerifyCertificate`**（除非 `BFNoPoWCheck`；SimNet 默认带此标志 `process.go:41-46`）。
3. 检查点相关检查（:191-237）、孤块处理（:240-250）
4. `maybeAcceptBlock` → `checkBlockContext`（`validate.go:742-760`）：`CheckBlockHeaderContext`（头部上下文，含期望难度）→ **`CheckCertificateRules`**（证书版本按高度、dense-only、rank-penalty，`validate.go:529-559`）。

Headers-first 同步（`netsync/manager.go:1079-1102`）顺序相反：先 `CheckCertificateRules`（廉价）再 `CheckHeaderSanity`（含 ZK 验证）。

---

## 8. 矿池份额验证

【代码】`verify_plain_proof_ffi(block_header, pp_bytes, pp_len, cert_version, nbits_override, err)`（`bindings/go/src/plain.rs:28-69`）：
- 空指针/空输入→2；`deserialize_compat`→失败返回 1；`check_cert_version_eligible`（MoE 需 ≥2，未知版本拒绝）；`nbits_override==0` 表示用头部 nbits，否则用矿池份额难度；调用 `verify_plain_proof`；panic→2。
- 文档要求（:18-26）："只做 blake3，不跑 plonky2"；**不修改 header 的 nbits**（proof commitment 依赖它）；**不限制 pp_len**，`deserialize_compat` 无内部字节上限，调用方必须先拒绝过大份额，建议上限约 8 MiB。`prove_plain_proof_ffi` 有同样警告（:78-81）。
- `_with_nbits` 系列是 ZK 证书的变体：`verify_zk_proof_v2_with_nbits` / `v3_with_nbits`（`bindings/go/src/verify.rs:139-179`），用于把 ZK 证书按份额难度校验；Go 侧 `VerifyZKProofFFI(..., nbitsOverride)`。
- Python 等价物：`verify_plain_proof_for_cert_version(cert_version, header, plain_proof, nbits_override=None)`（py-pearl-mining `lib.rs:471-488`），升级指南建议在份额入口使用（`docs/moe-fork-upgrade-guide.md:101-107`）。
- 【代码】份额验证路径不调用 rank-penalty 检查；网关挖矿任务目标使用 `penalized_target_bound`（`pearl_gateway/comm/dataclasses.py:213-222`）。

---

## 9. 编译与实验记录

| 步骤 | 结果 | 记录 |
|---|---|---|
| `cargo test --release --no-run`（默认 1.87） | **失败**：darling 0.23 / serde_with 3.18 需 rustc ≥1.88 | `experiments/verify/build_rust187_fail.log` |
| `cargo +1.88.0 test --release --locked --no-run` | **成功**，56 s | `experiments/verify/build_tests.log` |
| `tests/plain_proof_dimension_wraparound.rs` | 1/1 通过（0.01 s） | `experiments/verify/tests_run.log` |
| lib 测试 `ffi::`（plain_proof 11 项 + mine 5 项含 `test_verify_moe`） | 16/16 通过（0.07 s） | 同上 |
| `circuit::compat_test::test::test_proof_fixture`（dense fixture ZK 验证，含编译验证电路） | 通过，1.09 s，RSS 1.0 GB | 同上 |
| `test_moe_proof_fixture` | 通过，0.96 s，RSS 1.19 GB | 同上 |
| 自写示例 `examples/research_plain_vs_zk.rs`（仅在副本中）：`mine()` → bincode → `verify_plain_proof` ×20 → `zk_prove_plain_proof` ×2 → `verify_block` / `verify_block_cached_circuits_only` ×5 | 两组配置全部成功 | `experiments/verify/exp_node.log`、`exp_gpu4096.log`；产物 `experiments/verify/artifacts/` |

示例源码：`third_party/pearl/zk-pow/examples/research_plain_vs_zk.rs`。运行：`./target/release/examples/research_plain_vs_zk node|gpu4096`。
关键实测：CPU 参考 `mine()` 21 ms / 34 ms（小矩阵、难度 0x1E01FFFF）；PlainProof 26,193 B / 277,633 B；原始验证 0.38 ms / 1.62 ms；ZK 证明 5.5–14 s；ZK 证明大小恒为 59,560 B；ZK 验证（缓存）≈5 ms。未跑：py-pearl-mining（需 maturin 构建 Python 扩展，非必须，Rust 路径已覆盖同一函数）、Go 节点 cgo 构建、完整 84 项 lib 测试与 27 项 MoE 测试。

---

## 10. 疑点（留待问题阶段分析）

1. 原始验证与 ZK 验证检查集合不完全一致（见 §5：树大小绑定显式 vs 隐式；配置规范化、STARK 规模/FRI 白名单只在 ZK 侧；部分 MoE 结构检查只在原始侧）。一个通过份额验证的 PlainProof 是否总能生成可被共识接受的 ZK 证书（例如 degree_bits>19 的配置）？
2. 证明字节（proof_data，含 pow/rate/zeta）不进入区块哈希；同一区块头可有多份不同有效证明。
3. 验证顺序：`ProcessBlock` 中 ZK 验证在上下文难度检查（`CheckBlockHeaderContext`）与 `CheckCertificateRules` 之前执行，且只要求 bits ≤ powLimit；headers-first 路径顺序相反。
4. rank-penalty 不在 Rust 验证函数内，只在 Go 共识中按高度另行检查；份额验证 FFI 不检查。
5. pow_bits/rate_bits 由证明者自选（白名单内），Stage0 两种组合对应不同查询数；安全度基于猜想型 120 位。
6. 预处理列（含噪声）由证明者与轨迹一起承诺，仅靠 ζ/gζ 两点开值与验证者明文求值比对来绑定。
7. hash_jackpot 在 ZK 路径中由 public_data 提供并被电路约束；原始路径由验证者重算——两条路径的"难度对象"来源不同。
8. 全局 `Mutex<CircuitCache>` 串行化所有 V2/V3 验证，并与证明共用；`ProcessBlock` 持 chainLock 期间做 ZK 验证（单块约 5 ms 实测）。
9. `verify_plain_proof_ffi` / `prove_plain_proof_ffi` 不限制输入长度，依赖调用方截断。
10. 证明大小 59,560 B 距 60,000 上限仅约 440 B。
11. Go FFI 证明路径 `sanity_check=false`，Python 路径为 `true`。
12. 文档/注释与代码不一致：`PROOFDATA_PREAMBLE` 注释写 "preprocessed_digest(32)" 而实际是 zeta(16)（`proof_utils.rs:1416-1417`）；`pearl_circuit.rs:43,74` 仍描述 STARK_PREPROCESSED_DIGEST(32 B)，而实际 Layer1 PI 是 hash_public_data+zeta+预处理列开值；`proof.rs:98` 称 hash_jackpot 的 key 为 commitment_hash（实为 a_noise_seed，二者在代码中同义）。
13. `SecondCircuitKey` 的 `Hash` 只取 CommonCircuitData 的少数字段（`circuit_utils.rs:51-61`），Eq 用完整结构。
14. V1 证书使用独立的旧代码树 `zk-pow/src/v1`，与 V2/V3 的检查是否同步未逐项核对。
