# 组件 6：节点与共识层（node/，btcd 分叉）

> 研究对象：`third_party/pearl` @ `156ef9a8`（2026-09-29）。
> 下文路径默认相对仓库根目录 `pearl/`。
> 标记约定：**[代码]** = 源码中可直接核对的事实（附 文件:行号）；**[历史]** = git 历史事实（附 commit hash）；**[模拟]** = 本人脚本的数值结果；**[理解]** = 我的推断或解读，未经独立验证。
> 模拟脚本：`experiments/node/wtema_sim.py`（输出在 `wtema_sim.out`）、`experiments/node/emission.py`（输出在 `emission.out`）。两者只用标准库，运行方式为 `python3 -I <script>`。

---

## 1. 与 btcd 的差异总览

### 1.1 导入点 [历史]

| commit | 日期 | 内容 |
|---|---|---|
| `064708e6` | 2026-02-02 | `Initial commit`（Or Aharonee），空仓库 |
| `9a4bb1e8` | 2026-02-02 | **`Import btcsuite/btcd`**：merge commit，第二个父提交 `082820ed` 是上游 btcd master 上的 `Merge pull request #2296 from btcsuite/psbt-global-xpub`（2025-01-14）。整个 btcd 历史被并入 `node/` 子目录，新增 3388 个文件，共 195101 行 |
| `92474ba1` / `d93dc52e` | 2026-02-02 | 用同样方式导入 btcwallet 和 lightninglabs/neutrino（后者对应 `spv/`） |
| `960681d4` | 2026-03-17 | **`Initial Pearl implementation`**（Daniel Raskin），提交信息只有标题。仅在 `node/` 下就改了 639 个文件（+23169/−33322）；全仓库改了 1724 个文件 |
| `83d307cc` | 2026-04-27 | `updating genesis`（主网创世时间为 2026-04-27 09:00 UTC，见 `node/chaincfg/genesis.go:102`） |

- 6806 个提交里绝大部分来自 btcd/btcwallet/neutrino 上游。`960681d4..HEAD` 只有 100 个非合并提交，其中 48 个涉及 `node/`。[历史]
- [理解] Pearl 上线前的开发历史被压缩进了 `960681d4` 这一个提交，所以 **初始设计决策（WTEMA 参数、K 值、MaxBlockHeadersPerMsg=100、只允许 Taproot 输出等）在 git 里找不到单独的提交说明**。
- `5b09d844`（2026-09-16，#324）"absorb upstream btcd hardening"：把 btcd v0.26.0–v0.26.2 的改动逐条手工移植进来，两边历史没有合并。[历史]

### 1.2 Pearl 修改过的主要包（`git diff 9a4bb1e8 HEAD -- node`，单位为改动行数 = 增加 + 删除）[历史]

| 包 | 改动行数 / 文件数 | 主要改动（[代码]/[理解]） |
|---|---|---|
| `txscript` | 11659 / 59 | 只允许 Taproot/P2MR/OP_RETURN；新增 OP_CHECKXMSSSIG、P2MR（BIP 360）、`randevictcache.go`，删除 `txscript/data` 测试数据 |
| `blockchain` | 9025 / 84 | WTEMA 难度、新的 subsidy、证书校验（`CheckCertificateRules`）、vsize 取代 weight（`vsize.go`）、证书存储读取 |
| `wire` | 8541 / 84 | 区块头格式（`ProofCommitment` 取代 Nonce）、证书 V1/V2/V3、`MsgHeader`=证书+头、`MaxBlockHeadersPerMsg=100`、`MsgGetHeaders.IncludeCertificates`，删除 MsgAlert |
| `btcutil` | 7162 / 95 | 地址类型只保留 Taproot/P2MR，单位改为 grain（`GrainPerPearl=1e8`，`btcutil/const.go:12`） |
| 根目录 `*.go` | 4709 / 21 | `rpcserver.go`（GBT `requiredcertversion`、getheaders 返回证书）、`server.go` |
| `chaincfg` | 2020 / 15 | 网络参数、分叉高度、checkpoint、创世块 |
| `peer` / `v2transport` / `netsync` | 1954 / 1590 / 1176 | BIP324 v2 传输（`v2transport/` 为新增）、headers-first 带证书、同步节点选择加固 |
| `mempool` / `mining` | 1676 / 1194 | 新增 fee_heap；模板中的占位证书 |
| `zkpow`（新增） | 937 / 5 | Go↔Rust FFI：验证、挖矿 |
| `database` | 1059 / 43 | `FetchBlockSize`、按区间读取证书（`3fe22676`） |

---

## 2. 区块头与证书

### 2.1 BlockHeader [代码]

`node/wire/blockheader.go:21-41`，序列化顺序见 `:184-215`（小端）：

| 字段 | 字节 | 说明 |
|---|---|---|
| Version | 4 | int32 |
| PrevBlock | 32 | |
| MerkleRoot | 32 | |
| Timestamp | 4 | uint32 秒 |
| Bits | 4 | compact target |
| **ProofCommitment** | 32 | 取代 Bitcoin 的 Nonce(4)，承诺证书中确定性的那部分 |

- `MaxBlockHeaderPayload = 12 + 32*3 = 108` 字节（`blockheader.go:17`），Bitcoin 为 80 字节。
- `BlockHash()` = **SHA256d(全部 108 字节)**（`blockheader.go:44-48`），因此包含 ProofCommitment。
- `ProofCommitment = SHA256d(CertificateVersion_LE32 || PublicData)`（`certificate_v1.go:54-59`；V2 见 `certificate_v2.go:66-77`；V3 改成用版本号 3 做哈希，见 `certificate_v3.go:22-24`）。**ProofData（Plonky2 证明本体）不在承诺范围内。**
- 送进 ZK 电路的是 `IncompleteBlockHeader{version, prev_block, merkle_root, timestamp, nbits}`，不含 ProofCommitment（`node/zkpow/verify.go:225-238`；`zk-pow/src/api/proof.rs:10-`）。[理解] 所以工作量绑定的是"不完整区块头"。BlockHash 只是区块标识，不是 PoW 哈希；"哈希 < 目标"的比较发生在电路内部的 jackpot 上。

### 2.2 证书的线格式 [代码]

`MsgCertificate`（`node/wire/certificate.go:132-191`）先写 `Version(4)`，再按版本多态解码：

| 版本 | 布局 | 大小 |
|---|---|---|
| 0 Null | 只有 Version | 4 |
| V1 | Hash(32) + PublicData(164, 定长) + ProofLen(4) + ProofData | 4+200+len(proof)（`certificate_v1.go:62-107`） |
| V2 | Hash(32) + PublicDataLen(4) + PublicData(≤4807) + ProofLen(4) + ProofData | 4+40+N+len（`certificate_v2.go:79-134`） |
| V3 | 与 V2 逐字节相同，只有版本号与 ProofCommitment 的域分隔不同 | `certificate_v3.go:11-24` |

- 上限：`MaxZKProofSize = 60000`，`CertificateMaxSize = 65000`（`certificate.go:72,75`）；解码时套了一层 `io.LimitReader(r, 65000)`（`certificate.go:180`）。
- 证书里的 `Hash` 字段 = 区块头的 BlockHash。验证时要求两者相等（`zkpow/verify.go:59-62, 119-124`）。[理解] 该字段是冗余的自洽检查。
- 密集证明与 MoE 的区分：`IsMoE() = PublicDataLen > 164`（`certificate_v2.go:44-46`），由 `e72ad36c` 从 `!= 164` 改成 `> 164`。
- **实际证书大小 [代码/理解]**：`zk-pow/fixures/v2_stark_proof.bin` 是 59,724 字节，内容为 public_data(164) 加 proof，所以 proof ≈ 59,560 B。MoE 夹具为 59,795 B。V1 夹具 `zk-pow/src/v1/fixures/stark_proof.bin` 也是 59,724 B。整张证书约 4+32+4+164+4+59,560 ≈ **59.8 KB**，离 `MaxZKProofSize=60000` 只剩约 440 B 余量。（夹具是用测试参数生成的；主网实际大小没有实测。）

### 2.3 P2P 传输 [代码]

- `MsgHeader = MsgCertificate + BlockHeader`，**证书写在前面**（`msgheaders.go:18-41`）。`MsgBlock` 由 `MsgHeader` 加交易组成（`msgblock.go:46-49`），所以完整区块也带着证书。区块落盘时证书也一起存储，排在区块序列化的最前面（`blockchain/chainio.go:1416-1435` 按 `min(blockSize, 65000)` 读取前缀再解码）。
- `MsgGetHeaders` 新增 `IncludeCertificates bool`，线上用 1 字节表示（`msggetheaders.go:31-40, 97, 137`）。服务端 `LocateHeaders(..., includeCerts)` 在 chainLock 读锁下逐个从 DB 取证书（`blockchain/chain.go:1807-1871`，`server.go:994`）。
- 一个 HEADERS 批次里不能同时出现有证书和无证书的头（`msgheaders.go:109-112, 170-183`）。
- **`MaxBlockHeadersPerMsg = 100`**（`msgheaders.go:12-16`，注释为 "2000 is too many for our Block Header size"；Bitcoin 为 2000）。`MaxPayloadLength = 9 + (108+65000)*100 ≈ 6.51 MB`（`:154-157`）。全局 `MaxProtocolMessageLength = 8,000,000`（`wire/message.go:33`，`5b09d844` 的说明："8,000,000 bytes rather than Core's 4 MB because Pearl headers carry certificates"）。`MsgBlock.MaxPayloadLength = 4,000,000 + 65,000`（`msgblock.go:312-316`）。
- [理解] 按 59.8 KB/证书计算，满载的 HEADERS 消息约 6.0 MB；单个"头"约 60 KB，是 Bitcoin 的约 750 倍。
- P2P 协议版本 `ProtocolVersion = 2`（`wire/protocol.go:15`）；`bfd06471` 把最低接受版本提到 2。

---

## 3. 区块验证完整流程

### 3.1 入口与调用链 [代码]

```
netsync.handleBlockMsg (manager.go:720)  /  rpc submitblock (rpcserver.go:3702, BFNone)
  └─ BlockChain.ProcessBlock (process.go:154)            ← 在此获取 chainLock.Lock() (:155)
       1. blockExists / orphan 去重                       (:164-177)
       2. checkBlockSanity (validate.go:415)              (:180)
            a. flags |= NetBehaviorFlags (SimNet → BFNoPoWCheck) (:421, process.go:41-46)
            b. 交易数 >0、≤MaxBlockVsize；vsize ≤ 1M      (:424-440)
            c. 第一笔为 coinbase、只有一个 coinbase       (:443-455)
            d. 每笔交易 CheckTransactionSanity（含输出类型限制）(:459-464)
            e. Merkle root、重复交易                      (:473-494)
            f. CheckBlockHeaderSanity (validate.go:356)   (:499) ← 最后执行
                 - cert != nil、4+size ≤ 65000、版本 ∈ {1,2,3} (:360-377)
                 - 时间戳精度为秒、不超过 adjusted time + 5 分钟 (:384-395)
                 - checkProofOfWork (validate.go:309)     (:402)
                     · target>0 且 ≤ powLimit             (:311-323)
                     · cert != nil                        (:325-327)
                     · 若无 BFNoPoWCheck → zkpow.VerifyCertificate ★ZK 验证★ (:331-336)
       3. checkpoint 相关检查（存在 checkpoint 时）       (:192-237)
            - 在最后一个 checkpoint 之前分叉 → ErrForkTooOld (094dc96f 新增)
            - 时间戳早于 checkpoint → 拒绝
            - !fastAdd 时：bits 不得比 calcEasiestDifficulty(checkpoint.bits, Δt) 更容易
       4. 父块不存在 → 放入孤块池并返回                   (:240-250)
       5. maybeAcceptBlock (accept.go:24)                 (:254)
            - 父块未知或已知无效 → 拒绝                  (accept.go:28-35)
            - checkBlockContext (validate.go:742)         (accept.go:42)
                · CheckBlockHeaderContext (validate.go:661)
                    !fastAdd: Bits == calcNextRequiredDifficulty（WTEMA）(:669-684)
                    !fastAdd: Timestamp ≥ prev.Timestamp + 1s（取代 MTP）(:686-696)
                    checkpoint 哈希一致、不在 checkpoint 之前分叉 (:705-725)
                · CheckCertificateRules (validate.go:529)（不受 BFFastAdd 影响）(:756-759)
                    版本 == RequiredCertVersion(height)；DenseOnly 激活后拒绝 MoE；
                    RankPenalty 激活且无 BFNoPoWCheck 时 → zkpow.CheckRankPenalty
                · ValidateWitnessCommitment（不受 BFFastAdd 影响，1a987848）(:764)
                !fastAdd: 交易已 final、BIP34 高度、vsize      (:768-803)
            - dbStoreBlock 落盘                            (accept.go:58-63)
            - index.Add、connectBestChain（UTXO、脚本验证；fastAdd 时跳过）(accept.go:68-76; chain.go:1176-)
            - 暂时释放锁并发送 NTBlockAccepted 通知        (accept.go:81-85)
       6. processOrphans → 对子孤块递归调用 maybeAcceptBlock (:262)
```

### 3.2 ZK 验证在流程中的位置 [代码 + 理解]

- **ZK 验证在 checkBlockSanity 的末尾执行**（`validate.go:497-499`，注释写明"certificate verification is compute intensive, so we ensure it is performed last"）。这个顺序是 `52108b61`（#308，2026-09-06）调整的：在那之前，区块体检查排在 ZK 之后，`handleHeadersMsg` 里的连通性检查也排在 ZK 之后。[历史]
- 但在整个 ProcessBlock 里，**ZK 验证排在以下所有上下文检查之前**：
  - 父块是否存在（孤块判断，`process.go:240-250`）；
  - bits 是否等于 WTEMA 期望值（`validate.go:669-684`）；
  - checkpoint 的 easiest-difficulty 检查（`process.go:218-236`）；
  - 证书版本是否符合当前高度（`CheckCertificateRules`，在 `checkBlockContext` 里）。
- ZK 之前唯一的难度约束是 `target ≤ powLimit`（`validate.go:318-323`）。
- 验证分两层：先在 Go 侧做 hash/commitment 比对（`zkpow/verify.go:119-140`），再进入 Rust 侧。Rust 侧先做反序列化与尺寸检查，然后 `acquire_cache()` 拿全局 Mutex，最后执行 `verify_block_cached_circuits_only`（`zk-pow/bindings/go/src/verify.rs:26-90`）。

### 3.3 BFFastAdd / BFNoPoWCheck [代码]

`process.go:22-36`

- **BFFastAdd**：只由 headers-first 同步设置（`netsync/manager.go:752-760`，当区块等于 headerList 队首时），另外 `cmd/addblock/import.go:134` 也会用。它会跳过：
  - WTEMA bits 等值检查和时间戳单调性检查（`validate.go:669`）；
  - checkpoint 的 easiest-difficulty 检查（`process.go:218`）；
  - 交易 final、BIP34、上下文 vsize 检查（`validate.go:768`）；
  - connectBestChain 里的脚本/输入验证（`chain.go:1177`）。

  它 **不会跳过** ZK 验证（`checkProofOfWork` 不读这个标志）、`CheckCertificateRules`、`ValidateWitnessCommitment`。
- **BFNoPoWCheck**：跳过 `zkpow.VerifyCertificate`（`validate.go:331`）和 rank-penalty 规则（`:549`）。使用场景：
  1. `CheckConnectBlockTemplate`（`validate.go:1123-1157`），即区块模板和 GBT proposal（`rpcserver.go:2120`）；
  2. SimNet，经由 `NetBehaviorFlags`（`process.go:41-46`）。

  它不跳过 powLimit 和 cert 非空检查。

### 3.4 headers-first 同步（`node/netsync/manager.go`）[代码]

- 当前高度低于下一个 checkpoint 时，`startSync` 发 `getheaders(locator, nextCheckpoint, includeCerts=true)` 并进入 headersFirstMode（`manager.go:393-415`）。
- `handleHeadersMsg`（`:982-1163`），对每个头按下面的顺序检查：
  1. 非 headersFirst 模式下收到带证书的头 → 断开连接（`:1005-1011`）；只接受 sync peer 的头（`:1036`）。
  2. 连通性：`prevNode.hash == PrevBlock`（`:1068`），这是便宜的检查，`52108b61` 把它挪到最前。
  3. `CheckCertificateRules`（版本、dense-only、rank penalty）（`:1080-1092`）。
  4. `CheckHeaderSanity` → `CheckBlockHeaderSanity`，**对每个头完整做 ZK 验证**（`:1094-1103`；`validate.go:1168-1178`）。注意这里**不检查 WTEMA bits**，只检查 powLimit。
  5. 高度到达 checkpoint 时比对哈希（`:1109-1126`）。
- 收到 checkpoint 之后，按 headerList 拉取完整区块。区块到达时设置 BFFastAdd 并调用 `ProcessBlock`（`:752-776`）。**由于 BFFastAdd 不跳过 ZK，同一个证书会再被验证一次**（`checkBlockSanity` → `checkProofOfWork`）。[理解] 所以在最后一个 checkpoint（目前是 110000，`params.go:388`）以下，每个块的 ZK 证明被验证 2 次，证书也被下载 2 次（一次随 HEADERS，一次随 BLOCK）。仓库里没找到"已验证证书"缓存（在 `node/` 中 grep `certCache|verifiedCert|proofCache` 无结果）。
- 最后一个 checkpoint 之后切回普通模式（`:914-919`），走 inv/getdata → ProcessBlock(BFNone)，每块验证 1 次。
- 低质量 peer 的路径：inv → 不带证书的 getheaders 探测 → getdata（`5e4366bd`，`5b09d844`）。

---

## 4. 难度调整（WTEMA）

### 4.1 公式与参数 [代码]

`node/blockchain/difficulty.go:84-149`：

```
t        = lastNode.Timestamp − parent.Timestamp          (:116)   ← 上一个块的出块间隔（滞后 1 块）
new      = old + (t − T)·old / halfLife                    (:126-129, big.Int 整除)
new      = min(new, powLimit); new = max(new, 1)           (:132-139)
bits     = BigToCompact(new)                                (:141)   ← 23 位尾数，截断
```

- `T = 194 s`（3 分 14 秒，`params.go:357`）；`WTEMAHalfLife = 168h = 604800 s`（`:358`）。字段注释说明它其实是"time constant"（`params.go:208-210`）。[理解] 线性 WTEMA 在墙钟时间上近似 logistic 方程 `dx/dw = x(1−x)/HL`（x = 实际算力/当前难度对应的算力），所以 168h 是**时间常数**而非半衰期。误差真正的半衰期约为 HL·ln2 ≈ 4.85 天。`difficulty.go:79-82` 的注释里写的"600 秒""2 天"等例子是遗留文字，与实际参数不一致。
- `powLimit = 2^208 − 1`，`PowLimitBits = 0x1b00ffff`（`params.go:29, 355`）。创世块 bits = 0x1b00ffff（`genesis.go:103`），**主网从最低难度起步**。
- 时间戳规则：`ts ≥ prev + 1 s`（`MinTimestampDeltaSeconds = 1`，`validate.go:26, 686-696`），它取代了 MTP 规则；同时要求 `ts ≤ adjusted now + 5 min`（`MaxTimeOffsetMinutes=5`，`params.go:362`；`validate.go:390-395`）。
- 主网 `ReduceMinDifficulty` 设为 true 会直接 panic（`difficulty.go:95-98`）。
- **calcEasiestDifficulty**（`difficulty.go:168-212`）：给定 checkpoint 的 bits 和经过的时间 D，返回理论上最容易的目标 = `target·(87/32)^ceil(D/HL)`。87/32 = 2.71875，大于 e，用来作为 exp(D/HL) 的上界；超过 177 个周期时直接返回 powLimit。只用在 `ProcessBlock` 的 checkpoint 检查（`process.go:227`）。
- [理解] **单块难度最多只能上升 1/(1−(T−1)/HL) ≈ 1.000319 倍**（在 t 取最小值 1 s 时）。所以把难度追高 X 倍至少需要 `ln(X)·HL/(T−1) ≈ 3134·ln X` 个块，与墙钟时间无关。这个下界决定了上线初期的行为。

### 4.2 模拟 [模拟]（`wtema_sim.py`：精确复现整数运算与 compact 截断，并带上 1 s/5 min 时间戳约束，出块间隔服从指数分布）

**(a) 均衡状态下算力突降**

| 场景 | 初始期望出块时间 | 出块时间回到目标 +11% 以内（x≥0.9） | 回到 +1% 以内（x≥0.99） | 头 30 天出块数（计划 13,361） |
|---|---|---|---|---|
| −50% | 388 s | 解析解 15.4 天 / 5,017 块；MC 中位数 15.4 天 / 5,006 块 | 解析解 32.2 天 / 12,196 块；MC 29.6 天（23.9–42.2） | 11,243 |
| −90% | 1,940 s（32 分钟） | 解析解 30.8 天 / 6,850 块；MC 32.8 天 / 7,618 块 | 解析解 47.5 天；MC 中位数 80.8 天（59–117 天，接近均衡时受噪声影响大） | 6,547 |

解析解：`x(w) = 1/(1+(1/x0−1)e^{−w/HL})`；累计块数 = `(HL/T)·ln(1+x0(e^{w/HL}−1))`。

**(b) 上线初期（从 powLimit 起步）**

- 解析拟合（恒定算力，忽略 1 s 下限）：要在 5.9 天内出 40,000 块，需要算力为创世难度对应算力的约 **2.8×10^5 倍**。按此推算，第 5.9 天的出块时间约 110 s。
- MC（恒定算力倍数 x0，包含时间戳约束）：

| H/H_gen | 5.9 天时块数 | 第一周平均 s/块 | 5.9 天时的期望出块时间 | 30 天块数 | 难度追上（x≤1.1）所需天数 |
|---|---|---|---|---|---|
| 1e3 | 22,169 | 26.3 | 112 s | 34,603 | 14.9 |
| 1e4 | 29,379 | 20.0 | 111 s | 41,719 | 16.4 |
| 1e5 | 36,430 | 16.2 | 110 s | 48,674 | 16.2 |
| **3e5** | **39,634** | **14.9** | 105 s | 52,298 | 16.6 |
| 1e6 | 43,440 | 13.7 | 106 s | 56,077 | 15.7 |

- 对照公开数据（首周约 13 s/块，5.9 天 40,000 块）：**x0 ≈ 3×10^5 到 1×10^6 的恒定算力模型就能复现这组数据**。无论 x0 取多少，难度大约都要 **15–17 天** 才追平。[理解] 这是 3134·ln X 块的下界加上 logistic 动力学共同造成的结果。
- 算力渐增模型（几何增长后保持不变）：87 天累计 77k–88k 块。
- **git 中的旁证 [历史/理解]**：
  - `e72ad36c`（2026-07-23）写道 "Postpone DenseOnlyForkHeight ~1h: mainnet 91600->91630"，即 30 块 ≈ 1 小时，**说明当时出块约 120 s/块，仍快于 194 s**；
  - 主网在约第 87 天到达 91,630，而按计划只应到 38,764；
  - `bfd06471`（2026-08-12）称 salted-seed 分叉"has activated"，即 99,000 已达到。
- compact 截断的影响：在均衡状态下平均时间戳间隔为 194.0–194.1 s，可忽略（`scenario_quantization`）。

---

## 5. 发行

### 5.1 公式 [代码]

`node/blockchain/validate.go:160-191`；常量见 `:42-49`。

```
S = 2,100,000,000 × 1e8 grain   (totalSupply, :44)
K = (4·365·86400) / T_sec = 126,144,000 / 194 = 650,226   (:168-169，整除；defaultEmissionConstant 也是 650226，:49)
subsidy(0) = 0
subsidy(h) = floor( S·K / ((h+K)(h−1+K)) )
```

### 5.2 累计公式推导 [理解 + 模拟验证]

`S·K/((h+K)(h−1+K)) = S·K·[1/(h−1+K) − 1/(h+K)]`，求和时逐项相消（telescoping）：

`Σ_{h=1..H} = S·K·(1/K − 1/(H+K)) = S·H/(H+K)`

- `emission.py` 精确逐块求和后，与闭式结果的差值等于每块 floor 舍去部分之和，总是 < H grain。例如 H=K 时，累计 1,049,999,999.9967 PRL，闭式为 1.05e9，差 325,155 grain ≈ 0.003 PRL。[模拟]
- 数值：
  - subsidy(1) = S/(K+1) = **3,229.64 PRL**；subsidy(40,000) = 2,866.17；subsidy(K) = 807.41。
  - 按计划出块时，累计占比：1 年 20%，4 年 50%，12 年 75%，36 年 90%。
  - 曲线是双曲型平滑衰减，没有减半，**总量 21 亿只在 H→∞ 时渐近达到**。
- [理解] 发行完全由高度决定，与时间无关。所以出块过快不会增加总量，但会把发行"提前"到墙钟时间更早的地方。

### 5.3 上线初期的超前发行 [模拟]

| 情形 | 实际高度 | 按计划应有的高度 | 已发行 | 按计划应发行 | 超前量 | 占 S |
|---|---|---|---|---|---|---|
| 公开数据：5.9 天 40,000 块 | 40,000 | 2,627 | 121.70M | 8.45M | **113.25M PRL** | 5.39% |
| MC 恒定算力 3e5，5.9 天 | 39,634 | 2,627 | 120.65M | 8.45M | 112.20M | 5.34% |
| MC 渐增 1e3→1e6/7 天，第 30 天 | 55,902 | 13,360 | 166.25M | 42.28M | 123.97M | 5.90% |
| git 推算：91,630 @ 约 87.0 天 | 91,630 | 38,764 | 259.38M | 118.15M | **141.2M** | 6.73% |
| git 推算：99,000 @ ≤107.1 天（下界） | 99,000 | ≤47,693 | 277.49M | ≤143.51M | ≥134.0M | ≥6.38% |

[理解] 第一周超前约 1.1 亿 PRL（约 5.4% 的总量），之后差距在 1.2–1.4 亿之间。前几个月持续出块偏快，所以差距还在略有扩大。上表的时间点来自提交时间，属于近似值。

---

## 6. 分叉历史

### 6.1 四次高度分叉 [代码/历史]

参数见 `node/chaincfg/params.go:364-371`；规则执行见 `validate.go:529-559` 和 `params.go:330-339`。

| 分叉 | 主网高度 | 类型 | 引入 / 调整的提交 | 规则 | 官方说明的原因 |
|---|---|---|---|---|---|
| **MoE** | 71935 | 硬分叉（V1→V2 证书） | `77fef91e`（#173，2026-06-11，raskin1，"Moe fork changes"）；testnet 改为在第 1 块激活：`f00a8112`（#227） | 达到高度后必须用 V2（`RequiredCertVersion`） | 提交信息只有 "MoE Hard fork changes / set for crossover points"。`docs/moe-fork-upgrade-guide.md` 称其目的是 "adds support for **mining MoE models**"，MoE 挖矿可选，旧的 dense 证明继续有效。**为什么选 71935 这个高度：未找到说明**；该文档中主网高度一栏仍是 "TBD" |
| **DenseOnly** | 91630 | 软分叉（拒绝 MoE 证书） | `670da8c0`（#260，2026-07-23）初定 91600；`e72ad36c`（#261，同日）推迟到 91630；testnet2 修复：`b6d7e5e1`（#276） | V2/V3 证书的 `IsMoE()` 必须为 false | `670da8c0` 只写了"MoE certificates are rejected from the activation height on"。**为什么要禁用 MoE：未找到说明**。推迟的原因写明了：模板占位证书 `PublicDataLen==0` 被 `!= 164` 判成 MoE，导致激活后所有模板都会被拒，所以改成 `> 164` 并推迟约 1 小时 |
| **RankPenalty** | 96251 | 软分叉 | `51a6e11b`（#275，2026-08-05） | noise rank ≥ 128（`MIN_NOISE_RANK`），且 jackpot 必须满足按 128/rank 缩放后的难度界（`zkpow.CheckRankPenalty`） | 有说明。提交原文大意：jackpot 的界按单次尝试的工作量缩放，而 noise rank 会除这个工作量，所以用更大的 rank 挖矿，同等付出就能拿到更宽松的界；分叉要求 rank ≥128 并按 128/rank 缩放界，以抵消这个优势 |
| **SaltedSeed** | 99000 | 硬分叉（V2→V3） | `fc5ca65a`（#280，2026-08-11）初定 98900；`fadd42af`（#282，同日）推迟 100 块 | 必须用 V3。噪声种子先把 Merkle root 与维度一起做带密钥的 BLAKE3（`bound_a/bound_b`） | 有说明。提交原文大意：噪声种子由两个矩阵的 Merkle root 推导，旧做法直接使用原始 root，既没有把 root 与它承诺时的维度绑定，A 和 B 的 root 进入种子链时也没有区分。`docs/salted-seed-fork-upgrade-guide.md` 记录了推导方法，并要求矿工也必须升级。推迟 100 块的原因：**未找到说明**（提交信息只写了"delay … by 100 blocks"） |

其他网络的分叉高度：
- testnet：MoE=1，DenseOnly=28262，RankPenalty=36761，SaltedSeed=38648；
- testnet2：54869 / 80051 / 80627 / 83109（`params.go:571-578, 667-674`）。

`b6d7e5e1` 记录了一次事故：testnet2 的节点一直没有升级，结果在 dense-only 激活后接受了 12 个 MoE 块（78309–80050），导致"canonical chain is unsyncable"，于是把激活高度改到 80051。[历史]

### 6.2 其他与共识/验证相关的重要修复 [历史]

| commit | 日期 | 摘要 |
|---|---|---|
| `6789d6c3` (#9) | 04-27 | netsync：防止伪造 version 高度来劫持同步节点的选择 |
| `e7a5667a` (#29) | 04-28 | IBD 期间可回退到 inbound peer 同步 |
| `094dc96f` (#102) | 05-14 | ProcessBlock 拒绝在 checkpoint 之前分叉的块（新增 `process.go:196-208`） |
| `3be33a59` (#96) / `5e4366bd` (#91) | 05 | 不从高度不高于自己的 peer 同步；inv 驱动下载增加 peer 质量门槛 |
| `79433b6b` (#116) | 05-23 | 降低 UTXO cache flush 与 chainLock 的争用 |
| `15d39538` (#218) | 06-23 | wire：readScriptBuf 校验脚本数量与剩余缓冲区 |
| `1be35b74` (#182) | 06-14 | 修复多个 MoE 挖矿一致性问题 |
| `670da8c0`/`e72ad36c`/`51a6e11b`/`fc5ca65a`/`fadd42af`/`b6d7e5e1` | 07–08 | 见 6.1 |
| **`1a987848` (#266)** | 07-28 | 不论是否 BFFastAdd 都校验 witness commitment（此前 checkpoint 以下的 witness 数据完全没被校验；"Thanks to C3Pool for reporting"） |
| **`10a0ee18` (#281)** | 08-13 | zkpow：把 Merkle 树叶子数固定为所声明维度对应的值（"a miner cannot open one committed tree under several dimension interpretations"）。**这一改动没有对应的分叉高度** |
| **`4a0c24bc` (#299)** | 08-30 | zkpow：拒绝 usize→u32/u16 的窄化转换（维度的未检查强转）。同样没有分叉高度 |
| `bfd06471` (#283) | 08-12 | 最低协议版本提到 2，每 10000 块一个 checkpoint |
| `821c811d` (#298) | 08-31 | HashCache/SigCache 设上限，采用随机淘汰 |
| `e868134a` (#304) / `50d64268` (#294) | 08 | RPC：limited-user 白名单、searchrawtransactions 上限、WS panic 恢复 |
| **`52108b61` (#308)** | 09-06 | 先查头的连通性和便宜的区块体检查，再做 ZK（"a peer could force full recursive STARK verification on headers that don't connect"） |
| `5b09d844` (#324) | 09-16 | 移植上游 btcd 的加固：严格解码、消息长度上限、panic 恢复等；新增 110000 checkpoint |
| `3fe22676` (#334) | 09-24 | 按区间读取证书（之后不再做记录校验和检查） |

---

## 7. 节点验证性能

- **验证在锁内执行 [代码]**：
  - `ProcessBlock` 全程持有 `chainLock.Lock()`（写锁，`process.go:155-156`），ZK 验证就在这把锁里（`:180` → `validate.go:333`）。期间所有需要 chainLock 读锁的操作都会阻塞，例如 `LocateHeaders`（`chain.go:1866`）、`CalcNextRequiredDifficulty`（`difficulty.go:220`）、GBT 模板检查（`validate.go:1124`）。
  - headers-first 的逐头 ZK 验证不持有 chainLock，但它运行在 SyncManager 唯一的 `blockHandler` goroutine 里（`manager.go:1519`），所以会串行阻塞区块、inv、tx 等全部同步消息。
- **Rust 全局 Mutex [代码]**：`CIRCUIT_CACHE: Mutex<CircuitCache>`（`zk-pow/bindings/go/src/common.rs:38-46`）。验证（`verify.rs:71`）和证明（`common.rs:129` `zk_prove`）**共用同一把锁**，V1 另有一把（`:48-52`）。[理解] 所以同一进程内所有 ZK 验证完全串行，无法利用多核并行验证；如果同一进程还在跑内置挖矿证明，验证也要排队。注释说缓存是只读的，但实际用的是 Mutex，不是 RwLock。
- **构建标签 [代码]**：不带 `-tags zkpow` 编译时，`VerifyCertificate` 始终返回错误（`zkpow/zkpow_stub.go:20-22`），节点会拒绝所有块。正式构建都带了 `xmss,zkpow`（`node/Makefile:12`，`Taskfile.yml:41`）。
- **IBD 每块都验证 ZK [代码/理解]**：BFFastAdd 不跳过 ZK（见 3.3），checkpoint 以下每块要验证 2 次（头一次、块一次）。单次验证耗时**未实测**：仓库里没有相关 benchmark，也没有构建 Rust 库。
- **同步数据量估算 [理解]**（按证书约 59.8 KB、高度 110,000 计）：
  - 证书合计约 6.6 GB；
  - headers-first 阶段证书要传两遍，约 13 GB；
  - 区块落盘时也存证书，所以磁盘上光证书就约 6.6 GB；
  - 每天新增约 445 块 × 60 KB ≈ 27 MB/天（只算证书）。
  相比之下，如果 Bitcoin 用同样的 80 B 头，110k 个头只有 8.8 MB。
- **SPV 轻节点**（`spv/blockmanager.go`）[代码]：
  - 总是请求带证书的头（`PushGetHeadersMsg(..., true)`，`:390, 2157, 2410, 2762`）；
  - 对每个头执行 `checkHeaderSanity`（`:2827-2867`）：先做 **CheckBlockHeaderContext（含 WTEMA bits 等值和时间戳单调，checkpoint 检查跳过）**，再做 CheckCertificateRules，最后做 **CheckBlockHeaderSanity（完整 ZK 验证）**；
  - 证书不写入 headerfs，`spv/` 下除 blockmanager 和 chainimport 外没有任何地方引用 Certificate。

  [理解] 轻客户端也要下载约 60 KB/头，并对每个头做完整 STARK 递归验证，"轻"只体现在不下载交易。`spv/chainimport` 导入可信来源的头文件时，按设计跳过证书检查（`chainimport/block_headers_validator.go:1-8, 163-175`）。

---

## 8. 地址与签名（简述）[代码]

- **输出类型的共识限制**：每个 TxOut 的 scriptClass 只能是 `WitnessV1TaprootTy`、`WitnessV2MerkleRootTy`（P2MR, BIP 360）或 `NullDataTy`，否则返回 `ErrScriptMalformed`（`validate.go:221-234`，在 CheckTransactionSanity 中执行，因此 coinbase 也受约束）。GBT 不传地址时，coinbase 用 OP_RETURN 占位（`mining/mining.go:224-243`）。
- **P2MR**：`OP_2 OP_DATA_32 <merkle_root>`（`txscript/standard.go:162-176`），只能走脚本路径，没有 key-path（`txscript/engine.go:525-541`）。
- **OP_CHECKXMSSSIG = 0xde**（`txscript/opcode.go:264, 503, 2020-2105`）：
  - 只能在 tapscript 中使用，否则视为保留 opcode 报错（`:2025-2030`）；
  - 栈布局为 `sig1..sig5 pubkey`，5 段 × 468 B = 2340 B 签名（因为单个栈元素上限 520 B），pubkey 64 B；
  - 长度不对时压入 false，不报错（`:2051-2062`）；
  - 按 sigops 预算扣 1000（`tallyOpCost(1000)`，`:2065-2068`；预算模型同 BIP342：50 + witness 大小）；
  - sighash 固定为 `SigHashDefault`（`:2077-2079`），不支持其他 sighash 类型；
  - 签名无效时直接报 `ErrNullFail`（`:2096-2100`），不会压入 false。
- XMSS 参数：`PKLen=64, SignatureLen=2340, MsgLen=32, MaxSigns=32`（`xmss/xmss_stub.go:8-15`）。[理解] 这是有状态签名，**每个密钥只能签 32 次**，钱包必须维护签名计数（`Sign(msgUID, ...)`）。不带 `xmss` 标签编译时，`Verify` 总返回 false（`xmss_stub.go:27-29`）。

---

## 9. 挖矿 RPC [代码]

- **getblocktemplate**（`rpcserver.go:2141` 起）：
  - 新增字段 `requiredcertversion`（`btcjson/chainsvrresults.go:303-307`），取值为模板占位证书的版本（`rpcserver.go:1752-1754`），而占位证书版本来自 `chainParams.RequiredCertVersion(nextHeight)`（`mining/mining.go:656-667`）；
  - 两份升级指南都要求矿池"read the version from the template, do not hardcode the fork height"；
  - 可变字段为 `time, transactions/add, prevblock, coinbase/append`，capabilities 只有 `proposal`（`rpcserver.go:96-114`）。
- **模板中的占位证书**：`CertificateV{1,2,3}{Hash: msgBlock.BlockHash()}`，PublicDataLen=0、ProofData 为空，header.ProofCommitment 为全零（`mining/mining.go:646-667`）。之后用 BFNoPoWCheck 调用 `CheckConnectBlockTemplate` 自检（`:679`，`validate.go:1123-1157`）。BFNoPoWCheck 下 rank-penalty 也被跳过。[历史] 占位证书曾在 dense-only 分叉时引发问题，见 `e72ad36c`。
- [理解] 矿工/矿池的流程：用模板里的 IncompleteBlockHeader 做 PoUW 并生成 ZK 证明，得到 public data；然后填入 `ProofCommitment = SHA256d(ver||public_data)`，重算 BlockHash，写入证书的 Hash 字段，最后提交。修改 `time` 或交易会改变 IncompleteBlockHeader，需要重新证明。
- **submitblock**（`rpcserver.go:3702-3732`）：反序列化后调用 `SyncMgr.SubmitBlock(block, BFNone)`，与 P2P 区块走完全相同的规则（包括 ZK 验证，持有 chainLock）。
- `zkpow.Mine/MineMoE`（`zkpow/miner.go:60-118`）是节点内置的 CPU 挖矿 FFI。主网 `GenerateSupported: false`（`params.go:361`）。

---

## 疑点（留待问题阶段分析）

> 以下只列出，不展开，不下结论。

1. **验证顺序与 DoS**：ZK 验证（全局串行，并持有 chainLock）排在 WTEMA bits 等值检查、父块存在性（孤块）、checkpoint easiest-difficulty 检查、证书版本/高度规则之前。ZK 之前唯一的难度约束是 `target ≤ powLimit`。
2. **checkpoint 以下 ZK 双重验证**、证书双重下载，且没有验证结果缓存。
3. **证书可塑性**：ProofCommitment 和 BlockHash 只承诺 `version||public_data`，不承诺 ProofData。同一区块哈希可能对应不同的证明字节（取决于 Plonky2 证明是否可塑）；P2P 中继和存储也以 BlockHash 为键。
4. 证书 `Hash` 字段与 BlockHash 冗余；`CertificateV2.PublicData` 是 4807 B 的定长数组，内存占用固定。
5. **证书大小余量**：实测夹具 proof 约 59,560 B，而 `MaxZKProofSize=60000`，余量约 0.7%。
6. **Rust Mutex**：验证与证明共用一把锁，注释说"只读"却没用 RwLock，panic 后直接 `into_inner` 继续使用。
7. **难度调整慢**：时间常数为一周，算力 −90% 后约 1 个月才回到目标 +11% 以内；上线时从 powLimit 起步，单块上调上限约 1.000319 倍。
8. **上线超前发行**：约 1.1–1.4 亿 PRL（5–7% 的 S）在头几个月提前释放；没有 slow-start 或创世难度校准。
9. **时间戳规则**：用 `prev+1s` 加 `≤now+5min` 取代 MTP；WTEMA 只用单块间隔 t（滞后 1 块），矿工对时间戳的操纵空间和对难度的影响有待分析。
10. **分叉频繁**：上线约 3.5 个月内 4 次高度分叉（2 硬 2 软），外加 2 次没有分叉高度的 ZK 验证收紧（`10a0ee18`、`4a0c24bc`），以及 2 次当天推迟高度；testnet2 出现过因节点未升级而不可同步的事故（`b6d7e5e1`）。DenseOnly 禁用 MoE 的原因、MoE 和 SaltedSeed 选择具体高度的原因都未找到说明。
11. BFFastAdd 下 headers-first 不检查 WTEMA bits，完全依赖 checkpoint 哈希；checkpoint 由开发者手工添加（每 10000 块一个）。
12. SPV 轻节点要逐头做完整 STARK 验证并下载约 60 KB/头，对移动端是否可行存疑；chainimport 跳过证书。
13. 二进制若漏掉 `zkpow`/`xmss` 编译标签，行为是全部拒绝（ZK）或全部签名无效（XMSS），属于静默失败模式。
14. XMSS 每个密钥最多 32 次签名，OP_CHECKXMSSSIG 只支持 SigHashDefault，固定成本 1000。
15. Pearl 上线前的开发历史被压缩为单个提交 `960681d4`，初始参数（K、HL、100 头/消息等）没有设计说明。
16. `difficulty.go` 注释中的参数（600 s、2 天）与实际不符，"HalfLife"的命名与"时间常数"的含义不一致。
