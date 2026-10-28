# Pearl PoUW 问题分析

本文档汇总我们对 Pearl 有用工作量证明（PoUW / cuPOW）发现的问题，每个问题一节，逐个补充。每个问题都按同一格式写：问题是什么、原来的算法是什么、我们怎么验证的、benchmark 结果、会影响什么。

---

## 一、Benchmark 项目：pearl-pouw-analysis

### 1. 作用

`pearl-pouw-analysis` 是我们为这次研究建的公开仓库，用来让**任何人都能独立核对我们的结论**：

1. **放证据**：每个问题都写明 Pearl 源码中的具体位置（文件、行号），以及实验的原始日志；
2. **能复现**：每个经实验证实的问题都配一个 benchmark 程序，直接调用 Pearl 自己的函数，在固定版本的 Pearl 上一条命令就能重跑；
3. **不改 Pearl**：benchmark 只作为示例程序（example）复制进 Pearl 源码目录后编译，Pearl 自身代码一行不动，避免"改了代码才复现"的质疑。

### 2. 基本信息

| 项目 | 内容 |
|---|---|
| GitHub 地址 | https://github.com/hetu-project/pearl-pouw-analysis |
| 分支 | `main` |
| 可见性 | 公开 |
| 许可证 | MIT；benchmark 程序调用 Pearl 代码，附 Pearl 的 ISC 许可证（`benchmarks/LICENSE-PEARL`） |
| 研究对象 | [`pearl-research-labs/pearl`](https://github.com/pearl-research-labs/pearl) `master` 分支 @ [`156ef9a8`](https://github.com/pearl-research-labs/pearl/commit/156ef9a8eb9dc7c74fab8c502598b8b53d4be7b1)（主网当前版本：int7 方案、证书 V3、秩惩罚已生效） |
| FP8 新方案 | Pearl 的 `fp8-scheme` 分支（撰写时未在任何网络启用，只做源码分析） |
| 研究时间 | 2026 年 10 月 |
| 依赖 | git；Rust **1.88.0**（1.87 编译不过 Pearl 的依赖）；Python 3.10+（只用于 `experiments/`，仅标准库）；零知识证明一步需要约 6 GB 空闲内存 |
| 参考实验机器 | Apple M4，16 GB 内存，macOS |

### 3. 目录结构

```
docs/
  problem-analysis.md   本文档：问题汇总分析
  pearl-algorithm.md    Pearl 挖矿与验证算法（伪代码 + 每个参数的含义）
  problems/             每个问题的详细报告
  notes/                按组件写的源码研究笔记（01–08）
  methodology.md        研究方法、报告格式、严重程度定义
  references.md         论文与外部资料
benchmarks/
  P01-any-matrix/       每个问题一个目录：程序、运行脚本、参考结果
experiments/            研究笔记引用的辅助实验脚本与输出
scripts/
  setup-pearl.sh        拉取固定版本的 Pearl 源码到 third_party/pearl
  fetch-papers.sh       下载论文（论文不随仓库分发）
```

### 4. 怎么复现

```bash
git clone https://github.com/hetu-project/pearl-pouw-analysis.git
```

```bash
cd pearl-pouw-analysis && ./scripts/setup-pearl.sh
```

```bash
./benchmarks/P01-any-matrix/run.sh
```

`setup-pearl.sh` 会把 Pearl 固定版本拉到 `third_party/pearl`（不提交到 git）。已有 Pearl 源码的话，设置 `PEARL_DIR=/path/to/pearl` 即可复用，但必须是 `156ef9a8` 这个版本，本仓库所有行号都以它为准。

### 5. 严重程度

| 等级 | 含义 |
|---|---|
| 致命 | 协议的核心承诺不成立（如"有用"或"工作量不可伪造"），或可直接导致资金损失、链停止 |
| 严重 | 可被利用获得 10% 以上的不当收益，或显著削弱安全性、去中心化 |
| 中等 | 有实际影响，但可以缓解，或只影响部分场景 |
| 轻微 | 工程瑕疵、文档不一致，没有直接利用路径 |

### 6. 原则

- 只在本机和 Pearl 源码副本上实验，**不在任何真实网络上测试**；
- 涉及正在运行的网络、且有可利用细节的问题，先通过 Pearl 的安全渠道通知，再公开细节。

---

## 二、问题总览

| 编号 | 问题 | 严重程度 | 结论 | 详细报告 | Benchmark |
|---|---|---|---|---|---|
| P01 | 任意矩阵（包括全零矩阵）都能挖矿出块，"有用"无法验证 | 致命 | 成立（实验证实） | [P01](problems/P01-any-matrix-mines.md) | [P01-any-matrix](../benchmarks/P01-any-matrix) |

后续问题逐个加入。

---

## 三、P01：任意矩阵都能挖矿出块，包括全零矩阵

**严重程度：致命** ｜ **结论：成立（本机实验证实）**
详细报告：[`docs/problems/P01-any-matrix-mines.md`](problems/P01-any-matrix-mines.md) ｜ Benchmark：[`benchmarks/P01-any-matrix`](../benchmarks/P01-any-matrix)

### 1. 问题是什么

Pearl 宣称挖矿用的是 AI 推理里的矩阵乘法，所以挖矿是"有用"的。但验证者**根本不检查矩阵是不是来自 AI 推理**：我们用随机数、全零、全是同一个常数的矩阵挖矿，生成的区块都通过了 Pearl 完整的共识验证。"有用"在协议里无法验证，也无法强制。

### 2. 原来的算法是什么

完整算法见 [`pearl-algorithm.md`](pearl-algorithm.md)。

**挖矿**：矿工自选矩阵 A（m×k）、B（k×n），元素为 [−64, 64] 的整数 → 对 A、B 做 Merkle 承诺 → 由区块头和承诺推导种子、生成低秩噪声 → 计算 (A+E_A)×(B+E_B)，每个 h×w 小块算一张彩票 → 彩票哈希低于难度目标即中奖，交出该小块用到的 h 行 A、w 列 B 及 Merkle 路径。

**验证**（`zk-pow/src/api/verify.rs:101-134`）：

| 步骤 | 检查内容 | 代码位置 |
|---|---|---|
| ① | Merkle 路径对得上承诺 | `:108` |
| ② | 参数在允许范围内 | `:111` |
| ③ | **每个元素在 [−64, 64] 之间** | `:114-118` |
| ④ | 由种子重新生成噪声 | `:124` |
| ⑤ | 重算中奖小块的彩票 | `:127` |
| ⑥ | 彩票哈希与难度目标比较 | `:130-131` |

- 对矩阵内容**唯一**的检查是第 ③ 步的数值范围；
- 验证者只重算一个小块，节点默认配置下占全部计算的 1/4096；
- 去噪后的 A×B（真正"有用"的输出）从不提交、不检查；
- 链上只公开 164 字节（`zk-pow/src/api/proof.rs:78-80`），连那 h 行、w 列都看不到；共识层（`node/blockchain/validate.go:529-560`）也不看矩阵内容；
- 附带发现：Pearl 官方的 CPU 挖矿函数 `try_mine_one`（`zk-pow/src/ffi/mine.rs:18-129`）用随机数生成 A、B，中奖即返回，**从不计算完整的 A×B**。

### 3. 我们怎么验证的

- 环境：Pearl `156ef9a8`（未修改）、Apple M4 16 GB、Rust 1.88.0；
- 参数：节点默认配置 m=256、n=512、k=2048、r=128、小块 4×8；V3 加盐种子；低难度 nbits=0x1E01FFFF（只为几秒出结果，不影响结论）；
- 4 组矩阵：随机数 [−64, 64]、全零、全 7、全 −64；
- 每组走完区块上链的全部 5 道检查：`mine()` → `verify_plain_proof`（矿池验份额）→ `check_cert_version_eligible(3)` → `zk_prove_plain_proof` + `verify_block`（节点共识路径）→ `check_rank_penalty`。

### 4. Benchmark 结果

| 矩阵 | 挖矿耗时 | 原始验证 | 版本检查 | zk 证明生成 | zk 证明大小 | 共识验证 | 秩惩罚 |
|---|---|---|---|---|---|---|---|
| 随机数 | 38.9 ms | ✅ | ✅ | 5.68 s | 59,560 B | ✅ | ✅ |
| **全零** | 20.8 ms | ✅ | ✅ | 4.91 s | 59,560 B | ✅ | ✅ |
| 全部是 7 | 26.0 ms | ✅ | ✅ | 4.43 s | 59,560 B | ✅ | ✅ |
| 全部是 −64 | 29.1 ms | ✅ | ✅ | 4.38 s | 59,560 B | ✅ | ✅ |

- 4 组共 21.3 秒，内存峰值约 4.2 GB；
- 结果是确定的：重跑时 Merkle 根、中奖坐标与参考日志一致；
- 参考日志：[`benchmarks/P01-any-matrix/results/reference-apple-m4.log`](../benchmarks/P01-any-matrix/results/reference-apple-m4.log)；
- 与 Basu 论文的区别：该论文只验证到原始证明和矿池份额；全零、常数矩阵走完零知识证明和节点共识的完整流程，以及官方挖矿函数不计算乘积，据我们所知此前未见公开报道。

### 5. 会影响什么

1. **"有用挖矿"只是矿工自愿**：用随机数或全零矩阵挖矿，出块权和真正做 AI 推理的矿工完全一样；
2. **现实中大部分算力就是这么挖的**：Basu 论文发现最大矿池的主流挖矿软件没有任何推理代码；全网约 96% 是消费级显卡加第三方纯挖矿软件；
3. **核心卖点不成立**："挖矿的同时完成 AI 推理"无法被任何人验证，链上也分辨不出谁真的做了推理；
4. **加大抽样也解决不了**：即使验证整个矩阵，验证的也只是"带噪声的乘法算没算对"，而不是"这个乘法是否在为真实用户服务"。"有用"这个属性根本不在验证对象里。
