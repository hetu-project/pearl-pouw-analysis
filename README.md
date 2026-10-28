# pearl-pouw-analysis

Independent, reproducible analysis of [Pearl](https://github.com/pearl-research-labs/pearl)'s Proof-of-Useful-Work (cuPOW): source-level notes, problem reports, and benchmarks anyone can rerun.

对 Pearl 有用工作量证明（cuPOW）的独立、可复现分析：源码级研究笔记、问题报告，以及任何人都能重跑的 benchmark。

---

## Scope / 研究范围

| Item | Value |
|---|---|
| Target | `pearl-research-labs/pearl` @ [`156ef9a8`](https://github.com/pearl-research-labs/pearl/commit/156ef9a8eb9dc7c74fab8c502598b8b53d4be7b1) (master, int7 scheme, certificate V3) |
| FP8 scheme | branch `origin/fp8-scheme` (not activated on any network at the time of writing) |
| Date | 2026-10 |
| Language | Documents are written in Chinese; code and logs are in English |

This repository does **not** modify Pearl. Benchmarks are small example programs that are copied into a pinned checkout of Pearl and call Pearl's own public functions.

本仓库**不修改** Pearl 的任何代码。每个 benchmark 都是一个示例程序：脚本把它复制到固定版本的 Pearl 源码中，直接调用 Pearl 自己的函数运行。

## Problems / 问题列表

| # | Problem | Severity | Status | Docs | Benchmark |
|---|---|---|---|---|---|
| P01 | Any matrix, including all-zero, can mine valid blocks; "usefulness" is not verified / 任意矩阵（包括全零矩阵）都能挖矿出块，"有用"无法验证 | Critical / 致命 | Confirmed by experiment / 实验证实 | [docs/problems/P01-any-matrix-mines.md](docs/problems/P01-any-matrix-mines.md) | [benchmarks/P01-any-matrix](benchmarks/P01-any-matrix) |
| P02 | Useful-mining coverage is low: only steps with ≥1024 tokens mine; about 0–5% of inference compute on chat workloads / 有用挖矿覆盖率低：单步 ≥1024 个 token 才挖，对话类负载约 0–5% | Serious / 严重 | Rule confirmed by code; ratios estimated, GPU measurement pending / 规则由代码证实，比例为 CPU 仿真估算，待 GPU 实测 | [docs/problems/P02-mining-coverage.md](docs/problems/P02-mining-coverage.md) | [benchmarks/P02-mining-coverage](benchmarks/P02-mining-coverage) |
| P03 | Useful mining earns at most 44–61% of the shares pure mining earns on the same GPU; the official useful miner runs only on H100/H200 / 有用挖矿比纯挖矿更不划算：同一张卡上份额产出最多只有纯挖矿的 44–61%，官方软件只支持 H100/H200 | Serious / 严重 | Cost structure confirmed by code; ratios from CPU simulation, GPU measurement pending / 开销结构由代码证实，比例为 CPU 仿真估算，待 GPU 实测 | [docs/problems/P03-useful-vs-pure.md](docs/problems/P03-useful-vs-pure.md) | [benchmarks/P03-useful-vs-pure](benchmarks/P03-useful-vs-pure) |

Summary of all problems: [docs/problem-analysis.md](docs/problem-analysis.md). More problems will be added one per commit.

问题汇总分析见 [docs/problem-analysis.md](docs/problem-analysis.md)。后续问题会逐个添加，每个问题一个 commit。

## Repository layout / 目录结构

```
docs/
  problem-analysis.md   问题汇总分析（从这里开始读）
  methodology.md        研究方法、问题条目格式、严重程度定义
  pearl-algorithm.md    Pearl 挖矿与验证算法（伪代码 + 参数说明）
  notes/                源码级研究笔记（按组件）
  problems/             问题报告（每个问题一个文件）
  references.md         论文与外部资料链接
benchmarks/
  P01-.../              每个问题的 benchmark：程序、运行脚本、参考结果
experiments/            研究笔记引用的辅助实验脚本与输出
scripts/
  setup-pearl.sh        拉取固定版本的 Pearl 源码
  fetch-papers.sh       下载论文并提取文本（论文不随仓库分发）
```

## Quick start / 快速开始

Requirements / 依赖：
- git, Rust toolchain **1.88.0** (`rustup toolchain install 1.88.0`; Pearl's dependencies do not build with 1.87)
- Python 3.10+ (only for `experiments/`, standard library only)
- About 6 GB of free memory for the ZK proving step

```bash
./scripts/setup-pearl.sh
```

```bash
./benchmarks/P01-any-matrix/run.sh
```

`setup-pearl.sh` clones Pearl into `third_party/pearl` (ignored by git) and checks out the pinned commit. To reuse an existing checkout, set `PEARL_DIR=/path/to/pearl` (it must be at commit `156ef9a8`).

## Methodology / 方法

1. Read Pearl's source code, papers and whitepaper, and write component notes (`docs/notes/`).
2. Collect doubts, then analyse each one to a conclusion: holds / partly holds / ruled out / needs measurement. Ruled-out doubts are recorded with the reason.
3. Every confirmed problem comes with evidence (code locations, experiment logs) and, where possible, a benchmark that reproduces it.

See [docs/methodology.md](docs/methodology.md).

## Disclaimer / 声明

This is independent research, not a security audit. Findings reflect the pinned commit and may not apply to later versions. Corrections are welcome via issues or pull requests.

本仓库是独立研究，不是安全审计。结论针对上述固定版本，可能不适用于之后的版本。欢迎通过 issue 或 pull request 指正。

## License

MIT, see [LICENSE](LICENSE). Benchmark programs call Pearl's code and are derived from its examples; Pearl is licensed under ISC, see [benchmarks/LICENSE-PEARL](benchmarks/LICENSE-PEARL).
