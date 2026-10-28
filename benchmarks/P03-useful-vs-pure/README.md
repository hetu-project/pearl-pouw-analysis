# P03 benchmark：有用挖矿与纯挖矿的份额产出对比

问题报告：[`docs/problems/P03-useful-vs-pure.md`](../../docs/problems/P03-useful-vs-pure.md)

本 benchmark 三部分都在 CPU 上运行，**不使用 GPU**，只需要 Python 标准库：

| 部分 | 程序 | 性质 |
|---|---|---|
| 1. 代码事实 | `p03_code_facts.py` | **代码实测**：在 Pearl 源码中按模式查找 15 条事实（有用矿工每次挖矿的额外工作、纯矿工能省掉什么、份额只按乘加数计、硬件只支持 H100/H200），打印文件、行号和原文；找不到任何一条就报错退出 |
| 2. 每次挖矿的额外工作量 | `experiments/miner/gemm_overhead.py` | **按代码计数**：根据 `pearl_gemm_noisy` 的步骤计算额外运算量占主乘法的比例（运算量，不是耗时） |
| 3. 份额产出比 | `p03_share_ratio.py` | **CPU 仿真估算**：沿用 P02 的仿真模型，用 H100/H200 公开规格参数，计算有用矿工与同卡纯矿工的份额产出之比 R。未在 GPU 上实测 |

## 运行

先在仓库根目录执行 `./scripts/setup-pearl.sh`，然后：

```bash
./benchmarks/P03-useful-vs-pure/run.sh
```

约 10 秒。输出写到 `results/run-<时间>-*.txt`（不提交到 git）。

## 第 3 部分的定义与假设

- R = 有用矿工每秒的彩票乘加数 ÷ 同一张卡上纯矿工每秒的彩票乘加数。份额的期望产出与彩票乘加数成正比，因为中奖门槛只按"小块大小 × 点积长度"缩放（`zk-pow/src/api/sanity_checks.rs:191,195`）。
- 有用矿工：沿用 P02 的 vLLM 调度仿真，只在 m≥1024 的步、7 bit 层上挖，每次挖矿付出 `pearl_gemm_noisy` 的额外开销（A、B 加噪，去噪，B 侧显存流量）。
- 纯矿工：同样的内核效率；B 固定，每个区块只加噪一次；不去噪；自选大 n 的形状；剩余开销按 2% 计。
- 硬件为模拟的 H100 SXM（8B 模型）和 H200（70B 模型），参数取自公开规格；GEMM 效率 0.6、显存带宽效率 0.8 为假设值。

## 文件

| 文件 | 说明 |
|---|---|
| `p03_code_facts.py` | 第 1 部分 |
| `p03_share_ratio.py` | 第 3 部分（导入 `../P02-mining-coverage/p02_workload_sim.py`） |
| `run.sh` | 运行三部分 |
| `results/reference-code-facts.txt` | 第 1 部分参考输出（Pearl `156ef9a8`） |
| `results/reference-overhead.txt` | 第 2 部分参考输出 |
| `results/reference-share-ratio.txt` | 第 3 部分参考输出（固定随机种子，重跑结果应完全一致） |

## 尚需 GPU 实测的部分

在 1×H100 上实测同一难度下有用挖矿与纯挖矿的命中数/秒：纯挖矿基线用官方 `noisy_gemm` 内核跑固定 B 的循环，或用第三方挖矿软件；有用挖矿与 P02 的实测环境共用。可选 1×RTX 4090 做跨硬件对比。约 5 GPU 小时、20 美元左右。
