# P02 benchmark：有用挖矿覆盖率

问题：Pearl 的推理挖矿插件只在单步 token 数 m≥1024 时挖矿，对话类负载中参与挖矿的推理计算约 0–5%。

问题报告：[`docs/problems/P02-mining-coverage.md`](../../docs/problems/P02-mining-coverage.md)

本 benchmark 两部分都在 CPU 上运行，**不使用 GPU**：

| 部分 | 程序 | 性质 |
|---|---|---|
| 1. 触发规则 × 官方模型各层 | `p02_trigger_table.py` | **确定结果**：直接加载 Pearl 插件自己的 `config.py` 和 `config.yaml`，调用 `should_use_noisy_gemm(m, n, k)`；各层量化方式读取官方 checkpoint 的 `config.json`（固定版本） |
| 2. 负载仿真 | `p02_workload_sim.py` | **CPU 仿真估算**：用 H100/H200 公开规格参数估算每步耗时，按公开负载的长度分布模拟 vLLM 调度，计算被挖的计算量占比。未在 GPU 上实测 |

## 运行

先在仓库根目录执行 `./scripts/setup-pearl.sh`，然后：

```bash
./benchmarks/P02-mining-coverage/run.sh
```

- 首次运行需要联网：从 Hugging Face 下载两个 checkpoint 的 `config.json`（固定版本，脚本会校验 sha256），并在 `third_party/venv-p02` 里安装 PyYAML 6.0.2（Pearl 的 `config.py` 依赖它）。
- 已有 `config.json` 时，可以设置 `CHECKPOINT_DIR=/path/to/dir` 复用。
- 第 1 部分不到 1 秒，第 2 部分约 15 秒。输出写到 `results/run-<时间>-*.txt`（不提交到 git）。

固定的 checkpoint 版本：

| 模型 | Hugging Face 版本 | config.json sha256 |
|---|---|---|
| [pearl-ai/Llama-3.1-8B-Instruct-pearl](https://huggingface.co/pearl-ai/Llama-3.1-8B-Instruct-pearl) | `5dcb348a` | `35085ae9…` |
| [pearl-ai/Llama-3.3-70B-Instruct-pearl](https://huggingface.co/pearl-ai/Llama-3.3-70B-Instruct-pearl) | `6cc401ca` | `7a76c1e9…` |

## 第 1 部分怎么保证用的是 Pearl 自己的规则

- 触发函数不是我们重写的：脚本用 `importlib` 直接加载 `$PEARL_DIR/miner/vllm-miner/src/vllm_miner/config.py`，阈值来自同目录的 `config.yaml`。`config.py` 还导入了 `miner_base.settings.MinerSettings`，它只用于读取运行时开关，与触发规则无关，脚本用一个空类代替。
- 哪些层能挖：按 checkpoint 中每组量化配置判断，条件与 `PearlConfig._is_mining_layer`（`vllm_config.py:67-84`）一致：权重和激活都是 7 bit、激活按 token 动态量化。
- 层的形状按 vLLM `LlamaForCausalLM` 的做法：q/k/v 融合为 `qkv_proj`，gate/up 融合为 `gate_up_proj`；张量并行时列并行层切 n，行并行层切 k。

## 第 2 部分的假设（都是估算）

仿真程序在 CPU 上运行，不使用 GPU。模拟的硬件：8B 模型 + 模拟 H100 SXM（80 GB、3.35 TB/s），70B 模型 + 模拟 H200（141 GB、4.8 TB/s），参数取自公开规格。int8/fp8 稠密峰值按 989.5 TMAC/s 计，bf16 减半；实际效率为假设值：GEMM 效率 0.6，显存带宽效率 0.8。调度按 vLLM V1：每步 8192 token 预算，最多 1024 个序列，分块 prefill，按 KV 容量准入。请求按 Poisson 到达，模拟 300 秒。输入、输出长度服从对数正态分布（σ=1），均值或中位数与公开统计一致：

| 负载 | 输入 | 输出 | 来源 |
|---|---|---|---|
| Alpaca | 均值 19.31 | 均值 58.45 | vLLM 论文 Fig.11（arXiv 2309.06180） |
| LMSYS-Chat-1M | 均值 69.5/轮 | 均值 214.5/轮 | arXiv 2309.11998 Table 1 |
| ShareGPT | 均值 161.31 | 均值 337.99 | vLLM 论文 Fig.11 |
| Azure 对话 | 中位 1020 | 中位 129 | Splitwise（arXiv 2311.18677）§III |
| Azure 代码 | 中位 1500 | 中位 13 | 同上 |
| Mooncake/Kimi | 均值 8,590 | 均值 182 | arXiv 2407.00079 §4 |

另做了两项敏感性分析：每步固定开销 T_FIXED 取 0 或 5 ms；前缀缓存命中率取 50%。

输出列：`mined/lin` = 被挖乘加 ÷ 线性层乘加；`mined/all` = 被挖乘加 ÷（线性层 + 注意力 + lm_head）；`timeMine` = 处于挖矿步的 GPU 时间占比；`R_busy`、`R_wall` = 有用矿工与同卡纯挖矿的份额产出之比（对应后续问题 P03）。

## 文件

| 文件 | 说明 |
|---|---|
| `p02_trigger_table.py` | 第 1 部分 |
| `p02_workload_sim.py` | 第 2 部分（依赖 `experiments/miner/coverage.py` 中的模型形状） |
| `run.sh` | 下载并校验 checkpoint 配置、准备 PyYAML、运行两部分 |
| `results/reference-trigger-table.txt` | 第 1 部分参考输出 |
| `results/reference-workload-sim.txt` | 第 2 部分参考输出（固定随机种子，重跑结果应完全一致） |

## 尚需 GPU 实测的部分

在真实 vLLM 上测量比例：1×H100 跑 8B，1×H200 跑 70B，使用官方 Docker 并加插桩，约 20 GPU 小时、80–120 美元。触发规则、可挖层、"纯解码步不挖"三点由代码决定，不需要实测。
