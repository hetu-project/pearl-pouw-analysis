# P04 benchmark：主网区块的形状普查

问题报告：[`docs/problems/P04-usefulness-unmeasurable.md`](../../docs/problems/P04-usefulness-unmeasurable.md)

问题：协议里无法度量"有用"挖矿的比例。本 benchmark 用唯一能拿到的公开信息，即每个区块证书中的挖矿参数，给出"可能来自官方有用挖矿软件的区块"比例的**上界**。

**只在 CPU 上运行，只用 Python 标准库；只读取公开区块数据，不向网络发送任何交易。**

## 做什么

| 部分 | 程序 | 内容 |
|---|---|---|
| 1. 读取区块参数 | `p04_fetch_certs.py` | 从 PRL Labs 运营的公开区块浏览器（Blockbook）读取主网原始区块，按 Pearl 的格式解析证书中的公开数据：k、r、小块图样、m、n 等。只保存解析出的公开字段，不保存零知识证明 |
| 2. 形状普查 | `p04_census.py` | 逐块检查是否符合官方有用挖矿软件的全部特征（见下），统计比例和参数分布 |

官方有用挖矿软件（vLLM 插件）产生的区块必须**同时**满足：

| 条件 | 依据（Pearl `156ef9a8`） |
|---|---|
| S1 噪声秩 r = 128 | `miner/miner-base/src/miner_base/settings.py`：`noise_rank = 128` |
| S2 小块图样为 2 行 × 64 列的官方图样 | 同上：`rows_pattern`、`cols_pattern` |
| S3 不是 MoE | 主网 91630 高度起禁用 MoE |
| S4 m ≥ 1024 | `vllm_miner/config.yaml`：`min_m = 1024`（见 P02） |
| S5 n 和 k 都等于某个官方模型的层宽度（可除以张量并行度 1/2/4/8） | 4 个官方模型的 `config.json`（固定版本） |

脚本直接从 Pearl 源码的 `MinerSettings` 读取 r 和图样，不手抄。S5 故意放宽：四个模型所有线性层的输入、输出宽度（含融合后的 qkv、gate_up）都算，n 和 k 都可以取其中任意值。放宽只会让上界更高，结论更保守。

不满足任何一条的区块，**一定不是**官方有用挖矿软件产生的；全部满足的区块也只是"可能有用"，因为纯矿工可以模仿这些参数。所以结果是**上界**。

## 运行

先在仓库根目录执行 `./scripts/setup-pearl.sh`，然后：

```bash
./benchmarks/P04-usefulness-unmeasurable/run.sh
```

默认分析仓库中已提交的样本 `results/certs-mainnet.jsonl`，几秒完成。重新从区块浏览器读取样本（约 10–20 分钟，只读）：

```bash
./benchmarks/P04-usefulness-unmeasurable/run.sh fetch
```

- 首次运行需要联网：下载 4 个官方模型的 `config.json`（固定版本，校验 sha256）。
- 样本：V3 证书启用（高度 99000）后，每 50 块取 1 块，加上样本末端的连续 200 块。

## 文件

| 文件 | 说明 |
|---|---|
| `p04_fetch_certs.py` | 第 1 部分：读取并解析区块证书 |
| `p04_census.py` | 第 2 部分：形状普查 |
| `run.sh` | 下载并校验模型配置，运行普查 |
| `results/certs-mainnet.jsonl` | 样本数据：每行一个区块的高度、时间、哈希和证书公开参数 |
| `results/reference-census.txt` | 普查参考输出 |
