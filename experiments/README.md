# 辅助实验

研究笔记（`docs/notes/`）引用的实验脚本与输出。Python 脚本只用标准库（运行方式 `python3 -I <script>`），`.out`/`_output.txt` 为撰写笔记时的实际输出。

| 目录 | 对应笔记 | 内容 |
|---|---|---|
| `math/` | 01 数学构造 | 去噪恒等式数值验证 |
| `noise/` | 02-03 噪声、种子、彩票 | Pearl 种子链、噪声、彩票的 Python 复刻（`pearl_ref.py`，含纯 Python BLAKE3），测试向量，噪声统计；`rust_xcheck/` 调用 Pearl 自身代码做逐字节交叉校验（需先运行 `scripts/setup-pearl.sh`） |
| `config/` | 04 挖矿配置与门槛 | 模板展开、门槛与秩惩罚数值表、配置对比 |
| `verify/` | 05 验证与零知识证明 | 编译与测试日志、`research_plain_vs_zk.rs`（mine → 原始验证 → ZK 证明 → ZK 验证）及其产物 |
| `node/` | 06 节点与共识 | WTEMA 难度模拟、发行公式验证 |
| `miner/` | 07 挖矿软件 | 有用挖矿覆盖率与单次 GEMM 开销估算（无 GPU 实测） |
| `fp8/` | 08 FP8 新协议 | 量化、噪声线、逐比特累加实验 |

运行 `verify/research_plain_vs_zk.rs`：

```bash
cp experiments/verify/research_plain_vs_zk.rs third_party/pearl/zk-pow/examples/
cd third_party/pearl/zk-pow && cargo +1.88.0 run --release --example research_plain_vs_zk
```

说明：日志中的本机绝对路径已统一替换为 `third_party/pearl`，其余内容未改动。外部网页快照（模型卡等）不随仓库分发，链接见对应笔记。
