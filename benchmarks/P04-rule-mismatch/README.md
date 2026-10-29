# P04 benchmark：矿工、矿池、共识三方的规则不一致

问题报告：[`docs/problems/P04-rule-mismatch.md`](../../docs/problems/P04-rule-mismatch.md)

**只在 CPU 上运行，不连接任何网络节点。**

## 做什么

| 部分 | 程序 | 内容 | 性质 |
|---|---|---|---|
| 1. 证明走一遍三方检查 | `p04_rule_mismatch.rs`（Rust，作为 Pearl `zk-pow` 的 example 编译） | A：按官方插件参数（r=128、2×64 小块图样）在 k=1024 和 k=2048 各挖一个中奖小块，依次过矿工预检、矿池份额验证、网关零知识证明、节点共识验证。B：用 r=256 挖 6 个中奖小块，对比矿池份额验证和节点秩惩罚规则 | 本机实测 |
| 2. 多卡配置下注定作废的挖矿比例 | `p04_wasted_mining.py` | 用 Pearl 自己的 `should_use_noisy_gemm` 和验证端的 `k ≥ 16r`，代入 4 个官方模型每一层的形状和量化方式，按 TP=1/2/4/8 计算 | 由代码和官方配置算出 |
| 3. 代码事实 | `p04_code_facts.py` | 在 Pearl 源码中按模式查找 14 条代码事实，缺任何一条即报错退出 | 代码事实 |

## 运行

先在仓库根目录执行 `./scripts/setup-pearl.sh`，然后：

```bash
./benchmarks/P04-rule-mismatch/run.sh
```

只跑不需要零知识证明的检查（几秒，内存需求小）：

```bash
./benchmarks/P04-rule-mismatch/run.sh plain
```

- 完整模式约 2 分钟，峰值内存约 11 GB（Apple M4 实测）。
- 首次运行需要联网：下载 4 个官方模型的 `config.json`（固定版本，校验 sha256）和 PyYAML（Pearl 的 `config.py` 需要）。
- 第 1 部分 B 用随机矩阵挖矿，每次运行"通过节点秩惩罚"的个数会波动，期望是一半。

## 文件

| 文件 | 说明 |
|---|---|
| `p04_rule_mismatch.rs` | 第 1 部分 |
| `p04_wasted_mining.py` | 第 2 部分 |
| `p04_code_facts.py` | 第 3 部分 |
| `run.sh` | 下载并校验模型配置，编译并运行三部分 |
| `results/reference-apple-m4-proofs.log` | 第 1 部分参考日志（完整模式） |
| `results/reference-wasted.txt` | 第 2 部分参考输出 |
| `results/reference-code-facts.txt` | 第 3 部分参考输出 |
