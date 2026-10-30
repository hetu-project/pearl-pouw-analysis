# P05 benchmark：官方"挖矿不影响结果"的测试为什么不成立，以及正确的测法

问题报告：[`docs/problems/P05-result-accuracy.md`](../../docs/problems/P05-result-accuracy.md)

分两部分：

| 部分 | 程序 | 能不能跑 |
|---|---|---|
| CPU：官方测试和公开数据能证明什么 | `p05_code_facts.py`（由 `run.sh` 调用） | 任何机器都能跑，几秒 |
| GPU：真正挖矿时，结果和不挖矿时比有没有变 | `gpu/p05_gpu_check.py` | **只能在 H100/H200 上跑，我们还没跑过** |

## CPU 部分

```bash
./benchmarks/P05-result-accuracy/run.sh
```

首次运行要联网下载固定版本的 8B 模型卡。参考输出：`results/reference-code-facts.txt`。

## GPU 部分（需要 H100/H200，尚未运行）

Pearl 的挖矿计算库只为 Hopper 架构编译，没有 H100/H200 就跑不了。脚本写好了，但**没有在 GPU 上运行过**，第一次跑时可能需要按实际环境调整。

做法和官方测试的思路一样，只是保证挖矿真的发生：

- 同一个模型、同一批输入、贪心解码，两次运行之间只改 `MINER_NO_MINING`；
- 一次性提交全部 1319 道 GSM8K 测试题，`max_num_seqs=1024`，让每一步都达到 1024 个 token 的挖矿门槛；
- 统计挖矿乘法 `pearl_gemm_noisy` 实际被调用的次数：开挖矿时必须大于 0，关挖矿时必须为 0，否则直接报错；
- 比较准确率和逐题答案是否一致，而不是只比一句短答案的文字；
- 不提交区块，网关用 Pearl 测试自带的模拟区块模板，不连接任何网络。

在 GPU 机器上，按 Pearl 的 `miner/README.md` 装好挖矿环境，再装 `datasets`，然后：

```bash
python benchmarks/P05-result-accuracy/gpu/p05_gpu_check.py run --pearl $PEARL_DIR --mode no_mining --out off.json
```

```bash
python benchmarks/P05-result-accuracy/gpu/p05_gpu_check.py run --pearl $PEARL_DIR --mode mining --out on.json
```

```bash
python benchmarks/P05-result-accuracy/gpu/p05_gpu_check.py compare off.json on.json
```

默认模型是 `pearl-ai/Llama-3.1-8B-Instruct-pearl`，可用 `--model` 换成 70B（需要 H200 或多卡）。8B 两次运行预计不到 1 GPU 小时。
