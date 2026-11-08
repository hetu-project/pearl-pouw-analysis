# P05：官方"挖矿不影响 AI 结果"的测试从未真正挖过矿

**严重程度：中等** ｜ **结论**：官方测试不成立，由代码证实；我们在 H100 上补做了实测，挖矿不改变 8B 模型在 GSM8K 上的结果 ｜ Benchmark：[`benchmarks/P05-result-accuracy`](../../benchmarks/P05-result-accuracy)

Pearl 说挖矿的同时产出有用的 AI 结果。那么开着挖矿时，AI 的回答和不挖矿时一样准吗？协议本身从不检查有用结果（P01），这件事只能靠官方的测试来保证。我们看了官方的测试，它的思路是对的，但数据对不上：两次运行都没有真正挖矿。

## 官方是怎么测的

测试 `test_consistency`（`miner/vllm-miner/tests/test_vllm_execution.py:442-472`）：同一条提示，先开挖矿生成一次，再关挖矿生成一次，要求两次生成的文本和参考答案完全相同。参考答案（`reference_outputs.json`）开、关挖矿都是 "2 + 2 = 4" 和 "Le chat est sur la table."，测试通过。

开关 `MINER_NO_MINING` 只控制一件事：7 位量化的层走带噪声的挖矿乘法，还是走普通乘法（`vllm_kernels.py:207`）。只改这一个变量，思路是对的。

## 哪里对不上

**1. 开着挖矿的那次，其实没有挖矿。** 插件只在一步 ≥1024 个 token 时才走挖矿乘法（P02）。而测试：

| | 数值 |
|---|---|
| 4 条测试提示的长度 | 336、126、249、62 个字节，即最多 336 个 token |
| 每次发送的提示数 | 1 条 |
| 生成回答时每一步的 token 数 | 1 |
| 挖矿门槛 | 1024 个 token |

没有任何一步达到门槛，两次运行执行的是完全相同的代码，输出相同是必然的，什么也证明不了。测试里也没有检查挖矿乘法到底有没有被调用过，所以这个问题一直没有被发现。

**2. 就算真的挖了，比一句"2 + 2 = 4"也说明不了多少。** 几个 token 的短答案对数值误差不敏感，要看整套评测题的准确率才有意义。

**3. 其他能找到的数据也补不上：**

| 数据 | 问题 |
|---|---|
| 挖矿计算库的单元测试（`miner/pearl-gemm/tests/test_pearl_gemm.py`） | 只测到矩阵维度 k = 8192，容差为绝对 0.1、相对 1%；Gemma-4-31B 有层的 k 是 16384，没被测过 |
| 70B、Qwen3、Gemma 模型卡上的评测分数 | 都没说明评测时是否开着挖矿；评测通常一条一条跑，按门槛多半不会触发挖矿 |
| 8B 模型（官方示例模型、也是这个测试用的模型） | 模型卡上没有任何精度数据 |

以上都由 [`p05_code_facts.py`](../../benchmarks/P05-result-accuracy/p05_code_facts.py) 从 Pearl 源码和固定版本的模型卡里直接读出，输出见 [`results/reference-code-facts.txt`](../../benchmarks/P05-result-accuracy/results/reference-code-facts.txt)。

## 正确的验证方法

思路和官方一样，开、关挖矿各跑一次，只改挖矿开关，但要满足三点：

1. **保证真的挖了**：一次提交大量请求，让每一步都 ≥1024 个 token，并统计挖矿乘法实际被调用的次数。开挖矿时必须大于 0，关挖矿时必须为 0；
2. **用评测集比准确率**：例如 GSM8K 全部 1319 道题，比较两次的准确率，以及每道题的答案是否一致；
3. **记录条件**：模型、显卡、并发数、挖矿调用次数，让别人能重跑。

## 我们的实测

[`gpu/p05_gpu_check.py`](../../benchmarks/P05-result-accuracy/gpu/p05_gpu_check.py) 按上面的方法实现：用 Pearl 官方的方式启动挖矿（本地网关加官方测试自带的模拟区块模板，不提交区块，不连网络），一次提交 GSM8K 全部 1319 题，`max_num_seqs=1024`，开、关挖矿各跑一次。2026-11-08 在 1 张 H100 NVL 上运行（Pearl `156ef9a8`，vLLM 0.20.2，模型 `pearl-ai/Llama-3.1-8B-Instruct-pearl` 版本 `5dcb348a`）：

| | 关挖矿 | 开挖矿 |
|---|---|---|
| 准确率 | 84.31% | 84.31% |
| 挖矿乘法实际调用次数 | 0 | 13,840（最小 m = 1024） |
| 生成文本与另一次逐字相同 | 100% | 100% |

单个矩阵乘法也逐位一致：8B 模型三种可挖层、m 从 256 到 8192，挖矿乘法去噪后的输出与普通乘法完全相同（[`GPU-mining-impact/kernel_overhead.py`](../../benchmarks/GPU-mining-impact/kernel_overhead.py)）。

**结论**：挖矿去噪后的结果与不挖矿逐位相同，挖矿本身不改变 AI 结果。问题出在官方测试：它没有真正挖矿，所以本来证明不了这一点；现在由实测补上。

输出见 [`results/reference-gpu-compare.txt`](../../benchmarks/P05-result-accuracy/results/reference-gpu-compare.txt)，逐题结果在同目录的 `gpu-mining-*.json`。这次只测了 8B 模型；70B 需要 H200 或多卡。挖矿对延迟和算力的影响见 [`benchmarks/GPU-mining-impact`](../../benchmarks/GPU-mining-impact)。

## 对我们的启发

- **结果对不对要由协议保证**，而不是靠项目方的测试。PoCW 验证的是交付给需求方的结果，结果要经过验收。
- **"做有用功不影响结果"要在证明逻辑真正执行的条件下测**，并在测试里检查它确实执行了。
