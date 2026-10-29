# P05 benchmark：有用结果的正确性没有可信验证

问题报告：[`docs/problems/P05-result-accuracy.md`](../../docs/problems/P05-result-accuracy.md)

**只在 CPU 上运行。** 模型级精度需要 GPU，本 benchmark 不包含，见报告"哪些需要 GPU 实测"。

## 做什么

| 部分 | 程序 | 内容 | 性质 |
|---|---|---|---|
| 1. 官方测试和公开数据能证明什么 | `p05_code_facts.py` | 内核测试的容差和覆盖到的最大 k；端到端测试的提示长度是否达到挖矿门槛；挖矿开关实际改变了什么；8B 模型卡有没有精度数据 | 代码事实 |
| 2. 去噪声误差 | `p05_denoise_error.py` | 在 CPU 上按官方内核的常数和计算顺序仿真 fp16 去噪声，测误差大小，并与内核测试容差、7 位量化误差对比 | CPU 仿真 |

## 运行

先在仓库根目录执行 `./scripts/setup-pearl.sh`，然后：

```bash
./benchmarks/P05-result-accuracy/run.sh
```

- 约 30 秒。
- 首次运行需要联网：下载固定版本的 8B 模型卡和 NumPy。

## 文件

| 文件 | 说明 |
|---|---|
| `p05_code_facts.py` | 第 1 部分 |
| `p05_denoise_error.py` | 第 2 部分 |
| `run.sh` | 下载模型卡，运行两部分 |
| `results/reference-code-facts.txt` | 第 1 部分参考输出 |
| `results/reference-denoise.txt` | 第 2 部分参考输出 |
