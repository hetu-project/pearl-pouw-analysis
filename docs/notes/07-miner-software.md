# 组件 7：挖矿软件与"有用挖矿"（vllm-miner / pearl-gemm / pearl-gateway）

> 研究对象：Pearl 源码 `third_party/pearl`，commit `156ef9a8`（master）。下文的 `文件:行号` 都指这个 commit，路径相对仓库根目录 `miner/`，另有说明的除外。
> 估算脚本：`experiments/miner/coverage.py`（覆盖率）和 `experiments/miner/gemm_overhead.py`（单次 GEMM 运算量），输出分别存在同目录的 `coverage_output.txt` 和 `gemm_overhead_output.txt`。外部资料快照（HF 模型卡、config、vLLM `arg_utils.py`）存在 `experiments/miner/external/`。
> 标注约定：**【代码】**＝代码事实；**【估算】**＝估算结果（附假设）；**【外部】**＝网络资料（附链接）；**【理解】**＝我的解读，未经验证。

---

## 0. 一页总览

| 模块 | 作用 | 关键文件 |
|---|---|---|
| `vllm-miner` | vLLM 插件。注册 `pearl` 量化方法，用 7bit 量化加 NoisyGEMM 替换线性层 | `vllm-miner/src/vllm_miner/*.py` |
| `pearl-gemm` | CUDA/CUTLASS（sm_90a）内核，包括 int8 GEMM、noise_gen、noise_A/B、noisy_gemm（含 PoW 提取与去噪）、GPU BLAKE3 Merkle 哈希；另有 CuTe DSL 写的 Hadamard+SmoothQuant 量化内核 | `pearl-gemm/csrc/**`, `pearl-gemm/src/pearl_gemm/**` |
| `miner-base` | 异步循环（每秒拉取任务、轮询 CUDA event、提交）、承诺哈希、Merkle 树、PlainProof 组装；另有一个 CPU 参考实现 | `miner-base/src/miner_base/*.py` |
| `pearl-gateway` | 连接 pearld 节点与矿工进程：拉 `getblocktemplate`、缓存任务、经 UDS 提供 JSON-RPC、把 PlainProof 生成 ZK 证书后 `submitblock` | `pearl-gateway/src/pearl_gateway/**` |
| `miner-utils` | 日志 | `miner-utils/src/miner_utils/logging_config.py` |

数据流：`pearld --getblocktemplate--> gateway(WorkCache) --getMiningInfo(UDS, 每秒)--> vLLM worker(AsyncLoopManager)`。worker 在每次满足阈值的线性层前向里执行 `pearl_gemm_noisy`。GPU 内核如果命中目标，会写入 pinned host header；异步线程看到后复制 A/B 到 CPU，组装 `PlainProof`，再用 `submitPlainProof` 发回 gateway。gateway 生成 ZK 证明、拼出区块并 `submitblock`。

---

## 1. 整体架构

### 1.1 插件注册

- **【代码】entry point**：`vllm-miner/pyproject.toml:54-55` 把插件注册到 `vllm.general_plugins` 组：`register_pearl_mining_plugin = "vllm_miner:register_pearl_miner_layer"`。vLLM 会在每个进程（API server 和 EngineCore/worker）加载这类插件。
- **【代码】注册函数** `register.py:33-50`：
  - 用进程名判断自己是不是 vLLM worker（`EngineCore*` / `VllmWorker*`，`register.py:26-30`）。只有 worker 会 `init_async_manager()` 并预分配 pinned 池（`register.py:42-44`）。
  - 未设置 `MINER_NO_VLLM_PLUGIN` 时，执行 `register_quantization_config("pearl")(PearlConfig)`（`register.py:49-50`）。checkpoint 的 `quantization_config.quant_method` 是 `"pearl"`（见 §5.3），所以 vLLM 加载这些模型时会自动走 PearlConfig。
  - `init_async_manager` 构造 `AsyncLoopManager(MinerRpcConfig(transport="uds", socket_path=config.gateway_socket_path))` 并 `start()`（`mining_state.py:24-38`）。`start()` 会**同步**调用一次 `getMiningInfo`（`async_loop_manager.py:85-90`），所以 worker 启动前 gateway 必须已经就绪。`entrypoint.sh` 会先启动 gateway，并等待 `/tmp/pearlgw.sock` 出现。
- **【代码】PearlConfig 继承 CompressedTensorsConfig**（`vllm_config.py:33-65`）：`from_config` 先用父类解析 compressed-tensors 配置，再原样构造子类。它只覆盖两处：
  1. `_get_scheme_from_parts`（`vllm_config.py:168-206`）：
     - **挖矿层**（`_is_mining_layer`，`:67-84`）：权重和输入都是 7bit；权重是 TENSOR 或 CHANNEL 策略；输入是 TOKEN 策略；对称量化；权重静态、激活动态。命中时返回 `PearlScheme(mining_enabled=True)`。
     - **非挖矿 int8 层**（`_is_non_mining_layer`，`:86-103`）：条件同上，但位宽是 8bit。命中时返回 `PearlScheme(mining_enabled=False)`，走 int8 加 vanilla GEMM。
     - 其他层交给父类处理，例如 FP8 block 层由 vLLM 原生 compressed-tensors FP8 路径执行（`:205-206`）。
  2. `get_quant_method`（`:142-166`）：如果 FusedMoE 层的 gate/up 是 7bit 挖矿配置、down 是 FP8 block 配置（`_is_fp8_block_layer`，`:105-126`），就路由到 `PearlMoEMethod`。
- **【代码】PearlScheme / PearlKernel**：
  - `PearlScheme.get_min_capability()=9`（`vllm_scheme.py:57-60`）。
  - 权重以 int8 存储，形状为 (n,k)，不转置（`:85-94`）。
  - 每层额外注册两个参数：`smooth_quant_scale`（bf16，长度 k，默认全 1，`:123-129`）和 `hadamard_block_size`（int32，默认 0，`:131-136`）。
  - `PearlKernel` 继承 vLLM 的 `Int8ScaledMMLinearKernel`（`vllm_kernels.py:33`）。`apply_weights` 按 `mining_enabled` 分成两条路径（`:154-185`）。

### 1.2 哪些层挖矿、哪些不挖

- **【代码】决定层能否挖矿的只有位宽**：checkpoint 把某层标为 7bit int，它才可能进入 `_apply_weights_mining`（`vllm_kernels.py:187-224`）。即便如此，还要每次前向满足 m/n/k 阈值才真正走 `pearl_gemm_noisy`；不满足时仍按 **7bit 量化**执行 `pearl_gemm_vanilla`（`:218-224`）。也就是说，**这些层不论挖不挖，精度损失都是 7bit**。
- **【外部】pearl-ai 发布的 checkpoint 实际怎么分配**（HF `config.json` 的 `quantization_config`，2026-10-27 抓取，快照在 `experiments/miner/external/*.config.json`）：

| 模型 | 7bit int（可挖矿，W7A7，权重按通道、激活按 token 动态量化） | FP8 block 128×128，激活按 group 128（不挖矿，走 vLLM 原生 FP8） | 不量化 |
|---|---|---|---|
| [Llama-3.1-8B-Instruct-pearl](https://huggingface.co/pearl-ai/Llama-3.1-8B-Instruct-pearl) | 全部层的 `o_proj`、`gate_proj`、`up_proj`；第 16–31 层的 `q/k/v_proj` | 全部 `down_proj`；第 0–15 层的 `q/k/v_proj` | `lm_head` |
| [Llama-3.3-70B-Instruct-pearl](https://huggingface.co/pearl-ai/Llama-3.3-70B-Instruct-pearl) | 全部层的 `o_proj`、`gate_proj`、`up_proj`；第 40–79 层的 `q/k/v_proj` | 全部 `down_proj`；第 0–39 层的 `q/k/v_proj` | `lm_head` |
| [Qwen3-30B-A3B-Instruct-2507-pearl](https://huggingface.co/pearl-ai/Qwen3-30B-A3B-Instruct-2507-pearl)（MoE） | 全部 `q/k/v/o_proj`；每个专家的 `gate_proj`、`up_proj` | 每个专家的 `down_proj` | — |
| [Gemma-4-31B-it-pearl](https://huggingface.co/pearl-ai/Gemma-4-31B-it-pearl) | 其余所有 `Linear` | `q/k/v(qkv)_proj`、`mlp.down_proj` | vision tower 的大量线性层（列在 ignore 中） |

  - 结论：**所有已发布的 checkpoint 都没有用到 "8bit int 非挖矿" 那条 PearlScheme 路径**。非挖矿层一律是 FP8 block，交给父类 CompressedTensors 处理。
  - **【理解】** `down_proj`（k 等于 FFN 维度，最大）和前一半层的 qkv 保持 FP8，可能是为了精度，因为 down_proj 的输入是 SiLU·up 的乘积，离群值多。这只是推测，仓库里没有文档说明原因。
- **【代码】MoE**（`pearl_moe_method.py:43`、`pearl_moe_experts.py:184-260`）：只有 GEMM1（gate_up，即 w13）挖矿，并且是**每个专家单独调用一次 `noisy_gemm`**（`_apply_per_expert_gemm1`，`:262`）。是否挖矿用 `should_use_noisy_gemm(num_tokens, N, K)` 判断，`num_tokens` 是整批 token 数，**不是每个专家分到的行数**（`:184-190`）。GEMM2（down）始终走 Triton fp8 block GEMM（`:245-260`、`:381-431`）。中奖时按专家逐个查看 header，只提交第一个命中的专家（`callbacks.py:93-151`）。
- **【代码】注意力**：QK^T 和 softmax·V 在 vLLM 的注意力后端（FlashAttention/FlashInfer 等）里计算，不经过 `LinearMethod`/`CompressedTensorsScheme`。插件只拦截线性层，所以**注意力内部的矩阵乘不走插件，也不挖矿**。`lm_head` 在 ignore 列表中，同样不挖。

### 1.3 版本锁定与硬件

- **【代码】**
  - `vllm==0.20.2`、`torch==2.11.0`、`torchaudio==2.11.0`、`torchvision==0.26.0`，Python `==3.12.*`（`vllm-miner/pyproject.toml:39-44`）；`compressed-tensors` 未锁版本（`:33`）。
  - pearl-gemm 依赖 `torch==2.11.0`、`cuda-python>=13.0,<13.1`、`nvidia-cutlass-dsl>=4.4`（`pearl-gemm/pyproject.toml:9-20`）。
  - Docker 镜像基于 `nvidia/cuda:13.0.3-devel-ubuntu24.04`，使用 `UV_TORCH_BACKEND=cu130`（`vllm-miner/Dockerfile:3,22`）。`setup.py` 要求 torch 的 CUDA ≥ 13.0（`pearl-gemm/setup.py:453-459`）。
- **【代码】CUDA 架构**：`COMPUTE_CAPABILITY = "arch=compute_90a,code=sm_90a"`（`pearl-gemm/setup.py:89`），以 `-gencode` 传给 nvcc（`:352`, `:443`）。这里**只有 sm_90a 一个目标**，即 Hopper H100/H200 专用的 `a` 后缀特性（WGMMA、TMA）。内核源码直接 include `mma_traits_sm90_gmma.hpp`，例如 `csrc/gemm/pow_utils.hpp:4`。`miner/README.md:46` 写明 "Currently only **sm90** (H100 / H200) GPUs are supported"。
- **【代码】默认编译的内核网格**（`pearl-gemm-build-utils/.../default_compiled_kernels.py:15-62`）：matmul 只有 128×256×128、3 stage、cluster 1×1，R∈{64,128}；noisingA/B 为 64×64，R∈{64,128}，类型 fp16 或 int32。setup 默认 `DISABLE_R32=TRUE`（`setup.py:64`），输出类型只有 bf16（`:82`）。
- **【代码】运行时**：没有任何 CUDA Graph 防护（在 `vllm-miner/src` 中 grep `graph` 无结果）。官方示例和 E2E 测试都加了 `--enforce-eager`（`miner/README.md:129`；`vllm-miner/tests/test_vllm_execution.py:212`）。**【理解】** 原因是 `pearl_gemm_noisy` 里有 host 端操作：读取 Python 对象、`torch.frombuffer(...).to("cuda")`、`schedule_status_check` 回调，这些在 CUDA Graph 捕获下不成立。

### 1.4 主干外的新方案（作为背景，不属于 156ef9a8 的运行代码）

- **【代码/git】** 远端分支 `origin/fp8-scheme`（2026-09-09 起，最新为 `1280a21c` 2026-09-28）重写成 "FP8/FP10 scheme"，未合入 master。其中：
  - `0b18ef42 feat(fp8): Hopper (H100) support (#337)`；
  - `85216510 consumer blackwell support (#342)` 新增 `_arch.py`，`Arch` 只列 **SM100（B200）和 SM120（RTX PRO 6000 / GeForce RTX 50）**；
  - 该分支的 README 写明 "Blackwell only (SM100 and SM120): the `pearl` quantization method refuses to load on any other GPU"。
  - 该分支 README 还写明 "Decode, small batches, capture windows, no-job periods, and mining failures serve through the FP8 fallback GEMM"；阈值改成环境变量 `PEARL_MIN_MINING_TOKENS`（默认 1024）；"B-side per-job preparation runs ... off the serving path"，即权重侧的准备按任务缓存。
  - 读取命令：`git show origin/fp8-scheme:miner/vllm-miner/README.md`。

---

## 2. 一次挖矿 GEMM 的完整步骤（`pearl_gemm_noisy`）

记号：A 为激活，形状 (m,k)，int8，值域 [-63,63]；B 为权重，形状 (n,k)，int8，值域同上；r = `noise_rank` = 128（`miner_base/settings.py:9-10`）。下表 "量级" 一列是相对主乘积 m·n·k 的比例（**【估算】**按运算/字节计数，不是实测时间；见 `gemm_overhead.py`）。

| # | 步骤 | 代码 | 内容 | 量级（相对 m·n·k） |
|---|---|---|---|---|
| 0 | 7bit 量化 | `vllm_kernels.py:202`；`quantization_operators.py:72-79`；`pearl_gemm/quantization/hadamard.py:209-470` | 按 token 动态对称量化：可选 Hadamard，再乘 `smooth_quant_scale`，取行最大绝对值，scale=max/63，`rint` 后截断到 [-63,63] | O(m·k)，≈ 1/n |
| 1 | 取任务 | `gemm_operators.py:113`；`async_loop_manager.py:114-115` | 直接读 `_mining_job`，后台线程每秒刷新 | O(1) |
| 2 | 生成 MatmulConfig | `gemm_operators.py:110`；`gpu_matmul_config.py:22-41` | `MiningConfiguration(common_dim=k, rank=r, mma_type=Int7xInt7ToInt32, rows_pattern=[0,8], cols_pattern=64 个偏移)`（模式见 `settings.py:18-26`）。哈希 tile 每次尝试覆盖 2×64=128 个输出元素 | O(1) |
| 3 | 目标调整 | `gemm_operators.py:116`；`comm/dataclasses.py:212-225`；`zk-pow/src/api/sanity_checks.rs:192-196,239-245` | `adjusted = target × tile_size × (dot_len/r) × 128`，其中 tile_size=128，dot_len = k − k mod r。r=128 时系数为 128·k。【理解】这一步把区块难度换算成单次 tile 尝试的难度：每次尝试做的工作越多，单次目标越宽 | O(1) |
| 4 | job_key | `gemm_operators.py:118-122`；`commitment_hash.py:36-37` | `key = BLAKE3(incomplete_header_bytes ‖ mining_config.to_bytes())`，然后拷贝到 GPU，产生一次小的 H2D 同步 | O(1) |
| 5 | A 的 GPU Merkle 哈希 | `gemm_operators.py:124-130`；`csrc/tensor_hash/*` | `tensor_hash(A.to(torch.uint8), key, ...)`。注意 `.to(uint8)` 会**复制一份 m·k**。用 BLAKE3 keyed 哈希建 Merkle 树，得到 32B 根 | m·k 字节，≈ 1/n 字节/MAC |
| 6 | B 的 GPU Merkle 哈希 | `gemm_operators.py:132-138` | 同上，对象是**整个权重分片 n·k**，**每次调用都重新复制并哈希**，没有缓存 | n·k 字节，≈ 1/m 字节/MAC |
| 7 | 种子链（承诺哈希） | `gemm_operators.py:141-150`；`commitment_hash.py:80-108` | V3 证书（`cert_version.uses_salted_seeds`）先把根加盐：`root' = BLAKE3_keyed(salt, root‖dim_u32‖pad)`，A 用 m、B 用 n（`:16-27`）。然后 `seedB = BLAKE3(key ‖ rootB')`，`seedA = BLAKE3(seedB ‖ rootA')`。MoE 情况下 rootA 还要先折叠路由根（`:98-104`）。【理解】A 的噪声依赖 B 的承诺，两边的噪声都依赖 header，所以矿工在 header 固定后无法先选噪声再挑矩阵 | O(1) |
| 8 | 生成噪声 | `gemm_operators.py:153-170, 267-333`；`csrc/gemm/noise_generation_kernel.h:1-30, 85-95` | 用 BLAKE3 keyed 输出流生成：稠密的 EAL (m,r)、EBR (n,r)，值域 [-32,32)；稀疏的 EAR、EBL (k,r)，每个 k 行恰好一个 +1 和一个 −1，各存 R-major 和 K-major 两种布局；再加 EAL/EBR 的 fp16 副本 | (m+n)·r + 4k·r 字节 |
| 9 | 加噪 A | noisy_gemm 内部的 noise_A 内核（`pearl_gemm_interface.py:117-168`；`csrc/gemm/pearl_noisingA_kernel.h:99-117`） | `ApEA = A + EAL·EAR`（int8）。因为 \|A\|≤63、\|EAL·EAR\|≤63，结果在 [-126,126]，**正好放进 int8**。【理解】这就是激活只能用 7bit 的原因。同时算 `AxEBL = A·EBL`（m,r），用于去噪 | 2·m·k·r（稠密 WGMMA）≈ **2r/n** |
| 10 | 加噪 B | noise_B 内核（`pearl_gemm_interface.py:186-237`；`pearl_noisingB_kernel.h:101-117`） | `BpEB = B + EBL·EBR`，`EARxBpEB = EAR·BpEB`（n,r）。**每次调用都对整个权重分片重做一遍** | 2·n·k·r ≈ **2r/m** |
| 11 | 带噪 GEMM 与 PoW | `noisy_gemm`（`pearl_gemm_interface.py:325-497`）；`csrc/gemm/pow_utils.hpp:128-237` | int8×int8→int32 计算 `ApEA·BpEBᵀ`（tile 128×256×128）。k 方向每推进 r，每个线程把自己 128 个 int32 累加值 XOR 归约，再 `rotl13` 后 XOR 进 16×u32 的 transcript（`TileHashAccumulator::accumulate`，`:166-182`）。tile 结束时用 `pow_key = seedA` 做一次 keyed BLAKE3 压缩，若 `hash ≤ adjusted_target`（`check_pow_target`，`:202-237`）就加锁写 pinned host header（`write_host_signal_header`，`:239-325`），多个命中时**只记第一个** | 主乘积 m·n·k；XOR 约 m·n·k/r，即 1/128 次 ALU 操作/MAC；BLAKE3 压缩 m·n/128 次 |
| 12 | 去噪 | `csrc/gemm/collective_epilogue.hpp:360-455`；`kernel_traits.hpp:24-28,144-146`；`pearl_gemm_constants.hpp` | int32 累加值转 fp32 后除以 2¹²，做两次 **fp16×fp16→fp32 WGMMA**（秩 r）：减去 `EAL·EARxBpEB` 和 `AxEBL·EBR`，再乘回 2¹²，最后乘 A_scales·B_scales 输出 bf16。**精度：去噪因子以 fp16 存储（AxEBL 预缩放 2⁻¹⁴，EARxBpEB 预缩放 2⁻¹²），用 fp32 累加，所以不保证逐位精确还原**；kernel 测试容差为 `atol=1e-1, rtol=1e-2`（`pearl-gemm/tests/test_pearl_gemm.py:704-708`） | 2·m·n·r 次 fp16 MAC ≈ **2r/k**（H100 上 fp16 吞吐约为 int8 的一半） |
| 13 | 异步检查中奖 | `gemm_operators.py:219-234`；`async_loop_manager.py:148-167, 225-244` | 记录一个 CUDA event，回调放进队列；后台协程每 10ms `event.query()`，完成后执行回调 | 不阻塞前向；但 pinned 池（128 个）满了会阻塞（`pearl_gemm/helpers.py:43-54`） |
| 14 | 组装 PlainProof | `callbacks.py:21-64`；`miner_base/block_submission.py:14-62` | 只在命中时执行，见 §6 | — |

- **【代码】中间张量**：`ApEA` (m,k) 和 `BpEB` (n,k) 都以 int8 写回显存（`gemm_operators.py:173-178`），另有 scratchpad 大小为 `get_required_scratchpad_bytes(max(m·k, n·k))`（`:104-109`）。**【估算】** 每次挖矿 GEMM 的额外显存峰值大约是：B 的 uint8 拷贝（n·k）＋BpEB（n·k）＋A 拷贝和 ApEA（2·m·k）＋scratchpad。以 70B 的 gate_up 为例（TP1，n·k = 57344×8192 ≈ 470 MB），单层瞬时增加约 1 GB 以上。
- **【估算】额外张量核工作量**：合计 `2r/n + 2r/m + 2·(2r/k)`，最后一项按 fp16 吞吐折半计。代表值如下（`gemm_overhead_output.txt`）：

| GEMM | m=1024 | m=2048 | m=4096 | m=8192 |
|---|---|---|---|---|
| 8B gate_up (n=28672,k=4096) | 38.4% | 25.9% | 19.6% | 16.5% |
| 8B o_proj (4096,4096) | 43.8% | 31.2% | 25.0% | 21.9% |
| 70B gate_up (57344,8192) | 31.7% | 19.2% | 12.9% | 9.8% |
| 70B o_proj (8192,8192) | 34.4% | 21.9% | 15.6% | 12.5% |
| 70B o_proj TP4 (8192,2048) | 53.1% | 40.6% | 34.4% | 31.2% |

  主要来源是**每次都重新给权重加噪**（2r/m，m=1024 时为 25%）和 **fp16 去噪**（2r/k）。另外，B 侧的显存流量（复制、哈希读取、加噪读取、BpEB 写回）每次挖矿前向约为可挖矿 int7 权重量的 4 倍：8B 约 19 GB，70B 约 185 GB。按 H200 4.8 TB/s 计算分别约 4 ms 和 39 ms。**这些都是运算量和字节数上的上界估计，没有考虑与主 GEMM 的重叠和内核实际效率，不能当作实测开销。**

---

## 3. 触发条件

- **【代码】阈值**：`config.yaml:9-14` 为 `min_m: 1024, min_n: 256, min_k: 1024`。判定函数 `Config.should_use_noisy_gemm`（`config.py:36-44`）：
  ```python
  if (n == 1) or (k == 1) or (m == 1): return False
  return (m >= min_m) and (n >= min_n) and (k >= min_k)
  ```
  调用前另有 `not config.settings.no_mining`（`vllm_kernels.py:207`）。环境变量是 `MINER_NO_MINING`，`MinerSettings` 的前缀为 `miner_`（`settings.py:7`）。
  - git 历史：初版（`960681d4`, 2026-03-17）中 `min_n` 是 1024，在 `7f260586`（2026-06-22, "Added support for hadamard quantized models"）改成 256。
- **【代码】m、n、k 的含义**（`vllm_kernels.py:202-204`）：`m = x_q.shape[0]`、`k = x_q.shape[1]`、`n = w_q.shape[0]`。
  - **m 是本次前向里该线性层的输入行数**，即 vLLM 这一步调度的 token 总数。V1 引擎把所有请求的 token 展平拼在一起：prefill 时每个请求贡献整段（或一个 chunk）的 token，decode 时每个请求贡献 1 个 token。
  - **n 是本 rank 的输出维度**：qkv_proj 为 (n_heads+2·n_kv)·head_dim / TP；gate_up_proj 为 2·FFN / TP；o_proj 为 hidden。
  - **k 是本 rank 的输入维度**：qkv 和 gate_up 为 hidden；o_proj 为 n_heads·head_dim / TP。
- **【代码】与共识规则的关系**：verifier 要求 `k ≥ 16r`，r=128 时 k ≥ 2048；还要求 `k ≥ 1024`、`k % 64 == 0`、`k ≤ 4r²`（`zk-pow/src/api/sanity_checks.rs:33-37`）。插件阈值 `min_k=1024` 比 `16r=2048` 宽松。**【估算】** 8B 在 TP=4 或 70B 在 TP=8 时，o_proj 的本地 k=1024，满足插件阈值会被挖，但按上面的检查生成的证明无法通过（见 `coverage_output.txt` 的 TP 表）。
- **【代码】MoE 的触发**：`should_use_noisy_gemm(num_tokens, N, K)`（`pearl_moe_experts.py:184-190`），m 用整批 token 数，不是单个专家的行数。

---

## 4. "有用挖矿覆盖率"估算

> 脚本：`experiments/miner/coverage.py`，输出 `coverage_output.txt`。以下全部为**【估算】**。

### 4.1 模型结构（官方 config）

【外部】数值取自 pearl-ai checkpoint 的 config.json，与 Meta 官方一致：

| | hidden | FFN | 层数 | 注意力头 / KV 头 | head_dim | vocab |
|---|---|---|---|---|---|---|
| Llama-3.1-8B | 4096 | 14336 | 32 | 32 / 8 | 128 | 128256 |
| Llama-3.3-70B | 8192 | 28672 | 80 | 64 / 8 | 128 | 128256 |

### 4.2 假设

1. vLLM 把 q/k/v 融合成 `qkv_proj`、把 gate/up 融合成 `gate_up_proj`，所以 n 是融合后的维度。
2. 哪些层可挖按 §1.2 的 checkpoint 分配：down_proj 和前一半层的 qkv 是 FP8，lm_head 是 bf16，这些都不挖。
3. 只统计乘加数（MAC），int8、fp8、bf16 的 MAC 等权，不折算时间。
4. 注意力 MAC：prefill（因果，长度 L）每层为 2·L(L+1)/2·d_attn；decode（上下文 c）每层每 token 为 2·c·d_attn。
5. lm_head 只对每条序列的采样位置计算一次。
6. vLLM v0.20.2 的默认值：H100/H200（显存 ≥70 GiB 且不是 A100）上，API server 的 `max_num_batched_tokens=8192`、`max_num_seqs=1024`；LLM 类分别为 16384 和 1024。【外部】见 vLLM `vllm/engine/arg_utils.py@v0.20.2:2208-2217`（快照 `experiments/miner/external/vllm_arg_utils_v0.20.2.py`）。V1 默认开启 chunked prefill。

### 4.3 静态上限：m ≥ 1024 时可挖的线性层 MAC 占比

每层每 token 的 MAC：

| | qkv | o | gate_up | down | 可挖占比 |
|---|---|---|---|---|---|
| 8B | 25.2M | 16.8M | 117.4M | 58.7M | (16.8+117.4+½·25.2)/218.1 = **67.3%** |
| 70B | 83.9M | 67.1M | 469.8M | 234.9M | (67.1+469.8+½·83.9)/855.7 = **67.6%** |

TP 的影响：8B 在 TP=8 时 o_proj 的 k=512 < 1024，不再挖，占比降到 59.6%。8B TP=4 和 70B TP=8 时 o_proj 的 k=1024，会被挖，但不满足 verifier 的 k ≥ 2048（见 §3）。

### 4.4 (a) 预填充（单步，B 个长度为 L 的提示一起调度，m=B·L ≤ 8192）

| L | B | m | 8B 已挖/线性 | 8B 已挖/全部（含注意力和 lm_head） | 70B 已挖/线性 | 70B 已挖/全部 |
|---|---|---|---|---|---|---|
| 128 | 1–4 | ≤512 | 0% | 0% | 0% | 0% |
| 128 | 8 | 1024 | 67.3% | 67.1% | 67.6% | 67.6% |
| 512 | 1 | 512 | 0% | 0% | 0% | 0% |
| 512 | 2 | 1024 | 67.3% | 66.7% | 67.6% | 67.3% |
| 1024 | 1 | 1024 | 67.3% | 66.0% | 67.6% | 67.0% |
| 2048 | 1 | 2048 | 67.3% | 64.8% | 67.6% | 66.3% |
| 4096 | 1 | 4096 | 67.3% | 62.5% | 67.6% | 65.1% |
| 8192 | 1 | 8192 | 67.3% | 58.3% | 67.6% | 62.7% |

结论：只要单步 token 数 ≥1024，可挖层的 MAC 全部被挖，占线性层 MAC 的 ~67%；提示越长，注意力 MAC 越多，"已挖/全部" 越低。单步 token 数 <1024 时为 0。这是**阶跃函数**。

### 4.5 (b) 逐 token 解码（单步，B 条并发序列，m=B）

| B | 8B 已挖/线性 | 8B 已挖/全部（c=512 / 2048 / 8192） | 70B 已挖/线性 | 70B 已挖/全部（c=512 / 2048 / 8192） |
|---|---|---|---|---|
| 1, 32, 128, 256, 512, 1023 | **0%** | 0% | **0%** | 0% |
| 1024 | 67.3% | 61.5% / 58.4% / 48.7% | 67.6% | 66.0% / 64.1% / 57.7% |

- 纯 decode 步要挖矿，**需要恰好 1024 条并发序列**，而这正是 vLLM 在 H100/H200 上 `max_num_seqs` 的默认上限。
- KV cache 也限制并发数（每 token KV：8B 为 128 KB，70B 为 320 KB，bf16）。按显存利用率 0.9、预留 5 GB 激活粗算：
  - 8B 单卡 H100 80GB：c=512 时约 862 条，c=2048 时约 215 条，c=8192 时约 53 条；
  - 70B 单卡 H200 141GB（对应官方 DP=4 配置）：c=512 时约 293 条，c=2048 时约 73 条，c=8192 时约 18 条。
  - **按这个粗算，上述配置在常见上下文长度下装不下 1024 条并发序列，纯 decode 步基本无法触发挖矿**（未考虑 FP8 KV cache 等节省显存的选项）。

### 4.6 (c) 连续批处理的影响

- **【代码/理解】** vLLM V1 每一步把 "所有正在 decode 的序列各 1 个 token" 和 "新请求 prefill 的一个 chunk" 拼成一个大小为 m 的批，m ≤ `max_num_batched_tokens`。阈值按整步 m 判断，所以：
  1. 单个请求的短提示（例如 300 token）与别的请求拼起来后 m ≥1024，**也会被挖**；
  2. 在含 prefill chunk 的步里，**同批的 decode token 也会被挖**，因为整个 A 矩阵进入 NoisyGEMM；
  3. 纯 decode 步（没有新请求进来）m 等于并发数，通常 <1024，不挖。
- **【估算】玩具调度仿真**（`coverage.py: simulate`）：预算 8192 token/步，`max_num_seqs` 1024，Poisson 到达，FCFS 分块 prefill，**不模拟 KV 容量限制**。"已挖/线性" = 可挖层占比 × 处于挖矿步的 token 占比：

| 模型 | 提示/输出 | 到达率（请求/步） | 平均 token/步 | 挖矿步占比 | 挖矿步中的 token 占比 | 已挖/线性 |
|---|---|---|---|---|---|---|
| 8B | 512/256 | 0.05 | 36 | 0.1% | 3.3% | 2.2% |
| 8B | 512/256 | 0.2 | 158 | 2.0% | 14.2% | 9.6% |
| 8B | 512/256 | 1.0 | 767 | 25.9% | 50.0% | 33.7% |
| 8B | 2048/256 | 0.05 | 110 | 4.7% | 89.5% | 60.2% |
| 8B | 256/1024 | 0.2 | 247 | 0.0% | 0.1% | 0.1% |
| 8B | 256/1024 | 1.0 | 1196 | 87.4% | 93.5% | 62.9% |
| 70B | 4096/512 | 0.2 | 948 | 18.6% | 91.1% | 61.6% |

  （完整表格见输出文件；70B 和 8B 的 token 层面结果相同，只是可挖层占比略有差异。）

  解读：**长提示、短输出**（RAG、摘要）的负载里，大部分 token 处在挖矿步，覆盖率接近 60–65% 的上限。**短提示、长输出**（聊天、生成）且并发较低时，覆盖率接近 0。高负载时 decode 并发很容易超过 1024，但 70B 实际受 KV 容量限制，本仿真没有约束这一点。按"步"计，挖矿步通常只占少数。【理解】如果按时间算，decode 步占 GPU 时间的比例往往高于它在 MAC 中的比例，因为 decode 受显存带宽限制，所以按时间算的覆盖率会比按 MAC 算的更低。

### 4.7 注意力是否走插件

- **【代码/理解】不走。** 注意力在 vLLM attention backend 内部，插件只替换 `LinearMethod`（compressed-tensors 的 scheme）。长上下文下注意力 MAC 占比上升，例如 8B 在 L=8192 的 prefill 时 "已挖/全部" 从 67.3% 降到 58.3%。

---

## 5. 量化

### 5.1 激活 7bit 量化 `quant_7bit`

- **【代码】入口**：`quantization_operators.py:72-79` 调用 `quantize_kernel(x, max_val=63, ...)`（`:36-69`），实际执行 `pearl_gemm.quantization.quantize`（CuTe DSL JIT 内核，`hadamard.py:530-611`）。按 `(N, dtype, has_smooth, ..., max_val)` 缓存编译结果（`:575-609`）。
- **【代码】单个内核内的融合流程**（`hadamard.py:209-470`）：
  1. 从全局显存把一行 x（fp16/bf16）读进寄存器，转成 fp32（`:321-331`）；
  2. 若 `block_size>0`，在寄存器里做块对角 **Fast Walsh–Hadamard 变换**（`_fwht_reg_mut`，`:333-337`, `:476+`）。块宽必须是 2 的幂且整除 N（`:568-575`）；
  3. **乘以** `smooth_scale`（broadcast 形状 (1,N) 或按 token 形状 (M,N)），同时求 |max|（`_apply_smooth_chunk`，`:93-134`, `:347-416`）；
  4. 规约出行最大值；`scale = max/63`，`inv = 63·rcp(max+1e-30)`（`:424-446`）；
  5. `q = clamp(rint(x·inv), −63, 63)`，存为 int8；每行输出一个 fp32 scale（`:448-470`）。
- **【代码】参数来源**：`smooth_quant_scale` 和 `hadamard_block_size` 都是层参数，从 checkpoint 加载；不存在时默认值为全 1 和 0（`vllm_scheme.py:123-136`；`vllm_kernels.py:140-151`）。`DEFAULT_HADAMARD_BLOCK_SIZE=16` 只是 `quantize()` 的默认参数（`hadamard.py:525`），插件调用时显式传入层上的值。
- **【外部】已发布 checkpoint 里的实际情况**（读取 safetensors index/header，2026-10-27）：
  - **没有任何 checkpoint 含 `hadamard_block_size`**，所以运行时 block_size=0，Hadamard 不启用。
  - `smooth_quant_scale` 只出现在：Llama-8B 和 Llama-70B 的 `o_proj`（分别 32 层、80 层）；Qwen3 的 `q_proj`、`o_proj` 和专家 `gate_proj`（共享）；Gemma 的 `o_proj`（60 层）。
  - 其余 7bit 层（如 Llama 的 gate/up、qkv）的 smooth scale 为默认全 1，即**只做普通的按 token 7bit 量化**。
- **【理解】** smooth scale 在运行时是"乘"到激活上的，所以 checkpoint 里存的应该是 1/s，权重侧离线乘了 s。插件代码只负责乘，离线处理流程不在仓库中。

### 5.2 权重量化

- **【代码/外部】**
  - 7bit 层：权重 int7 对称、**按输出通道**（`strategy: channel`），存为 int8，scale 是 fp32 (n,1)（`vllm_scheme.py:85-108`）。config 中 observer 为 `minmax`（Llama）或 `memoryless_minmax`（Qwen/Gemma）。
  - FP8 层：`float8_e4m3`，权重 128×128 块量化，激活按 group 128 动态量化。
  - 量化流程本身（GPTQ/SmoothQuant 校准）**不在仓库中**。【外部】Pearl 官网写的是 "runs today on integer hardware via W7A7 quantization (GPTQ + SmoothQuant)"（[pearlresearch.ai/research/blog/proof-of-useful-work](https://pearlresearch.ai/research/blog/proof-of-useful-work)）。

### 5.3 公布的精度与基准

均为**【外部】**，来自 HF 模型卡，2026-10-27 抓取，快照在 `experiments/miner/external/*.README.md`：

- **Llama-3.3-70B-Instruct-pearl**（[模型卡](https://huggingface.co/pearl-ai/Llama-3.3-70B-Instruct-pearl)，"Launch Benchmark"，4×H200）：

  | 模型 | 并行 | MMLU | 吞吐 (tok/s) | 时间 (s) | Useful MADs (TMADs/s) |
  |---|---|---|---|---|---|
  | Meta 原版 | PP=4 | 0.8198 | 15,269.81 | 441.100 | – |
  | Meta 原版 | TP=4 | 0.8193 | 13,218 | 510 | – |
  | Meta 原版 | DP=2,TP=2 | 0.8197 | 13,162 | 512 | – |
  | Meta 原版 | DP=4 | OOM（bf16 约 140 GB 放不进单卡） | | | |
  | Pearl | PP=4 | 0.8190 | 17,206.26 | 391.457 | 806 |
  | Pearl | TP=4 | 0.8180 | 13,264.38 | 507.789 | 620 |
  | Pearl | DP=4 | 0.8198 | 18,291.66 | 368.229 | 981 |

  模型卡没有说明 MMLU 的 shot 数、prompt、输入/输出长度、并发数，也没有说明测试时挖矿是否开启。
- **Qwen3-30B-A3B-Instruct-2507-pearl**（[模型卡](https://huggingface.co/pearl-ai/Qwen3-30B-A3B-Instruct-2507-pearl)）：MMLU-Pro（12,032 题，5-shot CoT 生成式，`temperature=0.7, top_p=0.8, top_k=20`，lm-eval-harness on vLLM）：原版 77.88%，Pearl 77.33%。
- **Gemma-4-31B-it-pearl**（[模型卡](https://huggingface.co/pearl-ai/Gemma-4-31B-it-pearl)，lmms-eval + vLLM，全测试集）：

  | 版本 | GPQA | MMLU | HumanEval | MGSM3 | MMMU-Pro Vision | Video-MME(short) |
  |---|---|---|---|---|---|---|
  | 原版 | 77.27 | 90.93 | 94.70 | 88.62 | 54.57 | 79.0 |
  | Pearl | 77.37 | 90.56 | 94.15 | 89.09 | 54.45 | 78.2 |

- **Llama-3.1-8B-Instruct-pearl**：模型卡**没有给出任何精度或基准数字（未找到）**。
- 社区独立的精度复现：**未找到**。
- **【代码】仓库内的精度测试**：`vllm-miner/tests/test_vllm_execution.py` 用 8B 和 Qwen3 MoE 对比挖矿与不挖矿的生成结果，参考输出在 `reference_outputs.json`。kernel 级测试的容差为 `atol 0.1 / rtol 0.01`（`pearl-gemm/tests/test_pearl_gemm.py:704-708`）。没有模型级 benchmark 脚本。

---

## 6. 中奖后的处理

1. **【代码】GPU 端**：tile 的哈希 ≤ 目标时，在全局锁内写 `HostSignalHeader`（包括 status、tileCoord、每线程的行列坐标 `thread_rows/cols`、mma 尺寸、target）。header 用一次拷贝写进 pinned 内存，只记录第一次命中（`pow_utils.hpp:239-330`）。
2. **【代码】StatusCheckCallback**（`callbacks.py:21-64`）：
   - 由后台协程在 CUDA event 完成后调用（`async_loop_manager.py:225-244`）；
   - 若 `status == kSignalTriggered`：`extract_indices` 把 tileCoord 和线程坐标换算成 A 的行号、B 的列号（`pearl_gemm/helpers.py:73-82`）；
   - 把 `seedA/seedB` 从 GPU 拷回；
   - **`A.cpu()` 复制整个 m×k 激活，`B.cpu()` 复制整个 n×k 权重分片**（`callbacks.py:49-52`），然后放入 `OpenedBlockInfo` 交给 `handle_submit_block`；
   - 无论是否命中，最后都归还 pinned header（`:58`）。
3. **【代码】提交线程**（`async_loop_manager.py:120-146, 250-294`）：在 ThreadPoolExecutor 中执行 `create_proof`（`block_submission.py:14-62`）：
   - 在 **CPU** 上对完整的 A 和 Bᵀ 重新建 BLAKE3 Merkle 树（`MatrixMerkleTree`，通过 Rust 绑定 `pearl_mining`）；
   - 生成被打开的行和列的 multi-leaf proof；
   - 组装 `PlainProof(m,n,k,noise_rank,a_merkle_proof,bt_merkle_proof,moe)`；
   - 执行 `check_cert_version_eligible`，不合格就跳过；debug 模式下还会本地 `verify_plain_proof_for_cert_version`；
   - 新建一个 RPC 客户端发出 `submitPlainProof`。
4. **【代码】矿工与 gateway 的接口**（JSON-RPC 2.0，换行分隔，默认走 UDS `/tmp/pearlgw.sock`，也可用 TCP 127.0.0.1:8337；`pearl_gateway/config.py:27-49`；`miner_rpc/server.py:188-235`；`miner_rpc/schemas.py`）：
   - `getMiningInfo`，无参数，返回 `MiningJob.to_dict()`，包含 `{incomplete_header_bytes: base64, target: int, cert_version: int}`（`comm/dataclasses.py:176-201`）。没有模板时返回错误 `-32001 mining_paused`（`:228-236`）。
   - `submitPlainProof`，参数为 `{plain_proof: base64(PlainProof), mining_job: {incomplete_header_bytes, target, cert_version}}`（`schemas.py:27-46`；`gateway_client.py:31-41`）。服务端**立即返回 "submitted"**，然后异步处理（`server.py:214-220`）。如果 `mining_job.incomplete_header_bytes` 与当前模板不一致，就丢弃（`:247-252`）。
   - gateway 处理流程：`SubmissionService` 加锁、去重，检查 `check_cert_version_eligible`，然后 `ProofGenerator.generate_block`，即**在 gateway 端由 PlainProof 生成 ZK 证明**（`generate_proof_for_cert_version`，`proof_generator.py:20-57`）。最后拼出区块并 `submitblock`（`submission_service.py:29-74`）。
5. **【代码】轮询与换任务**：
   - **矿工端**：`_update_gateway_loop` 每 1 秒调用一次 `getMiningInfo`；如果与旧任务不同就替换 `_mining_job` 并触发回调（`async_loop_manager.py:200-223`）。前向里直接读当前对象（`:114-115`），没有锁。
   - **gateway 端**：`TemplateScheduler` 每 `refresh_interval_seconds=1` 秒（`config.py:23`，环境变量 `PEARLD_REFRESH_INTERVAL_SECONDS`）调用一次 `getblocktemplate`（`scheduler.py:56-110`；`pearl_client.py:93-97`）。
   - `WorkCache.update_template` **只在 `previous_block_hash` 变化时替换模板**（`work_cache.py:22-42`）。因此同一高度期间，timestamp、交易集合和 coinbase 保持不变，`MiningJob` 也不变。
   - **【理解】** 后果是：同一高度内到达节点的新交易，要到下一块才进入模板；矿工的 `incomplete_header_bytes` 在一个区块周期内固定，job_key 也随之固定。

---

## 7. 官方开销数据与 Basu 论文对照

### 7.1 官方说法

- **【外部】** [pearlresearch.ai/research/blog/proof-of-useful-work](https://pearlresearch.ai/research/blog/proof-of-useful-work) 的 "Measured in production" 部分：
  - "+5.08% on Llama 70B (DP=4 on 4×H200)"
  - "+3.9% on DeepSeek V3.2 (DP=8 + EP on 8×H200)"
  - 页面称之为 "end-to-end overhead versus stock serving engines"。
  - 页面**没有给出**：负载与提示/输出长度、批大小与并发、挖矿是否开启、指标定义（吞吐还是时延）、测量时长。同页另有理论开销 O(r/n) 的分析性说法（"a fraction of a percent"），不是实测。
  - 转述来源：[TokenPost](https://www.tokenpost.com/news/technology/24270) 写作 "Pearl measured overhead of 5.08% on Llama 70B using four H200 GPUs and 3.9% on DeepSeek V3.2 using eight H200 GPUs"，没有给出处。[TECHi](https://www.techi.com/pearl-prl-search-surge-ai-mining-together-api/) 明确表示没有自行复现。
- **【外部/理解】与 HF 70B 模型卡对照**：模型卡 DP=4 一行里，Meta 原版是 **OOM**，没有可比的原版基线；Pearl 版 DP=4 相对 Meta 的 PP=4 和 DP2/TP2 反而**更快**（18,292 vs 15,270 / 13,162 tok/s），因为 7/8bit 权重让单卡放得下 70B。所以 "+5.08% (DP=4)" 的基线**应该**是 "同一 Pearl 量化模型关闭挖矿" 或 "其他 W8A8 引擎"，页面没有说明。【理解】这一点未经验证。
- **【外部】DeepSeek V3.2**：HF 上 pearl-ai 组织只有 4 个模型（Gemma-4-31B、Qwen3-30B-A3B、Llama-3.3-70B、Llama-3.1-8B），**没有 DeepSeek checkpoint**。master 的 MoE 路径只支持 "int7 gate/up 加 FP8 block down" 的 FusedMoE（`vllm_config.py:142-166`）；DeepSeek V3.2 的 MLA/DSA 注意力是否被覆盖，仓库中没有相关说明。
- **【代码】仓库中的可复现基准**：
  - `pytest -m performance` 的标记（`pyproject.toml:49`）**只有** `pearl-gateway/tests/miner_rpc/test_performance.py` 一处在用（RPC 吞吐）；
  - **没有任何 vLLM 端到端吞吐或开销基准脚本**，也没有 MMLU 评测脚本；
  - 全仓库 grep `5.08`、`DeepSeek` 都没有结果。
  - 结论：**官方开销数据在仓库内无法复现**。

### 7.2 Basu 论文（arXiv 2606.04819，`papers/usefulness_gap_2606.04819.txt`）

- **【外部】第三方挖矿程序**：
  - 作者对闭源的 **alpha-miner v1.6** 做字符串分析：AI 框架相关字符串为 0（vllm、transformer、llama、checkpoint、safetensor），GEMM/挖矿相关字符串有 4,803 个，还包含 `matrix seed=0x`（`:250-254`）。
  - 运行时观测：RTX 3090 和 MI300X 上显存占用 **2.7–3.7 GB**，计算利用率 95–100%，显存带宽利用率 <30%。作者据此认为这是纯 GEMM 的特征，并对比说推理会占用 16–40 GB 显存（`:255-268`）。作者自己承认字符串分析和显存证据 "suggestive but not definitive"（`:263-268`, `:704-705`）。
  - 作者还写了开源的 stripped miner（随机矩阵，不做推理）：RTX 3090 为 2.85M tiles/s，约为 alpha-miner 3.29M 的 87%；MI300X（Triton/ROCm）为 10.6M tiles/s，获得 41 个被接受的 share；另有 CPU 和 Apple M2 数据（`:425-441`）。
  - 本地实验：20/20 份随机矩阵证明在降低难度后通过了 `verify_plain_proof()`（`:205-215`）。
- **【外部】"效率损失 10–30%"**：原文是 "absence of inference overhead (estimated 10–30% hashrate reduction)"（`:88`, `:332-334`, `:405-406`）。论文**没有给出这个区间的推导、实验或引用**，只是作为 "estimated" 出现在经济分析里。
- **【外部】需要注意**：论文中的 "5.08 EH/s" 是 AlphaPool 的算力（"5.08 EH/s of 24 EH/s"，`:219-220`），**与官方 "+5.08%" 开销是两回事**，只是数字巧合，引用时容易混淆。

### 7.3 对照

| 维度 | 官方 | Basu | 本笔记的代码与估算 |
|---|---|---|---|
| 开销 | +5.08%（70B, DP4, 4×H200）、+3.9%（DSv3.2），端到端，条件不详 | 推理耦合使算力下降 10–30%（估计，无依据） | 挖矿 GEMM 自身额外张量核工作量约 10–53%（视形状和 m 而定，§2）；但只有约 67% 的线性 MAC、且只在 m≥1024 的步才挖，所以端到端被稀释。两者口径都不清楚，**无法从仓库复现** |
| 两者的口径差异 | "服务引擎在开启挖矿时比不开慢多少" | "只挖矿的程序在加上推理后算力掉多少" | 【理解】两者衡量的方向相反，数值不能直接比较 |
| 显存 | 未提 | 第三方挖矿程序只占 2.7–3.7 GB | 官方路径需要装下整个模型、KV cache，以及每次挖矿 GEMM 的 B 拷贝和 BpEB 等瞬时缓冲（§2） |

---

## 8. pearl-gemm 的编译目标与第三方硬件支持

- **【代码】** master 只编译 `sm_90a`（`pearl-gemm/setup.py:89`），内核大量使用 SM90 的 WGMMA 和 TMA（`kernel_traits.hpp`、`collective_epilogue.hpp`），运行时要求 capability ≥ 9（`vllm_scheme.py:57-60`；`vllm_kernels.py:62-89`）。由于没有为 sm_100 或 sm_120 生成代码，【理解】Blackwell 卡上无法加载 master 的 `pearl_gemm_cuda`。
- **【代码/git】官方分支**：`origin/fp8-scheme` 是新的 FP8/FP10 方案。它支持 SM100/SM120（`85216510`, 2026-09-28），并有提交 "Hopper (H100) support (#337)"（`0b18ef42`）。截至 `156ef9a8` 未合入 master。
- **【外部】第三方挖矿程序**（都只做"纯挖矿"，不跑推理）：
  - **alpha-miner**（AlphaMine-Tech，[GitHub](https://github.com/AlphaMine-Tech/alpha-miner)）：README 写 "One Linux archive supports NVIDIA compute capabilities 8.6, 8.9, 9.0, and 12.0"，即 Ampere 消费卡、Ada、Hopper 和消费级 Blackwell；最新 v1.9.3；以二进制形式发布，附带 "ALPHAMINE-PUBLIC-TEST-LICENSE"。Basu 论文用的是 v1.6。
  - **MoneroOcean mo-miner**（[GitHub](https://github.com/MoneroOcean/mo-miner)）：开源，Node.js/SYCL，基于 xmrig，支持算法 `pearlhash`。README 列出的 PearlHash 实测：Intel Arc B580 86.75 TH/s，A770 89.07 TH/s，RTX 5060 Ti 80.49 TH/s，AMD RX 9060 XT 42.88 TH/s；显存建议 "2 GiB minimum; 8 GiB recommended"。相关 PR 包括 "Fix PearlHash V3 seeds and rank-penalized share targets"（[PR #5](https://github.com/MoneroOcean/mo-miner/pull/5)），说明它在跟进共识变更。
  - **Kryptex 矿池教程**（[how-to-mine-pearl](https://pool.kryptex.com/articles/how-to-mine-pearl-en)）：NVIDIA "RTX 2000-series and newer — RTX 3000-series and newer work best"；AMD RDNA2（RX 6600–6950）用 SRBMiner-MULTI；Intel Arc 用 ARCMiner；仅支持 NVIDIA 的程序还有 PeakMiner 和 ForgeMiner。
  - **Basu 的开源 stripped miner**：NVIDIA、AMD MI300X（Triton/ROCm）、CPU、Apple Metal（论文 Table 8，`:425-433`）。
  - 小结：**官方有用挖矿栈 = 仅 Hopper（master）且必须跑 vLLM 加 pearl 模型；第三方纯挖矿 = 覆盖 Turing/Ampere 到 Blackwell 的消费卡、AMD、Intel，以及 CPU/Apple（仅论文演示）**。

---

## 9. 疑点（留待问题阶段分析，此处只列不展开）

1. 纯 decode 步只有在并发数恰好 ≥1024（等于 vLLM 默认 `max_num_seqs`）时才挖矿；短提示、长输出的负载里，有用挖矿覆盖率可能接近 0（§4.5–4.6）。
2. 已发布的 checkpoint 中，down_proj、前一半层的 qkv、lm_head 和注意力都不挖，可挖 MAC 的上限约 67%（§4.3），与"推理 MAC 都在挖矿"的宣传口径之间的关系待核。
3. 插件阈值 `min_k=1024` 与 verifier 的 `k ≥ 16r = 2048` 不一致；在 TP=4（8B）或 TP=8（70B）下 o_proj 会做无效挖矿（§3）。
4. 每次挖矿 GEMM 都对整份权重重新复制、哈希和加噪（没有按 job 缓存），在 m 较小时额外工作量很大（§2）；新分支已改为按 job 预处理。
5. 去噪使用 fp16 因子和 fp32 累加，输出与无噪声 GEMM 不保证逐位一致；kernel 测试容差为 0.1/1%（§2 第 12 步）。
6. 7bit 量化的层不论是否挖矿都承受 7bit 精度损失；8B 没有公布任何精度数据；已公布的精度数据（MMLU、MMLU-Pro 等）缺少评测细节，也没有独立复现（§5.3）。
7. 官方 +5.08% 和 +3.9% 的测试条件、基线和指标未公开；仓库中没有可复现脚本；70B DP=4 的原版基线是 OOM；DeepSeek V3.2 没有公开 checkpoint（§7.1）。
8. Basu 的 "10–30%" 没有给出依据；它与官方数字口径相反；论文中的 "5.08 EH/s" 与官方 "5.08%" 容易混淆（§7.2）。
9. master 只支持 sm_90a，而第三方纯挖矿程序覆盖消费卡、AMD、Intel，硬件门槛不对称（§8）。
10. gateway 只在 prev hash 变化时更新模板：同一高度内不纳入新交易，timestamp 固定（§6）。
11. 中奖后要把整份 A 和 B 复制到 CPU、在 CPU 上重建 Merkle 树，再由 gateway 生成 ZK 证明；`submitPlainProof` 立即返回 "submitted"，提交结果不回传给矿工（§6）。
12. 没有 CUDA Graph 防护，官方示例必须加 `--enforce-eager`，这本身就会损失 decode 性能，"stock serving engine" 基线是否也用了 eager 不明（§1.3）。
13. MoE 用整批 token 数判断是否挖矿，即使单个专家只分到很少行也会跑 noisy_gemm；每层每次前向最多提交一个专家的中奖（§1.2）。
14. pinned 池（128 个）耗尽时，信号量会阻塞推理线程（§2 第 13 步）。
