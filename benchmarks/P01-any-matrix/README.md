# P01 benchmark：任意矩阵都能挖矿出块

问题报告：[`docs/problems/P01-any-matrix-mines.md`](../../docs/problems/P01-any-matrix-mines.md)

## 做什么

用 Pearl 自己的 `mine()` 分别以 4 种矩阵挖矿：随机数 [−64, 64]、全零、全 7、全 −64。每个结果依次通过一个区块上链要过的全部检查：

| 检查 | Pearl 函数 |
|---|---|
| 原始证明验证（矿池验份额） | `verify::verify_plain_proof` |
| 证书版本 | `check_cert_version_eligible(3)` |
| 零知识证明生成 + 节点共识验证 | `prove::zk_prove_plain_proof` → `verify::verify_block` |
| 秩惩罚 | `sanity_checks::check_rank_penalty` |

参数：m=256、n=512、k=2048、r=128、小块 4×8（节点默认配置），V3 加盐种子，nbits=0x1E01FFFF（低难度）。

## 运行

先在仓库根目录执行 `./scripts/setup-pearl.sh`，然后：

```bash
./benchmarks/P01-any-matrix/run.sh
```

```bash
./benchmarks/P01-any-matrix/run.sh plain
```

- 默认模式包含零知识证明，约 20 秒，需要约 6 GB 内存；`plain` 只跑原始证明验证。
- 脚本把 `p01_any_matrix.rs` 复制到 `$PEARL_DIR/zk-pow/examples/` 后编译运行，不修改 Pearl 的其他文件。`PEARL_DIR` 默认是 `third_party/pearl`。
- 输出写到 `results/run-<时间>.log`（不提交到 git）。

## 文件

| 文件 | 说明 |
|---|---|
| `p01_any_matrix.rs` | benchmark 程序（Pearl example） |
| `run.sh` | 编译并运行，保存日志 |
| `results/reference-apple-m4.log` | 参考结果：Apple M4、16 GB、Rust 1.88.0，含 `/usr/bin/time -l` 资源统计 |

## 判读

每组都应看到 `verify_plain_proof: ACCEPTED`、`verify_block (ZK, consensus path): ACCEPTED`、`check_rank_penalty: ok`。挖矿结果是确定的，各组的 Merkle 根与中奖坐标应与参考日志一致；耗时随机器不同。
