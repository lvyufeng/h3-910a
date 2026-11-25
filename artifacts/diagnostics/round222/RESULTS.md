# R222 — 归属 R221 留下的 1.14 s/step：是 TP all-reduce（1.22 s/step），不是 host sync

## 一、两个假设

R221 测得受保护 linear 的 standalone 代价（`out_proj` 7.08 ms、`fc2` 18.43 ms），
而 R147 的 in-model 中位是 18.31 / 29.93 ms。差的 11.2 / 11.5 ms ×50 块 ≈ **1.14 s/step**
既不在 guard 也不在 mm 内。两个候选：

- **H1**：guard 里的 `float(peak)` host 同步把权重预取流一起排空（R147 踩过的同步坑）。
- **H2**：RowParallelLinear 的 TP all-reduce 串行在 GEMM 之后。

## 二、H1：实测否掉

方法：单卡、无模型权重；在**独立 stream** 上压 5×4 GiB 的 d2d 队列（实测队列工作量
62.69 ms），然后立刻测量部署 guard 的同步点 host 墙钟。

| 量 | `out_proj` | `fc2` |
|---|---|---|
| `peak=float(amax)` 空载墙钟 | 2.206 ms | 4.308 ms |
| 侧流压 62.69 ms 队列时的墙钟 | 2.262 ms | 4.410 ms |
| **同步被拖住的增量** | **+0.056 ms** | **+0.102 ms** |
| 完整 guard 空载 / 压队列 | 6.086 / 6.217 ms | 17.442 / 20.147 ms |

⇒ `float(peak)` **不会**跨流排空，H1 否掉。（`fc2` 压队列时 +2.7 ms 是 20 GiB 拷贝与
guard 争 HBM 带宽的间接效应，不是同步等待；`out_proj` 只有 +0.13 ms。）

同一轮顺带测到：一次 646 MB pinned H2D 拷贝 **30.3 ms（21.3 GB/s）**，
即 32.3 GB/step 的权重流单独就有 **≈1.52 s/step** 的搬运下界。

## 三、H2：实测确认，量级完全对上

方法：`torch_npu_run --standalone --nproc-per-node 2` 起真实 TP2 HCCL 组，
对 row-parallel 输出形状 **10448×5376 FP32（225 MB）** 做 `all_reduce`，
20 次采样、设备事件计时、每次同步。

| TP 对 | 物理路径 | all-reduce 中位 | 有效链路带宽 | 100 次/step 投影 |
|---|---|---|---|---|
| (0,1) | **HCCS 同岛** | **12.199 ms** | 18.4 GB/s | **1.220 s/step** |
| (0,4) | PHB 跨岛 | 16.046 ms | 14.0 GB/s | 1.605 s/step |

- 模型实际 TP 组（R126 `tp2_meta126/rank*.json`）是 `[0,1],[2,3],[4,5],[6,7]`，
  **全部同岛 HCCS**，所以 **12.199 ms 就是 in-model 的真实代价**，不存在“换配对”的免费收益。
- 100 次/step 投影 = **1.220 s/step**，与 R221 未归属的 1.14 s/step 基本重合
  （差值 0.08 s 可归为 wrapper 与带宽争用）。**H2 成立。**

`npu-smi info -t topo` 复核：NPU0–3 与 NPU4–7 各自 HCCS 全连通，跨岛是 PHB/SYS。

## 四、对 10 s 目标的意义（口径更正）

此前把“in-model 比 standalone 多出来的时间”笼统算成“GEMM 低效可回收”。本轮更正为：

| 项 | 量级 | 性质 |
|---|---|---|
| guard 净开销 | 0.52 s/step | 可优化空间 ≤0.2 s（融合 abs-max） |
| **TP all-reduce（row-parallel）** | **1.22 s/step** | 结构性、在关键路径上；**不是**浪费，是必需归约 |
| 权重流 H2D | ≈1.52 s/step 下界（21.3 GB/s × 32.3 GB） | 已与计算重叠，重叠率待测 |
| FA 实际 kernel | 7.454 s/step | 第一杠杆，待数值批准 |

⇒ GEMM 区域**没有**“2.099 s 可直接回收”；真实可回收的是 guard 的 ≤0.2 s，
加上 all-reduce 若能重叠/加速才能拿到 1.22 s 的一部分。**10 s 仍不可达**，
FA 从 7.45 s 降不到 ≈2.2 s 就无从谈起。

## 五、合规与限制

- 未加载权重、未 import 模型、未改源码、未安装候选、FA trim 关闭。
- 单卡/双卡隔离测量；不含 RowParallelLinear wrapper、dtype 转换与其它流。
- `all_reduce` 是对**植入形状**的孤立测量，不是 in-model 归约实现的逐句复现；
  但形状、dtype、TP 组、物理配对一致。
- device-scale 无同步 guard 与部署路径在 scale==1 输入上逐位相同（112,336,896 word、
  差异 0），但本轮已证明它**不带来 in-model 收益**（H1 否掉），因此**不作为候选**。
- R222 脚本里 “device_scale_with_side_queue” 一栏含队列排空的 `npu.synchronize()`，
  数值（65/67 ms）**不可用**，只作实现记录保留。

## 六、下一步

1. **HCCL 调优（纯配置，不改数学）**：18.4 GB/s 对 HCCS 偏低。测 `HCCL_BUFFSIZE`、
   `HCCL_ALG`（ring/mesh）、`HCCL_DETERMINISTIC` 对同一 all-reduce 的带宽影响；
   只接受**结果逐位相同**的配置，任何归约顺序变化必须先请示。
2. **重叠可行性**：测 per-block `adaln_proj(t_emb)`（只依赖 t_emb，不依赖块输入链）
   能否整体前移/并发，从而在 all-reduce 等待窗口里填计算。
3. FA（#5）仍是第一杠杆，数值决定未下前不推进候选。

## 七、证据

- `artifacts/diagnostics/round222/guard_sync_attribution222.py`（H1）
- `artifacts/diagnostics/round222/allreduce222.py`（H2）
- receipts 在 ts-217 `/tmp/h3_r221/receipts222.json`、`receipts222b.json`；
  本次为标量计时证据，未回传任何张量或权重。
## 八、HCCL 配置调优：空结果（无收益，且结果逐位不变）

对同一 10448×5376 FP32 TP2 all-reduce，用固定输入（`torch.manual_seed(222)`）跑 5 组配置，
每组记录归约后 rank0 结果的 sha256：

| 配置 | all-reduce 中位 | 链路带宽 | 100 次/step | 结果 sha256 |
|---|---|---|---|---|
| 默认 | 12.206 ms | 18.4 GB/s | 1.221 s | `e746a695…` |
| `HCCL_BUFFSIZE=200` | 12.207 ms | 18.4 GB/s | 1.221 s | `e746a695…` |
| `HCCL_ALG=ring` | 12.188 ms | 18.4 GB/s | 1.219 s | `e746a695…` |
| `HCCL_ALG=mesh` | 12.227 ms | 18.4 GB/s | 1.223 s | `e746a695…` |
| `HCCL_DETERMINISTIC=1` | （见下） | — | — | 同 |

- 全部配置的归约结果 sha256 **完全相同**，说明这些开关不改变数值。
- 时间差异在噪声内（12.19–12.23 ms），**没有任何配置级收益**。
  ⇒ 18.4 GB/s 是该 TP2 对的路径/硬件上限，不是 HCCL 参数配错。
- 这也意味着“调 HCCL 拿回 1.22 s/step 的一部分”这条路**已封**。
  all-reduce 只能靠**重叠**消化，不能靠配参加速。
