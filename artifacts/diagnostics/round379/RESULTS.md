# R379 — B 方案（集合通信/计算流水）实测：机制在模型内不生效，净收益为 0

## 一、B 的设计与实现

`RowParallelLinear.forward` 行切成 4 块，每块 GEMM 后立刻发 `dist.all_reduce(..., async_op=True)`，
用下一块 GEMM 隐藏通信，最后 `cat`。逐元素求和不变 ⇒ 理论上 bit-identical。

**离线微基准（2 rank，真实形状）**：方案成立且 bit_equal=True

| 层 | 基线（GEMM+同步 AR） | 4 块流水 | 节省 |
|---|---|---|---|
| out_proj (K=3584) | 12.55 ms | 7.69 ms | 4.98 ms |
| fc2 (K=7168) | 18.82 ms | 8.68 ms | 10.14 ms |

`async_op` 重叠也单独验证过：顺序 22.9 ms → 重叠 12.6 ms。

## 二、整模实测：无收益（15.5436 vs 15.5489，噪声内）

profiling 对比（稳态 forward）：

| 指标 | 无 B | 有 B |
|---|---|---|
| allReduce 次数 | 98 | **245**（out_proj 被切成 4 份） |
| allReduce 总时长 | 1.256 s | 1.309 s |
| **集合通信与计算重叠比例** | 0% | **6.9%** |
| 暴露的集合通信时间 | 2.235 s | 2.194 s |

⇒ **模型内 async allreduce 几乎不与计算重叠**（只有 6.9%），切块反而增加 245 次小通信的开销，
两者抵消，净收益 0。

## 三、原因分析

1. 离线探针用的是**默认 2-rank 进程组**，async 重叠正常；模型里用的是 **vLLM TP 子组**
   （`get_tp_group().device_group`），子组的 async 语义可能不同，或 vLLM 的执行上下文/流绑定让它串行；
2. `fc2` 的输入是 **fp32**（`silu(gate.float()) * up.float()`），被我的 `x.dtype==fp16` 条件排除，
   因此只有 out_proj 被切块 —— 而 out_proj 的 GEMM 只有 2.7 ms，可隐藏量本来就小；
3. 每次调用多加一次 `abs().max()`（同步）也抵掉了一部分收益。

## 四、结论与建议

- **B 在当前 vLLM/HCCL 执行环境下不可行**（async 重叠不生效），不建议继续投入；
- 集合通信 2.235 s（14.7%）的剩余选项只有：
  - **fp16 allreduce**（224→112 MB，−0.58 s，需用户批准数值变化）；
  - 或接受现状。
- 当前最优交付组合（FA V1 + 快速审计 + 快速 GEMM）保持在 **15.5489 s/step**，
  B 的补丁全部 env-gated（`H3_OVERLAP_AR=1`），默认关闭，不影响交付。
