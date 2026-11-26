# R376 — 整模 op 级端到端归因（msprof `--task-time=l1`，稳态 forward = 15.20 s 窗口）

## 一、测量方式

- 候选配置（V1 FA + 128/384）跑 1 个被测步，`msprof --task-time=l1`，torchrun 8 rank；
- 用 **FA 内核出现位置**切分 forward：每 forward 50 个 FA kernel，窗口 = [首个 FA 开始, 末个 FA 结束]；
- 稳态 forward 窗口 15.20 s，窗口内 device-busy 15.06 s ⇒ 基本无气泡，**瓶颈是 op 本身**。

## 二、稳态 forward 归因（device 1）

| 类别 | 时间 | 占 wall | 次数 | 备注 |
|---|---|---|---|---|
| **FA 内核**（V1 128/384） | **5.805 s** | **38.2%** | 50 | 116 ms/call |
| **Cast** | **2.172 s** | **14.3%** | 2402 | 48 次/block，单次 ~0.9 ms |
| **MatMul** | **2.002 s** | **13.2%** | 147 | GEMM |
| **hcom_allReduce** | **1.256 s** | **8.3%** | 98 | TP 集合通信 |
| hcom_alltoall | 0.978 s | 6.4% | 196 | USP 相关 |
| MatMulV2 | 0.607 s | 4.0% | 98 | aclnn Mm |
| Mul | 0.348 s | 2.3% | 933 | elementwise |
| ReduceMaxD | 0.227 s | 1.5% | 100 | |
| Add | 0.170 s | 1.1% | 637 | |
| Equal | 0.148 s | 1.0% | 294 | |
| Transpose | 0.111 s | 0.7% | 637 | |
| ReduceMeanD | 0.098 s | 0.6% | 196 | |

合计约 13.9 s；余量为未列出的小 op。

## 三、结论与下一批优化目标（按占比 × 可改性）

1. **FA 38.2%（5.8 s）** — 仍是最大单项，已从 137.8→128.8 ms/call，继续按 pipe 数据压；
2. **Cast 14.3%（2.17 s，2402 次）** — 最大非 FA 单项。FP16 矩阵 + FP32 残差/归一化是硬约束，
   但 48 次/block 里很可能存在**重复 cast**（同一张量多次转换）或可融合的转换；
3. **MatMul+MatMulV2 17.2%（2.6 s）** — GEMM 形状/TP 切分/对齐问题；
4. **集合通信 14.7%（2.23 s）** — allReduce 98 次、alltoall 196 次，需确认是否与计算重叠；
5. 小算子群（Mul/Add/Equal/Transpose/Reduce*）合计约 1.1 s — 多为 elementwise，可融合。

## 四、重要副产物

- forward 0 窗口里出现 **4.935 s / 4.466 s 的单次 allReduce**（首轮，含权重同步/预热），
  稳态 forward 里 allReduce 恢复到 1.256 s —— 说明**首次 forward 有一次性开销**，测步时需丢弃首轮。
- 该表是后续每轮优化的对照基线：任何改动都应让对应条目下降，且总 wall 下降。

## 五、下一步（择一或并行）

- **A（首选）**：查 2402 次 Cast 的来源（模型 block 源码里的 `.float()/.half()` 与 norm/residual 路径），
  找出重复/可融合的转换；预期可省 0.5–1.0 s/step；
- **B**：allReduce/alltoall 与计算的重叠情况（是否串行等待）；
- **C**：继续 FA 内核（38% 仍是最大项）。
