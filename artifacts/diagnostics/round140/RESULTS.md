# R140 — DiT 单步 roofline 探测（10 s/step 目标的可行性依据）

用户 2026-11-23 把 DiT 目标上调为 **≤10 s/step**。本轮为纯测量：复用 R126 已验证的 replay harness
（同一 checkpoint、同一 `same_initial_sampling123` bundle、step 25、TP2/USP4、`--offload native`、
`--mode profile`），**不安装候选、不改数学、不采用 tolerance**。

## 实测（8 rank 全部完成，warm rank-max）

step = **19.906 s**（跨 rank 中位 19.914 s），peak allocated 7.13–7.15 GiB。

| 区域 | device 时间 | 占步比 | 该区域 FLOPs/rank | 达成 | 占 310 TF 峰值 |
|---|---|---|---|---|---|
| **FA 实际 kernel**（50 次调用） | **7.454 s** | 37.4% | 313.0 TF | 42.0 TFLOPS/卡 | **13.5%** |
| attention 合计（含 FA+USP+q/k norm+qkv/o） | 12.298 s | 61.8% | 330.6 TF | 26.9 TFLOPS/卡 | 8.7% |
| mlp（gate/up/down） | 4.420 s | 22.2% | 120.8 TF | 27.3 TFLOPS/卡 | 8.8% |
| adaln | 0.295 s | 1.5% | — | — | — |
| norm1+norm2 | 0.187 s | 0.9% | — | — | — |
| misc（refiner/patch/final/time） | 0.161 s | 0.8% | — | — | — |
| **未归因**（块间切换、hook 外的 collective、采样器） | **2.545 s** | 12.8% | — | — | — |

全步建模 FLOPs ≈ **451.4 TF/rank/step** → 实测
**22.7 TFLOPS/卡（≈7.3% peak）**。

## 10 s 意味着什么

| 目标 | 需要 | 相对今天 |
|---|---|---|
| **10 s/step** | **45.1 TFLOPS/卡 ≈ 14.6% peak** | **×2.0** |
| （对比）5 s/step | ≈96 TFLOPS/卡 ≈31% peak | ×4.2 |

即 10 s 只需要把整机效率从 7.3% 提到 14.6%——**两个 2× 就够**，不是重写全部 kernel。

## 权重 H2D 的硬底

- 每步要搬 **32.28 GB** 权重；本机 pinned→NPU 实测 **21.81 GB/s**
  → 纯传输 **1.480 s/step**，占今天 7.4%、
  **占 10 s 预算 14.8%**。
- prefetch 拷贝流实测跨 **7.62 s**（201 次调用）。
  说明 staging 并没有被计算完全吃掉；10 s 里必须把它压到 ≤1.5 s 并真正重叠，否则预算直接被吃掉近 15%。

## 由 roofline 得到的候选排序（下一步）

1. **FA（#5）**：7.454 s、13.5% MFU，单步最大单项。**2× → 省 3.7 s**；这是 10 s 路径上最关键的一步。
2. **MLP/GEMM（#6）**：4.42 s、8.8% MFU。**2× → 省 2.2 s**。
3. **未归因 2.545 s**：需要先拆分（块间切换 / hook 外 collective / 采样器更新），其中可能有可观的纯开销。
4. **H2D 重叠（#4）**：硬底 1.48 s，必须隐藏；这也解释了为什么 #4 从"可选"变成 10 s 的前置项。

粗算 10 s 预算：FA 3.5 + mlp 2.0 + 其余 ~1.0 + 未归因 ~1.5 + H2D（重叠后）≈ 8–10 s，**可达但需要 FA 与 GEMM 各自 2×**。

## 方法与限制

- 区域计时用 `synchronize`，属 coarse 归因；父子区间包含、**不相加**。FA 用的是 R126 同一 kernel 的 device 时间
  （本轮 attn 区域内含它，未单独重测）。
- 峰值 310 TFLOPS 是 910B 量级的**数量级假设**，不是厂商报价；MFU 只用于比较与排序。
- 本探测不改数学、未安装任何候选；结论**不构成任何数值批准**，后续候选仍需用户逐项批准。
