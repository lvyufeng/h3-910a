# FA kernel V1 — `PV(j)` 提到 `QK(j+1)` 之前（用户已批准 1-ulp 数值变化）

## 改动（相对交付基线）

唯一改动在 `compute_in_each_kv_block`：把当前块的 PV matmul 挪到下一块 QK matmul **之前**。

```diff
-        # do Sij = matmul(Qi, Kj.T) for next inner loop
-        with self.tik_instance.if_scope(kv_blk_idx < self.Tc - 2):
-            ... load K(j+1); Sij = matmul(Qi, KjT, ..., workspace=Sij_ub_zN)
-        Oi_l1_zN = self.output_compute(Oi_l1_zN, Pij_l1_zN, Vj_l1_zN, ...)
+        Oi_l1_zN = self.output_compute(Oi_l1_zN, Pij_l1_zN, Vj_l1_zN, ...)
+        # do Sij = matmul(Qi, Kj.T) for next inner loop
+        with self.tik_instance.if_scope(kv_blk_idx < self.Tc - 2):
+            ... load K(j+1); Sij = matmul(Qi, KjT, ..., workspace=Sij_ub_zN)
```

`flash_attention_fwd.py` 是本文件的完整版本；与交付基线 diff 仅 6 行（上述搬移）。

## 为什么有效（msprof `PipeUtilization` 证据，R374）

| 配置 | 内核时间 | vec | mac | scalar | mte2 | memory_bound |
|---|---|---|---|---|---|---|
| 基线 64/912 | 135.4 ms | 36.9% | 28.4% | 12.4% | 20.1% | **54.5%** |
| V1 128/384 | **126.5 ms** | 49.1% | 27.1% | 21.6% | 11.7% | 23.8% |

基线无单一 pipe 饱和、`memory_bound 54.5%` ⇒ 访存+依赖混合受限。Br 64→128 把 K/V 重读量减半
（mte2 20.1%→11.7%），但只有在 PV 前置后 128 的 PV（M=128）才不与下一块 QK 抢 L0C。

## 部署方式（opt-in，不改共享 SDK）

```bash
export H3_PERF_COMBO=1 H3_PERF_FA_TILES=1 H3_PERF_FA_TRIM=1 H3_PERF_GUARD_FAST=1
export H3_FA215_PKG=<包含本文件的 FA 包根>      # 内核源码（按需构建）
export H3_FA215_KERNEL_DIR=<内核缓存目录>
export H3_FA_BR_CAP=128 H3_FA_BC_CAP=384
```

## 实测

| 项 | 值 |
|---|---|
| FA 单次（N=41792, B=1,H=7,D=128） | 基线 137.79 ms → **V1 128/384 = 128.81 ms**（−6.5%） |
| Bc 敏感性 | 384/448/480 均 128.8 ms |
| 数值 vs 基线 | relRMS **1.290e-04**，max_abs 4.768e-07（1 个 fp16 ulp），6.86% 元素差 1 ulp |
| 用户裁决 | **允许**（2026-11-27） |
| Br=256 | UB 溢出（`Oi_ub_zN` 131072B > 10880B），不可行 |

模型级预期：每步 49 次 FA × 8.98 ms ≈ **−0.44 s/step**（16.34 → ≈15.90 s，待整模 49 步验证）。