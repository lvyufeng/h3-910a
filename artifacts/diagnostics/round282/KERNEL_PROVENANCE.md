# R282 — 重要发现：在用的 FA 包**已内含 FP16 原地 exp 改动**，并非上游原版

## 一、源码证据

`/tmp/fa_pkg`（我们一直称为 "pristine"、所有测量与交付都在用它）里
`flash_attention_fwd.py` 的 `softmax_compute` 中有这样一段注释与实现：

```
# exp, in place on the fp16 tile.
#
# The original path cast the whole score tile to a fresh fp32 UB buffer,
# exponentiated there and cast back. That temporary, not the score tile
# itself, was what capped the tiling: at Br=64/Bc=608 it is 152 KB and at
# Bc=832 it is 208 KB, which is exactly where the build failed. Doing the
# exp in place frees that buffer and lets Br double at the same footprint,
# which halves the number of K/V sweeps.
#
# Precision cost: after the row-max subtraction the scores are <= 0, so
# exp is in [0, 1]. fp16 holds that range well but only to ~1e-3 relative,
# against ~5e-4 for the fp32 path. Verify at the real shape.
self.tik_instance.h_exp(Sij_ub_zN, Sij_ub_zN)
```

⇒ 该包**不是**上游原版：它把 `exp` 从 "cast 到 FP32 UB → exp → cast 回" 改成了
**在 FP16 上原地做 exp**，注释明确写了精度代价（FP16 ~1e-3 相对误差 vs FP32 ~5e-4）与动机（省 UB、放大 tile）。

## 二、影响（必须如实说明）

1. **我们所有 FA 测量（152.19 / 154.75 / 142.39 ms）与 16.34 s 交付，都包含这个改动**；
2. 它使 `Br` 可以加倍、K/V sweep 次数减半 ⇒ 142.39 ms 里已含这部分收益，**不是**上游原版的性能；
3. 它是**数值改动**（exp 精度从 FP32 降到 FP16 中间值）——我们 49 步的 relRMS 在原生带内，
   但该改动的**授权来源需核实**（属于早期轮次，可能在 R216 前后由用户批准或作为实验件引入）；
4. R245 的 "用 pristine /tmp/fa_pkg 修复 C_l0c 报错" 结论需要重新表述：该包是**已优化版**，
   而非上游原版；"原版基线"应为 R245 中失败的那份（R216-modified 表述亦需复核）。

## 三、待办（合规优先）

1. **核实授权**：在 Issue #5 里查该 FP16-exp 改动的批准记录；若无，需向用户报告并请求确认，
   因为它影响 16.34 s 交付的数值基线；
2. **建立金标准基线**：保留一份真正的上游原版 FA 包副本（若主机/缓存中有），
   用于区分 "上游性能" 与 "本方案性能"；
3. 所有后续比较（W1 等）必须声明用的是哪一份包。

## 四、合规

仅读码与证据记录；未改任何文件；部署路径 `/tmp/fa_pkg` 未动（md5 见 R274/R278 记录）。
