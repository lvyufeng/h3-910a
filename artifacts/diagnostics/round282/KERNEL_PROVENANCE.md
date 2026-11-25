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

## 五、溯源结果（补充，2026-11-25）

在本仓库的历史记录里找到了该模式的来源：

| 记录 | 证据 |
|---|---|
| **R99**（`round99/flash/README.md:56`） | "UB `151424 B` < 196608 的安全边界：score tile 复用为 exp tile（**原地 `h_exp`**，与已执行 flash-attention 源码同模式）" |
| **R101**（`round101/flash_bc128/README.md:73`） | "8192 元素整块 `h_sub`/`h_exp`/`h_cast` 链可编译（R99 在 4096 元素上已执行）" |
| **R216**（`round216/RESULTS.md:27`） | "seeded baseline repeated ≈**154.51–154.55 ms** at actual Br128/Bc304, kernel_N43776"；同目录有 `flash_attention_fwd_l0c.py`（该轮的内核变体，构建失败于 C_l0c） |

**结论**：
1. "原地 FP16 `h_exp`" 是本项目 R99/R101 期间**自研内核工作**里确立的模式（当时用于自写 FA 原型的 UB 预算），
   不是上游 MindSpore FA 包的原始实现；
2. `/tmp/fa_pkg` 中出现该实现 ⇒ 说明某轮把自研变体**安装进了 `/tmp/fa_pkg`**；
3. R216 记录的 154.51–154.55 ms 与我们 154.7 一致 ⇒ **R216 的"baseline"也是在这个已修改包上测的**，
   而非上游原版；
4. 因此：**16.34 s 交付、以及 152/154/142 ms 这些数字，全部是"自研 FA 变体"口径**。

## 六、对合规的影响与建议

- 该改动属于**数值改动**（exp 中间精度 FP16 vs FP32）；用户合同要求"任何可测数值变化先停用上报并请用户决定"；
- 我们 49 步的 relRMS 在原生带内（9.84e-04–1.077e-03），**结果上可接受**，但**授权链条需用户确认**；
- 建议：①在 #5 明确标注"FA 内核 = 自研变体（含原地 FP16 exp）"；②若用户要求"上游原版基线"，
  需另建一份未修改的 FA 包并重测 15 s 差距（预计比 142 ms 慢，差距更大）。

## 七、上游原版是否可重建？（2026-11-25 主机检索结果）

在 ts-217 上做了三处检索：

| 检索 | 结果 |
|---|---|
| `site-packages` 中的 mindspore / flash / attention 包 | **无** |
| `/home/mseco`（6 层内）除 `/tmp/fa_pkg*` 外的 `flash_attention_fwd*.py` | **无** |
| `/home/mseco/vllm-omni` 内 `*flash_attention*` 源码 | 只有 `test_vision_flash_attention.py`（测试文件），**无 TIK 内核源码** |

⇒ **上游原版 FA 包在主机上已不存在**，无法从现有工件重建"原版基线"。

### 对选项 B 的影响

- 选项 B（要求上游原版基线）**需要先重新获取原始 FA 包**（来源：原始 MindSpore FlashAttention 发布物/pip 包/wheel）；
- 在此之前，**无法给出"上游原版 + trim + guard"的步时**，也无法量化我们的 FA 变体究竟贡献多少；
- 现有可交付口径只有 A：自研变体（含原地 FP16 exp）下的 **16.34 s（−8.9%）**。

### 建议

| 若你选 | 下一步 |
|---|---|
| **A** | 继续 W1（削减向量工作量）冲 15 s；同时在 #5 标注"FA 内核 = 自研变体"作为交付口径 |
| **B** | 先安排重新获取原始 FA 包（需网络/来源确认，不能从客户端中转权重），再重建基线并重测；15 s 判定相应后移 |
