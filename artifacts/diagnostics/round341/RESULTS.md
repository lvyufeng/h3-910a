# R341 — M=16 单块也失败：**A 的 M 方向在加载阶段就没走**（不是 IterateAll 的错）

## 一、实验

M=16、N=32、K=32，A=I(16×32)、B=arange、无转置，期望 C = B 前 16 行。

```
M16 maxdiff=96  bad=480/512
row0    = 1 2 3 4 5 6      ← 正确
row1    = 1 2 3 4 5 6      ← 应为 33 34 35 36 37 38
```

bad=480 = **30 行 × 16 列** ⇒ 仍然只有第 0 行对。

## 二、推论（重要，推翻了 R339 的假设方向）

M=16 已是**单个 M 块**，不存在 "IterateAll 不迭代 M 块" 的问题，但结果仍是"每行 = A[0]·B"。

⇒ 缺陷在更底层：**A 的 L1/L0A 加载把 M 方向的 stride 当成了 0**（同一行被复制 16 次），
而不是 `MatmulImpl` 的块迭代。

这与以下事实自洽：
- tiling 形状参数（SetOrgShape/SetSingleShape/enSequentialWrite）怎么改都**逐位相同**——因为它们不影响加载 stride；
- C 读取链路已由哨兵实验证明正常（R340）；
- 第 0 行精确正确。

## 三、下一步（两条，都是加载格式）

1. **把 A/B 的 `MatmulType` 改成 `CubeFormat::NZ`**，host 侧预先把矩阵按 NZ 分形摆好
   （M=16,K=32 ⇒ 1×2 个 16×16 分形，512 个 half）。ND 路径在 910A 的 `MatmulImpl` 上
   很可能没有正确设置 `dstNzC0Stride/srcDValue`；NZ 路径把这些交给调用方，反而是最可能的解法。
2. 若 NZ 仍不行 ⇒ 手写 `LoadData2dParams`，把 `srcStride` 显式设成 1（fractal 间隔），
   逐分形加载 + 逐块 `Mmad`，完全绕开框架。

## 四、状态

- 交付不变：**16.34 s/step（−8.9%）**，`H3_PERF_COMBO=1` 默认 OFF；
- cube 可算（行 0 精确）；缺陷收敛到 **A 的 ND→NZ 加载 stride**；
- 工程与两个判据 harness（ID32、M16）都在 `/tmp/h3_r221/`，可直接复用。
