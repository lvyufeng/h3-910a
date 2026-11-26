# R358 — 修正推断：MAD 出现 1 次 ≠ 只算一块（是循环体）；K=16 单步仍只对第 0 行

## 一、两处自我修正

1. **R357/R358 中"1 条 MAD = 只算一个 tile"的推断是错的**：
   反汇编里 MAD 出现在**运行时循环体内**，循环次数由 tiling 决定，因此 1 条 MAD 不代表只算一块。
   撤回该推断。
2. R341 的 M=16 结论曾被我怀疑是 tiling bug 导致——本轮用**修正后的 tiling** 重测：

```
[TILINGDUMP] M=16 N=32 Ka=16 Kb=16 scM=16 scN=32 scK=16
             baseM=16 baseN=32 baseK=16 stepM=1 stepN=1
K16 maxdiff=480 bad=511/512
row0 = 1 2 3 4 5 6        ← 正确（A=I ⇒ C=B）
row1 = 1 2 3 4 5 6        ← 应为 B 的第 1 行
```

⇒ **即使 tiling 完全自洽、K 只走单步，高层 `Matmul` 仍然只算出第 0 行**。
R341 的结论（与 tiling bug 无关）**成立**。

## 二、当前唯一站得住的描述

对 A=I 的用例，输出"每一行都等于 `A[0]·B`"⇒ **A 的 M 方向在搬入环节被压成了 stride 0**。
这在 ND（单分形）、NZ、K=16、K=32、M=16、M=32 下**全部一致**，
且与所有 tiling 参数无关。

## 三、下一步（如果继续 AscendC）

1. 反汇编**框架 Matmul 版 kernel 的 L1/L0A 搬入段**，看 `LoadData` 的 `srcStride` 立即数是多少
   （反汇编里应有立即数；若为 0 即确认）；
2. 对照 CANN Conv2D 的搬入段（144 MAD 那个）看它用什么 stride/分形参数；
3. 若确认是框架在 v200 上算错，则**唯一可用路径仍是 TIK**（CANN 自家 910A 算子即 TIK）。

## 四、状态

- 交付不变：**16.34 s/step（−8.9%）**，`H3_PERF_COMBO=1` 默认 OFF；
- AscendC 高层 `Matmul` 在 910A 上**只算出第 0 行**（多轮、修正后仍复现）；
- 底层 `Mmad/LoadData` 被编译器丢弃（0 指令，R356）；
- 结论方向不变：**转 TIK**。
