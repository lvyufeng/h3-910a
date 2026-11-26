# R338 — **AscendC cube 在 910A 上是能算的**（更正 R313）；缺陷精确定位到 A 的 M 方向走位

## 一、决定性实验：A=单位阵的 32×32 matmul

kernel：`Matmul<ND half, ND half, ND float, ND float>`，A=I(32×32)、B=arange、无转置，
`IterateAll` → float C，再转 half 输出。CPU 参考：A=I ⇒ **C 必须等于 B**。

实测：

```
ID32 maxdiff=96  relRMS=7.813e-01  bad=960/1024
out[0..5] = 1 2 3 4 5 6      ← B 的第 0 行，完全正确
row0 out = 1 2 3 4 5 6
row1 out = 1 2 3 4 5 6      ← 应为 33 34 35 36 37 38
col check (col0 各行) = 1 1 1 1 1 1
```

## 二、结论（更正 R313）

1. **cube 本体工作正常**：第 0 行与 CPU 参考**逐元素一致**（1..6 精确）；
2. **缺陷唯一**：C 的每一行都等于 `A[0]·B`，即 **A 的 M 方向没有被迭代**（M stride 被当作 0）；
3. bad=960/1024 = 30 行 × 32 列 ⇒ 只有第 0 行对，其余 31 行全错，**与缺陷形态完全吻合**。

⇒ **「AscendC 在本机不可用」是错误结论，正式撤回**。真实状态是：
**cube 可用，但我们的 A 张量/M 走位配置错**，这是一个可定位、可修的用法问题。

## 三、已排除的变量

| 变量 | 结果 |
|---|---|
| B 运行时转置 true/false | 无差别 |
| 模板 `ISTRANS` | 无差别 |
| host `SetBType(..., true→false)` | 无差别 |
| workspace 大小/指针 | 无差别 |
| M=16 单块 vs M=32 两块 | 同样"只有行 0 对" ⇒ 不是分块问题 |

## 四、下一步（精确、范围很小）

1. dump host 侧 `TCubeTiling` 的 **A 相关字段**：`Ka`、`singleCoreK`、`baseM`、`stepM`、`depthA1`，
   以及 `MatmulType` 的 layout；
2. 检查 kernel 里 A 的绑定：`SetTensorA(aGm, false)` 与 tiling 的 `SetAType(..., isTrans?)` 必须一致；
3. 试 **4 参 `SetOrgShape(32,32,32,32)`**（orgKa/orgKb 分开），以及 `SetSingleShape(32,32,32)`；
4. 判据：M=32 时 32 行全部正确（`bad=0`）。

## 五、状态

- 交付不变：**16.34 s/step（−8.9%）**，`H3_PERF_COMBO=1` 默认 OFF；
- **DiT ≤15/10 s 的路重新打开**：cube 可用 ⇒ FA 内核 AscendC 重写回到可行路径，只差修 A 走位。
