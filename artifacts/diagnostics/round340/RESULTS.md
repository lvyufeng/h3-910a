# R340 — 排除「kernel 缓存复用」：新内核确实生效；matmul 结果与配置无关

## 一、哨兵实验（关键排除）

在 kernel 里**跳过 matmul**，直接写 `o[i] = i % 1000` 的可辨识图案，然后重编译/重装/重跑：

```
out[0..5] = 0 1 2 3 4 5
row1 out  = 32 33 34 35 36 37      ← 正是索引图案
```

⇒ **内核确实被替换并执行了，不存在 stale kernel 缓存**；
同时证明 **C 的读取路径（workspace+16MB → UB → Cast → GM）完全正常**。

## 二、因此可下的结论

在 32×32 用例中，以下四种设置得到的 matmul 结果**逐位相同**（都是"每行 = A[0]·B"）：

| 设置 | 结果 |
|---|---|
| `SetBType(..., false)` | 相同 |
| 4 参 `SetOrgShape(32,32,32,32)` + `SetSingleShape` | 相同 |
| `IterateAll(cGm, 0, /*enSequentialWrite=*/true)` | 相同 |
| 默认 `IterateAll(cGm, false)` | 相同 |

⇒ 说明这不是形状/写出参数问题，而是 **`MatmulImpl`（910A 分支）在 M 方向只算/只动了一个块**。
与 R339 的假设一致，但排除了"缓存"这种更粗的可能性。

## 三、下一步（两条，二选一）

1. **手写 Mmad + 显式循环**（推荐）：
   - `LoadData2dParams{startIndex, repeatTimes=1, srcStride=?, dstGap=0}` 逐 16×16 fractal 载入 L0A/L0B；
   - `MmadParams{m=16,n=16,k=16}` 逐块 `Mmad`，用 `DataCopyCO12DstParams` 或手工 NZ 展开写回；
   - 这是最底层路径，不依赖 `MatmulImpl` 的 M 迭代；
   - 之前手工 Mmad 全 0，很可能就是 `LoadData` 的 `srcStride` 语义没对（本轮已证明 C 读取链路 OK，可专注 LoadData）。
2. **把 tiling 字段写文件**（不再靠 printf）：在 host 里 `fopen("/tmp/h3_r221/td.txt","w")` 写出
   `Ka/baseM/baseN/baseK/stepM/stepN/depthA1/stepKa`，确认 `baseM` 是否为 16、`stepM` 是否为 2。

## 四、状态

- 交付不变：**16.34 s/step（−8.9%）**，`H3_PERF_COMBO=1` 默认 OFF；
- cube 可算（R338 第 0 行精确）；`MatmulImpl` 的 M 迭代疑似不完整（本轮证据）；
- 工程与判据脚本：`/tmp/h3_r221/faA/proj` + `test_id32.cpp`（A=I、期望 C=B）。
