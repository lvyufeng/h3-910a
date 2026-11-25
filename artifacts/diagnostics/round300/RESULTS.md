# R300 — 双重转置已排除；症状指向 workspace 与系统工作区重叠

## 一、本轮实验

1. 查得 B 转置是**运行时参数**：`SetTensorB(const GlobalTensor<SrcBT>& gm, bool isTransposeB = false)`
   （`matmul_client.h:477`）；
2. 我此前在模板里写了 `<..., half, true>` 又在 `SetTensorB` 里传 `true`，疑似**双重转置**；
   改为模板不转置 + `SetTensorB(kGm, true)` 后重建重装重测。

## 二、结果：症状变了，但仍然错误

| 版本 | got[0][0..3] | nonzero | maxdiff | relRMS |
|---|---|---|---|---|
| 模板 true + SetTensorB(false) | -2.0000 -0.7368 -0 0.6650 | 24610 | 526.23 | 20.32 |
| **模板 false + SetTensorB(true)**（本轮） | **0 0 0 0** | 28067 | 22.82 | 1.340 |

- 数值幅度明显改善（526 → 22.8；relRMS 20 → 1.34），说明**双重转置确实是问题之一**；
- 但 `got[0][0..3]` 全为 0 而整体 nonzero=28067 ⇒ **第 0 行被写到了别处**，这是典型的
  「输出基址偏移」症状。

## 三、最可能的新根因：workspace 与系统工作区重叠

kernel 里同时做了两件事，**共用同一个 `workspace` 指针**：

```cpp
REGIST_MATMUL_OBJ(&pipe, GetSysWorkSpacePtr(), mm, &tilingData.mm);   // 用系统工作区
GlobalTensor<float> tGm; tGm.SetGlobalBuffer((__gm__ float *)workspace, kM*kN);  // 也写 workspace[0]
```

`REGIST_MATMUL_OBJ` 的系统工作区从 `workspace` 起始处开始，而我们把 C 矩阵也写在 `workspace[0]`，
两者**互相覆盖**，导致输出错位/前几行被冲掉。

正确做法：系统工作区在前，用户缓冲要**偏移 `sysWorkspaceSize` 字节**；
host 侧 workspace 申请量也应为 `sysWorkspaceSize + 用户缓冲`，并把 `sysWorkspaceSize`
通过 tiling 传给 kernel（或使用框架的 `GetUserWorkspace()` 接口）。

## 四、状态

- 交付不变：**16.34 s/step（−8.9%）**，49 步 ×2 验证，relRMS 在原生带内，`H3_PERF_COMBO=1` 默认 OFF；
- AscendC：S0 ✅、S0.5 ✅、S1a-0 ✅、S1a-1 高层 API 编译 ✅、**数值差一步（workspace 偏移）**；
- 机器 ts-217（910A）租约至 20:52 UTC。

## 五、下一步（R301）

1. host tiling 里加 `int64_t sysWorkspaceSize`（= `platform.GetLibApiWorkSpaceSize()`），
   workspace 申请 = `sys + 64*512*4`；
2. kernel 里 C 矩阵基址改为 `workspace + sysWorkspaceSize`，或改用框架提供的用户工作区接口；
3. 重测，目标 `relRMS ≤ 1e-3`；
4. 通过后进入完整 FA（softmax + PV + 循环），再进 S2 流水冲 ≤105 ms/块。
