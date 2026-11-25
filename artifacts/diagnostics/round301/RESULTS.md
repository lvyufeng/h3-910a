# R301 — workspace 偏移已修，但 kernel 现在触发 AICore 异常（疑似 UB 超限）

## 一、本轮改动（workspace 重叠问题）

按 R300 的诊断改了三个地方：

1. tiling 结构增加 `int64_t sysWs`（`{int32_t size; int64_t sysWs; TCubeTiling mm;}`）；
2. host：`tiling->sysWs = platform.GetLibApiWorkSpaceSize()`；workspace 申请 = `sysWs + 64*512*4`；
3. kernel：`GM_ADDR userWs = workspace + tilingData.sysWs;`，C 矩阵写 `userWs`。

**构建、安装均成功**（不再有编译错误）。

## 二、新错误：AICore 异常

```
ACL_FAIL 507015 line 35
```

- `507015 = ACL_ERROR_RT_AICORE_EXCEPTION`（`acl/error_codes/rt_error_codes.h:107`）；
- 失败点在第 35 行 = `aclrtSynchronizeStream(st)` ⇒ **kernel 在设备上崩溃**（不是 host 侧参数错误）。

## 三、最可能原因：UB 超限

kernel 里除 Matmul 自己要用的 UB（L0C→UB 中转、分形缓冲）外，我们还额外申请了：

| 缓冲 | 大小 |
|---|---|
| `inQ`（float C 矩阵） | 64×512×4 = **128 KB** |
| `outQ`（half 输出） | 64×512×2 = **64 KB** |
| 合计 | **192 KB**（910A UB 上限 192 KB，且尚未扣除 Matmul 自身占用） |

⇒ 几乎必然超限，导致 AICore 异常。

## 四、下一步（R302）

1. **不要一次性搬整个 C**：把 64×512 拆成 4 块（每块 16×512），
   循环 `DataCopy`→`Cast`→`DataCopy`，UB 峰值降到 32 KB + 16 KB；
2. 或者直接用 `Matmul` 的 C 类型设为 half（`cType = MatmulType<GM, ND, half>`），
   让框架完成 FP32→FP16 的输出量化，省掉我们自己的 Cast 与两块 UB；
3. 重测目标 `relRMS ≤ 1e-3`；通过后进完整 FA 与 S2 流水。

## 五、状态

- 交付不变：**16.34 s/step（−8.9%）**，49 步 ×2 验证，relRMS 在原生带内，`H3_PERF_COMBO=1` 默认 OFF；
- AscendC：S0 ✅、S0.5 ✅、S1a-0 ✅、S1a-1 编译 ✅、**数值：workspace 已修，卡在 kernel 崩溃（UB）**；
- 机器 ts-217（910A）租约至 20:52 UTC。
