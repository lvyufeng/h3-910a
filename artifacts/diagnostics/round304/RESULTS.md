# R304 — AICore 异常根因找到：**workspace 太小**；第 0 行已数值正确

## 一、二分定位结果

| 步骤 | 设置 | 结果 |
|---|---|---|
| R301/R302 | `currentWorkspace = sysWs(平台值) + 512 KB`，C 写在 `workspace+sysWs` | `507015 AICore 异常` |
| K1（本轮） | C 写在 `workspace[0]`，`currentWorkspace = 512 KB` | **仍 507015** |
| **K2（本轮）** | **`currentWorkspace = 32 MB`**，C 写在 `workspace + 16 MB` | **不再崩溃，且第 0 行数值正确** |

**结论：崩溃根因是 workspace 分配量不足**（Matmul 的框架级系统工作区需求远大于
`platform.GetLibApiWorkSpaceSize()` 所报数值 / 我的估算），而不是 UB、不是偏移语义、也不是 tiling。

## 二、K2 的数值结果（部分正确）

```
probe got[0][0..3] = 3.5547 -3.2852 -5.0898 -0.8999
probe ref[0][0..3] = 3.5545 -3.2861 -5.0882 -0.8997
probe nonzero = 32768/32768
RESULT M=64 N=512 K=128 maxdiff=27.4668 relRMS=1.395e+00 bad=32038
```

- **第 0 行前 4 个元素与参考吻合到 1e-4 量级** ⇒ **布局/转置/基址全部正确**（这是本轮最大进展）；
- 但整体 `relRMS=1.395`、`bad=32038` ⇒ 其余部分仍错，需要逐行定位。

## 三、下一轮定位（R305）

打印「逐行误差画像」：对每 16 行统计 `max|diff|`，判断是
(a) 只有第一个 16×512 分块正确（→ 单核只算了一块 / tiling 的 `usedCoreNum·singleCoreM` 只覆盖 16 行）；
(b) 前 N 行对、后面错（→ 分块迭代缺失）；
(c) 规律性错位（→ C 的 ND stride 与 GM 长度不匹配）。

最可能：`MultiCoreMatmulTiling` 默认把 N 或 M 切成多个 `singleCore` 块，而 kernel 只调用了
`IterateAll` 一次（单核）；需要 `SetDim`/`SetSingleShape` 与 `SetBlockDim` 对齐，或改用
`mm.SetOrgShape(M, N, Ka, Kb)` 的四参数形式。

## 四、状态

- 交付不变：**16.34 s/step（−8.9%）**，49 步 ×2 验证，relRMS 在原生带内，`H3_PERF_COMBO=1` 默认 OFF；
- AscendC：S0 ✅、S0.5 ✅、S1a-0 ✅；**S1a-1 已能跑、第 0 行正确**，剩余为分块/tiling 覆盖问题；
- 机器 ts-217（910A）租约至 20:52 UTC。
