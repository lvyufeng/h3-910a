# R312 — 架构分支确认 Matmul 在 910A 可用；显式 workspace 无效；补充判断

## 一、架构分支阅读（`asc/include/adv_api/matmul/matmul_intf.h:20-70`）

```cpp
#ifdef ASCENDC_CPU_DEBUG
#if defined(__NPU_ARCH__) && (__NPU_ARCH__ == 2201 || __NPU_ARCH__ == 3510)
    ... MatmulImpl / MatmulClient
#else
#else
    // Use MatmulImpl in other cases
    using Matmul = MatmulImpl<A_TYPE, B_TYPE, C_TYPE, BIAS_TYPE, MM_CFG, MM_CB, MATMUL_POLICY>;
#endif
```

⇒ **910A（非 2201/3510）走 `MatmulImpl` 分支，高层 Matmul 在 API 层面是可用的**，
此前「不支持 910A」的猜测被削弱。

## 二、本轮实验：显式传入 Matmul 工作区

`GetSysWorkSpacePtr()` 在非 2201/3510 架构返回 `g_sysWorkspaceReserved`（自定义算子可能未被注入），
因此改为显式传：

```cpp
REGIST_MATMUL_OBJ(&pipe, (GM_ADDR)workspace, mm, &tilingData.mm);
```

结果**与上一轮完全相同**（每行仍 = ref 第 13 行，relRMS 1.429）⇒ **不是工作区指针问题**。

## 三、当前为止对 S1a-1 的全部否证

| 变量 | 结果 |
|---|---|
| 运行时 `SetTensorB(gm,true)` | 每行=ref[0] |
| 模板 `ISTRANS=true` + 运行时 false | 每行=ref[13] |
| 预转置 B + 无转置 | rowVar≠0 但仍错 |
| A/B 对调 | 两种错误形态 |
| 显式 Matmul workspace | 无变化 |
| 32 MB workspace | 崩溃消失（必要但要非充分） |
| `mm.End()` | 无变化 |

**共同症状：C 恒为「单行重复」**。这说明 fixpipe 的 C 写出 **M 方向步长没有生效**，
而 M 步长来自 tiling 的 C 布局参数——目前我们对 `MatmulImpl` 在该 arch 上的 C 布局约定理解不足。

## 四、下一步建议（两条，需你选）

| # | 路径 | 预期 |
|---|---|---|
| **A** | 转**手工 `Mmad`**，完全自己控制 L0C→GM 的搬运（照 TIK 的 `data_move` 语义：`nBurst/burstLen/srcGap/dstGap`），绕开框架 C 写出 | 可控性最高；但需把 L0C 分形→ND 的搬运调对，估计还要若干轮 |
| **B** | 暂停 AscendC，回到交付与并行目标（encoder<2 s、VAE<20 s、A2/A3） | 确定性收益；15 s 保持未达成 |

## 五、状态

- 交付不变：**16.34 s/step（−8.9%）**，49 步 ×2 验证，relRMS 在原生带内，`H3_PERF_COMBO=1` 默认 OFF；
- AscendC：S0 ✅、S0.5 ✅、S1a-0 ✅、S1a-1 仍未通过（C 写出 M 步长未生效）；
- 机器 ts-217 直连正常；远端 master 含本轮记录。
