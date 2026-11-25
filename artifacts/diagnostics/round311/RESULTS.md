# R311 — 四种配置全部产出「常量行」⇒ 高层 Matmul 在本机 910A 上疑似不受支持

## 一、本轮四次对照实验（全部 relRMS ≈1.3–1.4）

| # | 配置 | 结果 |
|---|---|---|
| 1 | 模板不转置 + `SetTensorB(kGm, true)`（运行时转置） | 每行 = ref 第 0 行（rowVar≈0） |
| 2 | A/B 对调（k 作 A、q 作 B） | 行/列都非常量，仍然错 |
| 3 | 关运行时转置 + host 侧预转置 k（B 为 K×N） | rowVar=4.92，仍然错 |
| 4 | **模板 `ISTRANS=true` + `SetTensorB(gm,false)`**（官方写法） | **每行 = ref 第 13 行**（rms 5.14） |

**共同点：无论如何配置，C 都是「同一行被重复写」的形态** ⇒ 不是转置语义问题，而是 **C 的写出/框架 Matmul 本身在本机异常**。

## 二、新增证据

`asc/include/adv_api/matmul/matmul_intf.h` 里有 **`__NPU_ARCH__ == 2201 || 3510` 的架构分支**
（2201 = 910B 系列、3510 = 新一代），而本机是 **Ascend910A**（`AddConfig("ascend910")`）。
结合「编译通过但运行结果恒为单行重复」，高度怀疑：

> **高层 AscendC Matmul 不支持 Ascend910A**（只保证 310P/910B 等 arch）。

这也解释了为什么 S1a-0（纯 `DataCopy`）完全正常，而一旦用 `Matmul` 就出诡异结果。

## 三、下一步（两条，按优先级）

| # | 路径 | 说明 |
|---|---|---|
| **1** | **手工 `Mmad` + 照抄 TIK 内核的 buffer 布局** | 我们手里有**能跑通的 TIK FA 源码**（`/tmp/fa_pkg_r266/.../flash_attention_fwd.py`），它明确了 L1/L0A/L0B/L0C 的形状与 stride；AscendC 手工路径按同样布局写，成功率高 |
| 2 | 确认 910A 的 AscendC Matmul 支持情况 | 查 `ccec` 宏或 CANN 文档；若确认不支持，则路径 1 是唯一选择 |

## 四、状态

- 交付不变：**16.34 s/step（−8.9%）**，49 步 ×2 验证，relRMS 在原生带内，`H3_PERF_COMBO=1` 默认 OFF；
- AscendC：S0 ✅、S0.5 ✅、S1a-0 ✅；**S1a-1 高层 Matmul 路线判定为受阻（疑似 arch 不支持）**；
- 机器 ts-217 直连正常（bridge 误报 OFFLINE）；远端 master 含本轮记录。
