# R367 — 架构澄清 + AscendC Matmul 确实发射 cube；数值仍全 0（待修）

## 一、架构澄清（用户 2026-11-26 明确）

> 「910B 不带数字，就是 910A 那一代」

即 `npu-smi` 显示的 **`910B`（无数字后缀）属于 910A 世代（v200）**，不是 910B 系列（v220）。
因此：
- 计算目标保持 **ascend910 / dav-c100（v200）**；
- R366 中"机器是 910B，应改 v220"的判断**作废**；
- `CMakePresets.json` 已回退为 `"ASCEND_COMPUTE_UNIT": "ascend910"`。

## 二、AscendC Matmul 路径确实发射 cube 指令（更正 R356/R359）

把 `round299/fa_kernel_matmul_api.cpp`（`Matmul<aType,bType,cType>` + `REGIST_MATMUL_OBJ`）
放回 msopgen 工程（compute unit = ascend910）构建：

```
build_rc=0
设备对象: build_out/op_kernel/ascendc_kernels/binary/ascend910/flash_attention_custom/
          FlashAttentionCustom_fc666c715d92188ea50da7253934f7a5.o
反汇编: LOAD_L1_TO_L0A_2D, MMAD   ← cube 指令存在
```

⇒ **AscendC 高层 Matmul API 在 ascend910 目标上能生成 cube 代码**。
此前"AscendC 不发射 cube"的判断是基于错误的构建/对象（R356/R359 结论作废）。

## 三、数值验证仍失败（未解决）

harness `test_fa294.cpp`（M=64,K=128，A=I 语义）：

```
RESULT M=64 K=128 maxdiff_vs_q=255.9 bad=8190 head=(0.4231,...)->0.0000
```

输出**全 0**。已试：
- 加 `SyncAll()`（AIC/AIV 跨核同步）后重建 → 仍全 0；
- 设备二进制只有一个（含 MMAD），非 cube/vector 分裂。

**待排查方向**：
1. v200 mixed 模式的核心任务编码（是否需要 `KERNEL_TASK_TYPE_*` 或 AIC/AIV 分别启动）；
2. workspace 传递与 `IterateAll(tGm)` 输出是否真正落盘（可在 kernel 里写哨兵值验证）；
3. Matmul 的 tiling（`tilingData.mm`）是否被 host 正确填充（`MultiCoreMatmulTiling` 参数是否适用于 v200）。

## 四、CCE 路线证据（不变）

- `ccec --cce-aicore-arch=dav-c100` 编译 TIK 生成的 FA `.cce` → 设备段 **1215 `MMAD`**；
- TIK 默认构建的 16×16 matmul 内核 → 设备段含 `MMAD`/`LOAD_L1_TO_L0A/B_2D`/`MOV_OUT_TO_L1`。

⇒ "走 CCE 生成 cube 代码"已被三方独立证据确认（TIK、AscendC、手工 ccec）。

## 五、状态

- 交付不变：**16.34 s/step（−8.9%）**，`H3_PERF_COMBO=1` 默认 OFF；
- 结论更新：cube 代码生成**不是**障碍（R366/R367），当前障碍是**运行时数值正确性**；
- 下一步优先级：先让 64×128 matmul 输出正确（哨兵 + tiling 检查），再谈 FA 性能。
