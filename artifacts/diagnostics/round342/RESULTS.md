# R342 — NZ 路线被 op 原型挡住（msopgen 只生成 ND 签名）；剩两条路

## 一、本轮实验

把 kernel/host 的 `MatmulType` 与 tiling 全部改为 `CubeFormat::NZ`，host 侧把 A/B 按 16×16 分形摆好。

结果：**在执行前被 op 原型拒绝**

```
GWS rc=161002 Execution_Error(EZ1009):
  The dtype or format of the actual input or output parameter ... inconsistent with OpDef,
  q(DT_FLOAT16, FRACTAL_NZ), k(DT_FLOAT16, FRACTAL_NZ), v(DT_FLOAT16, FRACTAL_NZ), out(DT_FLOAT16, ND)
```

⇒ **msopgen 生成的算子原型只声明 ND**，调用方传 NZ 会被 aclnn 层直接拒绝。
要用 NZ，必须**重新生成 op（改 proto 的 format 列表为 FRACTAL_NZ）**，属于工程重建，不是改两行。

## 二、当前完整证据链（明确）

| 事实 | 证据 |
|---|---|
| cube 能算 | A=I 时 C 第 0 行**逐元素精确**（R338） |
| 内核替换生效、C 读取链路正常 | 哨兵实验（R340） |
| 与 IterateAll / 形状参数无关 | 四种设置结果**逐位相同**（R339/R340） |
| 单 M 块也一样 | M=16 仍只有第 0 行对（R341） |
| ND 路径的 A 加载把 M stride 当 0 | 上述综合 |
| NZ 路径需重建 op 原型 | 本轮 rc=161002 |

## 三、剩余两条路

### 路 A：重建 op 原型为 NZ（推荐优先，改动可控）
1. 改 `msopgen` 的 op json：input/output 的 `format` 从 `ND` 改为 `FRACTAL_NZ`；
2. host 侧对应 `SetAType/SetBType(CubeFormat::NZ)`，kernel `MatmulType<..., CubeFormat::NZ, half>`；
3. 实测判据：M=16、N=32、K=32，A=I 时 `bad=0`。

### 路 B：手写 Mmad（不依赖 MatmulImpl）
- 直接 `DataCopy` GM→L1（ND 线性）、`LoadData2dParams{srcStride=1}` 逐分形 → L0A/L0B、
  `MmadParams{m=n=k=16}` 逐块乘加、手动展开写回；
- 好处：完全绕开框架的 stride 推断；坏处：代码量大、要自己保证分形索引正确。

## 四、状态

- 交付不变：**16.34 s/step（−8.9%）**，`H3_PERF_COMBO=1` 默认 OFF；
- **AscendC cube 可算已确证**；缺陷定位到 ND 路径的 A 加载 stride；
- 下一步：路 A（重建 op 原型为 NZ），预计 2–3 轮内可判定 cube matmul 是否彻底打通。
