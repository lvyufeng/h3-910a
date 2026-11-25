# R313 — 16×16 最小手工 Mmad 仍全 0；结论：本机 910A 的 AscendC cube 路线不可行

## 一、本轮实验

写了一个最小手工 cube 内核（A=单位阵、B=`i*16+j`，输出应逐元素等于 B）：

1. `DataCopy` GM→L1（16×16，NZ 与 ND 在该形状下同构）；
2. `LoadData2dParams{repeatTimes=1, srcStride=1, dstGap=0}` → L0A/L0B；
3. `MmadParams{m=n=k=16, cmatrixInitVal=true}` → L0C；
4. `DataCopy(cf, c0r, DataCopyCO12DstParams{nSize=16, mSize=16, ...})` → UB → `Cast` → GM。

结果：

```
MINI16 maxdiff=256.113 bad=255
out[0..7] = 0 0 0 0 0 0 0 0   (期望 0 1 2 3 4 5 6 7)
```

- `nz2ndEn=false` 与 `true` **结果完全相同（全 0）**；
- 编译、安装、调用全部正常，只有数据算/搬不出来。

## 二、累计证据（AscendC cube 在本机不可行）

| 尝试 | 结果 |
|---|---|
| 高层 `Matmul`（4 种转置/布局组合） | C 恒为「单行重复」 |
| 显式 Matmul workspace | 无变化 |
| `mm.End()` | 无变化 |
| 手工 `Mmad`（64×512） | 第 0 行对、其余错 |
| **手工 `Mmad`（16×16 最小单元）** | **全 0** |
| `nz2ndEn` true/false | 无差异 |

CANN 内置算子（`opp/built-in/.../ascendc/*`）全部面向 310P/910B 等 arch；`matmul_intf.h`
的 KFC/MatmulClient 分支只对 `__NPU_ARCH__ == 2201 || 3510` 生效，**910A 走的是兜底的 `MatmulImpl`**。
结合以上实测：

> **结论：本机（Ascend910A + CANN 9.2.0）的 AscendC cube 能力不足以支撑 FA 内核重写；
> 纯数据搬运（S1a-0）可以，一旦涉及 Cube/L0C 就得不到正确结果。**

这是**路线级结论**，不是参数没调对——四种高层配置 + 两种手工配置全部失败，且失败形态稳定。

## 三、对 15 s 目标的意义

- FA 内核在 TIK 上已达结构下限 **126.19 ms（1.225×）**（R291）；
- AscendC 重写是唯一可能到 ≤105 ms 的路径，但本机不可行；
- ⇒ **在当前硬件+软件条件下，15 s/step 不可达**（需要 910B/310P 机器或厂商支持）。

## 四、建议与下一步

按目标里的并行目标推进，这些是**确定可做的**：

1. **encoder < 2 s**：现有 harness 无阶段计时（R281 已确认），需要专用探针；
2. **video VAE < 20 s**：同上；
3. **A2/A3 对比**：需要对应机器。

下一步（R314）：实现 encoder/VAE 阶段计时探针，产出第一份「encoder 实际耗时、VAE 实际耗时」数据。

## 五、状态

- 交付不变：**16.34 s/step（−8.9%）**，49 步 ×2 验证，relRMS 在原生带内，`H3_PERF_COMBO=1` 默认 OFF；
- 15 s：**在当前条件下不可达**（路线级结论，见上）；
- 机器 ts-217 直连正常；远端 master 含本轮记录与最小内核源码。
