# R310 — A/B 交换实验：现象随操作数改变，但问题未收敛

## 一、实验

交换 A/B：`A = k(512×128)`、`B = q(16×128)`（转置），tiling `SetShape(512,16,128)`，
输出 512×16。host dump 确认 tiling 正常：

```
[TILINGDUMP] usedCore=1 M=512 N=16 Ka=128 Kb=128 scM=512 scN=16 scK=128
             baseM=256 baseN=16 baseK=64 stepM=2 stepN=1 isBias=0
```

实测：

```
SWAP RESULT M=512 N=16 maxdiff=19.2035 relRMS=1.344e+00 rowVar=4.793e+00 colVar=3.071e+00
```

## 二、判读

- 原始方向（A=q 16 行）：输出**每行都等于第 0 行**（rowVar ≈ 0，rms 0.001 匹配）；
- 交换后（A=k 512 行）：输出既非常数行（rowVar=4.79）也非常数列（colVar=3.07），整体仍错。

⇒ 现象**随 A 的内容/形状改变**，说明不是单纯的「C 写出 stride=0」（否则交换后仍应常数行）；
但也未能定位到 A 加载 stride —— 交换后的错误是另一种形态。

## 三、当前判断（诚实）

高层 `Matmul` API 在本机的 910A 上，按我们目前的模板参数/布局组合**没有正常工作**。
可能原因：

1. `MatmulType` 的模板参数语义（`CubeFormat`/`LayoutMode`/`ISTRANS` 位）在该 CANN 版本与我理解的不同；
2. 910A（arch 2201 族）对 GM 输入的要求可能是 **NZ/fractal** 而非 ND，框架的 ND 自动转换未生效；
3. `IterateAll` + 单核 `SetDim(1)` 组合在该版本上对 C 的输出布局有额外要求。

## 四、建议的下一步（换策略，不再盲试）

**停止盲试模板参数**，改为：
1. 找 CANN 安装包/官网上的 **AscendC Matmul 官方样例**（`Samples` 目录、`ascendc` 示例仓），
   照抄其 host tiling + kernel 的完整调用序列（尤其是 `MatmulType` 模板与 `SetOrgShape/SetSingleShape`）；
2. 用最小形状 16×16×16 + 单位矩阵先对齐样例输出；
3. 再迁移到 64×512×128。

本会话已在 AscendC 上投入约 18 轮，卡在基础 matmul 语义；继续盲试的边际收益低。

## 五、状态

- 交付**保持不变**：16.34 s/step（−8.9%），49 步 ×2 验证，relRMS 在原生带内，`H3_PERF_COMBO=1` 默认 OFF；
- AscendC：S0 ✅、S0.5 ✅、S1a-0 ✅、S1a-1 未通过（matmul 语义未对齐）；
- 机器：ts-217 直连正常；**fleet bridge 将其误报 OFFLINE**（需用户侧修复 bridge 探测）。
