# R299 — 高层 Matmul API 版本编译通过，数值仍不对（relRMS=20）

## 一、本轮完成的事

1. 用高层 API 重写 kernel（源码 `round299/fa_kernel_matmul_api.cpp`）：

```cpp
using aType = MatmulType<TPosition::GM, CubeFormat::ND, half>;
using bType = MatmulType<TPosition::GM, CubeFormat::ND, half, true>;   // transpose B
using cType = MatmulType<TPosition::GM, CubeFormat::ND, float>;
Matmul<aType, bType, cType> mm;
REGIST_MATMUL_OBJ(&pipe, GetSysWorkSpacePtr(), mm, &tilingData.mm);
mm.SetTensorA(qGm); mm.SetTensorB(kGm); mm.IterateAll(tGm);   // F32 中间结果在 workspace
// 再 DataCopy + Cast 成 half 写回 o
```

2. host 侧用 `MultiCoreMatmulTiling` 生成 tiling（源码 `round299/fa_host_tiling.cpp`，
   `SetShape(64,512,128)`、`SetOrgShape` 同、A/B=FP16、B 转置、C=FP32、`SetDim(1)`）；
3. 修掉三个构建问题：
   - `adv_api/matmul/bmm_tiling.h` 找不到 → 改用**绝对路径** include（CMake `include_directories` 对 opbuild 阶段无效）；
   - kernel 里 `tilingData` 未声明 → 补 `REGISTER_TILING_DEFAULT(FlashAttentionCustomTilingData)`;
   - tiling 结构改写为 `{ int32_t size; TCubeTiling mm; }`。
4. **`build.sh` 通过并打包成功**。

## 二、数值结果（未通过）

```
probe got[0][0..3] = -2.0000 -0.7368 -0.0000 0.6650
probe ref[0][0..3] =  3.5545 -3.2861 -5.0882 -0.8997
probe nonzero = 24610/32768
RESULT maxdiff=526.23 relRMS=2.032e+01 bad=32447
```

比手工 cube 版更差（relRMS 从 1.0 变 20），说明 **tiling/转置语义仍不对**，而不是搬运路径问题。

## 三、最可能的三个原因

| # | 疑点 | 说明 |
|---|---|---|
| 1 | `MatmulType` 模板第 4 参数语义 | 我按 `isTrans` 用，实际可能是 `CubeFormat`/`Layout` 位；B 的转置应通过 `SetTensorB(kGm, true)` 而不是模板 |
| 2 | `SetOrgShape` 与 `SetShape` 组合 | 对 ND 布局 B（512×128）转置后 K×N=128×512，OrgShape 可能要求 (M,N,Ka,Kb) 四参数形式 |
| 3 | `IterateAll(tGm)` 的 C 布局 | C 是 ND float，但 workspace 的分配/对齐或 `GetLibApiWorkSpaceSize` 计算可能不够，导致越界写成垃圾 |

## 四、下一步（R300）

1. 查 `MatmulType` 与 `SetTensorB` 的确切签名（`matmul_intf.h`），确认 B 转置的正确表达；
2. 用**最小形状**（M=N=K=16，单位矩阵）先跑通，再放大到 64×512×128；
3. 每次只改一个参数，记录 maxdiff，直到 ndigits 与 CPU 参考一致（目标 `relRMS ≤ 1e-3`）；
4. 通过后进入完整 FA 与 S2 流水。

## 五、状态

- 交付不变：**16.34 s/step（−8.9%）**，49 步 ×2 验证，数值在原生带内，`H3_PERF_COMBO=1` 默认 OFF；
- AscendC：S0 ✅、S0.5 ✅、S1a-0 ✅、S1a-1 手工 cube ❌、**S1a-1 高层 API 编译 ✅ / 数值 ❌**；
- 机器 ts-217（910A）租约至 20:52 UTC；远端 master 将包含本轮源码与记录。
