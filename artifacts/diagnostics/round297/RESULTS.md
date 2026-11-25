# R297 — S1a-1 编译通过、数值未通过（relRMS=1.0）：cube 布局语义待对齐

## 一、本轮进展

1. 找到 L0C 出口的正确接口：`DataCopy(dst, src, DataCopyCO12DstParams)`
   （`kernel_operator_data_copy_intf.h:301-306`；结构定义 `kernel_struct_data_copy.h:127-150`，
   字段 `nSize / mSize / dstStride / srcStride / quantPre / reluPre / channelSplit / nz2ndEn`）；
2. 用它替换错误的 `DataCopy(ub, cL0, count)` 后 **`build.sh` 通过并打包成功**；
3. 安装后用真实形状（q 64×128、k 512×128 → o 64×512）跑数值测试。

## 二、实测结果（未通过）

```
RESULT M=64 N=512 K=128 maxdiff=15.2793 relRMS=1.000e+00 bad=32436
```

- `bad=32436` = 全部 32768 个元素里 99% 超差（阈值 5e-2）；
- `relRMS = 1.000` 说明输出与参考**完全不相关**（不是精度问题，是布局/步长问题）；
- 编译与执行都成功 ⇒ kernel 确实跑了，只是数据搬运/转置语义不对。

## 三、最可能的三个原因（按可能性排序）

| # | 位置 | 疑点 |
|---|---|---|
| 1 | `Nd2NzParams`（GM→L1） | q 是 M×K 行主序，`dstNzC0Stride = kM` 的语义是「NZ 分形里 C0 方向的 stride」，先行应取 `kK` 的分形数，需按 `dstNzC0Stride = K/16` 之类调整 |
| 2 | `LoadData2dParams`（L1→L0A/B） | `srcStride` 单位是 32B block、`repeatTimes` 是 16 行一组；k 的 `ifTranspose=true` 可能需要在 L1 里先按列主序摆好 |
| 3 | `DataCopyCO12DstParams`（L0C→UB） | `dstStride/srcStride` 取 0 是默认值，实际要按 16×16 分形的 block 粒度给；`nz2ndEn=false` 可能应为 true（分形→行主序） |

## 四、下一步（可执行的定位法）

**不要盲改三个地方**，而是先用「单块」把每一段单独验证：

1. **验证 Nd2Nz**：kernel 里只做 GM→L1→UB→GM 的搬运（不经 cube），用 python/CPU 对比，
   确认 q/k 在 L1 的 NZ 布局能被 UB 读回成原样（或确认转换后的形状语义）；
2. **验证 Mmad**：用已知常量矩阵（如 q 全 1、k 第 j 行只有第 j%K 列为 1）让 `S` 的输出可直接肉眼核对；
3. 三段都通后，再拼回完整链路并跑随机数值。

## 五、状态

- 交付不变：16.34 s/step（−8.9%），49 步 ×2 验证，数值在原生带内，`H3_PERF_COMBO=1` 默认 OFF；
- AscendC：S0 ✅、S0.5 ✅、S1a-0 ✅、**S1a-1 编译 ✅ / 数值 ❌（待对齐布局）**；
- 机器 ts-217（910A）租约至 20:52 UTC；源码 `round297/fa_kernel_s1a1_v2.cpp`。
