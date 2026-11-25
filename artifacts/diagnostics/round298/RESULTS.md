# R298 — 分段定位结果：nz2ndEn 无效；手工拼 cube 布局成本过高，改用高层 Matmul API

## 一、本轮两个实验结论

1. **输出并非全零**：`nonzero = 20780 / 32768`，但 `got[0][0..3] = 0/0/0/0` 而参考是 `3.55/-3.29/-5.09/-0.90`
   ⇒ 数据**被写入了，但位置错**（布局/映射问题，不是没写回）；
2. **把 `DataCopyCO12DstParams.nz2ndEn` 从 false 改 true 并重建重装后，输出 100% 相同**
   （包括 nonzero 计数与 maxdiff）⇒ 该开关不是问题所在，问题更早（L1/L0 布局或 Mmad 语义）。

## 二、判断：手工拼 cube 布局不划算

当前 kernel 里同时有三处布局假设未经验证（`Nd2NzParams` 的 `dstNzC0Stride`、`LoadData2dParams` 的
`repeatTimes/srcStride/ifTranspose`、`DataCopyCO12DstParams` 的 `dstStride/srcStride`），
每处都需要单变量实验才能定位，而每次实验 = 改码 + 编译(~2 min) + 安装 + 跑测试。

**更优路径：改用 AscendC 高层 Matmul API**，由框架负责 ND↔NZ、L0C→UB、分形拆分：

```cpp
#include "lib/matmul_intf.h"
using namespace matmul;
Matmul<A_TYPE, B_TYPE, C_TYPE, BIAS_TYPE> mm;
REGIST_MATMUL_OBJ(&pipe, GetSysWorkSpacePtr(), mm, &tiling);   // tiling 来自 host
mm.SetTensorA(aGm); mm.SetTensorB(bGm); mm.IterateAll(cGm);
```

代价：host 侧需要用 `MultiCoreMatmulTiling` 计算 `TCubeTiling` 并写进 tiling 结构
（当前算子还报过 `FlashAttentionCustom do not registe tiling struct!!!`，需一并补上）。
收益：布局/搬运/分形全部由框架处理，S1a-1 的正确性风险大幅下降，且这条 API 正是
S2 流水（`mm.SetTail`、双缓冲、`Iterate` 手动调度）的基础。

## 三、下一步（R299）

1. op_host 补 tiling：定义 `FlashAttentionCustomTilingData { TCubeTiling mm; }`，
   用 `matmul_tiling::MultiCoreMatmulTiling` 按 M=64/N=512/K=128 计算并写入；
2. kernel 改用 `Matmul` + `IterateAll`，先做 q@kᵀ（B 用 `SetTensorB(kGm, isTransposeB=true)`）；
3. 安装后用 `test_fa296` 验证 `max_abs_diff`/`relRMS`；
4. 通过后扩展完整 FA，再用 `Matmul` 的手动 `Iterate` 做 S2 流水。

## 四、状态

- 交付不变：**16.34 s/step（−8.9%）**，49 步 ×2 验证，数值在原生带内，`H3_PERF_COMBO=1` 默认 OFF；
- AscendC：S0 ✅、S0.5 ✅、S1a-0 ✅、S1a-1 手工 cube ❌（本轮定位清楚，转向高层 API）；
- 机器 ts-217（910A）租约至 20:52 UTC。
