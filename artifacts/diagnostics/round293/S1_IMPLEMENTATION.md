# S1 — AscendC FA 等价实现方案（骨架已生成，待写 kernel）

## 一、已生成的骨架

```
/tmp/h3_r221/faop/proj/
├── op_kernel/flash_attention_custom.cpp        # 入口（TODO: kernel impl）
├── op_kernel/flash_attention_custom_tiling.h
├── op_host/flash_attention_custom.cpp          # 原型/推断/tiling
└── build.sh
```

入口签名：

```cpp
extern "C" __global__ __aicore__ void flash_attention_custom(
    GM_ADDR q, GM_ADDR k, GM_ADDR v, GM_ADDR o, GM_ADDR workspace, GM_ADDR tiling)
```

## 二、S1 目标（等价、不做流水）

同形状 **B1 H7 N41792 D128**，dense、FP16 输入、FP32 softmax、FP16 输出、无 1/sqrt(d)
（调用方已对 q 做 scale，与 TIK 版约定一致）。

## 三、Tiling 方案（先保守，追求正确性）

| 参数 | 取值 | 依据 |
|---|---|---|
| Br（q 行块） | **64** | L0C 约束 `Br×Bc ≤ 65,536`（R291） |
| Bc（k/v 列块） | **512** | 64×512×4 = 131,072 ≤ 256 KB，留余量给双缓冲 |
| D | 128 | 固定，N0=16 |
| 核数 | 20（`GetBlockNum()` 可用核，AIV/AIC 混合时按实际） | 由 tiling 下发 |

## 四、AscendC 实现骨架（关键 API 映射）

```cpp
#include "kernel_operator.h"
using namespace AscendC;

class FaKernel {
  TPipe pipe;
  TQue<QuePosition::VECIN, 1> qIn, kIn, vIn;      // GM->L1 用 TQue<L1>
  TQue<QuePosition::VECOUT, 1> oOut;
  TBuf<TPosition::VECCALC> sBuf, pBuf, mBuf, lBuf, oaccBuf;
  GlobalTensor<half> qGm, kGm, vGm, oGm;

 public:
  __aicore__ void Init(GM_ADDR q, GM_ADDR k, GM_ADDR v, GM_ADDR o, ...);
  __aicore__ void Process();   // 每核处理若干 (q-block) 行块
  __aicore__ void ComputeOneQBlock(int qBlk);
};
```

单个 q 块的循环（与 TIK 版同构，但由我们显式控制缓冲）：

1. `DataCopy` q 块 GM→L1（L1 上做 NZ 变换由 `LoadData`/`SetFmatrix` 处理）；
2. 对每个 kv 块：
   a. `DataCopy` k 块 → L1，`Mmad`（QK^T）→ L0C → `DataCopy` 到 UB 得 `S`（FP32/FP16）；
   b. `ReduceMax`（行 max，`WholeReduceMax`/`ReduceMax`）→ `m_new`；
   c. `Sub` + `Exp` → `P`（FP16）；
   d. `ReduceSum`（行 sum）→ `l`；
   e. `Mmad(P, V)` → L0C → 累加器 `O`（需在 L1/L0C 间做 rescale：`O = O*alpha + PV`）；
3. `O / l` → `Cast` FP16 → `DataCopy` 到 GM。

**S2 的优化点**（TIK 做不到的部分）：把 b/c/d 的向量段与 a/e 的 cube 段
用**双缓冲 + SetFlag/WaitFlag** 交错，使 cube 与 vector 同时工作；Q/K/V 的 `DataCopy`
用 `TQue` 深度 2 预取。

## 五、验收判据

| 项 | 目标 |
|---|---|
| 数值 | 与 TIK 参考 `relRMS ≤ 1e-04`、`max_abs_diff` 记录、无 NaN/Inf（用 R283 闸门同 seed） |
| 性能 S1 | 记录 ms（预计慢于 126 ms，仅作基线） |
| 性能 S2 | **≤105 ms**（到 15 s 的必要条件） |

## 六、风险

- AscendC 的 FA 类算子无现成样例可抄，`Mmad`+`ReduceMax/Sum`+FP32 累加的语义对齐需多轮调试；
- UB/L1 预算需实测（S1 保守 tiling 先跑通，S2 再压缩）；
- 需要 CANN 的 AscendC 头（`kernel_operator.h`）与正确的 C++ 头路径（R292 配方已记录）。

## 七、当前状态

- S0 通过（工具链可编译可打包）；
- S1 骨架已生成，kernel 实现待写；
- 交付冻结 16.34 s（−8.9%），不受本项目影响直到 S3/S4 通过数值闸门。
