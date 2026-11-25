# S1a — 第一个可验证子步骤（CANN 无 FA 样例，需自研）

## 一、样例检索结果

在 `cann-9.2.0` 内检索 `*flash*attn*` / `*attention*` / `*sample*`：

- **没有任何 AscendC 的 FlashAttention 内核样例**；
- 只有 aclnn 的接口头（MultiScaleDeformableAttention 等，与 FA 无关）；
- `tools/op_project_templates/ascendc` 只有通用模板（add/abs 类）。

⇒ S1 必须**从零实现**，无现成代码可抄，这是本项目最大的工程风险点（已在 #18 标注）。

## 二、S1a 的定义（把大任务切成可验证小块）

**目标**：先跑通 AscendC 里"一块 `Mmad` matmul + UB 取回"的完整链路，形状用真实子问题：

- 输入：q 块 (Br=64, D=128) FP16、k 块 (Bc=512, D=128) FP16（GM 中为 ND 布局）；
- 计算：S = q × k^T，FP32 累加，输出 (64, 512) 到 GM；
- 验收：与 torch 的 `q @ k.T` 对比 `max_abs_diff` / `relRMS`（期望 ~1e-3 内，FP32 累加 vs torch FP16 GEMM 的差异需实测）；
- 意义：一旦 S1a 通过，FA 的 cube 部分（QK^T 与 PV）就有了可复用的代码骨架，
  剩余只是 softmax 向量段与循环控制。

## 三、S1a 涉及的 AscendC API（初版）

```cpp
#include "kernel_operator.h"
using namespace AscendC;

// 1) GM->L1
GlobalTensor<half> qGm, kGm, sGm;
TQue<QuePosition::A1, 1> qL1;      // A1 = L1 for A matrix
TQue<QuePosition::B1, 1> kL1;
// 2) L1->L0A/L0B
TQue<QuePosition::A2, 1> qL0A;
TQue<QuePosition::B2, 1> kL0B;
// 3) cube 计算 -> L0C
TQue<QuePosition::CO1, 1> cL0C;
// 4) L0C->UB (FP32) -> GM
```

需注意：910B 的 ND→NZ 转换由 `LoadData`/`Nd2Nz` 或 `SetFmatrix`/`SetLoadDataPaddingValue` 完成，
首个版本建议直接用 `AscendC::LoadData` 系列并接受较慢的搬运。

## 四、下一步

1. 在 `faop/proj/op_kernel/flash_attention_custom.cpp` 里先写 S1a（只需 q/k/s 三个 GM 参数即可测）；
2. 用 R292 的头文件配方编译；
3. 写一个 host 侧 python 测试（构造随机 q/k、调用算子、与 torch 对比）；
4. S1a 通过后再扩展为完整 FA（softmax + PV + 循环）。
