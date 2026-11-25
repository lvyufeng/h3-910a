# R255 — 立项 + S1：FA 内核是**向量受限**（vec 54.8% vs cube MAC 22.7%），三条新工具链全部可用

## 一、立项（用户 2026-11-25 决定）

用户选择「立项 FA 内核重写，冲 15 s」，并要求**充分利用一切可能手段**：
AscendC / TBE / Triton 算子、或把模型做成**整图**。
计划已建 Issue **#17**（S0–S5 分阶段，含数值闸门）；本地计划见 `round255/PLAN.md`。

## 二、工具链可用性（本轮实测）

| 路线 | 状态 | 证据 |
|---|---|---|
| TBE / TIK（现有） | ✅ 在用 | 现有 `fa910a` 内核即 TIK 生成 |
| **AscendC** | ✅ 可用 | `msopgen` 在 `/home/mseco/Ascend/cann-9.2.0/bin/`；头文件在 `.../aarch64-linux/include/ascendc` |
| **Triton（CANN 后端）** | ✅ 已装 | 站点包内有 `triton/language/extra/cann/...`（triton-ascend） |
| **整图模式** | ✅ 可用 | `torch.npu.NPUGraph`；`torch.compiler.list_backends()` 含 `npu`、`npugraph_ex`、`npugraphs` |

## 三、S1：FA 内核自身 profile（msprof `PipeUtilization`，8 次采样一致）

| 指标 | 值 |
|---|---|
| Task Duration | **152,202 µs/次** |
| aicore_time | 151,268 µs |
| **`vec_ratio`** | **0.548** |
| **`mac_ratio`（cube MAC 效率）** | **0.227** |
| `scalar_ratio` | 0.235 |
| `mte1/2/3_ratio` | 0.166 / 0.114 / 0.100 |
| `cube_utilization` | 99.39% |

### 判读（这是重写的直接依据）

1. **内核是向量受限**：vector 管 54.8% 忙，而 cube 的 MAC 效率只有 **22.7%**。
   `cube_utilization 99.4%` 只表示 cube 被“占用”，不代表在做有效 MAC
   ⇒ 现状是 **softmax/缩放/掩码等向量工作占主导，矩阵乘没有被流水起来**。
2. 这解释了 46 TFLOPS/卡（15% MFU）：cube 有效利用率被向量段串行化。
3. **重写主攻方向**：
   - **cube/vector 软件流水**（双缓冲：QK^T 结果在 cube 算 PV 的同时做 softmax；MTE1/MTE2 预取下一块 K/V）；
   - **削减向量工作量**：合并缩放进 exp、去掉冗余掩码/重复 cast、保持 FP32 softmax 语义不变；
   - **core task map 均衡**：30 核、tile 网格 342×144（trim 后 327×138），末块尾效应。
4. 实现载体的取舍：
   - **AscendC**：对 pipe/流水控制最直接，适合做双缓冲与手工调度（成本高、可控性最好）；
   - **Triton-CANN**：迭代快，适合先验证 tiling/流水收益（若 Triton 生成代码的 cube 利用率已明显更高，可先走这条路）；
   - **整图**：不改变单核效率，但可消除算子间调度/搬运的重复（R254 已知设备 100% 忙，
     整图主要收益在**跨算子融合与去冗余 cast**，属第二优先级）。

## 四、下一步（S2，下轮）

1. 写 FA 的 **Triton-CANN 原型**（同形状、FP16 输入、FP32 softmax 累加、dense 语义），
   在独立微基准上与 TIK 版对比 ms/块与 `mac_ratio`；
2. 若 Triton 原型显示 cube MAC 明显提升 ⇒ 继续优化并进入数值闸门①；
   若 Triton 生成质量不足 ⇒ 转 AscendC 双缓冲实现；
3. 并行评估整图：用 `npugraph_ex` 对单块做图捕获，测能否减少每块小算子/搬运（R251 的 N/A 群）。

## 五、合规

- 本轮仅 profile 与工具链探测；未改盘上源码、未装候选、无数值变更；
- 默认配置与 FA trim/tile 状态不变；只在 ts-217 运行。
