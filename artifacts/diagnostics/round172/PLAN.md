# R172 — 内核工作新阶段：授权、工具链侦察与实施计划

## 0. 授权变化（本轮起点）

用户 2026-11-23 明确授权：**"你可写ascendc tbe甚至直接逆向写cce指令都可以"**。

此前 #5 一直处于 `Awaiting approval`，理由是"新 FA 内核必然改变舍入"。该授权解除的是**实现手段的限制**
（可写 AscendC / TBE / 逆向 CCE），因此内核级工作可以开始。注意：**授权手段 ≠ 授权任意数值变化**，
所以下面每一步仍按既有纪律执行——候选单独测量、失败留痕、可测数值变化先报告。

## 1. 工具链侦察（ts-217，CANN 9.2.0，Ascend 910B × 8）

| 项目 | 结果 |
|---|---|
| `bisheng` | **clang 15.0.5，target aarch64** — AscendC/CCE 编译器，可用 |
| `ccec` | 在 `$ASCEND_TOOLKIT_HOME/bin/ccec` |
| `msopgen` | "Generate, **extend**, and compile Ascend **custom operator** projects" — 自定义算子工程 |
| `aoe` / `atc` | 在位（自动调优 / 图编译） |
| TBE python DSL | `tbe`、`te`、`tbe.dsl`、`tbe.tik` **全部可导入** → 可写 TBE 算子 |
| `topi` / `tik` / `ascendc`(py) | 缺失（不影响：AscendC 是 C++ 头文件 + bisheng） |
| AscendC 头 | `kernel_operator.h`、`tikcfw`、`basic_api`/`c_api`/`highlevel_api`/`tensor_api` |
| 自定义算子插件目录 | `$ASCEND_TOOLKIT_HOME/opp/vendors/`（含 `config.ini`、`customize`）→ **可安装自定义算子** |
| SOC | 910B × 8，每卡 32 GB HBM（npu-smi 25.5.2） |
| 参考实现 | 主机上有 `MindIE-SD/csrc/ops/prompt_flash_attention`（**24,506 行**，op_host tiling + op_kernel，支持 DT_FLOAT16 与 sparse mode）与 `incre_flash_attention` |

**结论：三条路都可行。** 自定义算子（msopgen + AscendC + bisheng + vendors）是风险最低的落地方式；
MindIE 的 FA 是 FA 阶段的高价值参考，但 2.4 万行、带自己的构建体系，需单独评估。

## 2. 关键发现：审计的成本不是带宽，是每次调用的固定开销

- 每步被审数据量：**24.44 GiB**（R170 实测形状：`qkv_proj` 10448×10752、`mlp.fc1` 10448×14336、`adaln` 3×96768）
- 按 HBM 带宽粗算只需 **约 20 ms**
- 实测审计耗时 **2.87 s/步**（R162）→ 有效带宽仅 **约 8.5 GB/s**
- 每步 **155 次**调用，约 **18.5 ms/次**

⇒ 成本来自**每次调用内部 5–6 个 kernel 的启动与依赖链**（format cast 全量拷贝、int16→int32 扩张、3 个全尺寸 mask、2 次求和），
而不是数据搬运本身。**这正是单趟自定义内核能吃掉的部分。**

## 3. 分阶段计划（按"价值 ÷ 风险"排序）

### 阶段 A（先做）：单趟统计内核 `h3_range_stats`

- **作用**：输入 FP16 张量的 int16 视图，一趟遍历输出两个 int32：非有限计数、饱和计数。
  （`magnitude >= 0x7c00` 覆盖 NaN+Inf；`magnitude == 0x7bff` 即 |x|==65504。位模式，**不使用浮点判据**——R165 已证本设备它们不可靠。）
- **预期收益**：每次 18.5 ms → 1–2 ms，**每步省约 2.5–2.9 s**
- **风险**：低。单算子、无矩阵运算、无累加顺序问题
- **验收**：与原生位模式计数**逐条位精确相等**——沿用 R166/R167 的双重验证（37 个对抗用例 + 465 次真实调用）
- **附带价值**：跑通 msopgen → bisheng → vendors 安装 → aclnn 调用的整条链路，为后续阶段铺路

### 阶段 B：guard 计算融合（上界 1.665 s）

`out_proj` 6.71×、`fc2` 2.45× 于独立速率（R147），源于 `_MiniMaxH3HalfInputFP32OutputLinearMethod`。
需要用自定义 GEMM（FP16 输入、FP32 累加与输出，与原数学一致）替代当前路径。
**风险**：中——涉及矩阵累加顺序，可能改变舍入，必须先做同输入比对并请用户决定。

### 阶段 C：FA 内核（上界 4.38 s）

当前 41.1 TFLOPS/卡（含补齐 45.1），本机实测 GEMM 最优 96.9。候选路线：
1. 先试 `aoe` 自动调优现有 `flash_attention` 的 tiling；
2. 参考 MindIE `prompt_flash_attention` 移植到 msopgen 自定义算子；
3. 必要时逆向 CCE。
**风险**：高（2.4 万行参考、构建体系不同、arch35 与 910B 的差异未核）。

## 4. 与 10 s 目标的关系（保持诚实）

R171 的上界计算是：所有已识别收益同时拉满 = **10.20 s**，其中审计按"不可回收 0 收益"计。
**阶段 A 改变了这个前提**——它可能把 2.87 s 的大部分变成可回收：

| 组合 | 步时 |
|---|---|
| 现状 | 17.90 s |
| + 阶段 A（审计融为单趟，乐观 −2.6 s） | **≈15.3 s** |
| + 阶段 B（guard −1.67 s） | ≈13.6 s |
| + 阶段 C（FA 达 96.9 TFLOPS/卡，−4.38 s） | **≈9.3 s** |

**即：内核路线的三阶段全部成功，10 s 才第一次真正可达。** 这也说明阶段 A 是性价比最高的一步：
它同时是"最容易验证"和"单独就能买 2.5 s 以上"的那个。
