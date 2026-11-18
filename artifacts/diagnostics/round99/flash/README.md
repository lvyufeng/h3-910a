# Round99 — 单 launch 流式视觉 flash attention 源码

**SOURCE ONLY。** 本目录里的文件没有被作者编译、启动或在 NPU 上导入过：无 SSH、无 CANN build、无 torch / torch_npu、无模型、无生产代码改动、无 git commit。所有数字属于 root driver；它负责编译这个 builder，并与**冻结的真实 Q/K/V + 原始 CPU Half MATH** 参考比对。本目录不声称精度、不声称加速、不声称默认启用。

## 目的

已执行的 Round97 视觉 SDPA 叶子每个 query chunk 需要**三次 kernel launch 加一次 GM 往返**：QK matmul → FP32 score tile 写回 GM → 独立 softmax/pack kernel → 独立 fused P@V kernel。本 kernel 把整个 row block 留在 UB 里，一个 launch 内完成完整 online softmax：

```
Q(64×72) , K(64×72) block , V(64×72) block  ->  O(64×72) Half
```

## 实际几何（H3 视觉注意力，2×2 merger 之前）

- H=16 heads，M=N=**4032** query/key 行（48×84 patch grid），原始 head dim **72** 零填充到 **80** = 5 个 16 的 NZ block。
- query 行按 **MB=BC=64** 分块：4032/64 = **63** 个精确 query tile、63 个精确 KV block。
- **tasks = 16×63 = 1008**，**30 cores**，**34 dynamic steps**，**63 次 dynamic KV 循环**（不展开）。
- 小尺寸 smoke（H=2, M=N=128, d=72）用同一份源码：tasks=4 → `block_dim=min(30,4)=4`，kv=2，编译便宜。
- 注意力**不加 mask（非 causal）**，缩放保持原始 `72 ** -0.5`：填充宽度 80 永远不是 scale。

## 数学（FP32 累加，无 Half 累积状态）

1. **QK** 一次 Half×Half→FP32 L0C matmul，覆盖完整 padded head dim（80）；score tile 全程 FP32 留在 UB，主分数不窄化。
2. 只用 **Half** 行最大值作为**稳定 shift 估计**：折叠 4 个 Half NZ key group 后 `h_reduce_max(axis=1)`。该部件支持 Half `vcmax`（Float `vcmax` 不支持）；shift 不需要精确，因为 softmax 在精确算术下平移不变。
3. shift 误差由 **host protocol** 约束（见下），因此 `exp(S - shift)` 落在 Half 范围内，two-Half 概率分解安全。
4. 跨 63 个 KV block 累积的一切都是 **FP32**：running shift `M`、分母 `L`、输出 `O`。三者用显式 `vector_dup`/逐 repeat 填充初始化，**不需要 first-block 的 `if`/`else` 向量分支**（910A 的 CANN SPR pass 曾在该模式崩溃）。
5. `P_hi = Half(exp)`、`P_lo = Half((exp - Float(P_hi)) * 1024)`；两个 FP32 L0C P@V 乘积，FP32 UB 合并 `hi + lo/1024`，累加器只在**最后一次**窄化。
6. 真分母是该 block `exp(S - shift)` 折叠 4 组后的 **FP32 求和**，不是 Half 求和。

## 行广播是显式的（不依赖隐式 broadcast）

本 SDK 没有 `h_broadcast`，源码**不依赖** high-level 算子自动广播。每个 per-row 值用 `_broadcast_row` 展开：对 64 行逐行 `Scalar.set_as`（UB→scalar 读），再用一次 `vector_dup` 把该 16 值行槽写到全部 `n1` 个 NZ group（`repeat_times=n1`、`dst_rep_stride=2*MB` blocks、32 B 对齐）。因此后续所有 `h_sub`/`h_mul`/`h_div` 的操作数**形状完全相同**。该标量循环是每处 64 次静态展开，**性能未测量、不做声称**；测试会验证 `2*MB` 的 repeat stride 恰好铺满 `(groups, MB, 16)` ——最后一行正好落在 tile 末尾。

## Host protocol（调用方强制，源码不 clamp）

`max |scaled QK| <= 16384`（Half 最大幅值的四分之一：即使 Half shift 短 8，所有 `exp` 仍在 `[e^-8, e^8]` 内，因而落在 Half 里）。调用方要么有限地实测 scaled QK 最大值，要么用 Cauchy-Schwarz 上界 `sqrt(72) * max|q| * max|k| <= 16384`。本模块提供 `check_host_protocol(...)`：越界即抛异常，**从不 clamp**，也不做事后容忍度放宽。

## 5 指针 ABI

GM 顺序 `[QHalfNZ, KHalfNZ, VHalfNZ, FloatAuxUnused, HalfOutNZ]`：

| 参数 | 形状 | dtype |
|---|---|---|
| `q_gm` | `(H, 5, M, 16)` = `(16,5,4032,16)` | Half |
| `k_gm` | `(H, 5, N, 16)` | Half |
| `v_gm` | `(H, N/16, 80, 16)` = `(16,252,80,16)` | Half |
| `aux_gm` | `(16,)`，**不使用**的占位 | Float |
| `out_gm` | `(H, 5, M, 16)` | Half |

`BuildCCE(inputs=[q,k,v,aux], outputs=[out])`：四个输入全部显式，否则自动收集会剪掉未使用的 Float aux。Half GM store stride 是 `M-64`，**不是** FP32 的 `2*(M-64)`。

## 预算（实测计划值）

- L1 `47104 B` = Q/K/V 3×10240 + P_hi/P_lo 2×8192。
- L0C `57344 B` = score 16384 + 2×PV 20480。
- UB `151424 B` < 196608 的安全边界：score tile 复用为 exp tile（原地 `h_exp`，与已执行 flash-attention 源码同模式），Half score tile 复用为 `P_hi`，行展开 tile 复用为 widen 后的 `P_hi`，PV low 复用为 rescale 展开，PV high 复用为最终商。
- 向量：group 内 `repeats=16`，broadcast `rep_stride=128 ≤ 255`，broadcast mask 16 值 = 64 B 且 32 B 对齐。

## 文件

| 文件 | 作用 |
|---|---|
| [flash_vision99.py](<flash_vision99.py>) | builder：`plan_kernel` / `build_kernel` / `check_host_protocol` / `capability_report`，CANN 懒加载 |
| [test_flash_vision99.py](<test_flash_vision99.py>) | 41 项 stdlib 契约测试（含 recording TIK stub 校验发射出的 operand 形状/覆盖/顺序） |
| [make_manifest99.py](<make_manifest99.py>) | 由交付文件重新生成 manifest |
| [manifest99.json](<manifest99.json>) | 文件 SHA/字节、几何、预算、assumptions 收据 |
| [../actual_sdk_highlevel99.json](<../actual_sdk_highlevel99.json>) | 只读的 CANN high-level API / intrinsic 实测表 |

## 测试

```bash
python3 -m unittest test_flash_vision99 -v
```

覆盖：真实与 smoke 几何、scale 是 72 而非 80/128、非 causal、head-dim 填充守卫、GM 形状/head stride、DMA 描述符（Q/K 5×64、V 4×80、P 4×64、output stride M-64）、L1/L0C/UB 预算、repeat 立即数与对齐、host protocol（边界内接受 / 越界与非有限拒绝 / Cauchy 路径 / 无 clamp）、capability 探测的 fail-closed 与 SoC 拒绝、5 指针 ABI、KV body 三个 matmul 形状、L0C→UB 覆盖、6 次 data_move 覆盖、Half shift + Float 分母、`h_exp` 为 Float、**所有 `h_*` 二元算子操作数同形同 dtype（无隐式广播）**、最终单次 Half cast、无 Half 累积、广播铺满、常量填充逐 repeat、64 次 inner loop 不展开、receipts 不声称任何结果。

## 待 root 验证的 SDK 假设（编译不是证明）

1. 真实两种几何 + H2/M=N128 smoke 都能编译；`block_dim=30`（real）/ 4（smoke）。
2. `h_reduce_max`/`h_reduce_sum` 在 `(MB,16)` tile 上以 `axis=1` 返回 `(MB,)` 行。
3. `reshape((n1,MB,16))` 是同一 UB buffer 的元数据视图，且复用顺序被尊重。
4. masked `vector_dup`（`repeat_times=n1`, `blk_stride=1`, `rep_stride=2*MB`）确实把 16 值写进每个 `(group,row)` NZ 槽，最后一行正好落在 tile 末尾。
5. 原地同形 `h_exp`/`h_sub`/`h_mul`/`h_add`（`dst==src0`）行为与 FA 源码对行向量的用法一致。
6. 融合输出与**冻结的真实 QKV CPU Half MATH** 一致，并在任何整塔结论之前先与已执行的 Round97 叶子对比。

## 不声称

不含 FP32 linear 权重 / teacher 张量 / BF16 gold 转换，不含缓存的 pixel/patch/hidden/DeepStack/conditioning，不含 Half 累积输出或 Half 分母，不 clamp、不做后验容忍度，作者未测任何速度或精度结果。

## 与 root 侧现状的关系

root 的 Round98 实测（4 个 case、8 rank 全完成）里当前路径 native 7.42%/2.173s、current 17.04%/2.034s、Round97 直连 NZ 的 native 9.918%/2.508s 与 precise 8.042%/3.063s，**没有一个低于 2s 或通过精度**。本 kernel 的目标是把"3 launch + GM 往返"换成结构上更省的一次 launch，但收益必须由 root 实测，不能从设计推断。
