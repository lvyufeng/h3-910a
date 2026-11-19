# Round103 / flash_bc256 — MB=32 × BC=256 单 launch 流式视觉 attention（仅源码）

**状态：SOURCE ONLY**。本目录的作者从未编译、启动或导入本 kernel 到任何 NPU：无 SSH、无 CANN
构建、无 torch/torch_npu、无模型、无 GPU/NPU、无 git、未改生产代码。所有数字都属于 root
驱动；本目录**不声明**任何精度、速度、容差或默认开启结论。

起始源码是 root 已审阅、执行中的 [R101 MB=64 × BC=128 kernel](../round101/flash_vision_mask_fixed101.py)
（SHA256 `ef21123261574bf31aa03e8d0ba8666dfc0f5fbb54a0c1d4694cce4c0749e69d`，本实验不改动它）。
R101 实测 BC=128 叶子约 23 ms（BC=64 约 25 ms，≈9%），完整视觉刚由 root 收集。

## 为什么做这个变体

R101 保留 MB=64 查询块、BC=128 KV 块，每个查询块 32 次 KV 迭代。本变体把查询块减半到
**MB=32**、KV 块加倍到 **BC=256**：动态 KV 循环降到 **16** 次，每行 Float 行广播与累积
`O`/`L` 更新处理的**行数减半**。代价是明确的、且**不假设为净收益**：查询块从 63 变 126，
每个查询块都要重读整个 K/V 流，**K/V 流量约翻倍**。是否更快只能由 root 实测。

## 几何与网格

| 项目 | 值 |
|---|---|
| H / M / N / D | 16 / 4032 / 4032 / 72（D 零填充到 80 = 5 个 NZ 块）|
| MB × BC | **32 × 256** |
| n_pad | 4096 = 16 × 256 |
| 查询块 / KV 块 | 126 / **16** |
| tasks / cores / steps | **2016** / **30** / **68** |
| 最后一块 | **192 真实 key + 64 phantom key**（组 12..15）|
| scale | 原始 `72 ** -0.5`，永远不是 padded 80 |

`30` 是原始 910A 代芯片 32 个物理核里可 launch 的数量，源码与文档都按 32 物理 / 30 可用表述，
不冒充 A2。

Smoke 几何 `(H=2, M=64, N=192, d=72) → n_pad=256` 被接受：**单个** KV 块、192 真实 + 64
phantom，tasks/cores/steps = 4/4/1，用来复现同样的 padding 路径。

## phantom key 必须从分母剔除（本变体的正确性关键）

真实 key 长度 4032 补到 4096，最后一块有 192 真实 + 64 phantom。零填充的 K 行使这些
score 恰好为 0，`exp(0 - shift)` 会给**每一行分母加 1.0**。**零填充的 V 单独无法修复**：
被放大的分母会缩小所有真实概率。因此：

1. 在 Half shift 估计与 Float 指数**之前**，把 phantom score 组显式置 `-FLT_MAX`；
2. 在 `exp` **之后**、Float 分母**之前**，把这些组显式置 `0.0`。
3. 两个 mask 都只在 `kv == kv_tiles-1` 的动态 `if_scope` 内执行（无 else），不展开尾块。

实现方式沿用 R101 已验证的 `_fill_flat` 形式：直接在基张量上按 64 lane 元素偏移寻址，
**不复用** R99/R101 那个导致编译失败的“单元素切片再索引”写法。

## 数学：什么保持 Float，什么只能是 Half

- `QK` 是 Half×Half → **FP32 L0C**（k=80）乘，score tile 在 UB 中保持 **FP32**，主 score 从不降精度。
- **Half 仅用于稳定 shift 估计**（本芯片无 Float `vcmax`）：显式 15 次 `vec_max` 折 16 组，
  再 `h_reduce_max` 得到行最大值。host 协议 `max|scaled QK| ≤ 16384` 保证 `exp(S-shift)`
  落在 Half 范围；更小的指数可正常下溢为 0。
- shift 是**已 Half 可表示**的值，才用 native `uint16 vec_or` + Half `vec_trans` 精确位拷贝
  广播成 Float（两个 16 行块：`shift_copy` 256 Half、`shift_broad` 512 Half、
  `shift_broad_float` 512 Float，再拷贝进 16 个 NZ 组，每组 512 个 Float）。
  **一般 Float rescale / 分母从不降 Half。**
- 跨 KV 累积的 shift `M`、分母 `L`、输出 `O` 全部 **FP32**；分母是 16 组显式 `vec_add`
  折 + `h_reduce_sum(axis=1)`（Float `vcadd`）得到的**真 Float 和**。
- 概率拆分：`P_hi = Half(exp)`；分母算完后把 `P_hi` 加宽回 Float，`(exp - Float(P_hi)) * 1024`
  **原位写回 score 缓冲**（省掉额外 Float low tile），再窄化成 `P_lo`。两个 FP32 P@V 后
  Float 合并 `hi + lo/1024`。
- 只有**一次**最终 Half cast（`O/L` 的 Float 商）。

## 内存预算（全部由 plan 推导，无假上限）

| 区域 | 字节 | 说明 |
|---|---:|---|
| L0C | **53248** | score 32768（16×32×16 FP32）+ 2 × pv 10240（5×32×16）|
| L1 | **119808** | Q 5120 + K 40960 + V 40960 + P_hi 16384 + P_lo 16384 |
| UB named | **141248** | score/exp/low 复用 32768、P_hi Half 16384、P_hi Float 32768、P_lo Half 16384、pv 3×10240、out Half 5120、7 个 128 B 行张量 |
| UB shift 附加 | **3584** | 512 + 1024 + 2048 |
| **UB 合计** | **144832** | < 安全上限 196608 |

Cube 操作数（每个 ≤ 64 KiB）：QK `L0A 5120 / L0B 40960`，P@V `L0A 16384 / L0B 40960`。
`informational_ub_aliases_not_summed` 里的项只是文档别名，**绝不再加进总量**。

## ABI / DMA / API

- 5 GM 指针，4 显式输入 + 1 输出，`aux` 为未使用 Float 占位以保持已验证 ABI：
  `Q HalfNZ(16,5,4032,16)`、`K HalfNZ(16,5,4096,16)`、`V(16,256,80,16)`、
  `aux Float(16)`、`out HalfNZ(16,5,4032,16)`。build 时 `inputs=[q,k,v,aux], outputs=[out]`。
- DMA：`q 5×32 stride M-32`、`k 5×256 stride Npad-256`、`V 16×80`、`P 16×32`、
  `out 5×32 stride M-32`（Half，无 Float 双倍 stride）、score L0C→UB 单 burst 32 KiB、
  P@V 单 burst 10 KiB。
- API：`flash_vision103.build_kernel(batch=16, m=4032, n=4032, d=72, out_dir=None, cores_max=30)`；
  metadata 保留 R99 已验证字段 `obj_path` / `obj_sha256` / `kernel_symbol` / `block_dim`，
  并额外记录 `GM_argument_count=5`、`kernel_arg_order`、`logical`、`grid`、`dma`、`buffers`、
  `vector`、`math`、`host_protocol`、`assumptions`、`capability`。
- 缓存/锁/不覆盖既有目录的构建行为与 R101 相同；`spec103.json` 明确
  `executed: false`、`not_a_speed_claim: true`、`numerical_tolerance: null`。

## 尚未在真实 CANN 上验证（root 必须逐项确认）

1. `burst=256`（8192 B）的 masked `data_move` 是否合法——**不预先缩到已验证的 128**。
2. Cube `m=32,k=80,n=256`（QK）与 `m=32,k=256,n=80`（P@V）的操作数驻留。
3. 32-lane `vector_dup` tail mask（Float 128 B / Half 64 B）。
4. 两次 16 行 `uint16 vec_or` + native Half `vec_trans` 的 shift 广播。
5. 16 组显式 `vec_max` / `vec_add` 折叠（关联性与 R101 的 8 组不同，低位可能变化）。
6. 单次 32 KiB `tensor_mov` L0C→UB burst。

## 文件与验证

| 文件 | 用途 |
|---|---|
| [flash_vision103.py](<flash_vision103.py>) | 唯一的 kernel 源码（plan + emitter + build），SHA256 `035b36c9e9ee68659668ae542221915905efc6e0c5b7ba67f37d57992422d78e` |
| [test_flash_vision103.py](<test_flash_vision103.py>) | 64 项**纯 stdlib** 契约测试，含从 R101 复用的记录式 TIK stub，真实运行 `_emit_kernel` |
| [make_manifest103.py](<make_manifest103.py>) | 生成 [manifest103.json](<manifest103.json>) |
| [manifest103.json](<manifest103.json>) | 源码 SHA、API、几何、内存、数学、未验证项 |
| [dryrun_build103.py](<dryrun_build103.py>) | 用**假** `tbe.tik` / `tbe.common.platform` 走通真实 `build_kernel` 代码路径（不是编译）|
| [dryrun103.json](<dryrun103.json>) | 该假 SDK 干跑的报告 |

本地验证（纯 CPU、无 CANN/torch）：`python3 -m unittest test_flash_vision103` → **64 tests OK**（干跑与 manifest 不覆盖既有结果：重跑前先删除 `dryrun103.json` / `manifest103.json`，与 `build_kernel` 不覆盖既有 kernel 目录的规则一致）；
`python3 flash_vision103.py --plan 16 4032 4032 72` 输出可解析的 plan JSON；
`python3 flash_vision103.py --plan 2 64 192 72` 为 smoke。stub 记录的真实发射结果：
3 个循环（core 30 / step 68 / KV 16，KV 动态未展开）、3 次 matmul
`[32,80,256] + 2×[32,256,80]`、15 次 `vec_max`、20 次 `vec_add`（15 分母折 + 5 合并）、
37 次 `vec_muls`（16 scale + 16 residual + 5 downscale）、2 次 `vec_or` / 2 次 `vec_trans`、
16 次 shift 广播拷贝（每次 64 个 32 B 单元）、64 次 phantom mask `vector_dup`、
8 次 `h_cast`（其中最终输出 1 次）、5 个同步 `new_stmt_scope`、3 个 `if_scope`
（1 个 task 边界 + 2 个 phantom mask）。

### 假 SDK 干跑（不是编译）

[dryrun103.json](<dryrun103.json>) 用纯 Python 的假 `tbe.tik` / `tbe.common.platform` 执行了**真实的**
`build_kernel` 代码路径（plan → `_emit_kernel` → `BuildCCE` → 复制 kernel_meta → `spec103.json`
→ 二次调用命中缓存 → block_dim 守卫）：

- 真几何：`kernel_symbol = flash_vision103_h16_m4032_n4032_np4096_d72_mb32_bc256_c30_kv16`，
  `block_dim = 30`，`GM_argument_count = 5`，`obj_sha256` 与文件自洽，
  engine/Runtime5GM 需要的 `obj_path` / `obj_sha256` / `kernel_symbol` / `block_dim` **全部存在**，
  缺失键为空；`spec103.json` 中 `executed: false`、`not_a_speed_claim: true`、
  `numerical_tolerance: null`。
- 二次同参调用命中缓存（同一 obj 路径与 SHA，`compiled_now: false`）。
- smoke 几何 `block_dim = 4`、`n_pad = 256`、phantom 64。
- 把缓存 meta 的 `blockDim` 篡改为 30 后，守卫按预期拒绝复用（root engine 依赖这个检查）。
- `mb=64` / `bc=128` / `m=48` / `d=64` 四种非法 plan 全部被拒绝。

**它证明的是**元数据装配、对象 SHA 自洽、缓存与守卫；**它不证明**发射出的程序能编译或运行——
写入的 `.o` 只是占位字节串，其 SHA 是文件哈希而非二进制身份。真实编译/执行只能由 root 在
ts-217 上完成。

这些是**结构证据**，不是编译器结果、不是精度结果、也不是速度结果。