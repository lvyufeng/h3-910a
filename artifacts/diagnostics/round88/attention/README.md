# Round88 / attention — accurate vision SDPA for the 72-dim H3 vision tower

接入文件：`accurate_vision_attention88.py`（root 要求的文件名），导出类
`AccurateVisionAttention88(candidate, runtime, kernel_dir)`，API 为
`prepare()` / `set_enabled(bool)` / `snapshot()` / `uninstall()`，另有
`install(...)` 工厂（默认 `enabled=False`）。**源码阶段（source only）**：未编译、
未上 NPU、未跑模型、未安装、未改生产代码、未提交；作者未使用 SSH/NPU/GPU/CANN。

## 要替换什么

R79 私有候选里的 `_vision_sdpa_bf16_compatible` 把 Half q/k/v 提升到 Float32 后做三次
Float32 torch matmul。R85 已在实际 text 塔上证明本栈 Float32 matmul 并不等于真正的
FP32 累加（注入精确 CPU 概率后 PV 仍差约 1.2e-4）。视觉塔用的是同一个 helper，因此它
的 P@V 精度同样未经验证。

本引擎只替换这一个**模块级** helper：`setattr(candidate, HELPER_NAME, self.forward)`，
不动 `candidate._scaled_dot_product_attention`，不 patch `torch.nn.functional`，不碰
biased Half Linear（R87 bias 单次 Half 转换仍是独立开关）。

## 算术路径（Half 进 → FP32 累加 → 一次 Half 出）

真实视觉注意力几何：16 heads、每图 4032 keys、head_dim 72（1152/16），两张图各调用一次，
27 层每层 2 次。

1. 原始 Half q/k/v 原值拷入零填充缓冲：head_dim 72→128、key 轴 4032→4096；
2. 复用**已实跑**的 R85 `batch_gemm85`（SHA `546cd211…`）与五 GM
   `runtime5gm83`（SHA `c8890dbd…`，库 `bd50d700…`）内核，H=16 一次算完：
   - QK：`(16, m, 128) × (16, 4096, 128)ᵀ → (16, m, 4096)` FP32；
   - PV：`(16, m, 4096) × (16, 4096, 128) → (16, m, 128)` FP32；
   - 每个 head 的 B 独立、单一 FP32 L0C 跨全部 K 块累加，无 Half 部分和、无 FP32 权重影子；
3. query 分块 7×512 + 尾块 448（共 8 块），尾块在 kernel 内补到 M=512、输出再裁剪；
4. **先裁掉 padding key 再做 softmax**：补齐的 key 分数恰为 0，`exp(0)=1` 会污染分母；
   所以分数 `[:, :, :4032]` 之后才 `torch.softmax(dim=-1)`，**无 causal mask、无 -inf**；
5. 缩放用**原始** `72 ** -0.5`（= 0.1178511301977579），绝不用 `128 ** -0.5`
   （否则真实分数被乘上 `sqrt(128/72)`）；
6. Float P 拆成 `P_hi = Half(P)` 与 `P_lo = Half((P - Float(P_hi)) * 1024)`，
   P 的 K 方向补零到 4096；两个 P@V 结果**都保持 FP32**，在 FP32 里相加后再**仅一次**
   转 Half，最后裁剪出 72 个真实输出列（不改变原值）。

零填充安全性：head_dim 多出的 56 列在点积里是精确的 0 乘积；PV 侧 P 的 padding key 行与
V 的 padding key 列都是 0，多出的收缩项也精确为 0。

## 预算（每次启用调用，确定性记账）

`padded_buffer_bytes()`：常驻约 **100,401,152 B（≈95.75 MiB）**（padded q/k/v、V 转置、
packed K/V）；单块临时峰值约 **341,835,776 B（≈326 MiB）**（QK 的 128 MiB FP32 C、
PV 的 4 MiB C、pack_a 缓冲、两份 padded P）。这些是精度实验的**临时**开销，不是常驻权重
影子；调用返回即释放，且 K/V packing 只在**单次调用内**复用，绝不跨层/跨输入缓存。

`planned_launch_counts()`：每次调用 **8 个 query chunk、2 次 pack_w、24 次 kernel 启动
（每块 1 QK + 2 PV）、8 个 Float-before-Half 块**。27 层 × 2 图 = 54 次调用 →
1296 次 kernel 启动。

## 准备好的四个二进制（`planned_kernel_shapes()`）

| role | batch | m | k | n | m_pad | k_tiles |
|---|---|---|---|---|---|---|
| score | 16 | 512 | 128 | 4096 | 512 | 1 |
| score | 16 | 448 | 128 | 4096 | 512 | 1 |
| pv | 16 | 512 | 4096 | 128 | 512 | 32 |
| pv | 16 | 448 | 4096 | 128 | 512 | 32 |

四者 `plan_tiling` 均为 30 核、均满足 L1/L0C/UB 预算（测试里对未被修改的 R85 plan 直接核验）。
`prepare()` 用共享 `fcntl` 锁串行构建（已预编译则直接命中缓存），并逐个校验
`dst_dtype=float32 / input_dtype=float16 / accumulator_dtype=float32 / GM_argument_count=5 /
block_dim=30 / gm_shapes / no-Half-partial / 单一 L0C / obj_sha256`。

## 门控与恢复

- 安装即可用，`enabled=False` 时**逐字节**转调捕获的原始 helper（`calls_disabled` 计数）；
- `set_enabled(True)` 在四个内核未 prepare 前**失败关闭**；启用后若 `_metas` 不全、
  dtype/device/shape 不符、输出不是 FP32、或单次调用启动数不等于 24，都立刻抛
  `VisionAttentionError`，不静默回退；
- `uninstall()` 用 `__self__`/`__func__` 判定（不是 `is` 比较新生成的 bound method），
  只有确认当前符号仍是本引擎才恢复；第三方改写则拒绝；**幂等**，重复调用不报错。
- `padding_selfcheck(q,k,v)` 供 root 在 warm 之外用真实 q/k/v 核验 padding 原值/零尾/分块，
  只返回布尔量，不选任何数值容差。

## 与 root 的接口约定

root 在**安装引擎之前**先装自己的捕获 wrapper（采真实 q/k/v），因此引擎捕获的
`_original` 就是那个 wrapper，`uninstall()` 恢复它也安全。root 侧流程建议：
先 `prepare()`（含编译，排在 warm 之外），需要时 `padding_selfcheck`，再 `set_enabled(True)`
做叶子探针与 27 层 A/B（native / PV-only / PV+bias），最后 `uninstall()` 并核对恢复。

## 非声明

源码不产生任何数字结论：没有精度结果、没有容差、没有性能结论、没有整编码器验收。
`full_encoder_precision_accepted=False`、`whole_goal_complete=False`。
真正的数值证据由 root 的驱动与回执给出：CPU Half MATH 参考（原始 72 维、原始 scale、
无 mask）、FP16 golden 未改动、真实权重未被读成 Float。