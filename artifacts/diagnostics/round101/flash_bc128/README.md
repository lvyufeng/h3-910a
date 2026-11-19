# Round101 / flash_bc128 — MB=64 × BC=128 流式 vision attention（仅源码）

**状态：SOURCE ONLY。** 本目录作者没有编译、没有上板、没有导入 NPU/CANN/torch，也没有测速或测精度。这里只有 builder 源码、纯 stdlib 契约测试和清单。所有真实数字由 root 驱动产生：编译 `flash_vision101.py`，对冻结的 Round88 真 Q/K/V 与 Round78 **原始 CPU FP16** 参考做比较。

起点是不可变的已执行 R99 kernel [flash_vision_shift_temp_fixed99.py](../round99/flash/flash_vision_shift_temp_fixed99.py)（SHA `d45cac79…`，本轮未修改）。本轮把 KV 块从 64 加倍到 128，KV 迭代 63→32，并复用一块 Float 概率缓冲以保住 UB 预算。

## 相对 R99 的两处实质变化

1. **KV 块翻倍**：查询块仍是 MB=64（63 个查询 tile 不变），KV 块 128（8 个 NZ key 组）。FP32 score tile 变为 32 KiB（UB 与 L0C 各一份）。`n_pad = ceil(4032/128)*128 = 4096`，于是**最后一个 KV tile 只有 64 个真实 key**，其 group 4..7 是由零填充 K 行产生的 phantom key。
2. **去掉一块 Float 概率缓冲**：真 FP32 分母折叠求和之后，`P_hi = Half(exp)` 再拓宽回 Float，然后把 **score Float 缓冲原地**覆写成 `(exp - Float(P_hi)) * 1024` 得到残差，最后 `P_lo = Half(残差)`。下一个 KV 迭代从 QK 的 L0C 结果重新载入 score，因此不再需要独立的残差 Float tile。

## phantom key 必须显式屏蔽（本轮的数值核心）

零填充的 K 行让 phantom key 的 score **恰好为 0**；缩放后仍是 0，`exp(0 - shift)` 会给每一行分母贡献 1.0。**只把 V 补零是不够的**：分母被抬高后每个真实概率都会被压小。因此：

- exp 之前（Half shift 估计与 Float exp 之前）把 phantom 组填 `-FLT_MAX`，它们既不会赢得行最大值，指数也下溢为 0；
- exp 之后、**真 Float 分母求和之前**再把 phantom 组显式填 0（双保险）。

两个动作都写在 `if kv == kv_tiles - 1` 的动态分支里（无 `else`，分支体内**只有 Float `vector_dup`**），这正是 R99 记录的高层 if/else 规范化会崩、而纯 vdup 分支可用的形态。真实 4032 几何下 phantom 组为 4（group 4..7），填充起点是 score 缓冲第 4096 个元素。

## 内存预算（精确推导，非估计）

| 项 | 字节 |
|---|---:|
| R99 已用 UB（named + shift 辅助） | 151 424 + 6 656 = 158 080 |
| 四块翻倍：score Float / P_hi Half / P_hi Float / P_lo Half | +4 × 16 384 = +65 536 |
| 删除 R99 独立残差 Float tile | −16 384 |
| **本轮 UB 合计** | **190 848** ≤ 196 608（余量 5 760） |

- L0C：score 32 768 + 2 × P@V 20 480 = **73 728 B（72 KiB）**
- L1：Q 10 240 + K 20 480 + V 20 480 + P_hi 16 384 + P_lo 16 384 = **83 968 B（82 KiB）**

`plan["buffers"]` 是唯一真值来源，`_check_budget` 会对 UB/L0C/L1、repeat 立即数（≤255）与 32 B 对齐做 fail-closed 检查；测试逐项对照上表。

## 保持不变的部分

- **主 score / softmax / 累加器 / 分母 / rescale 全部 FP32**；Half 只用于「稳定 shift 估计」。
- `QK` 是 Half×Half→FP32 L0C 乘积，覆盖 padded head 宽度 80；主 score 从不窄化。
- 双 Half 概率 `P_hi`/`P_lo` 喂两次 FP32 L0C P@V，UB 内 FP32 合成 `hi + lo/1024`，**全程只在最终输出 cast 一次**。
- 真分母是 `exp(S - shift)` 的 FP32 求和，绝不是 Half 求和。
- 尺度是**原始 head dim** `72 ** -0.5`，不是 padded 80，也不是 KV 块 128。
- 无 causal mask（vision attention 是 unmasked），最大 `|scaled QK| ≤ 16384` 由 host 协议保证，**不夹断**；`check_host_protocol` 违规即抛错。

## 展布与 ABI

- 5 GM 指针：`[Q HalfNZ(H,5,M,16), K HalfNZ(H,5,n_pad,16), V HalfNZ(H,n_pad/16,80,16), Float Aux×16 (未用), Out HalfNZ(H,5,M,16)]`；`BuildCCE(inputs=[4], outputs=[1])`，未用的 aux 占位仍显式列出，保持已验证的五指针 ABI。
- DMA：Q 5×64（源步长 M−64）、K 5×128（源步长 n_pad−128）、V 8×80 连续、P→L1 8×64、输出 5×64（目的步长 M−64，Half 不开双倍步长）。
- 网络：1008 个 (head, 查询 tile) 任务铺到 30 个可用核、34 个动态步；KV 循环是真动态 `for_range`，**不展开**。

## 保留 R99 的两处编译器规避

1. **不用高层 `h_max`/`h_reduce` 处理切片 NZ**：8 组 Half 折叠显式发 `vec_max`（mask 128，7 次）；Float 分母折叠显式发 `vec_add`（mask 64，7 次）。顶层 `h_reduce_max`/`h_reduce_sum(axis=1)` 只作用在最后一个 (MB,16) tile 上。
2. **不做大张量跨步 `h_cast`**：shift 广播沿用 R99 已实测位级精确的原生路径——uint16 `vec_or` 复制 + Half `vec_trans` 转置拼出 (64,16) 行块，`h_cast` 成 **1024 元素 Float 临时缓冲**，再逐组复制进 8 个 NZ 组。绝不把 Float rescale/分母降成 Half。

## 测试（65 项，纯 stdlib）

```
python3 -m unittest test_flash_vision101      # 65 tests, OK
python3 make_manifest101.py                   # 重写 manifest101.json
python3 flash_vision101.py --plan 16 4032 4032 72
```

覆盖：几何/尺度/pad 规则、内存与预算护栏（含「信息性别名绝不计入总量」的防重复计数守护）、DMA 描述子覆盖、phantom 屏蔽取值与**指令顺序**、数学顺序、ABI、循环结构、per-KV 静态指令计数、无切片 `h_max`、无跨步 `h_cast`、源码 import 纪律（模块级仅 stdlib，`tbe` 一律延迟）、能力表 fail-closed、以及 manifest 与活 plan/文件哈希一致性。

其中 `TikStub` 是**记录型假 TIK**（不计算、不导入 CANN），它运行真实的 `_emit_kernel` 并检查发出的指令序列。这验证的是**结构**，不是编译，也不是数值。

## 尚未验证、必须由 root smoke 确认的 CANN 行为

1. 32 KiB 单次 L0C→UB `tensor_mov` burst 可编译（R85 已执行过同形态 64 KiB 单 burst，见 `results/full01/prebuild_full85.json` 的 `l0c_to_ub.burst_len_kb=64`）。
2. 无 else、只含 Float `vector_dup` 的动态 `if_scope` 可编译（R99 SPR 崩溃只出现在 if/else + 向量/cast 形态）。
3. 8 组显式 `vec_max`（Half）与 `vec_add`（Float）折叠可编译。
4. 1024 元素 Float 承接 tile 的 `h_cast` + 8 次 NZ 复制可编译（R99 已实测位级精确）。
5. 8192 元素整块 `h_sub`/`h_exp`/`h_cast` 链可编译（R99 在 4096 元素上已执行）。
6. `-FLT_MAX` 经 `h_cast` 到 Half 为饱和/-inf 而非 NaN；即使饱和为 -inf 结论仍成立，因为真实缩放 score 都 > −16384，`max` 一定取到真实值。
7. 真实输入下与 Round78 CPU Half MATH 的对齐（root 用冻结 QKV `3fb5d524…`）。

**建议顺序**：先 `H=2 M=128 N=192`（`n_pad=256`，最后一个 KV tile 也是 64 真 + 64 phantom，所以 padding 路径被覆盖；`block_dim=4`），再跑真实 `16/4032/4032/72`。

## 非声明

不声称任何精度、速度、编译器或容差结果；不声称可直接设为默认；不使用 Float 线性权重教师、BF16 gold cast、隐藏 conditioning 缓存、clamp 或事后容差。

**也不承诺复现 R99 的位模式**：KV 分块变了，Float 分母折叠的结合顺序也变了（`a=g0+g1; b=g2+g3; a+=b; b=g4+g5; a+=b; b=g6+g7; a+=b`，记录在 `plan["math"]["denominator_fold"]`），低位可能不同。权威判据是 root 的冻结真实输入 CPU Half MATH 比较。