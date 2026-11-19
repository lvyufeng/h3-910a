# Round107 — 原始 Half 视觉 TP8 私有适配器（仅源码，未编译/未上板）

本目录只包含 **task-58** 的源码产物：[`VisionTPAdapter107`](vision_tp_adapter107.py)。
它把 27 个 Qwen3-VL 视觉 block 的 **attention 与 MLP 投影**改成 8 卡张量并行，
权重保持原始 Half、行并行部分和以 **Float32** 进入 `all_reduce`、每处只做一次 Half 收窄。

> **非声明**：这里的代码没有被编译、没有被 launch、没有跑模型、没有 SSH/CANN/NPU/GPU 操作，
> 也没有测过任何速度或精度。它是**结构契约**，不是精度结论、不是加速结论、不是官方 lossless
> 对齐，也不是整体目标完成。所有真实数字由 root 在租赁主机上用自己的算子/后端取得。

## 为什么只切 27 个 block

只在**实例级**替换 `blocks[i].attn.forward` 与 `blocks[i].mlp.forward`（`types.MethodType`），
**从不**修改任何类方法。因此：

- `_NpuBF16CompatibleLinear.forward` 的类级补丁（root 的 `VisionBiasAdapter87`）不被冲突；
  启用本适配器时 108 个 block linear 被绕过，8 个 merger linear 仍走既有适配器；
- patch embedding、4 个 merger（含 DeepStack）、LayerNorm、rotary 几何、`cu_seqlens` 构造
  全部保持已执行代码，所以 **R90 true-erf 的 2016×4608 merger 形状契约不变**；
- 全局 `_vision_sdpa_bf16_compatible` 不被替换（源码里只在文档中提到一次），
  注入的 2-head attention 由 root 作为独立 callable 传入。

## 分解方式（rank r / 8）

| 投影 | 原始形状 | 本 rank 取哪些 | 模式 | Float32 归约后 |
|---|---|---|---|---|
| `attn.qkv` | `[3456,1152]`（q/k/v 三段各 1152） | 每段取输出行 `s*1152+r*144 : s*1152+(r+1)*144`，拼成 432 | column | Half（含精确 Half bias 加宽到 Float32） |
| `attn.proj` | `[1152,1152]` | 输入列 `r*144:(r+1)*144` | row | `all_reduce`(Float) + 完整 bias + 一次 Half |
| `mlp.linear_fc1` | `[4304,1152]` | 输出行 `r*538:(r+1)*538` | column | Half（bias 同上） |
| `mlp.linear_fc2` | `[1152,4304]` | 输入列 `r*538:(r+1)*538` | row | `all_reduce`(Float) + 完整 bias + 一次 Half |

`4304 = 8*538`、`1152 = 8*144`，八个 rank 精确铺满真实通道，无缺口、无重叠、无遗漏；
只有 Cube 对齐用的补零通道（`k/n` 到 128 的倍数）是额外项，且**绝不进入 GELU、绝不进入分母**。

**行模式不带 bias**：如果每个 rank 都先加完整 bias 再求和，结果会被放大 8 倍，因此
`prime()` 对 row 模式显式传 `bias=None`，adapter 在 Float32 集合通信之后再加**完整原始 bias**
（Half→Float32 加宽）并只做一次 Half cast。column 模式则由 backend 内部完成
"Float32 累加 → 精确 Half bias 加宽 → 一次 Half"。两种模式都在源码和测试里断言。

## 逐块数据流

1. `qkv` column → Half `(8064,432)`；`reshape(m,3,2,72).permute(1,0,2,3).unbind(0)` 得到本地 2 头
   的 q/k/v（**按 head 分片，不是把 QKV 行整体取 `r*432` +432 的连续块**）。
2. 调用候选自己的 `candidate._apply_rotary_pos_emb_vision`（Float 中间值、原始 cos/sin），不改顺序。
3. 按原始 `cu_seqlens`（两图 `[0,4032,8064]`）切成 2 段，每段调用注入 attention：
   严格 `[1,2,4032,72]` Half、原始 scale `72**-0.5`、非 causal；返回 4D Half。
4. 段拼接 → `transpose(1,2)` → 展平 144 个本地 head 通道 → `proj` row：Float32 部分和 →
   `group.all_reduce`（Float）→ 完整 `proj.bias` 加宽 Float → 一次 Half → `(8064,1152)`。
5. MLP：`fc1` column → Half `(8064,538)` → **原始 `nn.GELU(approximate="tanh")`**（真实 538 通道）→
   `fc2` row → Float32 部分和 → `all_reduce`（Float）→ 完整 `fc2.bias` 加宽 Float → 一次 Half。

## 计数契约

一次完整视觉前向（2 图 × 27 block）必须产生：
**27** attn 调用、**27** MLP 调用、**54** 段级 attention 调用（2 图）、
**54** 次 Float 集合通信、**54** column 运行、**54** row 运行、**108** 个已 prime 的 linear 权重。
`expected_counts_per_forward()` 与 `snapshot()['expected_counts_per_forward']` 提供同一份数字，
测试逐项断言实测计数与之一致。

## 内存与缓存

适配器自身**不缓存任何 tensor 或激活**：Half 权重切片只在 `prime()` 里构造一次并交给
`backend.prime()`，随后丢弃自己的引用（任何打包副本属于 backend，由 backend 的 snapshot 报告）。
测试会断言 `_signatures`/`_prepared`/`_stats`/`_roles` 中没有任何 tensor，
并对 `snapshot()` 做 `json.dumps` 以证明它是纯数据。

供 root 记账的参考量（若 backend 保留打包副本）：每 block 的 Half 切片共 1,903,104 元素
（3,806,208 B），27 个 block 共 **102,767,616 B**；每次集合通信输出的 Float bias 加宽
4,608 B，均为瞬时。`m = 8064 = 63*128` 本身已 Cube 对齐，不需要 m 方向补零。

## 失败即拒绝（fail closed）

- 构造：`world_size != 8`、`rank_in_group` 非整数/越界、`m != 8064`、attention 不可调用、
  group/backend 缺方法、候选无 rotary helper。
- prime：block 数不是 27、权重/ bias 不是 Half、device 不是本 rank 的 npu、
  `attn.num_heads/head_dim/dim` 不是 16/72/1152、`mlp.act_fn` 不是 `approximate="tanh"`、
  形状不是 `(3456,1152)/(1152,1152)/(4304,1152)/(1152,4304)`、row 模式误带 bias。
  **中途失败会把已安装的实例补丁全部回滚**（测试覆盖）。
- 前向：激活不是 Half `(8064,1152)`、`cu_seqlens` 不是 `[0,4032,8064]`、
  某段不是 4032、`position_embeddings` 缺失、未知 kwargs、
  column 返回非 Half、**row 返回 Half（会在集合通信前丢低位，明确拒绝）**、
  `all_reduce` 返回非 Float、attention 返回非 Half 或形状不是 `[1,2,4032,72]`、
  最终输出不是 Half `(8064,1152)`、参数 identity（data_ptr/storage_offset/numel/shape/dtype/
  device/version）在 prime 后改变。

## 卸载与所有权

`uninstall()` 先产出快照，再逐个校验"实例上的 `forward` 仍是我方 wrapper"才恢复：
原先没有实例属性的模块**删除**该实例属性（类方法自然恢复），原先有实例属性的模块
**逐位还原**原来那个对象；若有人覆盖了我方 wrapper 则**拒绝破坏**并报错。
返回的快照是深拷贝，卸载后仍然有效（测试覆盖）。

## 测试

`python3 -m unittest test_vision_tp_adapter107 -v` → **36 项全过**，纯标准库：
几何/切片覆盖（全部 8 个 rank）、四个后端几何与 128 对齐表、108 次 prime 与 row 无 bias、
54 模块实例补丁且类方法未动、计数契约、冻结/回滚/拒绝分支、实例还原与拒绝破坏、
快照深拷贝、非零 storage offset 视图的密集化守卫、源码纪律（模块级只 import `copy/hashlib/types/pathlib`，`import torch` 只在
`_torch()` 内，无 `bfloat16`/`.clamp(`/`scaled_dot_product_attention`/`autocast`/`setattr(`，
无 `cls.forward` 赋值）。

**这些是结构测试，不是任何精度或速度结果。**

## 交付物与接口

| 文件 | 说明 |
|---|---|
| [vision_tp_adapter107.py](vision_tp_adapter107.py) | 适配器本体（`SHA256 691897d3…`，791 行） |
| [test_vision_tp_adapter107.py](test_vision_tp_adapter107.py) | 36 项 stdlib 测试（`SHA256 870a90fe…`） |
| [make_manifest107.py](make_manifest107.py) | 可再生 manifest 生成器 |
| [manifest107.json](manifest107.json) | 文件 SHA、API、几何/切片/对齐表、计数、内存、待 root 验证项 |

```python
adapter = VisionTPAdapter107(candidate, group, backend, attention, m=8064)
report = adapter.prime(encoder.vision)   # 27 blocks：切片 + 108 次 backend.prime + 54 个实例补丁
adapter.set_enabled(False)               # 默认关闭：wrapper 直接调用原方法（原生满 16 头基线）
adapter.set_enabled(True)                # 打开本地 2 头 + Float32 部分和路径
snap = adapter.snapshot()                # 纯数据，深拷贝，卸载后仍有效
adapter.uninstall()                      # 校验所有权后还原/删除实例属性
```

`backend.prime(name, weight_half, bias_half_or_None, mode, m=8064)`、`backend.run(entry, x)`、
`backend.snapshot()`、`group.all_reduce(FloatTensor)` 由 root 提供；后端须把 `k/n` 补到
`k_pad/n_pad` 并只返回**逻辑** n 列（`432/1152/538/538`），否则适配器会拒绝。

## 与 root 后端 API 的只读兼容性核对

撰写本目录后只读检查了 root 已经放在上一级目录的 [`VisionGemmBackend107`](../vision_gemm_backend107.py)，
逐条确认契约一致（这是兼容性核对，不是硬件结果）：

- `prime(name, weight, bias, mode, m=8064)` 位置参数顺序与本适配器调用一致；
- 该后端自带 `assert mode != 'row' or bias is None`，与本适配器"row 模式不传 bias、集合通信后再加"一致；
- column 的 `dst_dtype == 'float16'`、row 的 `dst_dtype == 'float32'`，正是本适配器 fail-closed 校验的两种 dtype；
- `run()` 返回 `(entry['m'], entry['n'])` 的逻辑形状（内部已切掉 128 补零），本适配器按
  `(8064, 432/1152/538/538)` 校验，二者一致；
- 后端每次调用重新打包 A（"no saved pixels/hidden states"），与本适配器不缓存激活的约束一致；
- 后端内部为 column 模式保留 Float bias（`pb`）属于硬件层设计；本适配器自身不产生 Float 权重影子。

若后端后续改动这些签名或 dtype，本适配器会在 `prime()`/前向时以 `VisionTPError` 拒绝，而不是静默跑错。

## 待 root 在真实硬件确认（本目录不证明）

`m=8064` 的 Cube 四种形状与 128 补零、column 模式的 Float32 累加 + 精确 Half bias + 单次 Half、
row 模式确实返回 Float32 部分和、`all_reduce` 保持 Float32、2-head attention 返回
Half `[1,2,4032,72]`、`cu_seqlens=[0,4032,8064]`、启用后计数为 27/27/54/54/108/54、
关闭后原生满 16 头基线逐位恢复。Float32 HCCL 部分和可能改变低位、本地 2 头改变 attention
分组，**必须**由 root 用原始 CPU FP16 参考在完整图上验证，不能据本文件宣布精度或加速通过，
也不承诺 8 倍。