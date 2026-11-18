# Round86 leaf — 真实晚层 FP16 归一化与 MLP 叶算子对照（仅源码）

本目录是一份**源码 + stdlib 测试**交付：作者没有 SSH、没有跑 NPU/GPU、没有加载模型、
没有安装、没有改生产代码、没有 commit。真实数字由 root 在已租用的 NPU 上执行
`run_controls` 产生；本目录**不含任何数值结果**，也不构成精度通过或加速结论。

## 为什么做这个叶算子对照

[Round85](../round85/README.md) 显示完整 TP8 文本图误差主要在晚段放大：精确 attention +
FP32 row 的联合路径 raw layer24 相对误差约 4.8e-4，而 raw layer49 增至约 2.71%。
这只把范围收缩到 24→49，并没有解释原因。因此这里对**单个真实晚层**（root 用 block25 起）
的叶算子做隔离测量，CPU 金标准一律用**实际捕获的原生 Half 值**现算，避免“看到 dtype 标签
就假设内部精度”。

## 交付物

| 文件 | 说明 |
|---|---|
| [late_leaf86.py](late_leaf86.py) | 叶算子对照实现；模块级仅 stdlib 导入，torch 全部惰性 |
| [test_late_leaf86.py](test_late_leaf86.py) | 23 个 stdlib 测试（几何/失效闭合/固定源码顺序/清单），无需 torch |
| [manifest86.json](manifest86.json) | 源码自描述：SHA、几何、pins、失效闭合项、测试数 |

模块 SHA256 `4a147c8a53d63a0d4745962cceb6c36f4db6ca492c2f2c22c17020f1b5b86322`。

## root 调用契约

```python
import late_leaf86 as leaf
report = leaf.run_controls(layer, block_input, mlp_input, compare, tensor_info)
```

* `layer` — NPU 上的真实 R79 `MiniMaxH3Qwen3VLTextDecoderLayer`，原始 Half 参数；
* `block_input` — 实际 Half `(1,2043,5120)`；block25 时就是**未改动的 CPU FP16 raw layer24**
  状态搬到设备上的张量；
* `mlp_input` — 首次原生 `mlp.forward` pre-hook 捕获的实际 Half `(1,2043,5120)`；
* `compare` / `tensor_info` — root 用 R84 不可变 helper（`relative_rms`/`raw_bits_equal`/`sha256`）。

返回纯 JSON 可序列化数据（不含张量）；冻结哈希在前后都记录，任何变化即抛 `LeafError`。

## 五个叶算子对照

1. **input_layernorm**：原生 FP16 两轮顺序 vs CPU FP16 同顺序
   （Float 归约 → Half 归一化值 → Half gamma 乘积）。**不**改成单轮 Float gamma。
2. **MLP gate/up 本地线性**：在实际捕获的 mlp 输入上，用原始 Half 权重切片
   `[0:3200]`、`[3200:6400]` 做 Half×Half `F.linear`，对 CPU Half `F.linear`。
3. **SiLU**：在实际**原生 Half gate** 上做原生 `F.silu`，对同一组值上的 CPU Half SiLU。
   `F.silu(x.float()).to(Half)` 只是**标注过的诊断**（衡量 pointwise 舍入代价），
   不是 FP32 权重影子、不是 FP32 教师。
4. **SiLU×up**：实际原生激活的 Half 逐元素乘，对同一原生值的 CPU Half 乘；
   另有一条“CPU 线性 + CPU pointwise”的组合上下文项，明确标注**不隔离任何单一算子**。
5. **down_proj 本地分片线性**：K3200→5120 的本地权重，对同一乘积上的 CPU Half 线性。
   **不执行** row-parallel collective（异步 HCCL 在单 rank 上会挂起或污染叶算子视角）。

## 与实际 checkpoint 对齐的几何（必须 25600，不是 27648）

| 量 | 值 |
|---|---|
| `intermediate_size` | 25600（实际 checkpoint） |
| TP8 每 rank `intermediate_size_per_partition` | **3200** |
| `gate_up_proj.weight` | `[6400, 5120]` |
| `down_proj.weight`（本 rank 分片） | `[5120, 3200]` |
| 序列 / 隐维 | 2043 / 5120 |

Round84 曾把 MLP 宽度误用 27648（per-rank 3456）导致 8 rank 在调用任何自研 kernel 前被
门禁拒绝。本模块把 27648 作为**必须拒绝**的陈旧常量显式编码，并在测试中覆盖
`(6912,5120)`/`(5120,3456)` 两种错误形状。

## 失效闭合（不是精度门槛）

唯一断言的是结构性质，不是数值容差：

- 激活/权重必须是原始 Half、激活必须在模型设备上，形状必须精确；
- 层/模块类名必须是 R79 的那几个类，`quant_method` 必须是 `UnquantizedLinearMethod`，
  且这些线性**无 bias**；
- `intermediate_size_per_partition` 必须 3200，norm gamma 必须 Half、eps 必须正且有限；
- 每次原生调用重复一次并要求**位级一致**，否则抛错；
- 捕获的 5 个输入/参数张量在前后 `tensor_info` SHA 必须相同，否则抛错；
- R79 两轮 FP16 norm 分支源码必须与 pin
  `da686cd5815ff31e13371f18ee2320232acc2be6d5da2bf2f7dccc4c2f1e8f14`
  （候选文件 `2141b29a…`）字节一致，语义改动即失败。

报告里显式带 `numerical_tolerance: None`、`numerical_pass_field_absent: True`、
`full_encoder_precision_accepted: False`、`whole_goal_complete: False`：本目录不做精度判定。

## 边界

只测层25（或 root 指定的一个晚层）；不覆盖 attention、不覆盖视觉塔、不覆盖完整 50 层、
不覆盖真机并发/共驻/端到端时延。`down_proj` 只测本地分片，因此**不能**用它的误差推断
row-parallel 整体误差。整图精度、text encode <2s、与官方 A2/A3 同工况比较仍未完成。