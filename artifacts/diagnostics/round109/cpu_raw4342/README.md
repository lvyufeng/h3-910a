# R109 — 原始 CPU FP16 第42/43层状态捕获（仅供 exact-input NPU 第43层诊断）

**状态：仅源码，作者未执行。** 本目录没有任何 CPU 模型、NPU、CANN、SSH 或 Git 操作；作者
不运行它。由 root 在某一个独立 private root 里、且自身 NPU 计时任务全部结束后执行。

## 它解决什么问题

R93 已经把原始 R78 CPU raw24 在原始 HF CPU FP16 上跑完整 25..49 层并验证 25 个边界位级
一致，但**没有保存** 42/43 层的张量。root 之后要做“用原始 CPU raw42 作为第43层输入”的
exact-input NPU 对比，需要这两个原始 CPU 张量本身。所以本脚本只补两个张量，不重做 R93 的
全部结论，也不引入任何新的数值口径。

## 会写出什么（全部在 `--root`，默认 `/tmp/h3_round109_cpu01`）

| 文件 | 内容 |
|---|---|
| `CPU_raw42_109.pt` | 原始 raw layer 42 之后的状态，**即 raw layer 43 的精确输入** |
| `CPU_raw43_109.pt` | 原始 raw layer 43 的输出 |
| `CPU_raw49_109.pt` | 仅 `--also-raw49` 时写出，用于 selected-50 佐证 |
| `CPU_raw_reference109.json` | 完成回执：门禁、张量/文件 SHA、参数与 R78 文件不变证明 |
| `CPU_raw_failure109.json` | 失败时保留阶段与异常，便于定位；不复用旧 root |

每个 `.pt` 是**普通 `torch.Tensor`**：`Half[1, 2043, 5120]`、CPU、contiguous，不是嵌套
bundle。回执为每层记录 `file_sha256` 与 `tensor` SHA256，并说明 42 是 43 的输入。位置/
rotary/mask 仍从冻结 R78 原始输入重算，不随张量另存。

## 复用与门禁（全部 fail closed，先验证后写文件）

复用不可变的 [R93 suffix93.py](../round93/cpu/suffix93.py)（SHA `6d062fdb…`）中的
`load_original_helper` / `boundary_truths` / `verify_conversion_inventory` /
`build_suffix_layers` / `layers_parameters_sha` / `propagate`，以及其固定依赖
[R92 original_cpu_block25_92.py](../round92/cpu/original_cpu_block25_92.py)（SHA `838027c8…`）。
两者都以字节 SHA 校验后按私有名导入，绝不编辑。

1. staged 两个 helper 的 SHA 与本轮 pin 必须一致；其内部 R78/HF 常量也要与本轮 pin 一致。
2. `reference.json`、`FP16_multimodal_inputs.pt`、`FP16_layer24.pt`、`FP16_selected50.pt`
   四个文件 SHA 必须等于 R78 pin；raw24 tensor `7a2178…`、raw49 tensor `829446…` 由 helper
   门禁校验；HF 原始建模源 `fcb556…` 必须存在且未变。
3. 用原始 CPU raw24 跑完整 25..49，**全部 25 个 R78 边界 SHA 逐一相等**；随后**第二次完整
   pass 的 25 个 SHA 必须与第一次逐位相同**；raw49 边界等于 selected-50 冻结 tensor。
4. 门禁通过后才写张量：写 `.tmp` 再 `os.replace`，只在 root 内、只允许声明的文件名；回读
   必须是普通 Tensor 且 SHA 一致。
5. 再取出磁盘上的 raw42，调用原始 `run_block25(raw layer 43)`，输出必须与磁盘 raw43
   **位级相同**——这就是“42 是 43 输入”的自证。
6. 结束时原始 275 个 Half 参数 SHA 不变、四个 R78 文件 SHA 不变；否则报错、不置 completed。

原始 Linear 的输入/权重/输出仍为 Half，允许原始模型的自然 Float norm/rotary 与
`SDPBackend.MATH` 注意力；不读 R92/R91 的 corrected NPU raw24，不写 BF16 gold，不加 final
norm/DeepStack，不做 Float 权重教师、隐藏 conditioning 缓存或任何数值容差，也不测速。

## 用法（root 执行，非作者）

```bash
mkdir -m 700 /tmp/h3_round109_cpu01
cp artifacts/diagnostics/round93/cpu/suffix93.py \
   artifacts/diagnostics/round92/cpu/original_cpu_block25_92.py /tmp/h3_round109_cpu01/
python3 newcapture_cpu_raw109.py \
  --root /tmp/h3_round109_cpu01 \
  --suffix-helper /tmp/h3_round109_cpu01/suffix93.py \
  --original-helper /tmp/h3_round109_cpu01/original_cpu_block25_92.py
```

需要 `torch 2.10.0+cpu`、`transformers 5.14.1`、MKLDNN 与 R78 gold 相同，CPU 线程数沿用
pinned helper 的 `CPU_THREADS`（实测为 16，非推测）。两次完整 pass 大约需要 20 分钟 CPU，
root 的 3600–5400s 上限内留有余量。失败会写 `CPU_raw_failure109.json` 并拒绝复用同一 root。

## 未验证的 CPU 假设（root 必须实测）

- 私有 root 里必须同时 staged 两个 helper，且此前没有任何本目录产物。
- 运行时库版本/线程/MKLDNN 与 R78 gold 一致（helper 会 fail closed，但仍需真实运行确认）。
- 两次完整 suffix pass 的实际耗时与内存符合预期。
- 真实执行由 root 完成，本目录**不包含**任何 CPU 运行结果或精度结论。

## 边界

本目录只是原始 CPU 参考张量的捕获工具：**不是新 oracle**、不是完整 CPU FP16 通过、不是
NPU 对比结论、不是速度或默认部署声明。后续 NPU 第43层探测与比较由 root 的另一轮完成。
文件 SHA/行数与外部 pin 见 [manifest109.json](manifest109.json)；stdlib 契约测试见
[test_newcapture_cpu_raw109.py](test_newcapture_cpu_raw109.py)。