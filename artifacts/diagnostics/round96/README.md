# Round96 — 融合 PV 保持全视觉位级结果，叶子加速约 10%

本轮按已固定的 **原始 HF CPU FP16** 参考验证，不使用 BF16 输出转换、Float Linear 权重教师或事后精度阈值。结论是一次受控 kernel 优化，**不是完整 encoder 精度通过，也不是 text encode <2s**。

## 实际结果

同一 owned ts-217、原始 910A 代卡、每卡 30 个可用核；两张实际图像，视觉输入 8064 patch tokens、每图 4032 tokens、16 heads × 72。原始 Half 权重/输入，27 blocks、4 mergers 全部执行。时间为模型与 kernel 已驻留后 5 次完整视觉调用的中位数，不包括 text/HCCL；叶子为 7 次交替 A/B 的中位数。

| 完整视觉配置 | 对原始 CPU FP16 的相对 RMS | 驻留视觉中位数 |
|---|---:|---:|
| 原生注意力/原生视觉 | 1.456805% | 1.453799 s |
| patch + biased Linear + true-erf GELU 修补，原生注意力 | **0.724382%** | **1.359864 s** |
| 同一修补 + R88 高精度注意力 | 0.736957% | 2.890330 s |
| 同一修补 + 融合 PV 高精度注意力 | **0.736957%，与上一行全塔位级一致** | **2.716301 s** |

融合版本比同精度旧版视觉快 **1.0641×（耗时下降 6.02%；跨运行、同机同口径比较，不是同轮配对计时）**，但仍明显慢于修补后的原生注意力，也没有降低最终视觉误差。因此保持诊断 opt-in，**不设置为生产默认**。高精度注意力对三路 DeepStack 的相对 RMS 为 0.139499% / 0.260542% / 0.447534%，比原生注意力修补版略低；不能据此推断 50 层最终文本误差单调降低。

### 两个实际冻结注意力输入

| 图像 | 旧 R88 叶子 | 融合 PV 叶子 | 加速比 | 对 CPU Half MATH 的相对 RMS |
|---|---:|---:|---:|---:|
| 0 | 47.511630 ms | 42.990640 ms | 1.1052× | 1.437132e-5，位级结果未变 |
| 1 | 47.265640 ms | 42.980940 ms | 1.0997× | 1.581622e-5，位级结果未变 |

两者均严格要求返回 `[1,16,4032,72]`，随后才比较位级结果；全部重复输出、输入 SHA 不变。全塔 **27 个 block 边界、3 路 DeepStack、最终 merger** 都与 R88 同组合逐项 SHA 一致。

主要证据：[摘要](<summary96.json>)、[旧注意力全塔](<results/vision01/vision96.json>)、[修复后融合注意力全塔](<results/fixed01/vision96.json>)、[修复后叶子 A/B](<results/fixed01/leaf_compare96.json>)、[小形状实际双 L0C 探测](<results/fused01/pv_smoke96.json>)。

## Kernel 改动与边界

[融合 PV kernel](<pv/pv_high_low96.py>) 保留原始 R85 QK 与 Float softmax，仅将两次 PV 合成一次：

1. `P_hi=Half(P)`；`P_lo=Half((P-Float(P_hi))*1024)`。
2. 两个独立 Float L0C 各自积累完整 K=4096，V 的 L1 tile 共用。
3. Float UB 中 `hi + lo/1024`，**只进行一次最终 Half 转换**。
4. 每图由 24 次 kernel launch 降至 16 次；没有 Float PV 中间 GM 和 Torch 侧组合。

实际 H=16、M=512/448、N=128；两个 Float L0C 共 131072 B，UB 163840 B。外层 tile/head 调度动态，30 cores；K 维固定 32 次累计。Half NZ store stride 是 `Mp-Mb=384`，不是 Float store 的双倍。5 GM 指针为 `[HalfHiNZ, HalfLoNZ, HalfVNZ, FloatAuxUnused, HalfOutNZ]`。

Q/K/V 的 D=72→128、key4032→4096 均零填充；仍按 **72⁻⁰·⁵** 缩放，scores 在 softmax **前**裁到 4032，原视觉注意力不加 causal mask。未缓存跨 block 或跨输入的 Q/K/V/conditioning。

正确的可组合 engine 是 [批维修复版](<pv/accurate_vision_attention_pv_fixed96.py>)；[第一版](<pv/accurate_vision_attention_pv96.py>) 保留供失败审计。运行入口：[精确形状叶子比较](<leaf_compare_fixed96.py>)、[完整视觉驱动](<vision_probe96.py>)。

## 真实失败与修复

- 根集成先修正了 R85 meta 字段：实际为 `logical_shape` 而非作者初稿的 `logical_geometry`；同时核查 runtime library SHA 并提供 `PV` 大写 API alias，未改 Cube/UB 数学。
- 初次融合 engine 遗漏 batch 维，返回 `[H,M,D]`。冻结比较工具可按元素数重排，误通过叶子；组合后 head concat 被打乱，最终视觉相对 RMS **6.019733（601.97%）**。这是形状集成错误，**不当作可接受加速或 kernel 数值结果**。
- [失败全塔证据](<results/vision02/vision96.json>) 完整保留。新文件只改 `return result.unsqueeze(0)`，新 root 运行；[新叶子驱动](<leaf_compare_fixed96.py>) 在任意比较前强制精确四维形状。随后全塔全部边界恢复 R88 位级结果。

## 验证与后续

源代码 52 项 stdlib contracts + 8 项根/实际结果 tests 通过；作者未运行 CANN/NPU，所有编译和测量由 root 在 owned 机器上执行。原始冻结 QKV file SHA `3fb5d524373c96de48a4be583cee87f171f5e20fdaa8ac57fd4a31cb709ce552`，原始 CPU receipt SHA `44acac61f9b4d9bc8afc91632392502ccaa29dee7bfab951e2b15902a0be74c9`。

[R97](<../round97/STATE.json>) 接着把 Float scores 的 NZ→softmax→双 Half 概率 NZ 打包搬进同一个 TIK kernel，避免整个 Float score unpack 与 Torch 概率转换/重排。它会改变 Float reduction 的低位，必须继续按实际 CPU Half MATH/全图验证，不能假定 R88 位级等价。完整 FP16 精度、text<2s、encoder/DiT 共驻留与匹配官方工况仍未完成。
