# Round97 — 直接 NZ softmax/概率打包，真实注意力叶子加速约 1.78×

新增自研 TIK kernel：**Float NZ scores → true-key Float softmax → 双 Half 概率 NZ 打包**。它与 [R96 融合 PV](<../round96/pv/accurate_vision_attention_pv_fixed96.py>) 组合，去掉 Torch 侧的大矩阵 score unpack、softmax、概率高低分解及重排。没有 Float Linear 权重教师、BF16 gold 转换或隐藏 conditioning 缓存。

**完整 CPU FP16 encoder 精度尚未通过；本轮时间是驻留视觉/注意力时间，不是 text encode <2s。** 新 Float reduction 的低位与 R88 不同，因此不称为原生或旧 R88 的位级 lossless。

## 实际叶子 A/B

同一 owned ts-217、原始 910A 代芯片、30 个可用 AI cores；原始冻结两组 `[1,16,4032,72]` Half Q/K/V。7 次交替、同步测量的中位数；shape 在数值比较前严格检查，原始 QKV file SHA 和每个输入 SHA 固定。

| 图像 | 旧 R88 | 新 NZ softmax + 融合 PV | 旧/新加速比 | 新版对原始 CPU Half MATH 相对 RMS |
|---|---:|---:|---:|---:|
| 0 | 47.232590 ms | **26.454500 ms** | **1.7854×** | 1.467854e-5，旧版 1.437132e-5 |
| 1 | 47.237790 ms | **26.767500 ms** | **1.7647×** | 1.600882e-5，旧版 1.581622e-5 |

两组全部重复结果位级稳定，输入未改。新版与旧版不完全同位，CPU 叶子误差略有增加；未为此设置或放宽任何精度阈值。

证据：[叶子 A/B](<results/leaf01/leaf_compare97.json>)、[主要摘要](<summary97.json>)、[精确形状驱动](<leaf_compare97.py>)。

## 完整 27-block 视觉塔

同一真实两图输入，8064 patch tokens、4 个原始 mergers；原始 Half 权重从固定 checkpoint 转换，原始 CPU FP16 参考不变。每配置 5 次驻留完整视觉调用。

| 配置 | 对原始 CPU FP16 相对 RMS | 驻留视觉中位数 |
|---|---:|---:|
| 全原生视觉 | 1.456805% | 1.449650 s |
| patch + biased Linear + true-erf GELU，原生注意力 | **0.724382%** | **1.338367 s** |
| 同一修补 + 新 NZ 高精度注意力 | **0.737915%** | **1.825096 s** |

与 R96 旧高精度注意力的 2.890330 s 相比，新版约快 **1.584×，耗时下降 36.86%（跨运行、同机同口径；同轮配对结论以叶子 A/B 为准）**。这是一项显著的高精度路径性能改进，但与同轮原生注意力修补版比较仍慢约 36.4%，最终视觉误差也略高，**保持诊断 opt-in、不提升为生产默认**。

三路 DeepStack 相对 RMS 为 0.139580% / 0.259952% / 0.455068%，比原生注意力修补版的 0.148065% / 0.277061% / 0.473018% 更低。更准确的某些叶子或 DeepStack 不保证 50 层文本输出更准确；[R93 CPU Half 自身的 layer43 放大](<../round93/README.md>) 已证明这种输入敏感性。

[完整视觉证据](<results/leaf01/vision97.json>) 包含所有 27 个 block 边界、DeepStack 和 merger；重复、原生恢复、原始参数、pixels、CPU gold SHA 全部守恒。[R98](<../round98/STATE.json>) 随后做真正 8-rank 全 encoder 比较，不用这些视觉单独时间替代完整计时。

## 新 kernel 与实际原始 910A 能力

当前实际成功源是 [第三版 softmax/pack](<softmax/softmax_pack_padded_fixed97.py>)，engine 为 [直接 NZ engine](<accurate_vision_attention_nz97.py>)。原始 QK 仍使用 [R85](<../round85/batch/batch_gemm85.py>)，PV 使用 [融合双 L0C kernel](<../round96/pv/pv_high_low96.py>)。每图 8 query chunks，每 chunk 为 QK + softmax/pack + fused PV，共 24 次 launch；**收益不是总 launch 减少，而是消除了大量 Torch 转换/重排及 Float GM 中间处理**。

- 原始 head scale 始终为 **72⁻⁰·⁵**，不因 D=128 零填充改为 128⁻⁰·⁵。
- Float score NZ 和 Half probability NZ 都是 `[16,256,512,16]`，无需 Torch score unpack 或 probability pack。
- 每行 Float load：256 bursts × 2 个 32 B blocks，source stride `2*(Mp-1)`。
- Half hi/lo store：256 bursts × 1 block，destination stride `Mp-1`，**不能带 Float 的双倍因子**。
- 4032 个真实 key，尾 64 在 max/exp 前屏蔽，exp 后、求分母前再次强制零；原始视觉 attention 无 causal mask。
- M=448 的尾 64 个 query rows 最终 hi/lo 全零，不能变成均匀 softmax。
- `hi=Half(P)`；`lo=Half((P-Float(hi))*1024)`。原始概率/中间值只在本调用内存在，未跨 layer 或输入缓存。
- 5 GM 指针顺序 `[FloatScoresNZ, FloatAux0Unused, FloatAux1Unused, HalfHiNZ, HalfLoNZ]`。

[实际 CANN 只读能力核查](<actual_sdk_reduction97.json>) 给出短 SoC `Ascend910`，物理 `CORE_NUM=32`，实际 launch 30 cores；**Float vcmax 不支持，Float vmax/vcadd 支持**。所以使用 Float vmax 精确最大值折叠，而非伪装 A2 的 Float vcmax；分母使用支持的 Float vcadd。倒数的两次 Newton 仅用于 softmax 正分母，不引入已在 R94 验证无益的 RMSNorm rsqrt 改动。

### 真/pad 端点实际探测

[实际 smoke](<results/nz03/softmax_smoke97.json>) 用真实固定几何，甚至把 64 个 padding key scores 设置为 +500，验证它们不能贡献 softmax。

| M | 对 CPU Float 概率公式相对 RMS | 行和最大绝对误差 | kernel launch 中位数 |
|---|---:|---:|---:|
| 512 | 2.656792e-7 | 2.384186e-7 | 1.436270 ms |
| 448 | 2.656171e-7 | 2.384186e-7 | 1.473480 ms |

两种 M 的 key 尾、query pad、重复、输入 SHA 全部通过。这是 vector softmax 的独立公式探测，不是 Float Linear 模型教师，也不是完整 encoder 精度验收。

## 保留的失败与根修复

作者为 source-only fresh fork；root 在实际 SDK 上核查并修复了能力/API 假设：实际 intrinsic 名是 vmax/vcadd，不能用不存在的 `Intrinsic_vec_reduce_*` 名；物理 32 核不等于只能 launch 32 核；elementwise API 参数数与 stride 按实际 SDK 核对。

1. [第一次失败](<results/nz01/run.log>)：连续 Float zero-row fill 错把 block stride 写为 8，造成 expected4488 / allocated4096 越界。失败 [源](<softmax/softmax_pack97.py>) 保留；[第二版](<softmax/softmax_pack_fixed97.py>) 改为 block stride1。
2. [第二次失败](<results/nz02/run.log>)：M512 编译/执行成功，M448 的 vector/cast if-else 在 CANN SPR 优化中触发 `int64 vs handle` 编译错误。没有绕过 ISA 能力门禁。[第三版](<softmax/softmax_pack_padded_fixed97.py>) 让 vector 指令全行一致，只在 scalar valid-row flag 中判断 padding；随后两种 M、实际 QKV 和整个视觉塔全部执行成功。

31 项 softmax stdlib contracts、4 项根/实际 smoke contracts 通过；所有测量由 root 单独执行，无作者 NPU/SSH/CANN/model 运行。历史 source-only [作者说明](<softmax/README.md>) 和 [manifest](<softmax/manifest97.json>) 保留假设与后续 root 修正记录，实际成功源以第三版和实际收据为准。

### 只读复核发现的记录缺陷

`prepare()` 的 softmax meta dict 被直接引用，`uninstall()` 清空后末次写盘的该字段变成 `{}`，而非实际没有内核。snapshot 中仍记录 2 kernels，实际身份由 [独立 smoke 对象 SHA](<results/nz03/softmax_smoke97.json>)、[R98 串行18对象 prebuild](<../round98/results/full02/prebuild98.json>) 与 [最终审计](<../round98/final_source_verification98.json>) 交叉验证。不修改本轮已执行源或原始收据；下一版 prepare/snapshot 必须深拷贝对象 metadata。原探针默认环境是 R88，当前实际运行显式设置了 NZ 且 snapshot 自证 `round=97`；重新运行必须同样显式设置，不能仅凭用例名判别引擎。
