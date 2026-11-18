# Round99 — 单 launch 流式 Float 视觉 attention 的实际验证

新增自研 TIK 流式 attention，保留 **Float QK scores、Float online output 和 Float denominator**，最终只转一次 Half。真实几何为 16 heads、每图 4032 queries/keys、原始 D72→D80，仅 63 个 M64/K64 tiles，**没有把4032 padding到4096、没有把D72过度 padding到128**。每图一次 kernel launch，而非 R97 的 24 次。

这是诊断 opt-in，**不宣称完整 CPU FP16 精度通过或 text encode <2s**。R96–R98 已提交为 `e46f054`；R99 的精确广播优化已完成真实叶子与全部27视觉边界验证，基线与优化后的数值均在下文明确区分。

## 同轮实际冻结 QKV 叶子

同一个 owned ts-217、原始 910A 芯片、30 个可用 AI cores，7 次交替同步调用的中位数。原始 R88 两组 `[1,16,4032,72]` Half Q/K/V 与原始 CPU Half MATH 相比；R97 NZ 对照输出 SHA 强制匹配已验证结果。输入和重复输出位级稳定。

| 图像 | R97 NZ 中位数 | 流式 Float flash 中位数 | 加速比 | 对原始 CPU Half MATH 相对 RMS |
|---|---:|---:|---:|---:|
| 0 | 27.301840 ms | **26.234530 ms** | 1.0407× | **1.502500e-5**，NZ 1.467854e-5 |
| 1 | 27.364510 ms | **25.665020 ms** | 1.0662× | **1.527378e-5**，NZ 1.600882e-5 |

消除了 score GM，但实际收益仅 4–6.6%，而不是由“24→1 launch”直接推断巨大加速；每行 Float rescale 的广播仍有大量 scalar/vector 指令。原始 score/概率数学低位因 block 在线归约不同而变化，CPU 叶子误差在约 1.5e-5，没有事后容差。

证据：[真实叶子比较](<results/leaf02/leaf_compare99.json>)、[摘要](<summary99.json>)、[精确形状驱动](<leaf_compare_guard_fixed99.py>)。

## 完整 27-block 视觉塔

实际 pixels 与 original Half 权重、27 blocks、4 mergers 都参与计算。5 次驻留完整视觉调用中位数，**不包括文本层/HCCL，因此不能代替完整 encoder 计时**。

| 配置 | 对原始 CPU FP16 相对 RMS | 驻留视觉中位数 |
|---|---:|---:|
| 原生视觉 | 1.456805% | 1.459682 s |
| patch/bias/true-erf 修补，原生注意力 | **0.724382%** | **1.360309 s** |
| 同一修补 + 单 launch 流式 Float flash | **0.726924%** | **1.636251 s** |

流式版比之前 R97 NZ 全视觉 1.825096s 少约 10.35%（**跨运行、同机同口径比较，不是同轮配对**）。误差比 NZ 的 0.737915% 略低，但仍比本轮原生注意力修补版略高、慢约 20.3%，没有默认开启的证据。

三路 DeepStack 相对 RMS 为 0.139716% / 0.260160% / 0.453123%。所有 27 个边界、merger/DeepStack 重复输出、原始参数、pixels、CPU gold SHA、恢复原生输出均核对成功。[完整视觉证据](<results/vision01/vision99.json>)、[完整视觉驱动](<vision_probe99.py>)。

## 精度保持的具体数学

- 每 KV block 的 Half×Half QK 在 Float L0C 中累计 D80，保留 Float scores 到 UB，按原始 **72⁻⁰·⁵** 缩放，无 causal mask。
- 原始 910A 不支持 Float vcmax；Half vcmax **只估计 softmax shift**，不是把主 scores/概率降为 Half。在线 max 是这些 Half 可表示 shift 的 Float max，score 主值不改。
- Float `exp(scores-newShift)` 后双 Half 分解：`hi=Half(exp)`，`lo=Half((exp-Float(hi))*1024)`，两次 PV 各自 Float L0C，Float UB `hiPV + loPV/1024`。
- Float `O=rescale*O+PV`、`L=rescale*L+sum(exp)` 累积，`O/L` Float → **一次最终 Half output**；不存 Half 累积 O 或 Half denominator。
- 参数/Linear没有 Float shadow、BF16 gold cast、CPU hidden teacher 或跨调用 Q/K/V/conditioning缓存。
- Host fail-closed 要求 scaled QK 绝对上界 `<16384`；先使用最大值保守上界，不够时用 Float Cauchy 上界。该界保证最大-score exponent 在 `[e^-8,e^8]` 内、双 Half exp 不溢出；其它小 exp 可以如正常 stable softmax 下溢为零。**没有 clamp 或缩放输入以逃避门禁**。
- GM5 `[QHalfNZ,KHalfNZ,VHalfNZ,FloatAuxUnused,HalfOutNZ]`，K/V共用块重载有显式同步，Half NZ output stride `M-64` 不带 Float 双倍因子。

已编译执行的基线 kernel 是 [显式折叠修复版](<flash/flash_vision_step_fixed99.py>)，host engine 为 [guard修复版](<flash_engine_guard_fixed99.py>)。prepare/snapshot 深拷贝 metadata，不再出现 R97 引用被 uninstall 清空的问题。

## 真实失败、修复与新的优化边界

1. 作者 fresh fork source-only，root 先修复了 host protocol 缺省/部分/负界值未拒绝的问题；46 项纯 stdlib 源 contracts 通过。
2. [初始编译失败](<results/smoke01/run.log>)：高层 `h_max` 对 NZ slice 推断 step 不同；原 [源](<flash/flash_vision99.py>) 保留。新 [显式 vector fold](<flash/flash_vision_step_fixed99.py>) 保留同一 pairwise Half-max / Float-add 序列，实际小形状 H2/M128/N128/D72 成功，CPU Half MATH 相对 RMS 1.222366e-5，重复位级稳定。[小形状结果](<results/smoke02/flash_smoke99.json>)。
3. [初次真实输入失败](<results/leaf01/run.log>)：Host global ReduceMax/offset view 触发 NPU buffer 错误，不是 attention 数学失败。新 [guard](<flash_engine_guard_fixed99.py>) 明确 dense/offset0 和 1D axis0 reduction，原 Half 值不变；之后叶子与完整视觉成功。
4. 读取 [原始 SDK transpose/copy 能力](<actual_sdk_copytranspose99.json>) 后，用 uint16 bit-copy + 支持的 Half transpose 实现 **精确 Half 可表示 shift → Float广播**。[独立 operator proof](<results/broadcast01/broadcast_copy99.json>) 位级等于直接广播，包含 `-0`。它**不能**用于把一般 Float rescale/denominator 降为 Half，当前只优化已经 Half 可表示的 shift。
5. 首次合并时 [cast目的 shape失败](<results/leaf03/run.log>)：1024-element Half row广播转到4096-element Float tile；失败 [源](<flash/flash_vision_shift_broadcast99.py>) 保留。接着 [cast view版本](<flash/flash_vision_shift_cast_fixed99.py>) 仍因 step 推断失败，保留 [失败收据](<results/leaf04/leaf_compare99.json>)。最终 [独立Float临时buffer版本](<flash/flash_vision_shift_temp_fixed99.py>) 先转到相同shape的1024-element Float UB，再原样复制到各NZ group；没有浮点数学变化，增加6656 B UB。

### 精确广播优化完成的实际结果

[同轮叶子 A/B](<results/leaf05/leaf_compare99.json>)：两图分别由 NZ **26.999570 / 27.185980 ms** 降到优化 flash **25.236630 / 24.683430 ms**，快 **1.0699× / 1.1014×**；与基线流式 flash 的输出 SHA 完全相同，CPU Half MATH 误差仍是 1.502500e-5 / 1.527378e-5。

[优化后的完整视觉](<results/vision02/vision99.json>)：驻留中位数 **1.582865s**、CPU 相对 RMS **0.726924%**；全部27个 block 边界、3路 DeepStack、最终 merger 与基线流式版本逐项 SHA 一致。对 1.636251s 基线的耗时下降约3.26%是跨运行同机同口径比较；不夸大为同轮配对收益。原生注意力修补版同轮仍为1.356696s/0.724382%，所以不升默认。

最新有效 kernel 是 [shift-temp版本](<flash/flash_vision_shift_temp_fixed99.py>)；完整TP8测量见 [R100](<../round100/README.md>)。其原生文本组合最终误差17.04%→8.17%，完整encoder2.406s，仍未CPU FP16对齐或低于2s。所有失败/负结果保留，最后已完成源/二进制/worker审计并释放owned机器；目标保持active。
