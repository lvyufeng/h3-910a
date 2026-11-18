# Round85 — 实际 attention 的 P@V 精度缺口与批量 FP32 累加

保持[CPU FP16 基准](<../round78/final_CPU_reference_verification78.json>)不变。
先用原始 checkpoint 的第0层、精确 CPU FP16 多模态初始条件、原有 TP8 loader/RMSNorm/RoPE
和 KV 零填充路径捕获真实 Half Q/K/V；没有重算整个 CPU 基准或替换金标准。

## 已验证的实际叶算子结果

[8 rank 捕获回执](<results/capture01/ranks/tp8_rank0.json>)全部重现 R84 的第0层原生输出，
Q/K/V 与 attention 输出重复位级一致，原参数 SHA 和 CPU 基准未变。

第0个512-query窗口，在所有8 rank：
- QK Float 相对诊断误差：约 **1.78e-7～2.06e-7**；
- Float softmax 概率相对误差：约 **5.0e-7～1.15e-6**；
- **即使注入精确 CPU 概率**，原生 Float P@V 仍有 **1.23e-4～1.67e-4** 相对误差。

这直接定位了该实际输入上的 P@V 精度缺口，不是仅凭返回 dtype=Float 推断内部运算正确。
Float 中间值只用于叶算子诊断；验收比较对象是同一组原始 Half QKV 的实际 CPU Half MATH SDPA。

[批量 TIK kernel](<batch/batch_gemm85.py>)让各 head 的 Half 矩阵独立累加到 FP32 GM，
不提前转 Half。概率 P 保持 Float；P_hi=Half(P)，缩放残差 P_lo=Half((P-Float(P_hi))*1024)。
两次 PV 的 FP32 结果在 FP32 中相加：PV_hi+PV_lo/1024，随后才转一次 Half。

[实际 batch 叶回执](<results/batch01/attention_result85.json>)：
- 同一 CPU Half MATH attention：误差 **2.3403e-4 → 1.5473e-5**，约降低 **15.1倍**；
- 四个真实 query chunk 的概率重构相对误差约 **3.45e-8～3.55e-8**；
- PV 对 CPU Float 诊断误差 **4.81e-7～9.73e-7**；
- 两个不同 head、两个 K 分块的硬件 smoke 也通过，输出重复位级一致；
- 完整 resident attention 叶计算 **15.737ms**，包含 padding、packing、softmax、概率拆分、
  两次 PV、FP32 相加和 Half 转换。此数不是完整编码器耗时，也不单凭它声称加速。

## 完整 TP8 对照已完成：叶改善不等于整图通过

[可复用 gated engine](<attention_engine85.py>)在新完整图目录串行预编译后，已先核验
与上面冻结叶输入的自研输出 **位级一致**。它只在候选模块中替换 text SDPA；禁用时
调用原来的 Float-math helper，不改全局 F.scaled_dot_product_attention，不缓存跨层激活
或 FP32 权重。

[完整驱动](<full_attention85.py>)和[全8 rank 汇总](<summary85.json>)均完成 exit0：

| 完整50层、与 CPU FP16 原始输出比较 | 相对 RMS |
|---|---:|
| 原生 attention + 原生 row，精确 CPU 条件 | 2.4757% |
| **精确 attention + 原生 row，精确 CPU 条件** | **1.8193%** |
| 精确 attention + FP32 row，精确 CPU 条件 | **2.7148%** |
| 精确 attention + FP32 row，原生视觉条件 | 7.6094% |

新的 attention 本身改善了完整 CPU 条件对照，但叠加 FP32 row 反而不如原生基线。
所有案例重复位级一致，原生哈希恢复、参数/输入 SHA 不变。这不是把误差回归隐藏成成功，
也不是选择新的容差放行。

真正的私有 encode_ids 在 **精确 attention + FP32 row + 原生视觉**路径上 warm 三次，
按各次8 rank最大值取中位数 **3.064183s**，包含视觉、全部50层、packing、collective、
最后D2H。未达到 <2s；不能把它说成加速。更好的“精确attention+原生row”路径在本轮
只作精度隔离，尚未单独取得完整 warm 编码时序。

[逐层漂移](<layer_drift85.json>)显示：组合路径第0层相对误差仅7.03e-5，第24层仍只有
4.80e-4，却在第49层增至2.71%。不能用初层或单叶算子15倍改善替代完整图验收。
下一步需用已有 CPU raw-layer24 状态重置晚层输入、验证晚段与实际 SiLU/MLP等叶算子，
同时保留 native row 作为默认对照；不要继续盲目叠加更多 FP32 替换。

[最终核验](<final_verification85.json>)确认23生产+13安装API文件、CPU基准与已验证运行库
未变，相关worker为零，owned ts217租约已释放。执行源码的SHA由
[只读来源核验](<executed_sources85.json>)保存。

整体精度、text<2s、真实服务共驻和官方 A2/A3 同工况比较仍未完成，目标继续 active。
[下一步协议](<NEXT.md>)保留本轮真实收益和全部回归，避免下一轮把组合路径当成默认优化。
