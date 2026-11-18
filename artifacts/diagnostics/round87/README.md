# Round87 — 真实视觉 bias 单次FP16转换与精确attention/原生row完整编码

[CPU FP16 基准](<../round78/final_CPU_reference_verification78.json>)和原始 checkpoint 保持不变。
目标是在真实视觉 biased Half Linear 中移除多余舍入：Half A/W → 全K的FP32 L0C →
FP32 UB bias加法 → 一次 Half输出，不使用硬件不支持的TIK bias参数或FP32权重影子。

## 已完成：精确 attention＋原生 row 的完整时序

[8 rank 实测汇总](<best_text_summary87.json>)补测了 R85 中较好的精度路径，**未安装FP32 row
适配器，也没有2.16GB/rank的row布局缓存**。真实私有 `encode_ids` 包含原生视觉、全部50层、
attention packing、HCCL和D2H，warm三次按各次8 rank最大值取中位数：**2.898743s**。

相同CPU初始条件下精确attention＋原生row误差重现 **1.8193%**，原生视觉条件下则为
**7.5915%**。完整text<2s及整图精度仍未过关。该路径比之前组合FP32row的3.064s少一些
开销，但不据此声称整体加速目标完成。

原始驱动沿用了早期 `tp8_deviation` 的静态文字标签；本轮不使用该标签作运行证据。
实际源码没有row安装/priming/enable操作，运行字段 `row_adapter_not_installed=true`，
原生恢复、重复及参数SHA核验都通过。该事实在汇总中明确记录，原始回执不覆写。

## 已完成：视觉单次bias舍入

[视觉kernel](<vision/vision_bias87.py>)是新fresh fork的独立源码，已review并通过38项
stdlib契约测试；root对其原子spec修复增加了准确解析绝对路径检查，算术不变。
[私有adapter](<vision_adapter87.py>)保留116个原始Half线性权重，仅缓存Half布局及原始
Half bias精确扩展的Float值；激活/权重4304维度只加零到4352，输出按原始维度裁剪。

实际6组kernel尺寸：M8064的K/N1152、3456、4352，以及M2016的4608/5120。M2016的
尾部填充行输出是**bias而非零**，host正确裁剪。不能复用FP32输出stride的2倍系数来
写Half GM；新kernel使用正确的Mpad-Mb、32B单位stride。

[视觉驱动](<vision_probe87.py>)先保留原生27层对照，再用真实第0块qkv/mlp1/mlp2输入
对照 CPU Half F.linear（同一Half权重和bias），核验padding/重复，随后测试完整27层、
三路DeepStack并恢复原生输出哈希。所有setup/compile都在warm之外，layout和kernel
都在resident视觉时序内。单rank视觉时序不会冒充完整编码器/TP8服务时序。

[实际汇总](<summary87.json>)核验结果：

| 同输入 CPU Half Linear 对照 | 原生相对 RMS | 单次bias舍入相对 RMS |
|---|---:|---:|
| 第0块 qkv | 2.620e-4 | 6.625e-6 |
| 第0块 MLP fc1 | 2.455e-4 | 4.780e-6 |
| 第0块 MLP fc2 | 3.219e-4 | 1.034e-5 |

真实叶算子误差降低约31～51倍，padding、重复、原参数检查通过。但完整27层 merged
误差只由 **1.4568%降到1.2887%**；三路DeepStack由0.2820/0.5658/0.9088%降到
0.2198/0.4359/0.7332%。原生输出完整恢复，CPU基准未变。

Resident视觉warm中位数：原生 **1.473723s**，候选 **1.488395s**，没有加速收益。
额外缓存为1,186,725,888B Half布局和1,247,744B Float bias；不能说它们对共驻服务免费。
这条bias修补修正了真实舍入偏差，但剩余视觉差异仍显著，**不作为完整精度通过**。

[最终R87核验](<final_verification87.json>)确认23生产+13安装API、CPU参考回执和视觉/选中50层
原始payload、运行库SHA均未变，相关R87worker为零。
[执行来源](<executed_sources87.json>)保存实际源码SHA。47项源码与实测证据测试通过。
Owned ts217租约暂保留用于下一条已授权的视觉attention对照，未重复启动R87任务。

下一步见[协议](<NEXT.md>)：真实vision head_dim72、每图4032token的QK/PV CPU Half MATH
对照以及零填充候选，分别开关bias/attention，避免将更精确的叶算子盲目叠加为全图通过。
