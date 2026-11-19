# 下一轮执行约束与优先事项（R107–R110之后）

## 已完成的真实性能，勿重复未必要的benchmark

- [R108全8rankstock摘要](<../round108/summary108.json>)：真正Vision block TP8+原生text TP8 **1.321200s**，5次每次取8rank最大值，全部<2s；高精度text组合1.860187s也全部<2s。包含实际两图/27vision/50text/HCCL/packing/D2H，不是单rank、仅vision、CPU条件缓存或最小一次。
- [R107视觉](<../round107/summary107.json>)：TP8 **0.675060s** vs同轮replicatedBC1281.483958s，约2.198×，原始Half参数/3DS/merger/NPU所有rank SHA/repeats/nativeRecovery通过。不是8×；Float部分和与重关联也改变低位，不能把精度收益全部归因于分片。
- 全encoder gap **4.395021%** /高精度text **5.125418%** 仍明显，**数值未验收、默认未升级、整体goal17仍active**。性能<2s不能自动解除conditioning质量/共驻/官方可比资格。
- 独立 [fresh-fork review](<../round108/review_findings108.json>)未发现P0/P1，接受真实性能measurement-only、拒绝数值完成、共驻未知；所有capture备份SHA已完成的followup见 [记录](<../round108/review_followup108.json>)。不恢复inactive旧reviewer。

## 精度结论：43不是孤立的百分之十几NPU爆炸，SiLU不支持盲改

- [R109原始CPU捕获](<../round109/README.md>)两次完整25..49全部25SHA精确匹配旧R78，补存原始raw42（input43）和raw43，无新oracle/teacher/FinalNorm。
- [R110同输入block43](<summary110.json>)：原生 **0.0270854%**、高精度attention **0.00687125%**，全8rank重复/原参数/gold/nativeRecovery一致。R93输入扰动放大与该同输入实验不能做RMS百分比相减归因。
- 实际43层SiLU误差9.51e-7，FloatSiLU再Half仍与原生NPU逐位相同；25层R86也已排除。**不要再假定Float SiLU或融合后取消Half边界就会提高精度**。Half GEMM gate/up/down同输入误差约1.1–1.3e-5，norm约3.46e-6，尚无单原语能解释全图4.4%。

## 下一步建议（尚未实现/测量）

1. **真实conditioning链**：先读 [已核查的DiT原始源码与计划](<../round101/CONDITIONING_NEXT.md>)，从官方未蒸馏、无LoRA的Half DiT checkpoint核实实际config/shapes/weight mapping，使用R108保存的真实selected50与R78 CPUselected50，经真正condition_proj→两个token_refiner→模型本来存在的final_norm。建立同一个CPU FP16接口（Linear输入/权重/输出Half，原始Float norm/rotary/MATH中间值允许），对投影、每refiner、最终norm逐段重复、原参数SHA验证。不能为encoder加假norm，不能事后改验收对象宣布旧encoder通过。下游agreement可以定位影响，不抹掉raw50gap。CPU精度控制绝不与NPUlatency并发。
2. **Vision same-operand residual来源**：TP8提升了运算和速度，但merger对CPU仍0.6985%。检查真实patch输出、原始HF bilinear indices/weights/position embedding、58个LayerNorm、27tanh+4erf GELU、merger的同输入对照，先验证CPU原始舍入顺序/输出dtype，不根据“Float更精确”盲改。源码R79position用Floatweights/sum、HF原版调用vision_utils，这两者差异先逐位检查indices/weights，不能缓存CPUgold hidden state来绕过。
3. **只在隔离证据支持后叠加**：可在新private diagnostic中测试VisionTP8+text Float row部分和（原R84权重仍Half、HCCL前Float），但过去更准叶子常让全图更差。所有候选需sameinputnative/disabledRecovery、CPU原始Half、全部27/DS/merger/50text、5paired8rankmaxstock，保留负结果；不先宣称lossless或完整成功。
4. **真正共驻与49steps**：R108allocated约9.263GB、diag峰值12.733GB（含所有baselines/captures），TP额外Halfweight127.4MB，不代表实际DiT/VAE共驻。未做service生命周期加载/卸载、encoder组与USP组、R107instancepatch准确接入、真正49步含新conditioning的privatepipeline运行。必须真实共驻/运行/计时/质量核验后给可比表。R76DiT49warm≈14.657s/step和VAE≈13.5s为旧已测成绩；A2/A3官方T2VA/不同硬件/BF16不能直接等同FL2VA/FP16。

## 当前资源/证据边界

[最终审计](<final_source_verification110.json>)：23production+13installed API+230执行源+38实际对象、原始HF/CPUFP16gold/frozenQKV/launcher全部未变，5个真实R108Half输出及原始CPU42/43file/tensorSHA核验，所有worker0。全部相关job需收齐后本轮owned217释放，状态见 [STATE](<STATE.json>)；下轮任何NPU运行先live claim owned，不取CI/dormant，不动其他会话CUDA租约。首次CPU109 libhccl环境失败、旧auditcore白名单失败保留，不覆盖oldsource，不盲重播。

所有新subagent必须fresh fork；task58 VisionTP作者、task60 CPU作者、task61 reviewer已完成/结果收集，不能唤醒旧agent。read后才能edit；已执行source不可覆盖，数学修复用新file+private root，失败/负结果留档。所有background IDs记录收集；source/oracle/object/workerzero最终审计后再释放本次owned机器。goal17仍active，不能因性能成功标记wholecomplete或将困难当blocked。
