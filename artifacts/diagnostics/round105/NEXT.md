# 下一轮：当前基线、负结果与精度优先计划

## 先读真实状态，别重新执行已结束的job

- [R102完整8-rank摘要](<../round102/summary102.json>)：BC128/原生文本相对CPU FP16 RMS **7.405683%**、完整stock **2.128423s**；同轮最快修补对照 **2.026669s**，所有样本仍≥2s。不是完整精度对齐或<2s，不升默认。
- [R103 BC256](<../round103/README.md>)两图慢0.5–1.0%；[R105精确Float逐字广播](<README.md>)虽然逐位一致，两图慢约1.5%。两个分支均不进入更贵的full benchmark；R104只是预备source，R106未创建/执行。别把更少scalar/循环或源码设计当作加速。
- [最终只读审计](<final_source_verification_cpu105.json>)：23 production、13 installed API、151执行源、24实际对象及原始HF/CPUgold/frozenQKV/launcher未变，全部相关worker0。所有相关job已收集、owned ts-217已释放，新一轮计算必须重新live claim owned；不取CI/dormant、不动其他租约。
- 大型真实selected50输出已下载本地，只用于精度控制，身份见 [captured outputs](<../round102/captured_outputs102.json>)。从来没被加载回任何warm stock输入；不能用它们冒充encoder速度。

## 精度：先尊重已完成的排除项，再定位真实42/43或下游conditioning

[R86实际第25层叶子](<../round86/README.md#L23-L38>)已经证明Half SiLU误差4.423e-7、SiLU×up3.132e-7，Float SiLU再Half与原生逐位相同。**不要重复这个已测对照或假定Float SiLU能修复全图；它目前不是有证据支持的精度修复。**

1. 先read当前R79候选与原始HF MLP代码，核查 `act_fn(gate_proj(x))*up_proj(x)` 的**实际**Half rounding顺序。CPU FP16接口的正常Float norm/SDPA中间值允许；Linear输入/权重/输出必须Half，不使用Float Linear权重影子、BF16 gold cast或事后norm。
2. 若继续晚层定位，优先从R93原始CPU Half suffix的新不可变脚本保存raw42/43，同时核对既有R78 SHA，**不能重定义oracle**；用同一CPU raw42对原始NPU layer43作norm/attention/row/MLP诊断，捕获真正42/43 gate/up而非重跑已排除的layer25 SiLU。精度隔离允许CPU参考输入，但不算完整encoder计时、不用于stock隐藏缓存。记录全部bits/误差与负结果。
3. 只有真实late gate/up给出额外证据时，才考虑自研SwiGLU：Float SiLU中间计算→**Half SiLU边界**→Half乘up→Half输出，必须复刻CPU顺序而非把乘法移到Half收窄之前。不能因“Float看起来更精确”就增加无效路径。廉价compile/probe→真实leaf→完整50层→真实5 paired8-rank-max encoder，再决定是否保留。
4. 后层输入敏感性已由 [R93原始CPU Half suffix](<../round93/README.md>)证实；不能从两个RMS百分比相减推断NPU算术贡献，也不能把某个叶子更准等同于完整模型更准。

## 真实下游conditioning（定位，不修改验收对象）

[原始DiT源码核查与计划](<../round101/CONDITIONING_NEXT.md>)已确认真正链条：condition_proj→2层token_refiner→模型本来就有的final_norm→SP选择/合并。使用实际checkpoint config/shape、原始Half weights和相同CPU FP16流程，比较R102保存的各路径在真实project/refiner输出处的传播。**尚未执行**，不得宣布下游误差已消失；也不得给encoder自行加fake final_norm来“通过”。CPU precision控制不得与NPU latency同时跑，避免污染计时。

## 更高收益的性能候选：真正Vision TP8（尚未实现）

本轮最后只读核查确认当前视觉是**replicated普通Linear**：[_NpuBF16CompatibleLinear继承nn.Linear且Half路径直接super.forward](<../round79/candidate/candidate_encoder79.py#L658-L674>)；[VisionAttention](<../round79/candidate/candidate_encoder79.py#L736-L774>)每rank投影全部16 heads并执行完整attention，MLP/merger也用同一普通Linear。它没有text模型的TP8投影；完整8-rank回执中的每rank全部27层/16 heads与此一致。这是源码事实，**尚未验证分片实现和加速**。

优先级可高于继续微调FA tile：用已有Text TP8 group，在私有Vision适配器里做原始Half权重的QKV按head column分片（每rank2 heads，Q/K/V各取144个输出通道：PyTorch weight/bias的**输出行**切片，合432个输出），独立当地attention，再out_proj Half×Half→Float局部和→Float HCCL→一次Half；MLP4304/8=538要精确logical slice并只为Cube pad到544，不能丢值或把padding算作真实通道。Norm仍完整1152，patch/merger需要对应真实column gather/row reduce，全部weight/bias值保持原始Half、rt SHA核验，不做Float权重影子。

先捕获/核查实际模型shape、通信group API与全视觉前向，不按defaults猜。廉价原语→原始FrozenQKV两head CPUHalf MATH→全27层/DS/merger重复和原参数SHA→真实8rankstock全计时，HCCL不能从计时里拿掉。Float部分和仍可能改变末位进而触发后层误差，**不能预先叫lossless或承诺8倍**；必须原始CPU FP16全图验证、保留负结果。源码工作用fresh fork、新private root，不能覆盖当前已执行candidate或原生forward。

## 较低优先级性能候选：MB128/BC64（尚未实现）

若后续继续tile探索，MB128/BC64需要减少PV UB占用，不能直接把现有3份Float PV/O buffer翻倍：会超原始196608B UB。可先廉价验证16-query-row的L0C→UB分块Float DMA，分别流式combine hi+lo/1024与O更新，精确记录C0 stride单位；还需处理4032→4096 query padding/最终64行丢弃、Float广播repeat_stride256超8-bit即时量问题。**所有CANN假设先真probe，旧source/newroot隔离**。R103/R105的负结果说明该路线净收益不能根据静态计数预判。

## 操作边界

所有新subagent必须fresh fork，不恢复任何inactive旧agent；本轮两个作者结果都已收集/完成。read文件后才能edit，已执行source永不覆盖；修复用新文件/private root，失败/负结果保留。追踪所有background IDs，丢回复先reconcile，不能盲重跑；最后job/source/gold/object/worker审计后释放本次owned机器。

整体goal17仍active：CPU FP16完整精度、text<2s、真正encoder/DiT/VAE共驻和匹配官方工况未达成。R76 DiT49warm≈14.657s/step、VAE≈13.5s仍是旧已测成绩，外部conditioning资格未解除；A2/A3官方T2VA/BF16/不同硬件不能直接等同当前FL2VA/FP16。不能complete或把困难当block。
