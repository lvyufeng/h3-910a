# 精度继续定位：真实 H3 conditioning 入口，而非人为加 norm

只读核查了当前已审计的 DiT 源：[H3 transformer](<../round62/source/minimax_h3_transformer.py>)。该本地文件 SHA `54e45b10c1650822cb6896df7bb90d107c15d3eea52d47b57a078de64c410d13` 与 R76/R100 23-source 审计记录中的生产源一致，不是未经核查的历史版本。

实际模型链是 `condition_proj` → 两层 `token_refiner` → 原模型已有的 `token_refiner.final_norm` → SP 行选择/合并。源码证据：

- [结构与 config defaults](<../round62/source/minimax_h3_transformer.py#L100-L143>)：text_dim5120、hidden5376、heads56、head_dim128、FFN14336、refiner2层；**实际实验仍必须读取 checkpoint config/weight shape，不能只用 defaults。**
- [condition projection](<../round62/source/minimax_h3_transformer.py#L1253-L1271>)：biased ColumnParallelLinear，gather_output=True。
- [两个原始 refiner blocks](<../round62/source/minimax_h3_transformer.py#L807-L853>)：真实 pre-norm attention + MLP residual，没有 AdaLN/RoPE，refiner attention 明确 skip_sequence_parallel。
- [真实 final_norm](<../round62/source/minimax_h3_transformer.py#L856-L905>)：它是原始模型的一部分，不允许给 encoder 自行添加一个假的 final_norm 以缩小误差。
- [实际调用](<../round62/source/minimax_h3_transformer.py#L1530-L1564>)：首先投影真实 text_embeddings_selected，再 token_refiner，随后按 text_local_indices 合并；数值基准须包含真实 selected rows/cu_seqlens/role tags。

这只是为下一步提供源码事实，**尚未构建或执行下游 CPU/NPU 数值比较，不宣称误差会被 norm 消除，更不能替代 R78 selected50 的当前精度结论。**

可执行的下一步：保存新完整 encoder 各配置的实际 Half selected50（在 timed stock calls 外保存，gold/weights/SHA固定），从无 LoRA 的原始 DiT checkpoint 取对应 Half condition_proj/refiner 参数，确认实际 config、映射、GQA、SiLU gate顺序与 packing。用原始 Half Linear 输入/权重/输出、原始 Float norm/MATH attention 中间值建立可重复 CPU FP16 conditioning 控制；分别比较投影、两层 refiner、真实 final_norm 输出。NPU使用真实原始入口而非 CPU precomputed conditioning，不能引入 Float Linear 权重教师、BF16 gold cast、隐藏值缓存或事后阈值。

CPU控制不得与完整 NPU latency样本并发，避免污染计时。下游 agreement与encoder raw selected50 gap应同时报告：它可以定位哪一段真的影响模型，但不能事后改验收对象宣布原来未通过的encoder通过。
