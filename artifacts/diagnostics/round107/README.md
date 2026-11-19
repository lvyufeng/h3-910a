# Round107 — 真正 Vision TP8：全部27层视觉1.484s→0.675s

本轮首次把原来8个rank都重复的VisionAttention/MLP改成真正TP8。只改27个block的attention+MLP，patch embedding、4个merger、norm/rotary/几何缓存保持原始replicated结构。所有参数仍为原始Half，生产源与CPU FP16 oracle未改；不是蒸馏、BF16 gold cast或Float权重教师。

## 实际视觉同轮结果

5次实际resident视觉前向，每次取8 rank最大值，再取中位数；计时包含HCCL、layout、packing、54次本地两头attention、全部27层与3路DeepStack/merger，没有CPU预计算conditioning。

| 配置 | 完整视觉中位数 | merger对原始CPU FP16相对RMS |
|---|---:|---:|
| 原生 | 1.472048s | 1.456805% |
| patch/bias/true-erf修补、原生attention | 1.372041s | 0.724382% |
| 同修补、replicated BC128 | **1.483958s** | 0.728894% |
| 同修补、真正block TP8+本地BC128两头 | **0.675060s** | **0.698518%** |

对replicated BC128同轮快约 **2.198×**，对修补原生attention快约2.032×；不是8×。TP8最终输出与replicated版本可能末位不同，不称原生-bit lossless；全8rank的全部27层、DeepStack、merger SHA相同，重复与原生恢复门禁通过。误差下降不等于完整encoder精度验收，完整测量另见 [R108](<../round108/README.md>)。

[完整可重算摘要](<summary107.json>)、[rank0](<results/vision_tp01/vision_tp_rank0.json>)/[rank1](<results/vision_tp01/vision_tp_rank1.json>)/[rank2](<results/vision_tp01/vision_tp_rank2.json>)/[rank3](<results/vision_tp01/vision_tp_rank3.json>)/[rank4](<results/vision_tp01/vision_tp_rank4.json>)/[rank5](<results/vision_tp01/vision_tp_rank5.json>)/[rank6](<results/vision_tp01/vision_tp_rank6.json>)/[rank7](<results/vision_tp01/vision_tp_rank7.json>)，执行 [whole-vision驱动](<vision_tp_probe107.py>)。

## 精确分片与算术顺序

- QKV原始weight `[3456,1152]`，Q/K/V各自的1152输出**行**按head分片，每rank144行，各段拼432；不是连续取整个QKV的432行。每rank2×72 heads，原始Float视觉RoPE顺序不变。
- out_proj原始weight的144输入**列**为本地row shard；Half输入/权重→Float L0C→Float GM部分和→**Float HCCL求和**→原始Half bias精确扩为Float相加→一次Half输出。
- MLP4304→每rank **538真通道**。fc1列并行，原始GELU(tanh)只对538真通道执行；fc2按538输入列row并行，Float求和后bias/一次Half。TIK K/N边界补零到640，不把640当作真实MLP宽度。
- GEMM几何：QKV K1152/N432→Npad512，proj K144→Kpad256/N1152，fc1 K1152/N538→Npad640，fc2 K538→Kpad640/N1152，M8064=63×128本来就对齐。
- 每次实际视觉前向验证计数：27 attention、27 MLP、54两头segments、54 Float collectives、54 columns、54 rows、108 primed original-Half权重；关闭后原生恢复。

[适配器](<tp_vision/vision_tp_adapter107.py>)只覆盖54个实例forward，不改全局Linear/SDPA，不干扰旧bias适配器，禁用时精确委托/卸载恢复。新source作者fresh fork，36项stdlib测试通过；真实编译与运行均由root完成。[矩阵backend](<vision_gemm_backend107.py>)复用已执行R87 biased-Half columns/R83 Float rows，缓存仅Half权重布局，Float bias，不缓存hidden。

## 先行硬件/精度门禁

[实际checkpoint/SoC metadata](<actual_preflight_fixed107.json>)核查hidden1152/intermediate4304/heads16/depth27，实际权重名带 `model.visual.`；原始full `Ascend910B`（无A2/A3后缀）alias短名Ascend910、30核，短alias另一入口显示32物理核，launch始终≤30。未伪装新硬件。

[两头probe](<results/head2_01/head2_probe107.json>)覆盖两图×8个head shards，严格 `[1,2,4032,72]`，与原BC128每个head slice逐位相同，对CPU Half MATH相对RMS8.17e-6–2.05e-5，中位约4.486ms/本地两头图。global原16-head helper未改。

[8-rank GEMM+Float-HCCL原语](<results/gemm_tp01/gemm_tp_rank0.json>)使用原始CPU Half F.linear控制，K144/538边界补零精确，最大相对RMS1.224e-5；rows通信前后Float、最终一次Half，全8rank row输出SHA一致，重复及原输入/权重未变。[串行小/全shape prebuild](<results/gemm_tp01/prebuild107.json>)在model workers前验证8对象，不并发重编译。

最终23 production/13API/60执行源/34实际对象与5份完整encoder输出审计见 [R108审计](<../round108/final_source_verification_grid_fixed108.json>)；随后CPU晚层精度对照与最后资源清理见 [R110](<../round110/STATE.json>)。完整CPU FP16精度及共驻/官方匹配工况未验收，持久目标仍active。
