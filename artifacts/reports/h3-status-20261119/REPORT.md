# H3 / 8×910A：现状复核与优化执行方案

**复核日期：2026-11-19（Asia/Shanghai）**  
**用途：回应“先系统梳理现状和优化方案”。不以这份报告宣称整体目标完成。**

## 0. 结论先行

**不是没有优化收益，而是“局部收益”还没有形成“精度可靠、阶段耗时可归因的统一端到端方案”。**

目前可以拿出的实测收益：

1. **完整编码器候选：2.156 → 1.321 s（同轮对照，耗时下降约38.7%，1.63×）。** 其中真正的视觉TP8把视觉塔1.484 → 0.675 s。不是仅文本、仅一个rank或缓存输入。尚未完成生产pipeline集成与精度验收。
2. **真实FL2VA形状的attention wrapper：约421 → 154–155 ms（约2.73×）。** 这包含padding/copy/trim；后续修正真实有效key尾部mask。旧N=63232的672 ms不是当前124帧基准的attention形状。
3. **官方未蒸馏FP16模型：热窗口测过14.657 s/步；VAE热态decoder reporter测过13.62 s。** 但该生成请求使用外部预计算conditioning，且视频质量未通过；不能作为完整pipeline完成证据。
4. **当前live-encoder服务路径：实测2169.11 s/请求，49步、124帧；去噪进度条稳态约33.45 s/步。** 这是“能生成、但质量和速度都未达标”的服务基线，尚未定位各阶段精确耗时。

**最优先的工作应是把已有1.32 s编码器收益真正接入、给阶段耗时补齐可靠埋点，并处理真实质量差异。不是继续无埋点地一遍遍跑49步、再从总耗时反推原因。**

## 1. 固定验收口径

- 模型：官方MiniMaxAI/MiniMax-H3，未蒸馏、无LoRA；官方原始BF16权重与可复现FP16转换分开记录。
- NPU：8×原始910A；设备长名称可能显示Ascend910B，不能与910B/A2当成同一代硬件。
- 工况：1344×768，FL2VA，第一/最后帧条件，名义5 s/24 fps，49次实际DiT评估，USP8/ring1，固定seed=1234。
- 现有采样接口将120帧对齐到124帧，实际视频时长约5.1667 s。报告必须列明这一点，不能写成严格5.0000 s。
- 分开报告：启动/加载、冷请求、暖请求、编码器、媒体VAE encode、conditioning refiner、每步DiT、VAE decode、uint8/D2H/音频/封装。
- 不混算：CPU/GPU参考构建时间不是NPU速度；算子时间不是服务总时间；不同轮次、不同conditioning、不同精度的时间不能简单相加当作完整测量。
- GPU权重只在远端设备保存；本地仅代码、日志、校验信息和小型输出证据，不新增任何权重下载或中转。

## 2. 实测台账：有成效的部分与边界

| 项目 | 原始实测 | 优化/当前实测 | 可以得出的结论 | 尚未满足的条件 |
|---|---:|---:|---|---|
| 完整encoder，同轮replica-BC128与VisionTP8 | 2.156184 s | **1.321200 s** | 真正27视觉块+3 DeepStack+50文本层；5次采样，每次取8rank最大值 | 还没集成到live服务；FP16 GPU全张量对照仍有差异 |
| 完整视觉塔，同轮replica与TP8 | 1.483958 s | **0.675060 s** | 2.20×实测加速 | 不能乘8当作收益；不代表整encoder或者DiT相同倍率 |
| Attention，真实41792 packed rows | 约421.151 ms | **约154.391 ms** | 2.73× wrapper加速，含搬运/填充/trim；v3修正非零无效key尾部 | 采样参考与端到端质量仍需分开；后续增益要在真实形状测 |
| Attention，旧63232形状30core修正 | 559.6 ms | **318.3 ms** | 同数学、输出bit-exact的core配置收益 | 历史接口返回209帧，不是当前124帧工况，不能并入当前收益总表 |
| 官方FP16 DiT，R76外部conditioning热请求 | — | **14.657157 s/步** | 第二个请求49步均低于15 s | 非live编码器完整服务；视频质量失败 |
| VAE，R76 decoder reporter | 冷态128.27 s | **热态13.62 s** | 20 s阈值在该decoder测量口径下可达到 | 不包括所有uint8/D2H/封装开销；冷暖差异原因没有隔离 |
| Live-encoder服务，R116 | — | **2169.110050 s/请求** | 完成49步与124帧，服务可运行 | 输出logo/光晕/拖影；未使用已验收的优化encoder集成；每步无独立同步埋点 |
| AdaLN精确保留边界，R62三块8rank | 约216–217 ms/完整块 | **约215.6–216.3 ms** | 实际捕获一致，块级只约0.3%收益 | 固定顺序3次，不是50层/49步/E2E新加速；融合Norm版本未过精度 |

**源码不是“还没做”：** RMSNorm/AddRMSNorm、SwiGLU、RoPE、AdaLN已经有自研TIK/TBE原型和部分bit-exact证据（R58–62）。缺口是适用于当前FP16生产路径的稳定语义与集成，不应再把它们全部从零重写。

### 主证据

- [R108完整encoder回执](<../../diagnostics/round108/summary108.json#L105-L135>)
- [R107视觉TP8回执](<../../diagnostics/round107/summary107.json#L43-L85>)
- [FA形状、core修正、padding与有效尾部更正](<../../design/ascendc_flash_attn/FINAL.md#L3-L101>)
- [R76完整生成状态与冷暖时序](<../../diagnostics/round76/STATE.json#L20-L35>)
- [R116 live-encoder原始回执](<../../diagnostics/round116/encoder_live_timing116.json#L23-L124>)
- [R62块级原始对照](<../../diagnostics/round62/summary.json>)

## 3. 精度与质量：还不能说已经成功

4090原始BF16/FP16 encoder参考已建成，固定了官方权重、输入与原始HF代码。这个工作有价值，避免继续在慢CPU上跑参考。但现有数字不支持“910A精度已经定论”。

| 比较 | 全部2043行的相对RMS误差 | 解释 |
|---|---:|---|
| 优化NPU FP16 vs GPU FP16 | **7.2472%** | 同精度全tensor仍未对齐 |
| 原生NPU FP16 vs GPU FP16 | **5.9994%** | 当前优化候选并非所有参考上都更接近 |
| 优化NPU vs GPU FP16，仅27个非视觉行 | 0.1534% | 部分行较接近，不代表全部conditioning行可忽略 |
| GPU FP16 vs原始GPU BF16 | **93.8709%** | 此报告固定以BF16参考RMS为分母；之前85.4%使用相反方向/分母，不能混用 |

- 按用户要求，**不强迫910A执行BF16，也不将“对齐BF16”作为FP16的验收定义。**
- 但GPU/CPU FP16彼此不同，不能逻辑推出NPU剩余误差一定无害。必须做**同输入阶段对照**，特别是视觉行→condition_proj→token_refiner→DiT预测→VAE。
- 已检查的transformer源码把text_embeddings送进condition_proj/refiner；没有证据证明那2016个视觉占位行被全部丢弃。不能只用27行宣布通过。
- 图片中logo/光晕/球体拖影在live-encoder请求也出现，说明“用实时编码替代外部conditioning”没有修好它。**不能据此断言这是官方模型学到的水印，更不能断言继续核对encoder毫无价值。** 尚缺完整GPU生成/DiT/VAE对照来定位原因。

证据：[GPU↔NPU精度回执](<../../diagnostics/round114/GPU_vs_NPU_encoder114.json#L33-L95>)、[实际conditioning投影源码](<../../diagnostics/round62/source/minimax_h3_transformer.py#L1541-L1548>)、[可见质量缺陷帧](<../../diagnostics/round116/frames/measured_frame048.png>)。

## 4. 必须撤回或降级的归因

以下更正不抹掉原始实验，旧文件保留；本报告优先采用原始日志和调用栈，不再继承旧summary中的过度推断。

### 4.1 “472 s就是VAE decode，约468 s/34×”——不成立

R116只有完整请求计时和去噪进度条，没有decode埋点。2169.11−1697≈472.11 s最多说明**去噪循环之外**的总成本，包含循环之前的live text encode、两张图的VAE encode、权重搬移、conditioning准备，以及循环之后的decode/音频/D2H。不能把残差都算到VAE decode。

因此468 s、34×、其中455 s为decode回归，以及9.8 GB/468 s≈21 MB/s的“传输速度”全部降级为**未经验证的猜测**。冷暖decode之差114.65 s也不能全部认定为加载时间，可能包含编译/缓存等成本。

### 4.2 “DiT始终整模常驻、没有offload选择”——与运行日志相反

`resident_dit_paths`是声明式placement计划，不等于所有DiT blocks永久常驻。当前R121日志明确显示：

- `DistributedLayerwiseOffloadBackend`
- `Applying distributed layer-wise offloading on ['transformer']`
- `Distributed layer-wise offloading enabled on 129 blocks across 4 group(s), transfers={dit: allgather}`

实际runtime确实有DiT层级流式权重/AllGather。不能从一个plan字段推翻运行时hooks。

证据：[运行日志](<evidence/r121_log_snapshot.log#L744-L771>)、[offloader实施源码](<evidence/distributed_layerwise_backend.py#L1690-L1803>)。

### 4.3 “14个shard是VAE，OOM证明DiT+VAE只能差74 MiB/103 MiB”——不成立

R119保存的OOM栈落在 `component.module.to(device)` → `minimax_h3_transformer._apply`。该失败是把**未选中的transformer搬到卡上**时发生，不能靠另一rank的14-shard进度推断是VAE加载。14-shard数量本身与text encoder相符。

失败的74/222 MiB是**下一次失败分配**，不是“完成全部加载只差这点内存”。关掉DLO后仍有后续权重待加载。另一方面，checkpoint磁盘字节数不等于GPU live权重/影子/缓冲总量，USP8切序列也不自动把DiT权重除8。不能用“10.43+9.8≈20 GiB”证明全驻留一定可行。

这些OOM证明**当前几种具体配置失败**，没有证明所有调度都不可行，也没有证明融合一个100 MiB buffer就能避免offload。

证据：[R119实际OOM调用栈](<../../diagnostics/round119/dit_resident_oom119.log#L684-L742>)、[未选DiT的device搬移逻辑](<evidence/component_utils.py#L187-L193>)。

### 4.4 “live encoder一定造成2.28×回归，优化候选1.32 s已包含在live结果里”——未隔离

R76与R116相差多个因素：conditioning来源、encoder计算路径、冷暖态、服务配置和计时方法。因此33.45/14.657≈2.28只能当**跨轮次差异**，不是干净的因果A/B。

更重要的是R119/R121启动日志仍写`vision replicated`。R108的真正VisionTP8候选尚不能视为已接入当前服务。现在最直接的收益机会是**接入已有候选并核验**，而非把当前长跑当成已经包含1.32 s编码的最优方案。

证据：[当前encoder运行模式](<evidence/r121_log_snapshot.log#L589-L597>)。

## 5. 优化优先级与交付物

### P0：一次补齐可信阶段profile，结束猜测（短任务，不再先跑49步）

**实现：** 在私有worker wrapper中覆盖实际pipeline的encode_prompt、_encode_local_media、condition_proj/refiner、denoise各步、decode、后处理边界；记录device event/必要的阶段同步、各rank max、allocated/reserved、权重H2D/AllGather次数与字节。每次启动记录所有source SHA、原始模型revision、各组件dtype、实际seq_len/valid_k、FA binary/core数与offload策略。

**工作上限：** 先1–2个DiT步骤与一次真实尺寸decode；profile用途可以改变步数，但不得拿它冒充49步总时长。拿到可信分解后才提交长跑。

**验收：** 能回答“33 s的一步到底多少是FA、GEMM、HCCL、H2D和CPU等待”“去噪之外472 s分别在哪些阶段”；计时范围闭合，异步重叠不得简单相加。

### P1：兑现已经测到的收益——encoder候选真实接入 + 分阶段释放

**第一项：** 接入R107/R108真正VisionTP8/FP16候选，固定已有权重和实际前向，不重写已有算子。先确认服务内的27视觉块确实每rank只做自己分片，而非`vision replicated`。

**第二项：** 生命周期做成**每请求实时编码 → 保留当次conditioning → 释放encoder权重/影子/临时buffer → DiT流式推理 → 释放DiT工作区 → VAE decode**。保存当次必要的conditioning是正常阶段数据流，不是复用旧输出跳过encoder。编码、释放、搬移全计入总时间。

**验收：** 模型权重/输入身份不变；新旧路径同输入输出对照；5次完整8rank-max encoder计时；当前encoder计算段<2 s，加载与释放另列。不能保证服务总编码<2 s，必须实测。

**预期收益：** encoder已有同轮1.63×证据；总请求节省多少取决于P0中真实编码占比。回到约14.66 s/步是目标，不是承诺。

### P2：对profile中的大头动手——权重流式与attention，两条可并行但分别验收

**DLO/加载：** 保留USP8工况，先优化重复materialization、dtype转换影子、非block常驻状态和预取buffer。用2槽复用/提前预取/同一节点不可变host权重共享；逐项测H2D/AllGather等待和峰值。尝试少量DiT layers常驻要按真实内存台账，不默认整模能装下。不能把故障超时加长算作加速。

**FA：** 真实124帧基准使用packed41792/valid41769（需当前启动再次核实），保留正确prefix/tail mask与30core。优先Cube与Vector重叠、K/V双缓冲、Q复用、NZ布局/拷贝消除，而不是又扫一轮未正确编译的tile。以目前154–155 ms为基线；首轮候选门槛为**至少10% wrapper收益**且实际tail样本/同输入输出不退化，才进入完整块。进一步希望提升20–30%是探索目标，不承诺。

**收益预算：** 50块每块154.4 ms的纯FA量级约7.72 s/步（算术量级，不是完整step分解）。若FA净提升20%，理论只省约1.54 s/步；它不能单独解释从33→15 s的18 s缺口，DLO/stage工作仍必须测清。

### P3：补齐融合清单，但避免为微小收益破坏精度

保留已验证SwiGLU/RoPE/AdaLN point kernels；FP16中间舍入边界和原Norm语义不能删。Norm融合未通过的版本继续默认off，不用“更高精度累加”替代实际合同。当前真实块R62是约216 ms，点融合只约0.3%，不能用旧869 ms profile泛化所有场景。先测launch/buffer是否真在critical path；除非P0证明明显内存收益，不再优先投入大规模Norm重写。

### 与P1/P2并行：质量定位门，不再因图像能解码就通过

- 用4090 FP16参考做**同一tensor**的vision/merger、condition_proj/refiner、一个DiT block/一步noise预测对照；不是新跑CPU整图。
- 找第一个分歧点；FP16 range策略可以用有证据的缩放/FP32累加，但禁止clamp、FP32权重teacher或改变语义。
- 用相同生成latents分别送GPU/NPU VAE，定位logo/光晕是否decode产生。若latents此前已出现偏差，再查DiT/attention/mask/schedule。
- 原始BF16 GPU只用于解释FP16范围/舍入差异，不强迫910A对齐BF16。
- 先一两个控制步骤；只有定位结果支持的候选才跑49步质量对照。

## 6. 如何让下一轮真的交付“有成效的结果”

**不再交付“又一个推测summary”，交付以下三件事：**

1. **一张可闭合的阶段耗时/显存表。** 能据此选方案，而不是不断推翻口述归因。
2. **一个已经接入的优化候选。** 首选现有VisionTP8；新旧同输入和服务内调用身份可复查，列出encoder实测节省及质量边界。
3. **同一原始模型、同一配置的完整冷暖对照。** 总墙钟、每rank同步步骤、VAE decode包含/不包含哪些后处理、完整124帧与音频质量。没有质量与精度验收的速度收益标记为measurement-only。

运行规则：短profile/单块候选→阶段A/B→最后49步。局部收益<5%或精度退化就停止该分支；若短测试连启动都失败，保留栈后先修加载生命周期，不继续盲目完整跑。最多保留一个当前完整生成任务。

## 7. 当前实验与官方比较

- R121双请求正在运行，快照时第1个请求推进到23/49，进度条稳态约33.3 s/步；还没有完整case回执。**尚无第二请求快了/解码13.6 s的结果。** 不把此前预测列入收益。
- 已启动任务只收集结果，不在本次梳理期间追加新完整生成。租约ts-217到2026-11-20 01:17:54 Asia/Shanghai；监控结束/失败后必须核对worker并释放。
- A2约150.06 s、A3约114.19 s来自暖态T2VA/原始支持精度/不同硬件；本任务FL2VA/FP16不能当成严格同工况。提供并列参考可以，宣称11×或4.8×因果差距不可以。
- 735.69+1.32≈737.01 s只是不同轮次两项数字相加，**不是实测完整服务基线、不是必然可实现的staged成绩**，报告中不作为已完成结果。

## 8. 整体目标状态

- 自研融合算子：**部分实现、部分合同通过；完整生产集成未通过。**
- Flash attention：**有真实加速与mask修复；距高利用率仍有余地。**
- Text encode≤2 s：**候选完整计算段已测到；live集成/精度尚未验收。**
- VAE decode≤20 s：**历史热态decoder reporter已测到；当前服务阶段未实测闭合。**
- DiT≤15 s/步：**外部conditioning热窗口已测到；当前live路径约33 s，质量未通过。**
- 官方可比、完整原始模型质量：**未完成。**

**后续方向：先兑现已获收益并给分阶段账算清，再做大头kernel/流式优化。无需继续强迫BF16，也不能因为不强迫BF16就跳过FP16精度与视频质量。**
