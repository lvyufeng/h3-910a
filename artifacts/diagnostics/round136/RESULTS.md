# R136完整49步质量对照：有数值变化，仍不采用

**Keep 完整质量测量；Reject 自动采用或“无损替换”；Awaiting approval 下一项定位。**
[Issue #15](https://github.com/lvyufeng/h3-910a/issues/15)仍开放，
[Issue #4](https://github.com/lvyufeng/h3-910a/issues/4)保持Blocked，copy候选未启用。

## 1. 本轮实际范围与授权边界

[用户授权](approval136.json)只允许完整49步质量对照：原默认False路径→临时globalTrue＋
最终AdaLN候选→恢复False原路径。没有采用、容差、copy或其他近似授权。

这三个请求都重算自己的实时FP16 encoder和原FP32 VAE conditioning/解码。**globalTrue作用于
整个pipeline**，不只DiT或AdaLN；最终质量差异是组合对照，不是AdaLN单算子的独立因果证明。
只有一次候选完整49步请求，不能声称完整49步候选已重复稳定；R135固定step25的三次重复
与本轮完整生成不是同一验证范围。

固定prompt红球、seed1234、1344×768、120→124帧/24fps、首尾keyframe、原initial/sigma fixtureSHA。
权重revision `0919fb684102ef115126612f3fbb84d1027e7317`复用已验证host副本，没有下载或中转client。

## 2. 原模型执行门未减

[执行汇总](summary_execution136.json)及[原始生命周期](executed_quality136a/engine_quality136a/lifecycle_result136.json)：

- 同8worker PID，三次完整请求，各rank每次**49个实际evaluation、50个sigma边界**，原50层DiT。
- 每step**155项严格Half检查**；各请求60,760项，总**182,280项**。没有BF16/clip/HF32/teacher/少步。
- DiT每forward50次实际FA ACL launch；每请求19,600次，总**58,800次，0个wrapper error**。
- DiT实际TP2＋USP4，FA输入BHND `[1,7,41792,128]`。原FP32 front/time/final/VAE/norm/residual
  与power-of-two range guards保留。原shared-copy baseline未改变。
- Live encoder实际TP8，组0–7；每rank实际NPU Half vocab shard `[18992,5120]`，partition rank0–7，
  不是仅传kwargs或空placeholder。Vision仍复制，不虚构vision已TP分片。
- 唯一最终AdaLN local候选为Half矩阵→FP32 GEMM输出＋原bias的FP32扩展相加→一次Half返回。
  原step0两time rows、其余48步三rows均实际执行，没有padding/subset。每rank候选rowhist
  `{'2':1,'3':48}`，共**49/rank、392次**；是Python调用路径计数，不是CANN内部归约机制证明。
- 50组原DiT CPU masters与R129引用以及前后resident参数/buffers一致；encoder/VAE原源码、dtype
  与保护保留，但没有把这些事实扩写成每个encoder/VAE参数都独立做完整前后字节比较。

## 3. 原始输出比较与独立核验

[主质量数据](quality_comparison136.json)在模型及私有worker退出后，CPU读取checksum核验的
原始数组与own-live文件，比较全部逻辑词/像素。没有对输出增益、移位对齐、重采样、归一化或clipping。

[第二实现核验](independent_quality_audit136.json)独立使用整数组FP64 NumPy norm、整数RGB
 delta/int64平方和与pixel mask，交叉核验四类float的完整word计数、relL2与maxabs，以及视频
 changed-channel/pixel计数、整体PSNR、MAE和最大差。音频Pearson与逐帧124项PSNR未由该第二
 实现逐项对主报告复算；SNR主指标可由已核relL2导出，但不称所有指标均有独立raw复算。
 统计归约roundoff一致性门只用于核验两种CPU统计，不是给模型误差的容差。没有再次运行模型。

MP4是CRF18/YUV420/AAC有损预览，**下列PSNR来自native UInt8 RGB数组，不是编码后视频**；
音频来自native FP32数组，不是AAC。raw NPY、latent、own-live encoder hidden都保留host，未传client。

### 最终latent及encoder特征

分母是表列每pair的reference，百分比与比例相差100倍：

| actual − reference | own-live encoder relRMS | video latent relRMS | audio latent relRMS |
|---|---:|---:|---:|
| 候选True − 原False | **2.530606%** | **0.184615%** | **0.042183%** |
| 恢复False − 原False | **3.696754%** | **0.027555%** | **0.020985%** |
| 候选True − 恢复False | **4.915210%** | **0.175480%** | **0.031750%** |

比较范围明确：encoder `[2043,5120]`为Half，共10,460,160词；最终video latent
`[1,24,37,48,84]`为FP32，共3,580,416词；保存的audio decoder latent实际shape
`[2,32,207]`为FP32，共13,248词，不误写成DiT预测布局。
这不是R135单step预测 `[39312,96]` 的3,773,952词，也不是R134小矩阵词数。

候选−原的video latent maxabs **0.51399046**、audio latent maxabs **0.00926912**；
恢复−原为 **0.06417537／0.00258452**。完整float词数／误差在两份数据中保存；float首批差异
坐标来自主指标，第二实现额外保存的是视频最大差坐标与worstframes，不假称float坐标独立复算。

**原False与恢复False的实时encoder输出本身不逐位同，relRMS达3.696754%**，maxabs1472。
权重或stream状态恢复成功不等于encoder特征重复。这个对照不能当候选误差许可，也不能把
组合对照最终质量全部归因最终AdaLN；encoder输入重复性仍需要独立固定输入诊断。

### Native解码视频，不只看整体PSNR

共124帧、**127,991,808个RGB像素／383,975,424个通道值**的完整原始数组比较；
准确分母按shape计算并存于[主质量数据](quality_comparison136.json)。

| actual − reference | native RGB整体PSNR | MAE（0–255） | 最大单通道差 | 改变RGB像素数 |
|---|---:|---:|---:|---:|
| 候选True − 原False | **55.957914dB** | **0.01039425** | **173** | **1,938,356** |
| 恢复False − 原False | **70.058231dB** | **0.00477375** | **21** | **1,584,389** |
| 候选True − 恢复False | **56.283526dB** | **0.01042064** | **174** | **1,968,309** |

候选−原最大差坐标 `[frame85,y624,x904,channel1]`，原值26、候选199。最坏帧是
**第85帧43.747848dB**，随后第86帧43.873362dB；不能只报55.96dB整体值就称画质无损。
恢复−原最坏帧77为63.034369dB，max21发生在 `[77,628,850,1]`。Frame index为0-based。

这些数字是完整native RGB数据比较，不是主观观看、感知质量接受或encoded PSNR自动门。
没有利用原恢复差异把候选差异“扣噪声”后接受。

### Native解码音频：近静音分母下的大相对差

原音频stereo FP32 `[2,165600]`共331,200词，reference RMS仅 **1.15307286e-5**，接近静音：

| actual − reference | audio relRMS | max abs | SNR | Pearson correlation |
|---|---:|---:|---:|---:|
| 候选True − 原False | **105.576768%** | **6.3570449e-5** | **−0.471367dB** | **0.435538** |
| 恢复False − 原False | **104.833356%** | **6.5607019e-5** | **−0.409990dB** | **0.445036** |
| 候选True − 恢复False | **104.222448%** | **6.2342500e-5** | **−0.359225dB** | **0.450293** |

候选−原absolute RMS差 **1.2173771e-5**，恢复−原 **1.2088050e-5**。绝对量很小，但
relative/SNR/correlation确实未重复；不能隐藏这些数字，也不能由“接近静音”自动判音频可接受。
两次原False路径也出现相似量级，只说明目前完整音频对照存在原生重复性混杂，**没有证明是
音频VAE本地同输入计算导致**，也没有证实所有差异都来自encoder或latent。

应在同一固定original latent上重放原decoder、并对固定prompt输入重放encoder以继续隔离，
而不是本轮就采用新容差。此为后续诊断建议，不是已获授权或已执行的新实验。

## 4. 耗时与内存：性能目标仍未达成

Engine初始化 **270.234s**，首请求是冷请求；一次候选请求与恢复请求是warm，不能用
1327s对1023s的冷／热差冒称算子加速。各stage可能嵌套、不同rank取max，不能加总冒充wall。

| 同范围观测 | 原False（cold） | 候选组合True（warm） | 恢复False（warm） |
|---|---:|---:|---:|
| 完整generate wall | **1327.452s** | **1023.054s** | **1018.765s** |
| DiT step0 rank-max | 91.174s | 20.659s | 20.118s |
| DiT其余48step：每step先取rank-max再median | **20.045s** | **20.111s** | **20.034s** |
| pipeline.encode_prompt rank-max | 62.312s | **3.172s** | **3.144s** |
| pipeline.diffuse rank-max | 1077.685s | 990.361s | 985.720s |
| video VAE decode_latent rank-max | 72.401s | **21.557s** | **21.524s** |
| audio VAE decode_latent rank-max | 0.450s | 0.212s | 0.226s |
| pipeline.decode rank-max | 135.499s | 23.420s | 23.315s |
| PyTorch peak allocated rank-max | 18.342815GiB | 18.341779GiB | 18.341893GiB |

本轮step时间包含原模型、range检查、预测D2H审计和FA/Candidate观测，属nonclean；不是纯kernel。
原stage时间按原wrapper范围记录，权重加载、tile同步和I/O是否包含须按对应范围，不偷换成
纯encoder或videoVAE kernel。raw数组保存、MP4/WAV导出与验证在generate wall之外。

每请求prefetch方法观测20,232次，属于method call数，不是copy-engine kernellaunch/真实搬运量
证明。TP/USP与原shared-copy保留，沒有新的独立collective/copy优化命中证据。
PyTorch allocator峰值是整pipeline范围，不可与R135的DiT-only 7.135GiB混称变差或改善，
也不是总HBM/CANN workspace完备峰值。

**Encoder<2s、videoVAE<20s、DiT<15s/step尚未达到；A2/A3实际同范围比较本轮未做。**

## 5. 状态、源码、资源与下一门

[精确执行源码](executed_sources136a/generate_quality136.py)、[source pins](launch_source_pins136a.json)、
[原始process](executed_quality136a/engine_quality136a_process.json)与输入/权重引用核验一致。
11项portablecontroltests、纯Mock、27项CPU指标unit通过；这是控制/指标测试，不是模型执行证明。
Metrics作者单元测试不冒充独立审查；lead用第二种CPU算法核了上述float词／误差、整体视频
指标及最大差，覆盖范围明确，不把尚未独立raw复算的Pearson等指标统称“全部质量指标”。

[R136清理](cleanup136.json)确认模型driver exit0、所有privateworker退出、8rank原方法和
False/warnFalse恢复，private authkey在绝对路径核验后移除；原DiTmasters/resident和source pins不变。
owned ts217租约已释放。CPU后处理已结束，没有新SDK、driver/device reset或sharedprocess信号。
权重、输入、activation和raw媒体都留host，仅source／JSON／text收集到client。

本轮质量对照授权已完成，候选**未部署／未采用／没有新数值阈值**，#15仍open/#4Blocked。
建议下一步先定位原生encoder与固定latent decoder重复性，保持原数学不变，区分输入特征漂移与
解码器自身漂移；是否做、何种范围或是否拒绝候选由用户决定。不能凭本轮的高整体PSNR，
近静音音频或单step稳定证据自动采用组合数值路径或开启copy候选。
