## R136完整49质量结果：保留测量，不自动采用；下一项待用户决定

原False→globalTrue+最终AdaLN→恢复False，3完整请求／同8worker，各49actualeval/50sigma/原50层。
实际**182280strictHalfchecks／58800DiTFA launches／0wrappererrors／392candidatePythoncalls**，
原TP2USP4＋liveencoderTP8实际Halfvocabshard `[18992,5120]` 各rank0..7。候选first2time rows＋其余48x3已真执行。

这只是globalTrue+AdaLN**组合**测试，一次候选完整请求不叫full49重复性证明，encoder/VAE也受flag影响。
所有quality来自nativeRGB/F32audio和own-livelatent/encoder，raw文件留host；MP4 CRF18/YUV420/AAC只是有损预览。
CPU主FP64chunk指标＋root独立F64norm/整数RGB第二实现的float全词/relL2/max、video整体PSNR/MAE/pixelcounts/max一致；
未把Pearson与124逐帧PSNR声称都有独立raw复算，统计roundoff一致性门不当模型容差。

| actual−reference | ownencoder relRMS | video latent relRMS | audio latent relRMS | rawRGB PSNR | maxchannel |
|---|---:|---:|---:|---:|---:|
| 候True−原False | 2.530606% | 0.184615% | 0.042183% | 55.957914dB | 173/255 |
| 恢False−原False | 3.696754% | 0.027555% | 0.020985% | 70.058231dB | 21/255 |
| 候True−恢False | 4.915210% | 0.175480% | 0.031750% | 56.283526dB | 174/255 |

候对原最坏frame85为43.747848dB，max坐标[85,624,904,1]原26→候199；不能只报global55.96。
原恢复own-liveencoderHalf也变化3.696754%/max1472，所以不是固定encoderconditioning的单AdaLN因果质量验证。
原恢复noise不能扣掉后当允许误差，也没有量化/skip/teacherfeatures。

Rawaudio候−原relRMS105.576768%／SNR−0.471367dB／corr0.435538／absmax6.357e-5，
恢−原104.833356%／−0.409990dB／corr0.445036／absmax6.561e-5。
referenceaudioRMS仅1.15307e-5接近静音，相对值很大但不能自动以静音允许；两原Falsecontrols相似级别，
存在原生重复性混杂，尚不能指认同输入audioVAE或encoder为根因。

Generate wall cold1327.452s／warmcandidate1023.054s／warmrestored1018.765s；init270.234s独立。
每DiTstep先取rankmax再median余48步：20.045/20.111/20.034s；编码warm3.172/3.144s，
videoVAE decode_latent21.557/21.524s。nestedstage不可加总，也不是纯kernel；原range/predD2H与观测nonclean。
整pipelinePyTorchpeak18.342815/18.341779/18.341893GiB，不和R135DiT-only7.135GiB混比或当总HBM。
prefetch方法calls20232/request不是copyenginekernellaunch，没有newcopy或A2/A3实测比较。
Encoder2s、videoVAE20s、DiT15s目标均未达成本范围，整体路线继续开放。

所有原DiTCPUmasters/resident/sourcepins保持、8rankFalse/warnFalse/method/FA恢复；modelexit0/privatechildren退出，
privatekey绝对路径验证后移除，CPU后处理完、ownedlease已release。weights/activation/rawmedia未下载到client。
**Keep质量证据／Reject自动采用和无损说法／Awaiting approval** 下一fixedprompt原encoder、fixednative latent原decoder
重复性定位(原数学不改)，或用户拒绝候选数值路径。#4copy继续Blocked，候选已撤销/无newbaseline/tolerance。
