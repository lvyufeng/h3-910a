## R135全50候选诊断完成：Video/Audio都重复稳定，但数值非无损；未采用

同8worker、原50层3原实现→3候选→3恢复，所有case同一temporaryTrue诊断条件，
不是accepted默认False基准。实际11,160strictHalfchecks／3,600FA ACLlaunches／0errors，
原TP2+USP4／FP32保护／50CPUmaster引用保持。input逻辑与allocationmetadata前后0diff。

Candidate视频、音频各8rank×3full50均selfrepeat0词；Original/Restored视频0词，
音频仍15FP32words maxabs2.384185791e-7。Time两阶段及原参考均0diff。
候选对原first／冻结warm-last分别核，数值表两anchor都保留：

| 模态 | 最大改变FP32词 | max abs | relRMS比例 | 百分比 |
|---|---:|---:|---:|---:|
| 视频[39312,96] | 3578877/3773952 | 9.776353836e-4 | 7.187914674e-5 | 0.007187915% |
| 音频[414,32] | 13245/13248 | 7.116198540e-4 | 8.823782567e-5 | 0.008823783% |

不是R134Half词计数，不拿原生15词噪声允许候选数百万／数千词变化。各最大metric可能来自不同rank。
恢复原视频对参考0词；音频仍原生残留，这是恢复math/flag与恢复outputbits的独立门。
**Keep受控稳定证据，Reject“无损替换”说法，Awaiting approval下一完整质量scope**。
候选方法已撤销，无newbaseline／误差阈值／copy采用，#15open/#4Blocked。

原／候选／恢复warm median(per-repeatrankmax)20.201323／20.179221／20.176944s；
原cold105.844233s、engineinit265.292022s分开。全部rankmaxPyTorchpeak7.134993553GiB。
计时是fullforward＋155检查＋video/audioD2H＋tinytimeclone；timeCPUread AFTERelapsed不包括该耗时。
没计纯kernel/copyengine/collective，也不是49step／encodedPSNR／视频音频qualityproof；≈.022s不当确定速度提升。
DiT15s目标、encoder/VAE目标及A2/A3实际同scope比较仍未达成或未测。

候选只finalAdaLNlocal sameHalfx/w/bias→FP32out/bias→Half，actual3/rank/24Pythonhits，
非CANN内部归约机制或kernellaunch证明。nestedRPC控制reply只显示rank0，真实8raw/PIDs为all8执行证据。
原CPUmasters/residentparams/buffers/sourcepins完全未变，descriptor／False/warnFalse恢复，全privateprocessexit0/IPCkey绝对路径核验后移除。
ownedlease已release，无SDK/driverreset/sharedsignal/CI/dormant，weights／inputs／activation从未下载/中转client。

下一若继续，需用户另行授权完整49eval/50sigma质量对照：**原接受False路径→temporaryTrue+候选组合→恢复False**，
保留原50/155/TP/FP32保护，报告最终latent／decodedvideo/audio差异。不能偷把本轮同True诊断
叫相对False接受参考的质量通过。或拒绝该数值路径，继续原算法内部定位；encodedPSNR等也非自动阈值。
