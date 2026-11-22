# R135：全50层候选重复稳定，但数值有变化；未采用

**Keep 全50层受控诊断证据；Reject“逐位无损替换”的说法；Awaiting approval 下一范围。**
[Issue #15](https://github.com/lvyufeng/h3-910a/issues/15)仍开放，
[Issue #4](https://github.com/lvyufeng/h3-910a/issues/4)保持Blocked，copy候选未启用。

## 1. 授权与比较对象

[本轮授权](approval135.json)是“只做全50层候选诊断”，不包含采用、允许误差、49步生成或copy优化。
R134局部试验已经披露：候选改变数千个Half词，不是无损实现。

本轮原实现→候选→恢复原实现，各做3次原50层forward，使用同一个own-live step25输入，
**三个阶段都使用同一临时True确定性条件**。因此下面的“原实现”是原最终AdaLN算法在True
条件下的输出，**不是默认False的R124接受参考**。结束后False/warnFalse另行恢复。

没有运行采样器、encoder或VAE解码；输出仅为完整50层的一次视频／音频预测tensor，
不是49步的最终latent、decoded媒体质量或跨输入确定性保证。

## 2. 实际8卡执行与原模型保护

[数值汇总](summary135a.json)、[实际验证](verified135.json)和
[原始rank0回执](executed_diagnostics135a/full50_rank0135.json)保留独立证据，全部8rank均已收集：

- 同8worker PID，9个50层forward/rank：**11,160项实际严格Half检查**，每次155项不减。
- **3,600次实际FA ACL launch，0个wrapper error**；每forward每rank50次。
- 实际TP2分组 `[0,1] [2,3] [4,5] [6,7]`，USP4分组 `[0,2,4,6] [1,3,5,7]`，
  TP local28heads、实际FA BHND输入 `[1,7,41792,128]`。未改TP／原50层／FP32保护和范围检查。
- 原input逻辑SHA/shape/stride及保存的allocation/pointer metadata前后完全一致。
- 50组原CPU masters的值和metadata与R129引用匹配，resident参数／buffer前后不变。
- 只唯一 `final_layer.adaln_proj.linear` 路由到[候选](fp32_bias_candidate134.py)，shared-method
  其他层保留原boundapply、args与return；未重绑任何参数。
- 每候选forward实际命中该Python候选一次，共**3/rank、24次**。返回契约确认矩阵输入／权重Half，
  GEMM返回FP32、bias扩展在FP32相加、最后一次Half回写。该计数／dtype不是内部CANN归约机制证明。

Driver的collective RPC返回只展示nested rank0摘要；实际8rank文件和8个不同worker PID才是
全8执行证据，没有把rank0的reply复制成“8份reply”。

## 3. 候选消除了本次原音频残留，但不是无损

所有计数均针对完整逻辑FP32词，不混用R134小组件的Half词计数：

- 视频预测shape `[39312,96]`，共**3,773,952个FP32词**。
- 音频预测shape `[414,32]`，共**13,248个FP32词**。

| 阶段 | 视频：8rank的3次重复是否全部逐位同 | 音频：8rank的3次重复是否全部逐位同 | 音频selfrepeat最大变化 |
|---|---|---|---|
| 原AdaLN、临时True | 是 | 否 | **15词，maxabs2.384185791e-7** |
| 候选AdaLN、同一True | **是** | **是** | **0词** |
| 恢复原AdaLN、同一True | 是 | 否 | **15词，maxabs2.384185791e-7** |

本轮原阶段只在rank6/7观察到上述音频self差异，恢复阶段只在rank0/1；其他rank在这3次
未观察到差异，不代表原实现每rank永久稳定，也不能写成“本轮原8rank全部漂移”。

原／恢复音频selfrepeat最大relRMS比例为 `3.099846733e-9`，百分比为 `3.099846733e-7%`。
这次15词残留小于某些早期轮次，不能据此发放新容差或宣称原路径对所有输入稳定。

候选对原实现首次输出，以及冻结的原warm-last（repeat2）分别比较：

| 完整预测 | 最大改变FP32词数 | 最大绝对差 | 最大relRMS比例 | relRMS百分比 |
|---|---:|---:|---:|---:|
| 视频 | **3,578,877 / 3,773,952** | **9.776353836e-4** | **7.187914674e-5** | **0.007187915%** |
| 音频 | **13,245 / 13,248** | **7.116198540e-4** | **8.823782567e-5** | **0.008823783%** |

本轮对original-first／original-warm-last所得上述最大值一致，但两个anchor仍单独保留，
候选未污染original-last参考。不同metric的最大值可以来自不同rank，不冒称是同一坐标。

恢复原实现的视频对原参考0词变化；恢复音频仍有15词原生残留。物理恢复成功与输出逐位
复原是不同的门，不能把音频残留当作恢复失败，也不能用它许可候选对原生数百万／数千词的改变。

**结论只限本输入／同一临时True条件：候选重复稳定，输出相对原算法有明确数值变化。**
不等于默认False接受参考的无损替换，也不等于49次评估后的视频／音频质量通过。

## 4. 同范围耗时和内存；没有性能达标结论

初始化单独为**265.292s**。原阶段首次forward为**105.844s**，不能与warm混合。
每个repeat先取8rank最大时间，再对后两次warm repeat取median：

| 阶段 | repeat0 | repeat1 | repeat2 | warm median(per-repeat rank-max) |
|---|---:|---:|---:|---:|
| 原AdaLN、True | 105.844s | 20.206s | 20.196s | **20.201s** |
| 候选AdaLN、True | 20.920s | 20.175s | 20.183s | **20.179s** |
| 恢复原AdaLN、True | 20.196s | 20.179s | 20.175s | **20.177s** |

计时包含原forward、155检查、最终video/audio D2H和time输出的当前stream clone；time的CPU
读回发生在elapsed读取之后，不包含在该列内。它是nonclean诊断，不是纯kernel或49步生成时间。
约0.022s的原／候选warm差不能当作确定加速，恢复时间也在相同范围；**DiT<15s/step仍未达到**。

三阶段rank-max PyTorch peak allocated均为**7.134993553GiB**；不是总NPU设备占用或CANN
workspace的完备峰值。没有新的copy-engine／collective计数或独立pure-kernel性能归因。
只确认保留了原shared-copy stream、原TP/USP与实际FA分片，不能冒称copy优化已验证。

Encoder<2s、videoVAE<20s以及实际A2/A3同范围比较仍开放，本轮未测试它们。

## 5. 恢复、源码与资源

[真实执行源码](executed_sources135a/diagnostic_rpc135.py)、[source pins](launch_source_pins135a.json)、
[driver/process回执](executed_diagnostics135a/engine_candidate135a_process.json)一致。
12项portable tests、pureMock target/shared-method委派和异常descriptor恢复检查通过；它们
不代替实际NPU验证，也不证明内部CANN算法。原模型源码和输入／master引用都保持。

Observer安装事务性持有outer handle；finally分别尝试adapter、flag、timehook、FA恢复，
失败需记录unknown/fatal而不能静默通过。本轮无runtime／cleanup失败；不宣称任意报告IO
失败下绝对保留primary exception。PIDfd、UID/env/starttick、总deadline和privatechildren兜底不变。

[清理](cleanup135.json)确认driver exit0、所有privatechildren退出、8rank原methoddescriptor
与False/warnFalse恢复、私有IPC authkey在绝对路径核验后移除，source pins不变。
owned ts217租约已释放。无SDK／driver/device reset或sharedprocess signals；不借CI/dormant。
所有weights、inputs、activations留host，client仅有源码／JSON／text，没有权重或activation中转。

## 6. 用户的下一项决定

本轮full50onlydiag授权已经完成。保留稳定证据，候选**已撤销、未采用、未放宽误差**。
#15仍InProgress，#4仍Blocked，不能因为本次candidate稳定就关闭原repeatability/数值决策前置。

若用户愿意继续质量验证，需要新授权的原49次实际评估／50 sigma完整输出对照：默认False原
接受路径与临时True＋finalAdaLN候选组合、以及恢复原路径；保持原50层、TP与FP32例外，
测最终latent／decoded视频及音频差异。True＋候选的组合变化必须明确，不能把本轮同True对照
偷偷改称相对默认False的质量证明。Encoded PSNR等指标也不是自动允许误差。

或者拒绝此数值路径，保留原算法并继续内部实现定位。任何选择都不意味着整体性能目标
已完成，不会自动开启copycandidate、少步、量化或其他近似方法。
