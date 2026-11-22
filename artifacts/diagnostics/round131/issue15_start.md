<!-- h3-910a:issue15:R131-start -->

## R131启动合同：用户只批准原50层同输入诊断，不采用新policy

R130提交7b32103后已向用户报数：时间guard privateTrue输出maxabs5.9604645e-8／
relRMS4.3729399e-8、2084/8064 FP32words不同；小组件稳定不等于可采用。
用户明确选择“只做全50层诊断 (Recommended)”，不是采用deterministic基线或新容差。

#15 live In Progress／executor lvyufeng，无blocked_by；#4仍Blocked。
本轮owned liveclaim只有ts217（其他GPU未使用，无dormant/CIfallback），租约至14:27+08。
原nativeinputs/weights均留host，仅源码与JSON/text回client。

### 原graph与严格保留

- 同一formalOmni/8真实DiffusionWorker线程，DiTTP2+USP4，actual28localheads；
  encoderTP8配置保留但encoder本轮不运行。原FP16matrices／FP32front/time/final/residual/norm，
  每次155strict Halfchecks、原acceptedFA50launch、原sharedcopy/nativeoffload，50blocks完整。
- 复用R128保存的ownlive step25输入，sourceSHA每rank重新核；不使用teacherfeatures，
  不采样/49生成/decodedVAE媒体，不新cache/NUMA/copy/precision参数重写。
- **本轮“最终video/audio输出”指原50层forward返回的预测tensor，非49步decoded视频。**
- NativeFalse3次→privateprocess-wideTrue3次→restoredFalse3次，三组同一NPUinputallocation。
  9×155×8＝11160额外strictchecks；不是49stepperformance/qualitypass。
- 每组自身first、globalnativefirst/nativewarm-last、restored数值独立比较全FP32words/maxabs/relRMS。
  input前后与CPUmasters/residentweights+buffers不变量记录，不把“False恢复”与“输出bit一致”混字段。

### 诊断干扰、预算与失败处理

一处timeoutput同currentstream独立clone（不参加数学/不提前D2H），fullgraph结束后读；
FA仅actualreturn-preservingcounter。计时是fullforward＋原检查＋outputD2H，含tinyclone，
不是purekernel或cleanproduction latency；逐repeat真实max_rank，再median按scope单列。
当前峰值memory也记录，无SDK／device reset／共享进程修改。

源/fixture/preflightpins先核；13portabletests和只读review后才launch新root/session，
privateIPC0600／UID+envmarker+PIDstarttick，TCPStorefatal/监控同时处理detachedchildren。
严格unsupported算子/FA异常或输入改写会停止全privatecohort，不拿fallback/失败轮当通过。
数值falsegate是已测结果，不抛异常隐藏；时间/最终误差先报用户，候选继续off。
预算≤90分钟、RPC≤2400s。结束恢复False/warnFalse、清理IPC/进程并releaselease、提交回写#1/#4。
