<!-- h3-910a:issue15:R132-start -->

## R132开始：音频残留首变化诊断，不接受新baseline/容差

R131已记录：True诊断下Video/Time位同，Audio仍6.0796738e-6／relRMS0.0000343518%，
1927/13248词变化。用户随后“继续吧”，本轮只沿此音频first-change任务，不采用True策略。
#15 live In Progress／executor lvyufeng，无blocked_by；#4继续Blocked。
owned liveclaim只有ts217用于NPU，extra4090立即release，租约至15:56+08；不借CI/dormant。

### 原代码范围

原FinalLayer先norm/AdaLN、将h转换F32，然后video_out/audio_out都对FULL41792行计算，
最后才select414音频行。TP2的audiohead local输出为41792×16，gather后41792×32。
本轮不把model计算改成414行、不改F32参数/输入，不调换GEMM/BIAS，不用新FP16近似替代。
原50block/155checks、TP2+USP4、原FA、FP32残差与原sharedcopy均保留。
权重及own-live NPU step25 input SHA每rank重核，只返回源码JSON/text，不下载activation/client。

### 分级诊断与配对

同一formal8worker：False参考1次→True控制3次→True tailtrace2次→原localaudio apply
固定FULLactual输入10次（不调用Column.forward/TPgather、不额外155checks）→撤销tail后True2次
→恢复False1次。原full50次数9×8，计划11160strictchecks；localmethodcalls另列80。

Tail沿sp_gather、time、finalnorm/AdaLN/final_layer及两head localapply-beforeTP/postTPgather，
记录FULL logical SHA与audio414-row SHA分开。video选择输出稳定不代表全部hidden稳定。
Audio local实际输入/原weight/bias、beforeTP完整/选audio输出，以及postTP完整/选audio输出
做完整FP32word配对；其余SHA只称fingerprint。原函数只调用一次，原return保持。
冻结FULLx入口即刻独立CPUref/firstNPUclone，不能保留事后可能被改写的alias；clone不参加原model数学。

大输入898,695,168B≈0.837GiB/rank，最多1CPUref+1NPUfrozen；CPU指標分262144word块，
不把整矩阵转double制造巨大临时。MemAvailable/devicefree未知或不足failclosed，不减行/禁TP。
固定x/bias clone对原CPUref做全范围word验证，实际layout与clone分别记录；physicalformat未测明确Unknown。
原dispatch actualcallable/sourceSHA记录，不从method类名猜CANNkernel；没有改新的数学实现。

### 失败、恢复、预算

return-preserving observers为private instance hooks/wrappers，构造异常事务回滚，shared quantmethod
按实际layer身份路由不伤其他层。SHA/D2H会改变调度，属于nonclean；control/撤销trace后控制保留。
计时scope注明observations，False单次不声称自身重复性。任何数值变化都报告，不把小残留当容差。

新exact root/session/0600authkey，继承R131 pidfd+UID/marker/starttick、parentgone/nonzero兜底与
总90mincompute deadline/有界cleanup，未知退出不claim成功，不reset设备/改SDK/杀共享任务。
源码/CPU单位测试/只读review通过并pin、commit后才launch；真实原full50数据而非tests证明。
结束撤销所有观测、恢复False/warnFalse、核原CPUmasters/residentvalues、清理IPC并releaselease，
提交数字与未决条件写回#15/#4/#1。无49步生成／encoder／VAE／decoded媒体质量声明。
