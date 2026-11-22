## R135全50候选仅诊断合同

R134局部候选稳定，但postgather最多改6484/32256Half词／abs9.765625e-4／relRMS0.01595574%，
明确非无损。用户随后选择“只做全50层候选诊断(Recommended)”，没有授权采用或容差／49生成/copy。
#15 live InProgress、executor lvyufeng/lead，无blocked_by；#4保持Blocked。
owned ts217新lease至16:46+08，extraGPU立即release，不借CI/dormant。

- 复用原singleformal8workers、TP2+USP4、原50blocks/155Halfchecks、原FP32例外/原FA/nativeoffloadcopy。
  input为R128 own-live NPU step25，fileSHA／原R129CPUmasterreferences与前后residentparams/buffers核验。
- 三case各3full50 forward：原finalAdaLN→FP32out/bias候选→恢复原finalAdaLN，
  全case同一privateTrue诊断flag；结束恢复False/warnFalse。总9×155×8计划11160检查。
- adapter只unique model.final_layer.adaln_proj.linear，根据actuallayeridentity路由；
  sharedmethod其他层必须strictdelegate原boundapply/args/return。退出完全恢复instanceoverride/descriptor。
- 候选与R134同source：同Halfx/w/bias，GEMM显式FP32out、bias.float精确扩展add、一次Half返回。
  不重绑参数、不改其他层／timeguards／norm/residual，不clip/BF16/approximatecache。
  每fullcandidate pass实际应只调用一次，共3/rank/24，这只是Python路径计数不是CANNlaunchproof。
- 比较samecasefirst、originalfirst、frozenoriginalwarm-last、candidatefirst；mode参考独立，
  candidate self稳定与candidate-v-original数值改变不混。物理method/False恢复与bit门分开。
- 输出是一次原50层forward预测tensor，非49step decoded媒体，不运行encoder/VAE/采样或copycandidate。
  timeobserver同streamtinyclone，不参与模型数学，FA仅原return-preserving计数；SHA/D2H/检查计时nonclean。
- 每passactual155/50FAcalls/0error、actualTP/SP/head分片、inputallocation/value/layout核；
  master/state before/after和原referenceSHA保持。任何runtimeunsupported/非finite/source不匹配停止全cohort。
- newprivate root/session/0600authkey、pidfd+UID/env/starttick、atomicfatal＋90min总deadline、
  parentexit/RPCtimeout/nonzero兜底detachedchildren；不device reset/SDK修改或sharedsignals。
- 先source/mock/只读review/commit/pin，后真实数据再报告；稳定或数值小不自动adopt。
  清理process/IPC、False恢复、lease释放、commit回写#15/#4/#1。任何更广验证要新用户决定。
