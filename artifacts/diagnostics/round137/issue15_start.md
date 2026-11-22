## R137原数学固定输入／固定latent重复性定位合同

用户直接批准：“继续 下一步保持原数学不变，分别做固定输入encoder、固定latent音频decoder的重复性定位”。
起点b271c75，#15保持open/owner lvyufeng、无blocked_by；由Awaiting approval回InProgress，#4Blocked。
owned ts217新lease至次日01:20+08，extra4090lease立即release。原onhost权重复用，client无权重／activation下载。
已有goal恢复API仍拒绝pausedgoal，继续直接授权工作，不新建goal绕过。

- 全程原defaultFalse/warnFalse/HF32off，原FP16 encoderTP8/all8、原F32audioVAE；不启任何R134/135候选、
  不新deterministic策略、不参数/激活cache/clip/BF16/teacher/少步，没有新的DiT/49sample或VideoVAEforward。
- 原source/context构造相同50DiT与F32保护，但component-only安装不复用旧request/49保存wrapper；
  DiT.forward若意外调用直接拒绝。155checks是原模型合同，不冒称这轮组件做了155/step。
- Encoder只用原rank0 presentation/processor准备一次，然后all8调用原pipeline.encode_prompt。
  在真正原encoder._encode前临时捕获1D Long ids与完整vision kwargs，冻结同shape/dtype/stride/可用format的
  NPU clone，原结果不替换。保存originalinputarchive仅host，恢复原instance方法后重复原pipeline._encode_text_hidden，
  保留encode_ids原Inference/cudnn SDP/CPU输出/staging语义；不直接用旧hidden作输入或绕过原entry。
- Audio使用R136 r01_original_false的已核wholefileSHA final_latents：原audio_latent F32[2,32,207]，
  固定原NPU clone，调用原audio_vae.decode_latent而不是跳过F32反归一化去remote.decode。
  原component-on-device/half_nan_scope语义保留（不是Half autocast），输出F32[1,2,165600]；原BigVGANclamp保留无新增clip。
- 每component每rank native8→trace2→observer-restored4，计划112encoder固定重放+8准备call／112audio固定重放。
  各次全部16/32位词计数与FP64误差、input完整指纹/format、参数/buffer/immutableCPUmaster、TorchCPU/NPU RNG可见状态核。
  本轮math始终原生，observer移除不代表numericcandidate切换；跨rank/first参考和每phase重复门分开。
- trace仅同输入两次高层/叶边界SHA，立即完整递归原input/output值，不持detachalias：BigVGAN首resblock返回会被后续xs+=改写。
  NativeAudio采用**parametrizations.weight_norm**非legacy；绝不另读动态module.weight属性以免多算norm，
  真derivedweights只在原ParametrizationList/weightNormchild正常forward POST观察，不加parametrize.cached。
  原参数g/v不变≠computedweight不变；SHA边界也不等于CANN内部kernel/physicalpadding证明。
- 旁路trace callback hash失败先标invalid/returnNone，待完整componentcall后end_call报错/cohortfail；不单rank先抛阻peercollective。
  模块native异常仍立即privateatomicfatal，全独立hook/method清理尝试，unknown恢复不能报成功；nonclean同步trace可能消噪。
- 原参数store/CPUmasters完整SHA（NPU_placeholder不当shard），all8真实encoder词表shard/rank/组/设备核；来源／clone／byte未改变方可解释首次变化。
  native/untraced及trace/restore次数、cold/warm时间/rank-max/allocator范围分别报，不做新的quality或losslessadoption。
- newprivate0700root/session/auth0600、90min总deadline+pidfd UID/env/starttick、parentexit childrenreconcile，不device reset/SDK/sharedsignals/CI/dormant。
- source/Mock/readonlyprelaunchreview/commit/pin后才执行；失败如实留存不未知replay。完毕恢复所有observer/原方法/flags、IPC/process/lease，
  记录#15/#4/#1和Project。任何数值修复或新容差仍须用户按数据决定。
