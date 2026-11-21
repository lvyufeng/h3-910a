# GitHub执行依据：MiniMax-H3／910A下一轮优化

本目录保存创建 GitHub Issues 的可复核正文。GitHub Project／Issues 是执行状态
的事实来源；此处不是第二套独立任务状态。创建回执将保存具体 URL、ID 与字段。

## 已知基线与用户约束

- 已提交R126证据：`4384e89`；共享已接受精度仍是R124＋FA。
- 原MiniMax-H3，50个主块，49次实际DiT前向、50个sigma边界；不切换student／LoRA。
- **TP始终开启**：当前DiT TP2＋USP4，live FP16 encoder TP8，8卡共同工作。
- 原FP32 patch／time／final／VAE参数与FP32残差、norm、wide GEMM输出保留。
- 155个Half范围检查／步、HF32／TF32关闭；不得BF16回退、clip或fake norm。
- 新候选数值变化先停用、记录并问用户；native重复误差不是自动接受阈值。
- 910A为两组4卡HCCS full mesh：0–3与4–7；跨组对应卡为PHB，其余为SYS CPU路径。
- 权重只复用经核验的主机缓存，新增权重去用户指定4090机器下载；不下载或中转本地。
- 只使用owned、实际claim的机器；不默认占用dormant／CI；禁止驱动重置、共享SDK修改。
- 新实验只写私有路径，结束后释放lease，清理私有进程／IPC，不删模型或他人数据。

## R126实测，禁止混淆计时口径

- 实际生成：1314.581s，初始化286.238s；此前FA生成1329.961s。
- 完整8rank×49原50块前向，60760 strict Half checks通过，124帧／24fps原生音视频；
  精度许可仍未自动改变，R126是执行证据而非新基线。
- 同一whole-forward hook：19.732→19.372s／步；并非17.556s的isolated replay。
- 完整冷VAE阶段142.143s；独立热态video20.565s、audio0.210s，加载另计。
- FA实际kernel约7.454s，USP前后约1.077s，包含packing／等待，不是纯HCCL。
- TP2／4／8实际参数30.890／15.446／7.723GiB／rank；常驻TP4／8反而更慢。
- 新安全判定遍历65536种Half位型，保持检查数和严格stop位置；isolated18.059→17.556s。
- VAE Half operand副本不改原FP32 parameters：1152 Linear叶子和196完整prequant tile pairs逐位相同。
  只省约0.20s，多4.502GiB，不能说已<20s或native融合命中。
- 当前geometry cache2.340→2.339s，无实用收益。
- encoder组内TP4重复／跨岛逐位一致，但vs fresh TP8 hidden relRMS2.783%，不晋升。
- 全生成video encoded PSNR最低51.519dB；native audio relRMS差1.057、绝对RMS差1.2195e-5。
  不宣称独立轨迹完整精度等价，低音量不是忽略音频差异的理由。

## 下一轮任务排序

1. P0：正式服务cold／warm／reload状态与执行线程／stream／NUMA事实。
2. P0：真正的worker-local异步H2D与生命周期、精确fence和full-engine证据。
3. P1：实际形状FA内核IO／tiling／30核流水线；真实GEMM布局预排布与融合。
4. P1：VAE同形状tile microbatch与时间chunk调度，窗口／overlap／blend语义不变。
5. P2：小区域编译／NPUGraph能力验证，不能全局盲开。
6. P2：TP2×组内Ulysses2×跨岛Ring2研究；先910A partial-output／LSE能力与精度门。
7. 决策门：8卡encoder可复现性、任何数值变化、阶段目标范围；待用户决定才推进。
8. 最终集成：仅有证据合格且精度许可的项，8rank×49真前向与全部原生媒体回归。

## 主流方案的实际适用边界

- FlashAttention是IO-aware exact attention设计；不同实现仍需核对舍入，FA3／4不能直接装910A。
- Diffusers低精度／compile／QKV融合路线：DiT已有packed QKV和gate/up，不重复算收益。
- xDiT USP：更多rank不必更快；Ring2需要当前910A FA的partial／LSE适配，不是改degree即可。
- 普通PP单视频可能有bubble；PipeFusion复用previous-step KV且官方当前不支持Ascend，默认不采用。
- DistVAE常规row-adapter不是H3原生ViT tile路径；改tile窗口／overlap可能改变输出。
- TeaCache／PAB／TaylorSeer／Cache-DiT、稀疏注意力、INT8／FP8、减少步数、FastH3蒸馏：
  均进入单独的“等待精度许可”分支，不能当原49步无损优化；当前不执行。
- LLM自回归KV缓存不能直接套到随latent／timestep变化的DiT KV。
- 官方资料：
  - https://github.com/xdit-project/xDiT/blob/main/docs/methods/usp.md
  - https://github.com/xdit-project/xDiT/blob/main/docs/methods/pipefusion.md
  - https://github.com/xdit-project/xDiT/blob/main/docs/methods/parallel_vae.md
  - https://github.com/huggingface/diffusers/blob/main/docs/source/en/optimization/fp16.md
  - https://github.com/huggingface/diffusers/blob/main/docs/source/en/optimization/cache.md
  - https://github.com/Dao-AILab/flash-attention
  - https://github.com/Ascend/pytorch
  - https://haoailab.com/FastVideo/cookbook/minimax-h3/

## 每个Issue的执行合同

- 开始前读取主Issue、当前Issue、依赖、最新评论与Project字段；从GitHub恢复上下文。
- Ready／Todo且依赖满足才claim、指定实际执行人，并改为In Progress。
- 测试前评论：hypothesis、源码／权重SHA、设备／TP组、输入、预算、干净计时与门禁。
- 测试中评论实际日志／receipt进度；遇到transport不确定性先回查，不盲目重建／重跑。
- 结束评论全部结果：基线／候选／恢复基线，rank-max与冷暖口径、内存、命中、精度及失败。
- 引用关联commit／PR；证据通过且依赖条件满足才能Done。无收益也能完成实验但必须写“Reject”。
- 可测数值变化转Awaiting approval，不把其依赖项自动推进；用户决策写回Issue。
- 临时无法继续转Blocked并写具体条件／恢复步骤；不能只留一句“优化失败”。
- 主Issue保持子Issue清单和最新结论；Project不取代正文精度合同。
- 每次会话结束回写状态、结果、下一步，避免仅在聊天或本地todo中保留任务。
