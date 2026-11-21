<!-- h3-910a:issue4:R128-start -->

## R128开始：worker-local独立copy stream，不全局关闭安全shim

依赖#3已关闭并验证同worker三请求。确认GitHub master已推送至7a73205；
Project #4为Todo、未分配。现由lead／lvyufeng认领并转In Progress。

### 当前事实与首批工作

- 正式worker的所有layerwise copy handle都等于运行线程的current stream；
  原shim的shared默认是线程/allocator安全措施，不是异步预取重叠。
- 同卡TP2全weights30.89GiB，不盲目全模型驻留。原pinned masters/native flattened dtype copy已存在。
- R127原数学warm whole-forward约19.90s，video/audio全warm stage约23.17s；
  R12617.556s单测不能代替formalwholehook。原始FP32例外和155strict checks都保留。

### 分级门禁

1. **小输入线程安全门**：在实际运行线程上创建、注册并持有私有torch.npu.Stream；
   验证同线程default/current、allocatorwarmup、H2D、eventwait、消费、错误/线程切换/close。
   无globalplatform.Stream覆盖；任何不支持的能力fail closed，不能swallow error当warmup成功。
2. **native hook窄适配**：在已排空边界给当前worker实例的native hook绑定私有流，
   ready与consumer-complete明确。CPUmasters/shape/stride/format/存储/参数dtype/版本不改；
   nativecopy与offload顺序保留，prepared/slot事件不因流切换失效。
3. **真实模型A/B**：同输入baseline→candidate→restored，保留TP2+USP4／encoderTP8、50主块、
   155strictchecks。实际stream handle／线程ID、全部104wide-GEMM叶子/完整输出逐位比、
   原始masters完整byteproof、serialoverlap0与asyncprealloc events。非位同关闭候选并问用户。
4. **正式engine gate**：只有上面合格，才将私有候选接入formalworker，测warmwhole-step和
   cold/reloadscope、HCCL争用、峰值显存与恢复状态。收益不清楚Reject，不继续叠加新候选。
5. NUMA/CPU affinity为独立后续子试验，先观测再A/B，不与copy流同时改以避免混淆归因。

### 不改变的边界

原MiniMax-H3/R124+FA，original50×49语义，HF32/TF32关闭；
原FP32 front/time/final/VAE/norm/residual、Half范围检查和二次幂保护保持。
不启用R126operandcache/device-range/geometry/pool或BF16/clip/student/LoRA/近似cache。
只复用已核验on-hostweights、不下载/中转本地；仅owned liveclaim资源、不借CI/dormant。
sharedSDK/driver不改。nativeCPU Half初始化−11在#14仍未解决，不盲目full49循环重跑。

### 预算与退出

首批small synthetic≤10分钟，模型固定input replay/完整formal证据总预算≤2小时后review；
每次launch前追加actualsource/weights/inputSHA、physicalcards、PID与资源lease。
同进程原非位同native repeat不作为误差许可，任何可测候选差异先restore。
结束回写Keep／Reject／Awaiting approval、失败回执、commit/PR并清理privateIPC/进程与lease。
该Issue不因为小unit通过而关闭；总体<2s/<20s/<15s和A2/A3目标仍开放。
