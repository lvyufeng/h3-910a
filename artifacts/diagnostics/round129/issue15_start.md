<!-- h3-910a:issue15:R129-start -->

## R129开始：先解释原native repeat差异，copy候选仍关闭

R128结果已在#4/#1和commit51db4b4中记录。#15初始Todo／无assignee，
由lead／lvyufeng认领为In Progress；#4保持Blocked，精度门不放宽。

### 同一测试基础

- 用R128保存在host的8份ownlive NPU pipeline step25输入（不是GPU/teacher features）。
- 原MiniMax-H3/FL2VA revision0919fb684102ef115126612f3fbb84d1027e7317，
  originalR124 source＋acceptedFA；DiT TP2+USP4，encoderTP8配置仍保留。
- 原50块forward，所有155strict Halfcheck/次、原FP32 front/time/final/VAE/norm/residual；
  不clip/BF16/HF32/LoRA/student/少步/新cache，不换数学，候选copy/pool/NUMA不启用。
- 标尺：R128 unchanged native第二次video/audio relRMS0.0261953%/0.0411016%；
  原CPUmasterSHA不变，但输入是否改写、非参数状态与首次变化算子仍Unknown。

### 分级诊断，不为取日志重跑完整49请求

1. 先只读检查fixture/sourceSHA、结构/dtype/stride和实际worker RPC入口；
   用私有method在原formal scheduler线程执行，不能把parent／toy线程当正式证据。
2. 无额外module-sync的native重复控制：同一input在每次forward前后做value和metadata指纹；
   CPUmaster／参数/非参数buffer-state前后核对，原输出都保留，不回送sampler。
3. 单独noncleantrace轮：模块顺序和输出指纹定位首次不同；分开区分SHA指纹与逐word比较，
   禁止称只测root或hash就覆盖全部叶子无损。
4. 原wide GEMM、FA本地kernel、FP32norm/residual、TP/USPcollective分区重复，同实际input/weight
   执行原算法并返回原结果；记录首次不同操作及差异，不自动改kernel／collective／stream。
5. 如果诊断同步会隐藏race，报告与unsynced控制的差别，不能把instrumented time当clean加速。

### 资源／失败／退出

仅owned liveclaim ts217（extraGPU立刻release），weights/fixture只留host、不client下载/中转。
sharedSDK/driver/CI/dormant不碰。已知nativeCPU init−11在#14保留；失败回执先保存再修，不盲重试。
setup/trace failure要cohort一致，继续沿用带外状态＋private process abort，不让其他rank独自进入HCCL。
预算source/preflight≤15分钟、模型诊断≤90分钟后review，打印phase/thread/actualgroups/source/inputSHA。
找到rootcause后才提出最小修复；数值改变写出delta与用户决定，绝不把native漂移变成默认容差。
结尾报告未知项/Keep/Reject/decision、commit/PR和Project状态，清理privateIPC/进程与ownedlease。
总体目标和#4尚未完成。
