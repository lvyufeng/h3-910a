# Round109 — 原始CPU FP16 raw42/43已固化，与旧oracle逐位相同

这是补存 R78/R93未持久化的原始CPU晚层张量，**不是重新定义参考、不是Float权重教师、不是新的宽松验收**。

在原始官方dense权重确定性转Half、原始HF source、CPU `SDPBackend.MATH` 和原始norm/rotary流程下，从旧R78精确raw24出发重建原始25..49：

- 两次完整pass的全部25个边界SHA都与旧R78逐位相同；raw49继续是旧selected50 tensor SHA。
- 275个参数都原始Half，参数SHA和4个R78 gold文件SHA前后不变。
- 两个普通CPU `torch.Tensor` `Half[1,2043,5120]`保存到新私有root，回读tensorSHA一致。
- 磁盘raw42→原始CPU layer43→磁盘raw43逐位相同，确证输入/输出配对，无DeepStack-after24或final norm注入。

| 边界 | Tensor SHA256 | 最大绝对值 |
|---|---|---:|
| CPU raw42（block43精确输入） | `58af7295b0544fcf8c68994d769a6495402d30cf24c8b884ff3cf3b67847335f` | 15744 |
| CPU raw43（原始输出） | `7600cf45e5230d4aa707d85f719a2a982bede75632a8fd86fb39f6826eecc23e` | 18736 |

File SHA与Tensor SHA是不同东西；[实际完整回执](<results/cpu02/CPU_raw_reference109.json>)逐项记录，不把压缩序列化bytes当tensor身份。两份原始CPU文件的本地备份已完成并SHA核验，身份见 [捕获清单](<captured_outputs109.json>)；大型再生Tensor不进入Git。原始pair用于 [R110](<../round110/README.md>)的同输入NPU43精度控制，不被用作完整encoder warm隐藏conditioning。

## 环境失败和资格边界

首次 [失败回执](<results/cpu01/CPU_raw_failure109.json>) / [run log](<results/cpu01/run.log>) 在任何layer模型build/forward之前失败：HF可选NPU集成显式import torch_npu，即使关闭autoload仍寻找libhccl。失败source/目录保留。新private CPU02使用完全相同3份不可变source，只source CANN环境使可选import成功；所有真实params/inputs/output仍CPU Half，没有NPU计算。CPU任务只在 [R108完整NPU时序](<../round108/README.md>)全部结束/收集后才启动，不污染<2s性能样本。

源码：[fresh-fork捕获程序](<cpu_raw4342/newcapture_cpu_raw109.py>)，复用固定 [原始R93 suffix](<../round93/cpu/suffix93.py>) / [原始R92 helper](<../round92/cpu/original_cpu_block25_92.py>)。16项source契约和 [实际gate测试](<test_actual109.py>)通过；作者没有CPU/Torch/SSH/model执行，root才运行。

最终230执行源/38对象/CPUgold/worker审计与资源清理见 [R110](<../round110/README.md>)。整体完整CPU FP16精度仍未对齐、目标active，不能凭重现CPU自己就宣告NPU成功。
