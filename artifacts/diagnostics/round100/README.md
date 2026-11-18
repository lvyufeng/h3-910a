# Round100 — 流式注意力真实完整 TP8：最终误差17.04%→8.17%，仍未验收

在同一 owned ts-217、8×原始910A、固定原始 HF **CPU FP16** oracle 上完成四配置 A/B。R99 的单 launch 流式 attention 使用 Float主 scores/O/L、Half仅用于稳定shift估计；精确native shift广播不改变任何已验证视觉输出位级结果。

## 完整 encoder 同轮结果

真实两张图、27 vision blocks、3 路 DeepStack、50 text layers、HCCL、layout/packing 和最终 D2H 全包含。每配置5次同步stock调用，每次取8 rank最大时间，再取中位数。没有CPU预计算conditioning、Float Linear权重教师、BF16 gold cast、假 final norm或事后容差。

| 配置 | selected50 对原始CPU FP16相对RMS | 完整stock encoder中位数 |
|---|---:|---:|
| 原生视觉 + 原生文本 | **7.420123%** | 2.179350 s |
| patch/bias/true-erf修补视觉 + 原生文本 | 17.037422% | **2.166834 s** |
| 同一修补 + 单launch流式视觉attention + 原生文本 | **8.171271%** | **2.406153 s** |
| 同一流式视觉 + 高精度文本attention | 8.207468% | 2.867527 s |

新流式视觉把修补路径最终误差从17.04%降到8.17%，但仍比原生路径7.42%更差，**不是CPU FP16精度对齐**。高精度文本attention在这个组合里略微恶化最终误差且更慢，保留负结果、不升默认。

本轮全部20次8-rank-max样本都≥2s。跨轮最快已测完整encoder仍是 [R98](<../round98/README.md>) 的2.034344s；不能因只看视觉1.582865s就宣布text encode<2s，也不能把不同时段的完整时间当作同轮配对加速。

原始结果：[可重算摘要](<summary100.json>)、[聚合程序](<collect100.py>)、[rank0](<results/full01/tp8_rank0.json>) / [rank1](<results/full01/tp8_rank1.json>) / [rank2](<results/full01/tp8_rank2.json>) / [rank3](<results/full01/tp8_rank3.json>) / [rank4](<results/full01/tp8_rank4.json>) / [rank5](<results/full01/tp8_rank5.json>) / [rank6](<results/full01/tp8_rank6.json>) / [rank7](<results/full01/tp8_rank7.json>)。

## 传播与结构门禁

原始block24相对RMS：原生0.509612%、修补0.253075%、流式/原生文本 **0.243241%**；但block49对应7.42%/17.04%/8.17%。与 [R93原始CPU Half的layer43放大](<../round93/README.md>) 一致：早层更准并不保证最终精度单调改善，不能把两个误差百分比相减当作某个NPU算术的贡献。

- 每rank所有27个vision边界、DeepStack和merger都精确匹配 [R99实测输出身份](<expected_vision99.json>)。
- 8 rank间selected50和全部50个text block边界SHA完全相同；每配置重复、stock/equivalent位级相同，最后原生恢复成功。
- 参数、fixture、原始CPU gold均不变。R91 position/rotary几何缓存始终启用且位级原生，CPU text-RoPE精度缓存和Float row adapter未安装。
- 独立CPU-conditioning-only精度控制仍为原生文本2.475700%、高精度文本1.819336%，**仅精度控制，不属于完整计时**。
- [串行prebuild](<results/full01/prebuild100.json>) 在8-rank priming前验证13个既有对象：4 text attention、7 vision/patch bias、1 true-erf、1 streaming flash。全部launch30可用核，原始物理32核不能被误标为A2。

执行源：[完整TP8驱动](<full_encoder100.py>)、[串行13对象验证](<prebuild100.py>)、[source-only contracts](<test_full100.py>)。有效视觉kernel是 [R99 shift-temp](<../round99/flash/flash_vision_shift_temp_fixed99.py>)，host engine为 [guard修复版](<../round99/flash_engine_guard_fixed99.py>)。

## 最后审计与资源释放

[只读最终审计](<final_source_verification100.json>) 成功核查 **23 production源、13 installed API源、89份执行源映射、23个实际记录对象**、原始HF源、CPU FP16 gold、冻结QKV与launcher全部未变。全部相关worker为零、全部相关background job已收集，无running/pending job；随后已释放本次owned ts-217，未请求CI/dormant，也未动其他会话租约。

[审计程序](<verify_final100.py>) 与 [source映射](<expected_execution_sources99_100.json>) 保留失败和新修复的每个private root对应关系。没有覆盖已执行源或补写gold来隐去失败。所有新subagent为fresh fork；耗尽上下文的原始flash作者留下零文件，未恢复，改由新的fresh fork完成kernel，root才编译/实测。

## 仍未完成的目标

完整CPU FP16精度、text<2s、encoder/DiT/VAE真实共驻留与匹配官方工况比较仍未达成，持久目标继续active。下一步见 [执行约束与计划](<NEXT.md>)：优先探索MB64/BC128减少Float scalar广播成本、原位复用概率Float缓冲，严格屏蔽最后64个phantom keys；它尚未实现或测量。精度继续针对实际输入传播及真实下游conditioning，不新增人为norm、不放宽oracle。
