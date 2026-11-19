# Round102 — BC128 完整TP8：7.405683% CPU FP16 gap，2.128423s；未达验收

同一份原始HF **CPU FP16** oracle、8×原始910A、实际两图、27 vision/3路DeepStack/50 text layers、HCCL/layout/packing及最终D2H。每配置5次真实stock入口，每次取8 rank最大值，再取中位数；没有CPU预计算conditioning替换视觉或计时。

| 配置 | 最终selected50对CPU相对RMS | 完整encoder中位数 |
|---|---:|---:|
| 原生视觉/文本 | 7.420123% | 2.143904s |
| patch/bias/true-erf修补视觉、原生文本 | 17.037422% | **2.026669s** |
| 同修补 + BC128 Float流式视觉、原生文本 | **7.405683%** | **2.128423s** |
| 同BC128 + 高精度文本attention | 8.230224% | 2.691181s |

BC128把修补路径最终误差17.04%降到7.41%，稍低于原生7.42%，但**仍不是CPU FP16精度对齐**。高精度文本attention在这个组合里反而更差且更慢；保留负结果，不升默认。全部20个8-rank-max样本都≥2s，最快样本2.010223s；不能宣布text encode<2s。

对 [R100](<../round100/README.md>) 的BC64流式2.406153s→2.128423s是跨运行比较，控制基线也随时段改变，不能全部归因于内核。实际受控attention收益以 [R101同轮QKV](<../round101/results/leaf01/leaf_compare101.json>) 的8–10%为准。

## 结构与数值证据

- 8 rank间selected50及全部50 text边界SHA完全相同；每配置重复/stock-equivalent位级相同，最后原生恢复成功。
- 全部27 vision边界、3路DeepStack与merger精确匹配 [R101实际输出身份](<expected_vision101.json>)。
- BC128原生文本raw24误差0.240285%、raw49为7.405683%，继续说明早层误差不单调决定最终结果。原始CPU-only的后层放大见 [R93](<../round93/README.md>)，不得把两个误差百分比相减当作某个NPU算术贡献。
- 参数、fixture、原始CPU gold不变；所有Linear权重Half。只有原生位置/rotary几何缓存，没有CPU text-RoPE精度表、Float row adapter或隐藏状态缓存。
- CPU-conditioning-only控制仍是原生文本2.475700%/高精度文本1.819336%，只是精度隔离，不属于上表计时。

[可重算摘要](<summary102.json>)、[聚合器](<collect102.py>)、[rank0](<results/full01/tp8_rank0.json>)/[rank1](<results/full01/tp8_rank1.json>)/[rank2](<results/full01/tp8_rank2.json>)/[rank3](<results/full01/tp8_rank3.json>)/[rank4](<results/full01/tp8_rank4.json>)/[rank5](<results/full01/tp8_rank5.json>)/[rank6](<results/full01/tp8_rank6.json>)/[rank7](<results/full01/tp8_rank7.json>)；执行 [完整驱动](<full_encoder102.py>) 与 [串行13对象prebuild](<prebuild102.py>)，[实际证据测试](<test_actual102.py>) 和 [入口契约](<test_full102.py>) 均通过。

## 实际输出捕获与只读审计

rank0四种配置的真实selected50仅在第一次完整forward后、所有warm计时前保存，从未加载回stock推理；用于以后真实DiT condition_proj/refiner精度控制，不构造新的hidden teacher。4份Half tensor/file SHA均经 [CPU-only审计](<final_source_verification_cpu102.json>) 核验，且已下载本地，校验记录见 [捕获身份](<captured_outputs102.json>)。大型再生tensor不进入Git，校验记录保留。

阶段审计核查23 production源、13 installed API源、33执行源、17实际记录对象及原始HF/CPU gold/frozenQKV/launcher，worker为零。首次audit因未source CANN导致Torch默认autoload寻找libhccl失败；只读重查使用 `TORCH_DEVICE_BACKEND_AUTOLOAD=0`，没有重跑模型、修改源码或gold。失败摘要见 [审计环境失败记录](<audit_failure102.json>)。

全部后续实验结束后的151执行源/24对象审计及owned资源释放见 [R105](<../round105/README.md>)。完整CPU FP16精度、text<2s、真正encoder/DiT/VAE共驻及官方匹配工况仍未完成；[下一步精度计划](<../round101/CONDITIONING_NEXT.md>)不能被误解为已完成下游验收。
