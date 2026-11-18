# Round98 — 真正完整 TP8 encoder：NZ 注意力减少误差，但未达精度/2s 目标

本轮用 [R97 新 NZ 高精度视觉](<../round97/README.md>) 在 **8×原始 910A** 上跑完整 `encode_ids`：实际两图、27 vision blocks、3 路 DeepStack、50 text layers、原有 TP8 HCCL、布局/packing、最终 D2H 全部包含。全部 Linear 输入/权重/输出为 **Half**；参考为固定的原始 HF CPU FP16，而非 BF16 或 Float 权重教师。

## 四配置同轮实测

每配置 5 次同步驻留调用；每次取 **8 rank 中最慢时间**，再取 5 次中位数。GPU 同步和 HCCL barrier 在 timer 外；timer 内是没有 hooks、没有预计算 conditioning 的完整 stock 入口。CPU text-RoPE 精度缓存与 Float row adapter 均未安装；R91 位级原生的几何常量缓存在所有配置中一致启用。

| 配置 | selected50 对原始 CPU FP16 相对 RMS | 完整 encoder 中位数 |
|---|---:|---:|
| 原生视觉 + 原生文本 | **7.420123%** | 2.173329 s |
| patch/bias/true-erf 修补视觉 + 原生文本 | 17.037422% | **2.034344 s** |
| 同一视觉修补 + NZ 高精度视觉注意力 + 原生文本 | **9.918301%** | 2.507635 s |
| 同一 NZ 视觉 + 高精度文本注意力 | **8.041802%** | 3.062943 s |

新视觉注意力使修补路径的最终误差 **17.04%→9.92%**，再加高精度文本注意力到 **8.04%**。但仍比本轮原生路径的 7.42% 更差，不能称为 CPU FP16 精度对齐。最快配置 2.034 s、最快单次 2.025508 s，**所有 20 次 8-rank-max 样本均 ≥2s**。不提升为生产默认，不改变精度阈值，不宣布完整目标成功。

主要结果：[可重算摘要](<summary98.json>)、[聚合代码](<collect98.py>)、[完整 rank0](<results/full02/tp8_rank0.json>)、[完整 rank7](<results/full02/tp8_rank7.json>)。其它 rank 的原始收据在同一 [结果目录中的 rank1](<results/full02/tp8_rank1.json>) / [rank2](<results/full02/tp8_rank2.json>) / [rank3](<results/full02/tp8_rank3.json>) / [rank4](<results/full02/tp8_rank4.json>) / [rank5](<results/full02/tp8_rank5.json>) / [rank6](<results/full02/tp8_rank6.json>)。

## 精度仍非单调，误差在后层放大

| 配置 | 原始 block24 相对 RMS | 原始 block49/selected50 相对 RMS |
|---|---:|---:|
| 原生 | 0.509612% | 7.420123% |
| 修补视觉、原生注意力 | 0.253075% | 17.037422% |
| 修补 + NZ 视觉、原生文本 | **0.246675%** | **9.918301%** |
| 修补 + NZ 视觉、高精度文本 | 0.247083% | **8.041802%** |

50 层结束没有人为加 final norm；真实结构保留的是 pre-final-norm selected50。不能用假 norm、BF16 gold 转换、Float 线性权重教师或减去两组百分比来宣布误差消除。输入敏感性与 [R93 原始 CPU Half 自身的 layer43 放大](<../round93/README.md>) 一致；本轮仍不把它归结为某个已唯一定位的 NPU primitive。

独立的“CPU conditioning 输给 NPU text-only”控制仍为原生文本 2.475700%、高精度文本 1.819336%，两者重复稳定。这些 **只是精度控制**，没有纳入任何完整 encoder 计时，也不作为替代真实图像计算的验收。

## 结构、重复与源身份门禁

- 每 rank 所有 27 个视觉边界、DeepStack 和最终 merger 都强制匹配 R97 单卡实际收据的精确 SHA；新的几何缓存 + NZ 注意力组合没有放宽门禁。
- 每配置重复与 stock/equivalent 输出位级相同；**8 rank 之间 selected50 及全部 50 个 text block 边界 SHA 也相同**，不只报告 rank0。
- 所有 rank 原生恢复成功，输入、参数、原始 CPU gold SHA 不变。
- [预构建收据](<results/full02/prebuild98.json>) 在 8-rank model priming 前串行验证 **18 个既有对象**：4 text attention、7 biased vision/patch、1 erf、2 vision QK、2 fused PV、2 NZ softmax；全部 30 可用 cores。
- [只读最终审计](<final_source_verification98.json>) 通过：**23 production + 13 installed API 源、全部执行源、18 对象、原始 HF 源、CPU gold、冻结 QKV、launcher** 均未变；当时全部相关 worker 为零。

源：[完整 TP8 驱动](<full_encoder98.py>)、[串行 prebuild](<prebuild98.py>)、[视觉身份门禁](<expected_vision97.json>)、[执行源映射](<expected_execution_sources98.json>)、[只读审计程序](<verify_final98.py>)。

## 已审计的记录缺陷与修复边界

新 fresh fork 只读 reviewer 核对了本轮之前的 99 项 stdlib tests、实际源 SHA 与收据，未找到改变上述数值结论的错误，但指出：R97 engine 的 `prepare()` 把 softmax meta dict 直接存为引用，`uninstall()` 清空后末次写盘变成 `{}`。**这不是零内核或未执行**：case snapshot 仍为 2 kernels，独立 smoke 与本轮 18-object prebuild 记录了实际对象 SHA，最终审计也验证对象未改。

不修改已执行源或补写原始收据来掩盖这个问题；把身份依据明确改为 [独立 nz03 smoke](<../round97/results/nz03/softmax_smoke97.json>) + [本轮 prebuild](<results/full02/prebuild98.json>) + [最终审计](<final_source_verification98.json>)。后续新 engine 的 prepare/snapshot 必须深拷贝 metadata，保留 uninstall 前对象身份。R96/R97 全视觉跨运行比较均已标明“同机同口径、非同轮配对”；同轮叶子 A/B 和本轮完整 encoder 表不受该标注问题影响。

## 调度与后续

首次 root 只有 staging：假设旧目录存在某个不需要的 helper，copy 失败，**没有 prebuild 或 model worker 启动**。[协调回查](<staging_failure_reconciled98.json>) 保留该事实；旧 root 不删除。自动续轮时旧 owned lease 已过期，重新 live-claim owned ts-217 后从本地已检查的源部署至新 root，未使用 CI/dormant；新拿到但不用的两台 CUDA 主机立即释放，没有运行计算。

本轮核查完毕后，仍在同一 owned lease 上推进 [R99 单 launch 流式注意力的实际 SDK 约束](<../round99/actual_sdk_highlevel99.json>)：进一步消除 score GM 与 D128 过度填充，让高精度路径接近原生速度。它尚未测量，不纳入本轮结论。完整 CPU FP16 对齐、text<2s、与 DiT/VAE 真实共驻留以及匹配官方工况的完整比较仍待完成，持久目标保持 active。
