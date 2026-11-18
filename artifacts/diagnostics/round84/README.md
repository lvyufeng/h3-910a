# Round84 — FP32 部分和接入真实 TP8：结构修复跑通，但不是完整优化成功

在同一份[CPU FP16 基准](<../round78/final_CPU_reference_verification78.json>)、
[私有 R79 编码器](<../round79/candidate/candidate_encoder79.py>)、
[R81 KV 重复性修复](<../round81/kv/kv_padding81.py>)上，只改变行并行投影：
原始 Half 输入/权重 → 自研 Cube FP32 GM 部分和 → **FP32 HCCL 求和 → 单次 Half 转换**。
没有 BF16 验收门槛、FP32 权重影子、量化或生产代码修改。

## 实测结果

[8 rank 核验汇总](<summary84.json>)：全部 50 层执行、各对照重复位级一致，禁用适配器后
恢复原生输出哈希，原参数与输入未变。比较对象始终是原始 CPU FP16 计算，以下相对 RMS
是误差描述，不是自行选定的容差。

| 同输入比较范围 | 原生 Half 部分和 | 自研 FP32 部分和 | 结论 |
|---|---:|---:|---|
| 真实 decoder 第 0 层，CPU FP16 输入，DeepStack 注入前 | 1.457e-4 | 1.034e-4 | 单层误差降低约 29% |
| 完整 50 层，CPU FP16 初始条件和 DeepStack | 2.4757% | 2.1902% | 误差有所降低，残差仍明显 |
| 完整 50 层，原生 NPU 视觉条件 | 7.4201% | **7.5777%** | **整体误差反而略升，不能只报告有利对照** |

使用自研行并行路径的真实私有 `encode_ids`（视觉、全部 50 层、布局转换、FP32 HCCL、
最终 Half 转换、D2H 均包含在内）三次 warm 测量，按每次 8 rank 最大值取中位数：
**2.433648 s**。没有达到 <2 s，也不据此声称加速。CPU 预计算条件只用于精度隔离，
没有替代本次 warm 时序中的视觉计算。

这条路径修复了“部分和先转 Half、随后 FP32 collective 无法恢复精度”的结构偏差，
但不是剩余完整图误差的唯一原因。**暂不部署为默认优化路径**。

## 数据类型与内存证据

- 每 rank 实际执行 **804 次 kernel / 804 次 FP32 collective / 804 次最终 Half 转换**；
  collective 前后均实测为 `torch.float32`，不是只根据配置推断。
- 100 个原始行权重仅缓存 Half 布局排列，值往返校验全部通过；无 FP32 权重影子。
- 缓存 **2,162,688,000 B/rank**，scratch **101,187,712 B/rank**，这些额外占用尚不能
  无条件加入 DiT 共驻服务预算。诊断峰值还含 proof capture，不冒充服务稳态内存。
- 真实 `o_proj` 和 `down_proj` 输入的 A packing 对照、往返以及 5 行零填充均核验通过。
- 两种 kernel 在模型加载前串行预编译；8 rank priming 使用 cache hit，不计入 warm 时间。

证据见[真实 rank0 回执](<results/row_fp3202/ranks/tp8_rank0.json>)和
[执行源码、二进制 SHA](<executed_sources84.json>)；相关断言由
[实际回执测试](<test_actual_rows84.py>)覆盖。

## 实际模型尺寸修正与保留的失败

首次完整尝试在**任何自研 row kernel 调用之前**被形状门禁拒绝：准备时误用了
intermediate size 27648 → TP8 K3456。根因是 root 沿用了未经 checkpoint 核实的规划假设。
[官方 config 和 safetensors 的只读核验](<actual_shape_receipt_fixed84.json>)确认实际为：

- attention `o_proj.weight`：[5120,8192] → 本地 K1024；
- MLP `down_proj.weight`：[5120,25600] → **本地 K3200**。

[首次 8 rank 失败](<results/row_fp3201/ranks/tp8_rank0.json>)、执行源码和
[形状诊断](<shape_diagnosis84.json>)均保留，没有原目录重放或覆写结果。
R83 的 K3456 原语测试仍是有效的通用形状实验，但不是这份实际模型的 MLP 宽度。

随后[真实 attention-row 原语](<results/o_row01/blocked_o_row_result83.json>)和
[真实 MLP-row 原语](<results/actual_down01/blocked_actual_down_result83.json>)均执行并重复验证：
FP32 accumulator 相对诊断误差分别 5.815e-7、1.038e-6。

修正后的[适配器](<row_fp32_adapter_fixed84.py>)和[完整 A/B 驱动](<full_row_fp32_fixed84.py>)
在新目录完成 exit0，而未修改第一次执行的源码。新的 fresh fork 的源码贡献已收集、
review 并测试；真实硬件执行与最终核验由 root 完成。

## 验收边界、清理和下一步

[最终核验](<final_verification84.json>)确认 23 个生产文件、13 个安装 API 文件、CPU 金标准
回执和已验证运行库 SHA 均未变，相关 worker 为零。本任务 owned 910A 租约已释放；最初
未使用的 CUDA grant 在计算前即释放，没有修改其他会话后来取得的 CUDA/CI 租约。

完整 CPU FP16 精度、text encode <2 s、实际编码器与 DiT 共驻、官方 A2/A3 同工况比较
仍未完成。不能把本轮“F32 部分和结构正确”或者“重复稳定”当作全目标通过。
下一步针对剩余误差作[实际 attention / 视觉 bias 的叶算子核验](<NEXT.md>)，再决定哪条
自研路径值得保留和融合，而不是继续只替换 row GEMM 或放宽精度门槛。
