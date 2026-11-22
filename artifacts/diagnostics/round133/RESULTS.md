# R133：原最终AdaLN本地Half投影自身非位同

**Keep 根因范围定位，未实现／采用任何替代算子或新容差。**
[Issue #15](https://github.com/lvyufeng/h3-910a/issues/15)仍开放，
[Issue #4](https://github.com/lvyufeng/h3-910a/issues/4)的copy候选继续关闭。

## 从R132区域SHA进一步做原小组件逐词配对

R132在rank0/1首次观察到finalAdaLN输出不同，音频头输入391个FP32词先变；
固定完整输入的原FP32audio_out重放稳定。[R132结果](../round132/RESULTS.md)仍独立保留。

本轮只加载原time与finalAdaLN组件，不运行50层／155checks／49步生成。
重建的True模式`t_emb [3,2688]`必须完整SHA/shape/stride匹配R132正式worker的实际输入，
原finalAdaLN Halfweight/bias也与R132resident参数fingerprints匹配，fixture文件SHA再次核验。

使用原worker config factory并在构造前设置当前VllmConfig；Python default dispatch
callable/sourceSHA与R132已记录的default实现匹配。R132完整finalAdaLN runtime config未独立
记录，physical格式与内部CANNkernel仍未知；这些限制没有被time/inputSHA匹配替代。

[实际验证汇总](summary133.json)记录所有8rank：

- **32次原模块调用＋4次撤销观察器后的模块调用/rank**，含原SiLU、Half转换和TP2gather。
- **128次原bound local apply/rank**，输入固定，不调用Column模块／TPgather。
- 合计**288个含gather模块调用、1,024个本地算子调用**。这是小组件计数，不是full50检查数。
- 输入是Half `[3,2688]`，localweight Half `[5376,2688]`、bias Half `[5376]`；
  localoutput `[3,5376]`，gather后 `[3,10752]`，拆为shift/scale各 `[3,5376]`。
- 原结果在local返回／TPgather前即刻独立冻结，观察器返回原对象。所有32次模块实际输入逐位同；
  固定clone与原输入全词一致，固定input/weight/bias前后fingerprints均不变，未重绑活模型参数。本轮完整after参数map未单独落盘；
`original_parameters_unchanged`来自已执行、已pin源码的before/after断言，而非外部重新测量。

## 首变化存在于本地投影，早于TPgather

8个rank在128次固定input的原本地apply中都出现差异，每次相对首次参考至多**1个Half词**变化：

| TP分片 | local输出变化坐标 | 对首次参考最大绝对差 |
|---|---|---:|
| even ranks 0/2/4/6 | `[1,1024]` | **7.62939453125e-6** |
| odd ranks 1/3/5/7 | `[1,1390]` | **4.8828125e-4** |

Half数据按uint16完整词比较，不把它误报为FP32词。每rank变化重复次数不同；这不是对全部
输入的概率模型，也不是新的容差。模块postgather差异可以来自任一local分片变化，不能把
它直接归因TPgather本身。本地apply尚包含原GEMM与bias行为，内部哪个阶段不确定仍未拆分。

此结果将R132首次区域变化进一步收窄到**原最终AdaLN的本地Half线性投影执行路径**。
不能由此宣称具体atomic、split-K或CANN内部workspace已经被证实。原算子仍未替换。

## 为什么本次残留只在音频预测中出现

[原始索引元数据核验](time_row_mapping133.json)从CPU读取已核SHA的own-live input，未运行模型：

- 音频414行全部使用时间行 **1**。
- 视频39312行中37296行使用时间行 **0**，2016行使用时间行 **2**；没有使用行1。
- 文本2043行使用时间行0。

所有测得的local变化都位于第1号时间行。这与“True视频预测逐位同，而音频仍有残留”的
结构一致；没有把视频行都误写成index0，也没有把word-change占比混为relRMS。
它不是所有潜在音频差异的完备排除证明，后续最小修复仍须原图回归和最终输出对照。

## 精度／执行／恢复范围

本轮没有启用新的数学实现，所有time原FP32参数与保护、finalAdaLN原Half输入/weight/bias
保持。True仅用于原问题诊断，没有成为正式模型baseline，也没有任何允许误差阈值。
没有newcopy/pool/NUMA/quantization/teacher/少步／decoded媒体质量结论。

立即CPU快照和cohort控制改变调度，属nonclean，不把小组件重复耗时当model speedup。
本轮未采纯kernel事件或NPU峰值memory；原R132 fullscope时间／内存另见其独立记录。
8项portable tests检查source/控制，不代替实际NPU证明。

[真实执行源码](executed_sources133/component133.py)、[source pins](source_pins133.json)、
[process回执](component_process133.json)与输入/参数引用检查匹配。全8cohort门、原子fatal与
20min总deadline、pidfd-bound private cleanup均保留；没有因某rank结果不同提前停止别的rank。
[清理](cleanup133.json)确认exit0、所有私有process退出、False/warn设置回原值、原source／参数
未变、无authIPCkey或device reset／sharedprocess signal。owned ts217租约已释放。
weights、inputs、activations均留host；client只有源码／JSON／text。

## 下一项需要的用户决定

原最终Half线性投影存在逐位不确定性，所以不能把仍有audio漂移的True模式当#4严格参考。
建议只做一个最小候选的**局部对照试验**：仍使用同一Half输入与权重，显式得到FP32 GEMM输出，
用FP32添加原Half bias的精确扩展值，最后只做一次Half回写。它会改变原rounding／bias路径，
**不是自动逐位无损修复**；尚未实现或测试，也不承诺已稳定。

必须先得到该试验的具体授权，再报告原生／候选／恢复的所有Half词和误差、实际重复性。
即便局部稳定，也不代表采用、允许误差或完整50层质量通过。原50层155checks/TP/FP32例外
不可减，copy候选继续关闭；随后是否进入完整原图回归由用户按数据决定。
