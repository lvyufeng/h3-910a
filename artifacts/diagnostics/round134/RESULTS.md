# R134局部候选：重复稳定，但改变大量原输出词；未采用

**Keep 受控局部试验数据；Awaiting approval 任何下一项完整图诊断；不自动合并或采用。**
[Issue #15](https://github.com/lvyufeng/h3-910a/issues/15)仍开放，
[Issue #4](https://github.com/lvyufeng/h3-910a/issues/4)copy候选继续关闭。

## 1. 本轮明确授权和原输入门

[用户选择](approval134.json)只批准“局部候选试验”。没有许可采用数值基线、容差、
全50层回归、49步生成或copy优化。候选方案的rounding／bias路径改变在执行前已披露。

同一originaltime/finalAdaLN小组件，实际8rank TP2+USP4；原time重建`t_emb`完整fingerprint
与R132实际正式worker输入匹配，原AdaLN Halfweight/bias与参考参数一致，fixtureSHA核验。
所有case使用同一True诊断flag，不拿flag差异掩盖原生／候选对比；结束恢复False/warnFalse。

[候选实现](fp32_bias_candidate134.py)保留同一Half输入与权重、原Halfbias参数：

```python
output = torch.mm(x, layer.weight.t(), out_dtype=torch.float32)
output.add_(bias.float())
result = output.to(torch.float16)
```

没有BF16、clip、FP32weight替换、少行／新activation cache。FP32是实际返回dtype／bias相加
dtype；**它不证明内部CANN归约格式或精度提升**。候选只用于隔离的小组件，不部署正式worker。

## 2. 实际调用与自身重复性

[实际验证汇总](summary134.json)：original→candidate→restored每case每rank分别
32个原小模块调用（含TPgather）和128个本地apply调用（无TPgather）。总数：

- **768个含TPgather模块调用、3072个无TPgather本地调用**。
- 其中candidate成功请求为160/rank、共**1280**，是Python路径调用计数，非CANNkernellaunch证据。
- 每次模块的Half输入逐词匹配原模块first；固定NPUclone对实际原输入逐词一致。
- 所有input／weight／bias fingerprints前后不变，全部原参数before/after map保留且相同。
- 本轮没有full50／155新检查或49步媒体；不能把小组件counts替代完整模型门。

| case | 本地128次/rank自身逐位同 | 模块32次/rank local及postgather自身逐位同 | **本地输出**自重复最大Half词变化 |
|---|---|---|---:|
| Original | 否 | 否 | 1 |
| Candidate | **是，8个rank全部** | **是，8个rank全部** | **0** |
| Restored | 否 | 否 | 1 |

这证明本次候选的受控局部重复性改善，且撤销后原一词漂移重现；它不是任意输入／所有kernel
或全50层音视频重复性的保证。没有因局部稳定自动采用新baseline。

## 3. 对原输出的数值变化不是原生噪声

每个metric均比较全部Half词。两种原参考各自独立：本地case引用原本地首次输出，
postgather case引用原模块首次完整输出；不是同一anchor，也不混成一个误差。

| 比较范围 | 最大改变Half词数 | 最大绝对差 | 最大relRMS比例 | relRMS百分比 |
|---|---:|---:|---:|---:|
| Candidate local对Original local-first，shape `[3,5376]` | **5447 / 16128** | **9.765625e-4** | **3.2833464631e-4** | **0.03283346%** |
| Candidate postgather对Original module-first，shape `[3,10752]` | **6484 / 32256** | **9.765625e-4** | **1.5955736446e-4** | **0.01595574%** |

Original/Restored各自**本地输出**selfrepeat至多1Half词，maxabs4.8828125e-4。
TPgather后的模块输出合并两个分片，本轮Original／Restored模块postgather selfrepeat实测
最多**2个Half词**改变；不能由“本地至多1词”推断为全输出也至多1词。
候选却改变数千词，所以**不能当作原生单词噪声中的无损替换**。

以上词数／maxabs／relRMS是跨rank／repeats分别取最大值，未必出现在同一个rank或坐标。
不是全模型视频／音频误差，也不是decoded媒体质量。不能利用0.01596%或0.03283%自动
授予新的误差阈值，后续必须由用户按scope决定。

## 4. 同范围局部时间，不是模型速度

各rank重复1–127本地调用的median，再取各rank最大值（`max_rank(median_local)`）：

| Original | Candidate | Restored |
|---:|---:|---:|
| **0.72238ms** | **0.96192ms** | **0.69964ms** |

计时包含explicit device同步、输出D2H、finite检查；不是pure devicekernel，也不是
coordinated per-repeat rank-max模型时间。模块的TPgather路径未进行该纯局部计时。
本轮未采纯kernel trace或peak memory，不能由此宣称总体DiT速度或HBM改善。

## 5. 执行指纹、恢复和资源

[源pins](source_pins134.json)与[真实执行源码](executed_sources134/candidate_probe134.py)、
[process回执](candidate_process134.json)一致。8项portable tests和pureMock检查调用顺序／dtype／
identity通过；它们不代替本轮实际NPU数值，也不证明内部GEMM算法。

newexactroot、全部cohort gates、atomicfatal、20min总deadline和PIDfd-bound private清理
保留；noSDK改动／device reset／sharedprocess signal／CI/dormant。原生source及参数均未变。
[清理回执](cleanup134.json)确认exit0、所有privateprocess退出，8rank恢复原methoddescriptor
和False/warnFalse；没部署正式模型或创建authkey。owned ts217租约已释放。
所有weights／inputs／activations只在host，client只收到源码／JSON／text。

## 6. 下一项由用户决定，不自动扩大授权

本轮授权已完成：候选局部重复稳定，数值与原生有明确差异。当前**未采用、未部署**，
#15保持InProgress，#4仍Blocked，copycandidateoff。

若用户另行批准，可只做原50层same-input诊断：在temporaryTrue基准下对原finalAdaLN→
候选finalAdaLN→恢复原实现，保留155checks/TP2+USP4/原FP32例外，测最终video/audio误差
与重复性，恢复所有改动；不开49步生成、不将结果直接当允许误差或模型adoption。
这是新scope，需要明确授权，不能由本轮“仅局部试验”推断已允许。

用户也可以拒绝该数值路径，继续原算法内部实现定位。任何选择都不把这组局部结果
称为完整模型质量／性能达标或自动无损修复。
