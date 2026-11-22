# R132：音频残留沿最终AdaLN输出传播，音频头固定输入重放稳定

**Keep 首变化定位；不采用True为新baseline，不放宽audio逐位门。**
[Issue #15](https://github.com/lvyufeng/h3-910a/issues/15)仍InProgress，
[Issue #4](https://github.com/lvyufeng/h3-910a/issues/4)copy候选继续关闭。

## 实际执行，不重跑49步媒体

同一组8个正式worker，每rank原50层9次／155checks每次；本轮**11,160 checks、
3,600原FA ACL launches、0wrapper错误**。TP2+USP4与原FP16/FP32例外保持，
input前后及原CPUmasters/residentfingerprints不变。[完整汇总](summary132a.json)和
[验证回执](verified132.json)记录exactsource pins与实际parallelgroups。

模式依次：False参考1、True控制3、True尾部trace2、原localaudiohead固定输入10、
移除tail观察后的True2、False最终1。局部10×8＝**80个local方法调用**，无新TPgather、
无额外fullmodel155检查，不用于sampler或activationcache。False单次只作参考，不当自重复证明。
输出是step25的原full50预测，非decoded49步视频／音频；encoder与VAE未运行。

## 首个变化边界：不是audio_out局部GEMM

两次tail trace的对齐SHA边界在**rank0/1**首先不同：`final_layer.adaln_proj`输出。
它的输入、sp_gather输出、finalnorm输出和更早观测边界fingerprint均相同。
另外6个rank这**两次**观测没有变化，不能扩展为永久可重复保证。

SHA只是区域fingerprint；本轮还未对AdaLN内部的linear／bias／SiLU／gather逐词拆分。
不能单凭这个边界指认某个CANNkernel、原子归约或通信机制。

尾部音频头完整逐词配对更清楚地区分了输入传播与局部不确定性：

| rank | 完整head输入变化词 | 其中414音频行输入变化词 | local音频输出变化词 | post-TP音频输出变化词 |
|---:|---:|---:|---:|---:|
| 0 | **391** | **391** | 5 | 15 |
| 1 | **391** | **391** | 10 | 15 |
| 2–7 | 0 | 0 | 0 | 0 |

实际head输入shape为`[41792,5376]`，共224,673,792个FP32词；选音频414行输入为
2,225,664词。模型仍对**全部41792行**执行原audio projection，未改成只算414行。
原FP32weight `[16,5376]` 与bias `[16]`逐词不变；全域preTP/local和postTP输出另有完整证明。
[rank0原始回执](executed_diagnostics132a/audio_tail_rank0132.json)保留这些wordcount与SHA。

固定第一次trace捕获的**完整**实际输入，先将NPUclone与原CPUreference输入／weight／bias
逐词确认一致，再直接调用原bound`quant_method.apply`10次：**全部8rank的所有local输出逐位同**，
音频414行亦同。每次重放仍是原41792×5376→41792×16，未调用Column模块或TPgather。

因此本轮不支持将首残留直接归因于FP32audio_out或TPgather：变动已出现在该head的输入。
固定输入重放只是测得的局部重复证据，冻结布局／调度不是所有production路径的完备证明；
original/frozen stride分别记录，physicalNPUformat明确未测。

## 观察干扰和数值：不能拿较小误差当容差

True控制3次：视频全位同；音频最多15/13248词变化，maxabs **2.3841858e-7**，
relRMS **3.0998467e-7%**（ratio3.0998467e-9）。两次tailtrace观测到同一幅度，
不意味着原问题已解决；移除tail后True2次又有**1927/13248词**变化、maxabs
**6.0796738e-6**、relRMS **0.0000343518%**，与R131残留一致。

True各phase对第一个Truecontrol的跨phase比较最多1938词／6.0796738e-6，参考不同不能
混为selfrepeat误差。视频在本轮True所有控制／trace／restore输出均位同，不能由此断言
整个hidden矩阵原本同值，也不能采用完整baseline绕过audio gate。

即刻SHA/D2H／冻结输入会改变调度、allocator寿命及执行顺序；trace为**nonclean**。
首变化定位保留，仍需小AdaLN组件的同输入逐词因果配对，才可能提出最小修复。

## 时间、内存和恢复

coordinated per-repeat真实maxrank，Truecontrol热态repeat1/2 median **19.839s**；
tailtrace首次 **65.764s**、第二次 **39.299s**；移除trace后True第二次 **19.854s**。
最后False单次 **19.853s**，首False冷次 **112.717s**。上述scope包含full50/原audit/D2H，
trace还含observations与CPUword比较；不冒称clean生产速度或优化收益。

trace最大PyTorch peak allocated **7.973GiB/rank**，控制 **7.135GiB/rank**；不是全部driverHBM。
大fullinput约0.837GiB/rank，CPU只保存一份reference，另一个NPUfrozeninput在local重放后释放。
CPUMemAvailable/devicefree guards先核，uint32字比较与float64metric按262144词分块，
没有把整hidden转double制造多份巨型临时。

[source pins](launch_source_pins132b.json)、[真实运行源码](executed_sources132a/audio_tail_trace132.py)
和回执一致。15portable source tests与CPUmetric/constructorrollbackunit通过，不代替NPU检查。
准备commit检查因尾部空行失败两次：第二次原文本归档亦触发检查；
[原失败记录](precommit_failure132a.json)及[原source字节prefix／SHA](prelaunch_state132a.txt)保留。
修正仅空白／collector manifest名，模型尚未launch时改pins，旧pins132a不覆盖；无未知重放。

[清理回执](cleanup132.json)确认本次exit0、所有privateprocess退出，False/warnFalse恢复、
observers撤销、exactIPCkey移除、原sourcepins不变。owned lease仅为接下来原小组件诊断继续持有；
无SDK更改／device reset／共享进程信号，权重与activation未转client。

## 下一项

只加载原time和final AdaLN小组件，使用同own-live timesteps，重建t_emb必须匹配本轮
真实AdaLN输入SHA后才测试。原finalAdaLN参数／linear输入为Half，保留原bias、TP2gather
和SiLU，不用新FP32近似、减行或替代GEMM。定位local linear输出与postgather是否改变，
先给同输入/权重逐词数据，再讨论最小修复。所有数值策略仍需用户逐项决定。
