# R131：确定性稳定了视频预测，音频仍有残留漂移

**结论：Keep 诊断证据；Reject 将此开关直接当作完整逐位基线。**
[Issue #15](https://github.com/lvyufeng/h3-910a/issues/15) 尚未完成，
[Issue #4](https://github.com/lvyufeng/h3-910a/issues/4) 的 copy 候选继续关闭。

## 1. 已获授权的范围和实际执行

[用户授权](approval131.json)仅允许同输入原50层诊断，不允许采用确定性基线、
新的误差阈值或copy候选。本轮使用R128保存的own-live NPU step25输入：

- 一个正式Omni引擎、同一组8个真实worker，每rank原生3次→确定性3次→恢复3次。
- **72次原50层forward，11,160项严格Half检查，3,600次原FA ACL launch，0个wrapper错误。**
- [实际结果](summary131a.json)和[校验回执](verified131.json)记录TP2、SP4/Ulysses4、Ring1，
  每TP分片28heads，实际FA入口BHND为7heads；不是只读取配置。
- TP组为`[0,1] / [2,3] / [4,5] / [6,7]`；SP组为`[0,2,4,6] / [1,3,5,7]`。
- 输入前后logical words/layout不变；原CPUmaster同时匹配R129原参考与本轮前后状态，
  resident parameters/buffers的fingerprints不变。
- 本轮输出是**一次原50层forward返回的视频／音频预测张量**，不是49步采样后的媒体。
  没有运行encoder或VAE，没有新生成视频，没有teacher features或权重下载。

## 2. 自身重复性：视频与音频必须分开报告

下表为各rank、各次重复对**同一case首次输出**的最大值；不是candidate对native的误差。
每case每rank仅3次重复，本轮结果不构成所有输入／全部内核的确定性保证。

| 模式 | 时间嵌入逐位重复 | 视频预测逐位重复 | 音频预测重复relRMS | 音频重复max abs |
|---|---|---|---:|---:|
| 原生False | 否 | 否，最大relRMS 0.02634784% | 0.05045982% | 0.00850129 |
| 确定性True | **是** | **是，全部比较0个词变化** | **0.0000343518%** | **6.0796738e-6** |
| 恢复False | 否 | 否，最大relRMS 0.02613451% | 0.04425284% | 0.00684009 |

确定性case中，音频仍最多有**1,927 / 13,248个FP32词**改变（原输出shape为`[414,32]`），
所以两个输出的组合bit gate仍失败。
时间／视频稳定是实际进展，但不能隐去音频的非位同，也不能因为它比native噪声小就放宽门。

这支持原执行路径中受确定性开关影响的非重复性判断。开关作用于**整个私有process**，
并不只作用于time projection；因此不能据此宣称“所有视频差异只由时间嵌入引起”。
音频残留的首个不同算子尚未定位，不能直接归因FA、HCCL、norm或最后projection。

## 3. 对原生参考的数值变化：不是新的允许容差

确定性输出对**最后一次原生warm输出**的最大差异：

| 输出 | relRMS | max abs | 最大改变FP32词数 |
|---|---:|---:|---:|
| 视频预测 | **0.02605133%** | **0.02469236** | 3,579,889 |
| 音频预测 | **0.04004749%** | **0.00684388** | 13,247 |

[完整汇总](summary131a.json)另保留对native first、同case first和deterministic first的比较，
不要把其中不同分母／参考混成一个误差指标。确定性time输出与native也不同，max abs
1.1920929e-7；只是其自身重复已经逐位一致。

**没有采用任何新numeric policy。** 用户授权只覆盖诊断，而非这些数值变化的接受。
完整49步媒体质量没有测量，本轮不能用于宣称生成质量通过或总体性能目标完成。

## 4. 同范围时间与内存

每次forward都有cohort阶段协调。先对同一repeat序号取真正的`max_rank(time)`，
再对repeat1/2取median；repeat0单列。这不是R130的`max_rank(local_median)`。

| 模式 | 第一次rank-max | 热态median(per-repeat rank-max) |
|---|---:|---:|
| 原生False | 118.631s | **19.927s** |
| 确定性True | 20.017s | **20.013s** |
| 恢复False | 19.946s | **19.905s** |

计时含原50层forward、原严格检查、输出D2H与tiny time snapshot clone；不含输入／master
fingerprinting和CPU数值比较。它仍是**诊断scope**，不是pure kernel或生产无干扰时间。
时间边界snapshot只在同consumerstream排队clone，直到fullforward完成才D2H，未替换模型返回。

各模式最大PyTorch allocator peak allocated均约**7.135GiB/rank**；这是PyTorch统计，
不是driver总HBM或全部外部ACL workspace占用。引擎初始化237.380s，整个RPC351.269s
包含SHA、CPU比较与cohort协议，不应代替forward延时。DiT<15s目标仍未达到。

## 5. 失败、控制与恢复事实

唯一实际失败的准备步骤是system Python3.10没有`hashlib.file_digest`：
[原失败回执](preflight_failure131a.json)保留；当时没有启动模型。
改用兼容的streaming SHA后，[8份输入重新校验](preflight131.json)通过。
实际模型仍使用已验证的Python3.12解释器，原R129file_digest逻辑没有换成有损实现。

只读prelaunch review发现的PID-number signal竞态、parent-only清理、总deadline和FA shape
口径问题都在运行前修复；[CPU pidfd unit](private_abort_unit131.json)验证exact-capability
children的TERM/KILL、错误starttick拒绝、无关control不动。15项portable source tests通过，
它们不代替本轮实际模型检查。

[真实启动源码](executed_sources131a/diagnostic_rpc131.py)和[source pins](launch_source_pins131a.json)
与执行receipt匹配。总compute deadline90分钟，pidfd绑定信号，所有parent退出路径都会
reconcile精确私有children；不以设备reset、共享进程信号或SDK改动处理失败。

[清理回执](cleanup131.json)确认：本轮exit0，所有私有process退出，8rank恢复False/warnFalse，
observer撤销、原source pins未变，0600私有IPC key删除。owned ts217租约已释放。
所有权重、inputs与activation都留在host；client仅收到源码及有界JSON/text。

## 6. 下一步严格边界

Keep 视频／时间稳定的证据；**不把仍有音频漂移的True模式直接用作#4逐位参考基线**。
#15保持InProgress，#4保持Blocked，正式模型开关False，copy候选未安装。

下一项是定位确定性诊断下音频残留的第一处变化：分离共享主干、audio final norm/projection
与TP/SP输出路径；先核各节点实际输入是否位同，再谈最小修复。不能把0.00003435%当新
容差，也不能未经逐项报数接受新算子。后续诊断／数值策略与完整媒体质量验证仍需按
用户已要求的GitHub合同分别记录，不能因本轮视频位同就关闭整体路线图。
