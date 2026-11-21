<!-- h3-910a:followup:native-sameinput-repeat128 -->

主路线图#1；发现于#4，须先解决#4的逐位候选门基础。P0 · Runtime · 逐位门禁。
不要求也不许可放宽数值容差，先建立可重复、可归因的baseline。

## 实际证据

R128在正式worker、同一ownlive NPU request step25 input上连续执行原50块forward。
TP2+USP4／encoderTP8与原FP16/FP32例外保持；没有安装任何serial/async adapter。

- 第二native forward vs第一次：video relRMS 0.0261953%，maxabs0.0168597；
  audio relRMS0.0411016%，maxabs0.00865376。
- 第三native forward vs第一次：video relRMS0.0263200%，maxabs0.0253544；
  audio relRMS0.0357664%，maxabs0.00477457。
- 每次155strict Halfchecks、全8rank原CPUmaster SHA不变、actualTP2group/28localheads已确认。
- 原prediction引用和CPU快照在replay后仍相同，extra replay不用于sampler。
- 尚未检查输入tensor是否被某处改写、模型非参数状态、确定的首个变化算子或collective；
  根因Unknown。不能直接指认encoder、FA、HCCL、allocator或copy stream。
- gate在第一个native repetition就停止，所以104wide候选proof／独立copy模型门未执行。

参考：[R128代码](https://github.com/lvyufeng/h3-910a/commit/087c5fd)、
[正式worker gate](https://github.com/lvyufeng/h3-910a/blob/087c5fd/artifacts/diagnostics/round128/formal_replay128.py)。
完整原始小JSON会链接到本Issue。输入/weights/activation仅在host，禁止下载/中转client。

## 工作清单

- [ ] 读取#4实际gate和host保存的8份原NPU input；核验source/input/权重/参数/stride/format/sourceSHA。
- [ ] 证明每次forward前后输入tensor的值/metadata是否变化，区分输入变化与模型/运行时非确定性。
- [ ] 检查非参数buffer、workspace、位置/attention状态和常驻缓存；不启用新的activation cache。
- [ ] 独立诊断轮逐50块记录首次不同节点；对104wide原算法／相同权重输入做成对FP32word比较。
- [ ] FA本地kernel重复、FP32norm/residual、TP/USP collective分开检查；不相加嵌套event spans。
- [ ] 非clean诊断sync可能隐藏race，必须和无额外sync的重复控制分开，不以它冒充性能证明。
- [ ] 仅在实际数据指向具体root cause后修复；原155checks、FP32例外和真实TP都不减。
- [ ] 如果确定属于原算法数值非确定性，报告重复分布/幅度与候选因果配对方案，由用户决定；
      原native noise不是自动容忍阈值，不直接把0.026%/0.041%写成允许候选误差。

## 验收／退出

- [ ] 有界same-input重复与首个不同节点/状态证据，或清楚列出尚不能定位的具体阻碍。
- [ ] candidate未启用与native-self-repeat分类分离；driver的恢复状态与输出位同比较字段分离。
- [ ] 提供#4可用的严格候选配对门；未解决前#4不合并/不声明加速。
- [ ] GitHub+Project记录Keep/Reject/用户决定、回滚与commit/PR；own资源privateIPC/进程清理。

依赖：无（发现于#4，不能再blocked_by #4而形成环）。
仅owned liveclaim资源，复用host原权重，不碰共享SDK、driver、设备reset或CI/dormant。
