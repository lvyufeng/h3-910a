# 当前性能状态与证据索引（截至 R221）

GitHub [路线图 #1](https://github.com/lvyufeng/h3-910a/issues/1) 与 [Project #10](https://github.com/users/lvyufeng/projects/10) 是实时状态源。本文件是证据索引，不把历史实验、拟合或情景算术升级为采用/达标。此前版本中的“10s可达”“FA实测证明1.89×”“encoder99%搬运/0.5s”等过强结论撤回。

## 1. 目标与已经实测的范围

用户2026-11-23将DiT目标从<15s **上调**为 **≤10 s/step**（旧记录常写<10 s/step）。原50块、49实际前向、TP2+USP4、FP16矩阵、原FP32例外与全部155条Half检查不变；live encoder TP8始终开启。**总体目标未完成。**

| 目标 | 有效历史/当前测量 | 当前结论 |
|---|---|---|
| DiT ≤10s/step | R126 whole-forward19.372s；R136逐step rank-max中位20.03/20.11/20.03s | 未达标；不可替换为isolated replay17.556s |
| 审计候选（R185） | 同轮配对17.987→15.473s，节省2.514s，10测量step×8rank | 候选实测，非本轮采用或完整49步10s证明；H3_FAST_AUDIT默认关 |
| 真实形状FA局部（R217） | wrapper154.57ms，ACL kernel152.18ms | 原版复现；R217候选全部Reject |
| padding-task裁剪FA单项（R218/R219） | R218局部kernel152.235→140.853ms；8rank replay17.944→17.376s，恢复17.977s | 实测节省0.568s；steps0/25/48共1,200真实FA对逐位同，8finite边界测试同；完整49step未验，默认关，待用户决定 |
| encoder<2s | 历史in-pipeline rank-max3.14–3.17s；隔离live FP16约2.34s | 未达标；驻留/NUMA等仍待实测 |
| video VAE<20s | 历史warm rank-max21.62–21.76s | 用户已暂停此路，未达标 |
| A2/A3同口径对比 | 未完成 | [#13](https://github.com/lvyufeng/h3-910a/issues/13)仍开放 |

[R217完整结果](round217/RESULTS.md) · [R217原始基线](round217/restored_final217.json) · [R217恢复](round217/restored_end217.json)。历史R126整体证据以[封存说明](round126/README.md)为准；不同scope或冷热数据不相减作因果归因。

## 2. R219补精度证据，R218单项实测收益未晋升

### R221 受保护 linear 精确分解：guard 只占0.52s/step，另1.14s/step未归属

按部署副本`transformer.py`（sha256`5d94ba09…`）第495–517行的逐op序列，在真实形状
（M=10448；`out_proj`K3584/N5376、`fc2`K7168/N5376）上做**免权重、单卡、仅计时**复现
（[R221结果](round221/RESULTS.md)、[receipt](round221/receipts221.json)）：

| 变体 | `out_proj` ms | `fc2` ms |
|---|---|---|
| `mm_only`（含x cast，FP32 out） | 2.952 | 11.613 |
| `amax_cast_mm`（去掉/scale与×scale） | 6.556 | 17.491 |
| `full_guard`（部署序列原样） | 7.080 | 18.429 |
| R147 in-model中位 | 18.306 | 29.934 |

- **guard净开销≈0.52 s/step**（×50块×2层），不是R147推的1.665 s；scale==1分支跳过两次恒等运算
  只值73 ms/step，且两case在112,336,896 word上逐位同——仅是分支证据，**未采用**。
- **in-model比standalone多11.2/11.5 ms/层，合计≈1.14 s/step既不在guard也不在mm内**；
  候选主因是RowParallelLinear的TP all-reduce，或guard内`float(peak)`把权重预取流一并排空。
  这是R222要归属的对象，不能计入已回收。
- GEMM达成率（`mm_only`口径，310 TF为沿用假设）：`out_proj`136.4 TFLOPS/卡（44.0%）、
  `fc2`69.3 TFLOPS/卡（22.4%）。
- **10 s仍不可达**：GEMM区域即使拿掉1.14 s，仍需FA从7.45 s降到≈2.2 s。

**结论**：GEMM/guard路径的可回收量级是0.05–0.2 s/step，不改变10 s判断；FA仍是第一杠杆。

- R219只做gate-only步骤，model始终返回original FA output。step0和48新增800次原版/candidate真实QKV对比，29,956,505,600FP16word零差异；与R218 step25合计1,200对/44,934,758,400word零差异。
- 同一candidate源码与binary SHA052c49f2…重编译复核；8个finite mask/极端值测试（signedzero、subnormal、maxinvalidKV、negative scores ×exact/partialkey边界）14,680,064word零差异、nonfinite/saturation=0。
- 这是局部精度补充证据，未采用、组合或重新测candidate speed，不证明全49step轨迹；R218完整预测variation仍不作为自动容差。原版restore、privatecache/tensor清理、lease释放。

[R219完整结果](round219/RESULTS.md) · [实际input gates](round219/gate_summary219.json) · [finite边界gate](round219/edge_summary219.json)。待用户对该项精度门与完整候选验证作决定，默认OFF。

### R218独立候选实测（未组合/采用）

- 当前同run8rank DiT replay step25：clean17.997s、只读profile18.080s、恢复18.000s；全部155strict检查保留，实际50FA launches/rank。不是执行live encoder的完整pipeline。
- 同runnative ACL FA kernel约7.627s/步；qkv/fc1/adaln检查subspan约1.244/1.578/0.066s；H2D约32.3GB/rank、50次prefetch，3.381s span包含stream waits并与compute重叠，不能与父子span相加。
- 精确UB广播位拷贝候选small/real逐位同（含signedzero/subnormal/NaN payload unit），kernel152.256→153.152ms，无收益Reject。
- **padding任务裁剪**：保留GM storage shape/stride、Br128/Bc304、mask/FP32累加、自动同步与所有真实token pair，仅Tr342→327、Tc144→138。真实几何局部kernel152.235→140.853ms（1.0808×），输出37,445,632word零差异。exact/partial有效key边界测试通过。
- **单独8rank测试这一项**：原版17.944s→候选17.376s→恢复17.977s，节省0.568s（3.17%时间减少），没有与R185审计/guard候选组合。原50块、155strict检查保留；首次gate forward返回原版输出，400真实attention输入对比较14,978,252,800word零差异。
- 完整预测candidate与native仍有video relRMS2.56–2.58e-4、audio3.99–4.44e-4，native自身也variation。这不是自动容差；候选默认关，R218具体测量时等待step0/48补证据；现R219已补，但**仍Awaiting user decision/full49step evidence**。

[R218完整结果](round218/RESULTS.md) · [当前同run预算](round218/path_summary218.json) · [独立裁剪单步测量](round218/trim_model_summary218.json) · [完整预测数值](round218/trim_prediction_metrics218.json)。不能直接把这0.568s从R185不同run15.473s里扣成组合结果；10s仍未达标。

### R217的前一轮工作（均Reject）

| 单项变体 | 真实形状kernel / wrapper ms | 数值/执行结果 | 决定 |
|---|---|---|---|
| 原版，候选前 | 152.180 /154.571 | 原版基准 | Keep baseline |
| 禁用matmul scope自动同步+外部PIPE_ALL | 无有效数字 | 编译成功，small形状507015 L0A/L0B冲突 | Reject，停用 |
| P×V直接改官方matmul | 仅small约1.579 /2.010 | NZ输入方向错误，relative RMS1.41094，99.989%word不同 | Reject实现错误，未测真实形状 |
| P×V，K分组16→128，保留NZ方向和自动同步 | 155.342 /157.652 | 37,445,632个输出word全部相同，慢2.08% | Reject性能 |
| 同分组+thread_num2 | 155.331 /157.701 | 与上项相同binary SHA；输出逐位同，慢2.07% | Reject性能 |
| 原版，候选后恢复 | 152.168 /154.510 | 所有输出word与前基准同 | Keep baseline restored |

[逐word比较](round217/exact_comparison217.json) · [错误P×V数值](round217/comparison_pv217.json) · [同步运行失败](round217/scoped_failure_raw217.json)。全部是固定合成输入的局部算子结果；不是全rank真实模型精度验收。该局部诊断没有改变TP配置，也不加载模型权重。

R215的tile测量仍只有约1.12×局部wrapper收益，原始输入未固定、缺配对精度验证；[原始receipt错误字段说明](round215/RESULTS.md)。R216共享L0C脚本没有成功生成，不能声称实现完成；[失败清单](round216/RESULTS.md)。

## 3. Roofline口径已更正：当前FA是dense，不是causal

编译实例真实字段：`has_attn_mask=False`、`only_causal=False`、`prev_block_num=next_block_num=65536`。策略名“sparse”并不产生causal mask。

- B1H7，input_N41792→kernel_N43776，D128；valid keys41769。
- Br128/Bc304，Tr342/Tc144，30core；dense tiles49,248/head，344,736/七head。
- QK+PV的padded dense矩阵工作量约6.868156 TFLOP/次；kernel152.18ms，约45.13 TFLOPS/卡。
- 历史最优GEMM96.9 TFLOPS/卡是其他工作负载的经验参考，**不是物理上限**；上述比率也不是硬件occupancy/MFU，更不能排除vector/HBM瓶颈。
- 原报告的causal FLOPs、24%利用率、按causal推导的36%目标率均不可靠，撤回。原有效key mask始终保留，没有稀疏化。
- torch队列事件围绕直接ctypes ACL调用出现0.005ms假kernel值，已拒用；最终用相同ACL stream上的原生计时事件，NZ预先构造，kernel与clean wrapper分测。

[可重复的计算与实测摘要](round217/summary217.json) · [修正计时探针](round217/probe217.py)。

## 4. 10s的算术仅是情景，不是证明

R185的**配对**收益是17.987→15.473，不能用17.902作为其同轮基线。原155检查不得删除；R162关检查的15.035s是**不合规诊断**，不能当“模型真正时间”与其他历史scope混用。验证脚手架2.87s是该诊断差值，并非所有历史步时统一可扣项。

仅为核算方便，设A=15.473，历史F=7.454，假设guard_gain=1.665、other_gain=0：

```text
new_step = A - guard_gain - F + F / FA_speedup - other_gain
```

| 假设FA加速 | 假设guard节省后的step情景 |
|---|---:|
| 1.5× | 11.323s |
| 1.89× | 10.298s |
| 2.044× | 约10s |

若不假设guard收益，固定其他项的预算需FA约3.763×。这不是取消guard的必要性证明；其他项仍可优化。A/F/guard来自不同scope和不同运行，尚未形成当前同run关键路径分解，**不能据此宣称可达/不可达**。

R193拟合`83.53ms+2.922µs×tiles`中的截距和斜率没有被证明是可移除的开销或真实性能下限。移除斜率的1.89×是拟合反事实，**不是已实现加速**。禁止“全部为实测”的混用。

## 5. 候选与数值门禁

- [#5 FA](https://github.com/lvyufeng/h3-910a/issues/5)：R217/PV与R218广播候选Reject；R218仅padding-task裁剪拿到实测单项收益，Awaiting user decision/additional evidence，默认未启用。R219已补step0/48实际输入门禁；待用户decide/full49step evidence，默认OFF。不能把单buf源码依赖定位等同于测得瓶颈，也不能称“唯一剩余路线”。
- R185单趟审计内核：配对节省2.514s；R183对抗15/15、R186实际张量310/310判定相符。不是“zero risk”；未由本轮推广到默认路径。内核阶段历史代理0.351 s与推演2.52 s仅是R178诊断；实际收益以R185配对为准。历史构建脚本[overlay179.sh](round179/overlay179.sh)不代表当前FA采用。
- [#6 GEMM](https://github.com/lvyufeng/h3-910a/issues/6)：仍需当前same-run布局/保护/传输分解；低“剩余速率百分比”不能证明没有优化空间。guard默认保留；不以native噪声带自动接受差异。
- [#4 Runtime](https://github.com/lvyufeng/h3-910a/issues/4)与[#15重现性](https://github.com/lvyufeng/h3-910a/issues/15)：R213 legacy也出现variation，只能说明async prefetch不是variation的必要条件；isolated collective稳定不证明全模型上下文无问题。历史26.923vs20s未配对，不能称prefetch收益6.9s。
- encoder：63GB/3s与单链20GB/s相近忽略TP8聚合链路，不证明99%搬运或3.15s物理下限。取消text_encoder offload的0.5s预测未实测、内存与合并运行未验证，不能称达标。
- VAE暂停。任何可测候选数值偏差先停用、上报并请用户决定；精度门是输入/算子归因问题，不是全模型先验“必然不可满足”。历史guard对照失败为INCONCLUSIVE，非候选通过。

## 6. 下一轮与资源

step0/25/48的局部FA门禁已通过；下一步等待用户对R218这一项的精度门和完整49step独立候选验证作决定。未批准前不采用/组合，可继续单测其他布局/向量或GEMM候选。当前same-run数据已建立区域范围，但精确关键路径仍需trace；父子与copy span不相加，目标达成前不关闭路线图。

本轮结束原版已恢复；私有候选cache/package与合成输出删除，未修改SDK/driver或reset设备。owned ts-217 lease已释放；没有使用CI/dormant，其他会话lease不操作。[清理回执](round219/cleanup219.json)。所有权重与激活留在主机，本轮只下载scalar JSON、源码及文本；专有SDK参考仅保留本地hash，不公开源码。
