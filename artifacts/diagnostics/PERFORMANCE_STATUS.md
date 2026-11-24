# 当前性能状态与证据索引（截至 R217）

GitHub [路线图 #1](https://github.com/lvyufeng/h3-910a/issues/1) 与 [Project #10](https://github.com/users/lvyufeng/projects/10) 是实时状态源。本文件是证据索引，不把历史实验、拟合或情景算术升级为采用/达标。此前版本中的“10s可达”“FA实测证明1.89×”“encoder99%搬运/0.5s”等过强结论撤回。

## 1. 目标与已经实测的范围

用户2026-11-23将DiT目标从<15s **上调**为 **≤10 s/step**（旧记录常写<10 s/step）。原50块、49实际前向、TP2+USP4、FP16矩阵、原FP32例外与全部155条Half检查不变；live encoder TP8始终开启。**总体目标未完成。**

| 目标 | 有效历史/当前测量 | 当前结论 |
|---|---|---|
| DiT ≤10s/step | R126 whole-forward19.372s；R136逐step rank-max中位20.03/20.11/20.03s | 未达标；不可替换为isolated replay17.556s |
| 审计候选（R185） | 同轮配对17.987→15.473s，节省2.514s，10测量step×8rank | 候选实测，非本轮采用或完整49步10s证明；H3_FAST_AUDIT默认关 |
| 真实形状FA局部（R217） | wrapper154.57ms，ACL kernel152.18ms | 原版复现；本轮候选全部Reject，无新whole-step收益 |
| encoder<2s | 历史in-pipeline rank-max3.14–3.17s；隔离live FP16约2.34s | 未达标；驻留/NUMA等仍待实测 |
| video VAE<20s | 历史warm rank-max21.62–21.76s | 用户已暂停此路，未达标 |
| A2/A3同口径对比 | 未完成 | [#13](https://github.com/lvyufeng/h3-910a/issues/13)仍开放 |

[R217完整结果](round217/RESULTS.md) · [R217原始基线](round217/restored_final217.json) · [R217恢复](round217/restored_end217.json)。历史R126整体证据以[封存说明](round126/README.md)为准；不同scope或冷热数据不相减作因果归因。

## 2. R217的实际优化工作

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

- [#5 FA](https://github.com/lvyufeng/h3-910a/issues/5)：本轮具体候选Reject。后续需正确NZ布局、内部依赖同步与尾块处理；不能把单buf源码依赖定位等同于测得瓶颈，也不能称“唯一剩余路线”。
- R185单趟审计内核：配对节省2.514s；R183对抗15/15、R186实际张量310/310判定相符。不是“zero risk”；未由本轮推广到默认路径。内核阶段历史代理0.351 s与推演2.52 s仅是R178诊断；实际收益以R185配对为准。历史构建脚本[overlay179.sh](round179/overlay179.sh)不代表当前FA采用。
- [#6 GEMM](https://github.com/lvyufeng/h3-910a/issues/6)：仍需当前same-run布局/保护/传输分解；低“剩余速率百分比”不能证明没有优化空间。guard默认保留；不以native噪声带自动接受差异。
- [#4 Runtime](https://github.com/lvyufeng/h3-910a/issues/4)与[#15重现性](https://github.com/lvyufeng/h3-910a/issues/15)：R213 legacy也出现variation，只能说明async prefetch不是variation的必要条件；isolated collective稳定不证明全模型上下文无问题。历史26.923vs20s未配对，不能称prefetch收益6.9s。
- encoder：63GB/3s与单链20GB/s相近忽略TP8聚合链路，不证明99%搬运或3.15s物理下限。取消text_encoder offload的0.5s预测未实测、内存与合并运行未验证，不能称达标。
- VAE暂停。任何可测候选数值偏差先停用、上报并请用户决定；精度门是输入/算子归因问题，不是全模型先验“必然不可满足”。历史guard对照失败为INCONCLUSIVE，非候选通过。

## 6. 下一轮与资源

下一轮优先以当前同run数据预算排优先级；FA可从减少vector/layout内存pass、正确tail tiling和保留自动同步的布局安排开始，逐项build→small数值→真实geometry数值/时间。未通过的候选不组合，目标达成前不关闭路线图。

本轮结束原版已恢复；私有候选cache/package与合成输出删除，未修改SDK/driver或reset设备。owned ts-217 lease已释放；没有使用CI/dormant，其他会话lease不操作。[清理回执](round217/cleanup217.json)。所有权重与激活留在主机，本轮只下载scalar JSON、源码及文本；专有SDK参考仅保留本地hash，不公开源码。
