# 下一步执行约束（R99/R100 之后）

持久目标仍为 active，精度依据为原始 HF CPU FP16，不强制 BF16，不允许 Float Linear 权重教师、BF16 gold cast、假 final norm、事后容差或隐藏 conditioning 缓存。

1. **首先读取真实状态**：[R100 STATE](<STATE.json>)、[R99 STATE](<../round99/STATE.json>)、[R98 已验证摘要](<../round98/summary98.json>)。不要把较早“running”的叙述当作现实，也不要重复未收齐的 job。所有新 agent 都必须 fresh fork，不能唤醒 inactive 的旧 agent。
2. 当前计算完成但收集/最后审计可能仍在进行；以 STATE 中 job id、`run_finished`、8 份 `completed=true` 收据与原始 SHA 为准。若回复丢失先 reconciliation，不能盲重跑。
3. R99 当前有效 kernel 为 [exact-shift-temp 修复版](<../round99/flash/flash_vision_shift_temp_fixed99.py>)，host engine 为 [guard 修复版](<../round99/flash_engine_guard_fixed99.py>)。较早失败版本保留，不得拿其性能或编译记录当作可接受结果。Host bound `<16384`、Float主 scores/O/L、只有 Half shift estimate；不夹断真实输入。
4. R99 单 launch 不是“24倍加速”：已经实测叶子仅比 R97 快约 7–10%，全部27视觉边界/DeepStack/merger位级相同；优化视觉时间 1.582865s。R100必须按5次**配对8-rank-max完整 stock入口**聚合，不能报告最快rank或视觉单独时间。
5. 继续精度定位时，优先围绕真实输入传播及实际下游 text-conditioning projection/refiner：若要比较原始 H3 的真实条件投影，读取真实代码与原始 Half权重、使用同一个CPUFP16 oracle，不能新增人为norm或更改oracle取巧。R93证明仅CPU原始Half就会把0.253%输入误差在layer43附近放大到17.16%；不能把误差百分比相减当作NPU算术贡献，也不能假定更准叶子必然更准全图。
6. 继续性能时，Float rescale/denominator 的64行 scalar广播仍是流式FA成本；精确 shift已用native uint16 copy+Half transpose取代，**一般Float rescale/denominator不能直接降Half**。下一个有明确依据的实验是 **MB64/BC128**：KV迭代从63降到32，复用 Float scores 在求完真实分母后原位生成 P_low，去掉独立 P_low Float UB，使约188KiB UB 可行。4032→4096 的最后64个padding keys必须在softmax前排除、exp后归零，V padding也为零；不能仅因为V为零就让phantom keys进入分母。此方案改变在线归约顺序，必须新source/root和实际CPUHalf叶子/全图验证，尚未实现或测量。原始910A没有Float vcmax、Floattranspose或vec_brcb；[真实SDK能力](<../round99/actual_sdk_copytranspose99.json>) 和 [broadcast能力](<../round99/actual_sdk_broadcast99.json>) 为准，禁止伪装A2。
7. 必须继续保留失败与负结果。每次新 source/runtime ABI先stdlibcontracts、廉价compile/launch、原始真实QKV×CPUHalfMATH/repeat/inputSHA，再完整vision的27+DS+merger门禁，最后全TP8 encoder。旧已执行源不可覆盖；修复用新文件与新private root。
8. 下次使用ts-*机器必须重新live `ts_claim(pool="owned")`，默认绝不取CI/dormant。只释放自己本次新拿到的机器，不动其他会话租约。计算完先所有job/worker/source/gold/binary审计，随后释放owned机器；不要在跨自动续轮期间留长任务超过租期。
9. 整体尚须完整CPUFP16精度、text<2s、encoder/DiT/VAE真实共驻留与可比官方工况验证。R76 DiT49steps约14.657s、VAE约13.5s已是旧测量，但外部condition/co-residency资格没有解除；官方A2/A3 T2VA/不同硬件/精度不能直接当同一FL2VA FP16行比较。无完整证据时不要 complete/blocked goal。
