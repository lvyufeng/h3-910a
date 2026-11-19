# Round105 — 原始910精确Float逐字广播：逐位通过，但FA实测慢约1.5%

原始910没有Float transpose或vec_brcb。这里用uint16 word copy、16位transpose和masked `vor` 做 **真正32位Float逐字广播**，不把一般Float rescale/denominator数值转成Half，不丢低16位。它是一个已在硬件验证的可复用原语；融合到本次FA没有提速，因此不升默认。

## 位级验证与工作原理

[实际SDK接口](<../round103/actual_sdk_vor_stride103.json>) 的低层 `vor` 支持block stride0：

1. 每8个Float行值对应16个uint16 word，复制成16×16字矩阵并transpose；
2. `vor` 的even/odd掩码将low/high word planes交错还原每行16份Float，repeat stride2、src block stride0；
3. 复制到各NZ group，没有Float→Half算术转换。

[实际operator probe](<results/broadcast01/broadcast_float105.json>) 与 [源码](<float_row_bits105.py>)验证：一般finite Float、真实分母/rescale范围，以及含±0、subnormal、最大finite、Inf、quiet/signalling NaN payload、非Half可表示值的字模式，全部Float bits精确相同，重复和输入也不变。bitcopy代码不包含 `h_cast` 或数值运算；[字地址测试](<test_word_broadcast105.py>)覆盖两个掩码恰好写每个word一次。

## 融合到BC128的负结果

[融合内核](<flash_vision_bitcopy105.py>) 只替换Float rescale/denominator广播；复用已经死的原shift scratch/broad buffers，**UB仍190848B，零额外UB**。Float主QK、概率分裂、PV、onlineO/L算术顺序未变。

[combined FA smoke](<results/smoke01/bc128_bitcopy_smoke105.json>)严格匹配R101 BC128的所有输出bits，包括poison key-padding；[真实QKV A/B](<results/leaf01/leaf_compare105.json>)同样匹配旧BC128 bits，CPU误差完全相同。

| 图 | 原BC128 scalar广播 | 精确word广播 | 原/新速度比 |
|---|---:|---:|---:|
| 0 | **22.815790 ms** | 23.191830 ms | **0.9838×** |
| 1 | **23.236160 ms** | 23.592180 ms | **0.9849×** |

7次同轮配对warm、旧R101 SHA、严格 `[1,16,4032,72]`、重复及输入全部核验。虽然64次scalar load/vdup替换成约32个native bit-ops及group DMA，**真实FA慢约1.5%，不能根据静态指令数宣布加速**。没有因此启动更贵的完整vision/encoder R106测量或默认部署；最新完整实际结果仍是 [R102](<../round102/README.md>) 的7.405683% gap/2.128423s，最快修补对照2.026669s仍≥2s。

## 最后审计与清理

[最终CPU-only审计](<final_source_verification_cpu105.json>)核查 **23 production源 + 13 installed API源 + 151份执行源映射 + 24个实际记录对象**，原始HF/CPU FP16 gold、frozenQKV和launcher未变，R102四份实际Half输出SHA核验通过，全部相关worker为零。相关job全部收集后已释放本次owned ts-217；未请求CI/dormant，未动其他会话租约。

[审计程序](<verify_final105.py>) 与 [新source映射](<expected_execution_sources105.json>)保留失败/修复路径；首次审计因新目录缺少CPU helper import path失败，使用同一不可变程序加 `PYTHONPATH` 后通过，未重跑模型或更改gold，见 [环境失败记录](<audit_failure105.json>)。所有NPU编译/推理都已结束，没有留下后台计算。

[host engine](<flash_engine105.py>)、[真实叶子](<leaf_compare105.py>)、[原语probe](<broadcast_float105.py>)、[实际与融合测试](<test_fusion105.py>)均保留。无Float权重教师、BF16 gold cast、假final norm、hidden conditioning cache或事后容差。整体目标保持active；下一步 [执行计划](<NEXT.md>)利用已核实的replicated视觉源码探索真正Vision TP8，并定位真实42/43或原始下游conditioning。R86已经排除了第25层SiLU的大精度缺口，不重做无效Float SiLU，不假设扩大tile或减少广播必然更快。
