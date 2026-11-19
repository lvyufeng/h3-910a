# Round101 — BC128 与原位概率缓冲：真实 attention 叶子快8–10%

从已执行的 [R99 BC64内核](<../round99/flash/flash_vision_shift_temp_fixed99.py>) 出发，改用 **MB64/BC128**，K/V4032→4096，Q仍为4032：每query tile的KV循环63→32。Float scores求完真正Float分母后原位生成低残差概率，删除独立Float P_low缓冲，UB精确为190848B（余量5760B）、L0C73728B、L183968B。原始910编译SoC，32物理核、30可用launch核，没有伪装A2能力。

## 已完成的实际结果

[同轮真实QKV A/B](<results/leaf01/leaf_compare101.json>)：输入为原始Half FrozenVisionQKV，CPU参考为原始Half MATH，形状严格先检查 `[1,16,4032,72]`。

| 图 | R99 BC64 | BC128 | 同轮加速 | BC128对CPU Half相对RMS |
|---|---:|---:|---:|---:|
| 0 | 24.645160 ms | **22.672270 ms** | **1.0870×** | 1.354055e-5 |
| 1 | 25.149060 ms | **22.893480 ms** | **1.0985×** | 1.556695e-5 |

旧BC64结果精确匹配已有SHA，7次配对warm、全部重复输出及输入SHA门禁通过。新KV折叠/在线更新次序改变，**不承诺与BC64位级相同**，没有事后容差。

[全部27层视觉](<results/vision01/vision101.json>)：BC128 **1.486846s**、merger对CPU相对RMS **0.728894%**；DeepStack分别0.139311%/0.262697%/0.456835%。原生注意力修补版同轮仍为1.380578s/0.724382%，因此这不是对原生注意力的无条件加速或默认升级。对R99准确流式1.582865s的改善约6.07%是**跨运行同机同口径**，不冒充同轮配对收益。完整encoder必须另测，见 [R102](<../round102/README.md>)。

## Phantom key 与失败保留

末KV块64真key+64phantom。**V补零不够**：phantom score在Half shift/Float exp之前填−FLT_MAX，exp后在Float分母之前显式归零。Float主scores/O/L、Float rescale/分母均保留；Half仅作为稳定shift估计与双Half概率乘积接口，最终一次Half输出，不夹断实际输入。

首次 [编译失败](<results/smoke01/run.log>) 的原因是把一个TIK单元素slice再次索引为64元素向量；未执行矩阵或更改参考。保留 [失败源](<flash_bc128/flash_vision101.py>)，新 [mask修复源](<flash_vision_mask_fixed101.py>) 直接从原始Float S基址寻址，不改变数学。

[实际H2/Q128/trueK192/pad256 smoke](<results/smoke02/bc128_smoke101.json>)：非零poison K/V尾与zero尾输出位级相同；constant-uniform在真正192 keys下与CPU Half **逐位一致**；随机对CPU相对RMS7.083577e-6，重复及输入SHA不变。65项作者stdlib测试、root寻址回归和实际证据测试均通过；所有新作者均fresh fork，未恢复inactive旧agent。

执行入口：[host engine](<flash_engine101.py>)、[真实叶子](<leaf_compare101.py>)、[完整视觉](<vision_probe101.py>)、[摘要](<summary101.json>)。作者 [源码说明](<flash_bc128/README.md>) 的“未编译”指作者自身，硬件执行及修复由root完成。

没有Float Linear权重教师、BF16 gold cast、隐藏conditioning缓存、假final norm或精度豁免。最终源/对象/worker审计和资源释放见 [R105最终审计](<../round105/README.md>)；整体目标仍active。
