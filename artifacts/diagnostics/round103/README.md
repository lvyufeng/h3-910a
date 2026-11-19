# Round103 — MB32/BC256 编译与精度结构通过，但真实叶子略慢

这是保留的负结果，不是默认优化。相对 [R101 BC128](<../round101/README.md>)，MB64/BC128→**MB32/BC256**：KV循环32→16，query tiles63→126，总task1008→2016，30核、68步。Float主scores/O/L、真Float分母与双Half概率PV保持；M=N4032、K/Vpad4096，最后192真keys+64phantom。

## 实际同轮结果

[真实Half QKV、原始CPU Half MATH、7次配对warm](<results/leaf01/leaf_compare103.json>)：

| 图 | BC128 | MB32/BC256 | BC128/BC256速度比 | BC256对CPU相对RMS |
|---|---:|---:|---:|---:|
| 0 | **22.788680 ms** | 23.025960 ms | **0.9897×** | 1.478438e-5 |
| 1 | **22.505170 ms** | 22.620390 ms | **0.9949×** | 1.583094e-5 |

两图都慢约0.5–1.0%，两图的CPU误差也略高；旧BC128叶子输出SHA、严格4D形状、输入及重复SHA通过。减少scalar广播/O+L更新并不等于加速：总K/V reload约翻倍，真实测量不支持这条改动。因此没有执行 [预备的R104完整encoder](<../round104/STATE.json>)，没有把更少循环或源码设计包装成速度收益。

[真实H2/Q64/trueK192/pad256 smoke](<results/smoke01/bc256_smoke103.json>) 编译运行exit0：poison K/V尾与zero尾输出逐位相同，constant-uniform与CPU Half逐位相同，随机相对RMS7.918091e-6，重复及输入不变。实际确认了256-row DMA、32-row Cube、32-lane vdup、2块Half-word transpose等原始910能力，而非假设A2。

## 容量和代码身份

UB144832B、L0C53248B、L1119808B；各Cube操作数≤64KiB。Float分母求完后原位复用S来生成P_low，没有额外Float低残差缓冲、Half累计状态或Float权重教师。phantom score在shift/exp前−FLT_MAX、exp后分母前零，V补零本身不够。

64项作者stdlib结构测试和 [root实际证据测试](<test_actual103.py>)通过。源码：[内核](<flash_bc256/flash_vision103.py>)、[host engine](<flash_engine103.py>)、[真实叶子驱动](<leaf_compare103.py>)；所有新增作者是fresh fork，未恢复inactive旧agent。作者 [source-only说明](<flash_bc256/README.md>) 不表示root未执行；硬件数字只来自上述实际回执。

后续 [R105逐字Float广播](<../round105/README.md>) 同样实测为负的速度结果，但证明了原始910没有Float transpose时仍可做精确32位广播。最终源/对象/worker审计与owned释放在那里记录。无精度豁免、BF16 gold cast、隐藏conditioning或整体完成声明，目标保持active。
