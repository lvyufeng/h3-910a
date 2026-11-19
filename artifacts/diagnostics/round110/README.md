# Round110 — 第43层同输入诊断：0.0271%→0.00687%，不是4%全图缺口的单算子爆炸

使用 [R109保存的原始CPU FP16 raw42](<../round109/README.md>)作为block43精确输入，原始CPU raw43作输出参考。只加载这一层官方11份原始Half参数，加载前逐份校验转换tensor SHA；实际TP8输入/权重不变，没有BF16 gold、Float权重教师、假final norm或DeepStack注入。

## 全8rank同输入结果

| NPU block43路径 | 相同CPU raw42输入后，对原始CPU raw43相对RMS |
|---|---:|
| 原生attention/row | **0.0270854%** |
| 高精度attention/原生row | **0.00687125%** |

所有8rank各案例输出SHA相同、重复逐位相同、原生恢复、11原始Half参数和CPU input/gold SHA不变。它不是完整50层encoder计时，也不是把CPU预计算条件塞进stock入口。

这与 [R93原始CPU Half input敏感性](<../round93/README.md>)的0.253%→17.16%、layer43附近15.8×放大是**不同实验**：R93故意传入已改变的早层输入，R110输入与原始CPU逐位一致。R110不支持“第43层NPU单算子自身造成百分之十几误差”的说法；也不能把两个RMS百分比相减当作NPU贡献分解。

完整 [R108](<../round108/README.md>) 仍有4.395% gap。虽然同输入43高精度attention更准，完整VisionTP8+高精度text却5.125%更差，数值传播仍非单调，不据局部结果升级默认或放宽验收。

[摘要](<summary110.json>)、[rank0](<results/block43_01/block43_rank0.json>)/[rank1](<results/block43_01/block43_rank1.json>)/[rank2](<results/block43_01/block43_rank2.json>)/[rank3](<results/block43_01/block43_rank3.json>)/[rank4](<results/block43_01/block43_rank4.json>)/[rank5](<results/block43_01/block43_rank5.json>)/[rank6](<results/block43_01/block43_rank6.json>)/[rank7](<results/block43_01/block43_rank7.json>)；执行 [11-weight-gated驱动](<block43_verified_weights110.py>) / [四对象prebuild](<prebuild110.py>)。

## 实际第43层same-operand leaves（rank0）

| 叶子 | 对CPU Half相对RMS |
|---|---:|
| 原始两舍入点RMSNorm | 3.46322e-6 |
| 本地gate / up Linear | 1.29834e-5 / 1.30992e-5 |
| 原生Half SiLU | **9.51329e-7** |
| SiLU×up | 1.30916e-6 |
| 本地down Linear（不含HCCL） | 1.14833e-5 |

Float SiLU→Half与原生NPU SiLU仍逐位相同，和 [R86第25层](<../round86/README.md>)的排除项一致；不给已有相同结果增加“更准Float激活”路径。没有证据把SiLU当作剩余4%缺口的主要来源。叶helper [原R86源码](<../round86/leaf/late_leaf86.py>)复用不改，回执保留其历史round标签，但实际layer index由root43指定。

## 最终审计/资源

[最终只读审计](<final_source_verification110.json>)核查 **23 production源 + 13 installed API源 + 230份执行源映射 + 38个实际对象**，原始HF/CPU FP16 gold/frozenQKV/launcher未变，R108五份真实Half输出及R109原始CPU42/43 file/tensor SHA验证，全部相关worker0。全部相关计算和只读备份job已收集/核验，本次owned ts-217在计算结束且worker0后已释放；之后不再进行远程计算或root清理，CPU42/43本地镜像也已完成并逐份SHA验证，见 [捕获身份](<../round109/captured_outputs109.json>)。未请求CI/dormant，其他会话CUDA租约未动。

[审计程序](<verify_final110.py>)、[新source pins](<expected_execution_sources110.json>)及 [实际gate测试](<test_actual110.py>)保留。R109首次环境失败、R108 metadata whitelist失败均保留，修复不改oracle/已执行source或原始数据，不盲重播模型。

[独立fresh-fork复核](<../round108/review_findings108.json>)未发现P0/P1：支持真实<2s测量、数值不验收、共驻仍未知；local输出备份P2已逐份SHA完成，见 [follow-up](<../round108/review_followup108.json>)。

## 下一步

优先从真实下游condition_proj/refiner与视觉norm/position/merger的same-operand原始CPU FP16控制继续定位；性能<2s已经得到，不再追逐静态tile/Float广播计数或重复已排除的SiLU。具体约束见 [NEXT](<NEXT.md>)。整体goal17仍active：完整CPU FP16精度、真正encoder/DiT/VAE共驻和匹配官方工况未完成。
