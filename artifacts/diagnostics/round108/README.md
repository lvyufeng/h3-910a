# Round108 — 完整TP8 encoder首次稳定<2s：1.321s，CPU FP16 gap仍4.40%

在同一owned8×原始910A上，**真正Vision block TP8 + 原始text TP8** 完成真实 `encode_ids`：实际两图、27 vision blocks、3路DeepStack、全部50 text layers、HCCL、layout/packing和最终D2H全部计入；没有CPU预计算条件代替视觉，没有隐藏conditioning缓存。

## 真实同轮完整encoder

每配置5次stock调用，每次取8 rank最大时间，再取中位数。

| 配置 | 完整encoder中位数 | 最终selected50对原始CPU FP16相对RMS |
|---|---:|---:|
| 原生视觉/文本 | 2.119597s | 7.420123% |
| patch/bias/true-erf修补、原生视觉attention/文本 | 2.013348s | 17.037422% |
| 修补、replicated BC128视觉attention、原生文本 | **2.156184s** | 7.405683% |
| 修补、真正VisionTP8/本地BC128、原生文本 | **1.321200s** | **4.395021%** |
| 同VisionTP8、高精度文本attention | **1.860187s** | 5.125418% |

**性能项已真实演示<2s**：VisionTP8/原生文本5次最大rank时间为 **1.311685 / 1.243577 / 1.322445 / 1.364717 / 1.321200s**，全部<2；高精度文本组合5次也全部<2。不是只看单rank、只看视觉0.675s、CPU预计算或最小单次。对同轮replicated BC128中位数约快 **1.632×**，完整耗时降低38.73%。

**完整CPU FP16精度仍未对齐**：最终4.395%仍明显，高精度文本attention反而5.125%且更慢。保留负结果、不自行放宽容差、不默认部署为lossless、不标记整体goal完成。

[可重算摘要](<summary108.json>)、[聚合器](<collect108.py>)、[rank0](<results/full01/tp8_rank0.json>)/[rank1](<results/full01/tp8_rank1.json>)/[rank2](<results/full01/tp8_rank2.json>)/[rank3](<results/full01/tp8_rank3.json>)/[rank4](<results/full01/tp8_rank4.json>)/[rank5](<results/full01/tp8_rank5.json>)/[rank6](<results/full01/tp8_rank6.json>)/[rank7](<results/full01/tp8_rank7.json>)。同轮修补原生对照有一次1.994422s，但中位2.013348s，不能据一次快样本替代真实TP8收益。

## 数值/结构门禁

- 每rank全部27vision/DS/merger与 [R107实际输出身份](<expected_vision107.json>)精确一致；全8rank的selected50、全部50text边界SHA一致。
- 每配置stock/equivalent与重复逐位相同，最后原生恢复。原参数、fixture、原始CPU FP16 gold未变；所有Linear weights仍Half，无Float权重教师、BF gold cast、假norm。
- TP8每rank2 heads/538真MLP通道，Float row部分和在HCCL前保持Float并在bias后一次Half；规范/patch/merger仍replicated，模型语义与checkpoint结构不改。
- 原生几何缓存只缓存positions/rotary，没有pixel或hidden状态。CPU-conditioning-only控制原生2.4757%、高精度attention1.8193%继续单独记录，绝不混入stock计时。
- [18对象串行prebuild](<results/full01/prebuild108.json>)在8rank model priming前验证旧13对象+2-head attention+4个TP GEMM，不并发编译。执行 [完整驱动](<full_encoder108.py>)、[prebuild](<prebuild108.py>)、[实际证据测试](<test_actual108.py>)、[入口契约](<test_full108.py>)均保留。

## 已验证内存，不冒充共驻留

rank0末allocated约9.263GB，诊断峰值12.733GB、reserved12.778GB。包含所有baseline与proof capture，不是service稳态；TP布局额外Half权重 **127,401,984B/rank**、Float bias124,416B、Float权重shadow0。**尚未实际encoder/DiT/VAE共驻**，不能仅凭空余HBM数字宣告兼容或用它代替真正运行。

## 只读审计与后续精度

[grid-aware审计](<final_source_verification_grid_fixed108.json>)核查23production、13installed API、60执行源、34实际对象及原始HF/CPU gold/frozenQKV/launcher，5份实际输出身份，worker0。第一次审计旧白名单只允许1/4/30核，遗漏合法小M128、N640/1152的5/9核；核验真实plan后新 [审计源](<verify_final_grid_fixed108.py>)按 `block_dim==grid.cores<=30` 验证，不修改任何已执行source/binary/gold/数值。失败记录见 [grid audit失败](<audit_failure108.json>)，原 [失败审计源](<verify_final108.py>)保留。

rank0真实selected50输出只在第一次forward后、所有warm前另存，没有加载回stock输入。之后 [R109原始CPU raw42/43捕获](<../round109/STATE.json>) 与 [R110 exact-input block43](<../round110/STATE.json>)用于定位剩余精度；CPU任务只在本轮NPUtimers结束并收集后启动，避免争用污染benchmark。

持久目标保持active：CPU FP16完整精度、真正共驻留、官方匹配工况仍未完成。DiT≈14.657s/step和VAE≈13.5s是此前已测性能，不因本轮encoder<2就自动解除外部conditioning质量/共驻资格。
