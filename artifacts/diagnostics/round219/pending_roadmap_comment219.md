<!-- h3-910a-original-optimization-20261121:comment:R219-roadmap-gates-finish -->

## R219：R218裁padding候选的首/末步补精度证据，目标未完成

> 待回写草稿，未发布。须先核远端推送与Issue最新评论，替换EVIDENCE_COMMIT。

[完整结果](https://github.com/lvyufeng/h3-910a/blob/EVIDENCE_COMMIT/artifacts/diagnostics/round219/RESULTS.md)。

- Gate-only step0/48原版与候选相同真实QKV，新增800FA对/299.57亿FP16words零差异；与前step25合计1,200对/449.35亿words零差异。
- 8finite边界/极端值test同，candidate binary与R218一致。模型从未使用候选返回值，原TP2+USP4、50blocks、155strictchecks、FP32guards保留。
- 这是精度补充，不是candidate采用/组合/完整49step验证。R218已测单项−0.568s/step未重测，fullprediction variation仍待用户数值决定，默认OFF。
- 私有process/cache/tensor已清理，originalsources unchanged，ownedts-217released，权重/activation未下载客户端，VAE暂停，encoder/A2A3/≤10s仍未达标。
- 后续用户决定是否允许仅此项进入完整49step验证。可继续单项GEMM/布局候选，但未经用户决定不组合。

Project #5补证据完成后应回Awaiting approval，主路线图保持In Progress；发布并read-back后保存回执。
