<!-- h3-910a-original-optimization-20261121:comment:R219-supplemental-gates-finish -->

## R219 补充精度门：step0/48与8finite边界测试通过，仍未采用

> 本文件为待GitHub回写正文，不表示评论已发布。证据提交 `EVIDENCE_COMMIT` 的推送未获确认；必须先回读远端提交后再发布。

- R218候选仍是唯一padding-task裁剪：GM storage/stride、Br128/Bc304、原mask/FP32accum/自动同步、所有真实tokenpair保留，仅Tr342→327/Tc144→138。默认OFF，未组合。
- R219只做gate-only模型前向，永远返回original FA结果（candidate结果使用次数0）。Step0/48各8rank×50=400对，合计新增800真实QKV对/29,956,505,600FP16word零差异。
- 与R218 step25合计1,200对/44,934,758,400FP16word零差异；不是完整49step轨迹证明。全部原50blocks/155strictchecks/TP2+USP4保留。
- 相同candidate源码52365cbf…重编译得到相同binarySHA052c49f2…；nativeSHA937671ef…与原源码/weight/snapshot pins复核未变。
- 8finite边界test：signedzero、subnormal、finite-max invalidKV、skewed负分数 ×validkeys1792/1791，14,680,064word零差异，nonfinite/saturation=0。没有注入nonfinite输入；这些测试不声称覆盖所有IEEE输入。
- 无新增candidate步时。本轮没有返回candidate model结果。之前R218单项实测仍是17.944→17.376s/step（−0.568s）、恢复17.977s；不与其他候选相减相加为组合达标。
- R218 fullprediction variation仍公开；native noise不是允差。候选默认OFF、Awaiting user precision/adoption decision；完整49step/media/liveencoder scope未验，10sgoal不完成。
- model/test process exit0，原版恢复，privatecandidatecache/package/tensoroutputs清掉，ownedts-217released=true；无CI/dormant、SDK/driver reset或其他进程操作，无weights/activation客户端传输。

证据入口（仅推送确认后替换EVIDENCE_COMMIT并发布）：
- [R219完整结果](https://github.com/lvyufeng/h3-910a/blob/EVIDENCE_COMMIT/artifacts/diagnostics/round219/RESULTS.md)
- [实际step0/48门禁](https://github.com/lvyufeng/h3-910a/blob/EVIDENCE_COMMIT/artifacts/diagnostics/round219/gate_summary219.json)
- [finite边界门](https://github.com/lvyufeng/h3-910a/blob/EVIDENCE_COMMIT/artifacts/diagnostics/round219/edge_summary219.json)
- [cleanup](https://github.com/lvyufeng/h3-910a/blob/EVIDENCE_COMMIT/artifacts/diagnostics/round219/cleanup219.json)

用户待决：是否接受这些相同真实输入FA逐位门禁作为该候选的局部判据，并允许**仅此项**进入完整49step候选验证（不是默认采用，也不是与阶段A/guard组合）。未收到决定前不晋升。
