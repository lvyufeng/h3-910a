## R134局部候选试验：仅获授权测试，非采用新精度基线

R133报告原finalAdaLN sameHalfinput/w/b有一词漂移后，用户明确选择“仅做局部候选试验(Recommended)”。
候选是同Half输入/权重，显式torch.mm FP32out＋原Halfbias精确扩展到FP32相加→一次Half回写。
改变rounding/bias路径已披露；本轮不授权采用/新误差阈值/全50回归/49生成/copycandidate。
#15继续InProgress、#4Blocked，executor lvyufeng/lead。ownedts217新lease至16:00+08，extraGPU立即release。

- 原time＋finalAdaLN小组件、TP2+USP4/world8；同actualR132t_embSHA/原FP16paramsref/fixtureSHA门。
  NativeConfigfactory/setcurrent、原F32time guards、原SiLU/Column gather不改。
- Original→candidate→restored三case，各32Module(withTPgather)＋128fixedlocal(noTPgather)，
  全部privateTrue仅用于sameinput诊断；结束False/warn原值。无full50/155新计数。
- 候选只finalAdaLN oneLayer apply，矩阵输入与weight始终Half，bias仍原Halfparameter；
  无BF16/clip/FP32weight改写或近似cache。actualcandidate调用160/rank，dtype/shapecontract记实。
- 用全uint16词比较selfrepeat/candidate-v-originalfirst/restored，两种reference分开。
  原native噪声不是允许候选误差，稳定也不自动adopt。参数before/after完整fingerprints存，
  cloneinput逐词对原输入，实际每moduleinput也核全词；descriptor恢复与outputbit门分开。
- 函数/候选返回Half，每case原Module其他逻辑保持，timings含明确sync+D2H不当cleanmodel性能。
- source/helper沿R133原组件复用，cohort原fatal/pidfd+UID/marker/startticks/20min totaldeadline。
  所有targets仅ownprivateprocess，noSDK/driverreset/CI/dormant；weights/inputs/activation留host。
- 先source/mockunit与readonlyreview/commit/pin，随后真实8rank数字写回。mock不证明数值或稳定。
  候选执行异常停止全privatecohort、精确恢复或报告unknown，不重复未知launch隐藏失败。
