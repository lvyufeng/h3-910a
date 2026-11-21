---
name: H3优化实验
about: 同输入、同口径、可回滚的原MiniMax-H3／910A性能实验
title: '[组件／优先级] '
labels: h3-roadmap, optimization, precision-gate
---

## 主Issue与Project

- 主路线图：#1
- Project：从主Issue链接进入。
- 优先级／组件／阶段：
- 前置Issue：请同时添加GitHub原生blocked-by依赖。
- 执行人／资源／预算：

## 假设与作用范围

- 预期优化哪项实际瓶颈？
- 是cold、warm、reload，还是whole-generation？
- 不改变哪些数学／参数／范围检查？

## 不可变实验输入

- 原checkpoint／revision／weight SHA：
- 源码／config／kernel SHA：
- prompt／seed／sigma／初始noise／conditioning：
- 主机、卡型、physical IDs、TP／USP／encoder TP：
- baseline为R124＋FA；R1264384e89只是执行证据，不是全局新精度接受。

## 实验步骤

- [ ] 开始前读主Issue、当前Issue、原生依赖、最新评论与Project字段。
- [ ] claim并评论合同；先live claim owned机器，不默认借CI／dormant。
- [ ] cold与warm分开，baseline→candidate→restored；clean计时不逐module同步。
- [ ] 核验实际kernel／copy stream／collective／TP shards，不只记录开关。
- [ ] 同输入叶子和完整prequant输出逐位比较；仅媒体PSNR不是精度证明。
- [ ] 数值变化先恢复／停用，报数字并等待用户；noise band不是许可。
- [ ] 记录失败／回滚／源SHA／内存／清理／lease释放，关联commit或PR。
- [ ] 回写Project与主Issue，再结束会话。

## 不得越过的门禁

- [ ] TP持续开启：baselineDiT TP2＋USP4／encoderTP8，8卡；新拓扑必须核验真实分片。
- [ ] 原50块／49真前向、FP16矩阵与原FP32例外、155strict checks不减。
- [ ] 不BF16／HF32／clip、不teacher隐藏替换、不student／LoRA／少步数。
- [ ] 不下载／中转本地weights或activation，不改共享SDK，不做device reset。

## 结果与退出

- actual scope、rank-max、cold／warm／reload、内存：
- actual operator／stream命中与numeric结果：
- 基线／候选／恢复基线：
- **Keep／Reject／Awaiting approval**：
- 证据、commit／PR、失败回执、清理：
- 下一步／阻塞恢复条件：

完成实验不等于精度／性能目标达标；Reject可关闭实验但不得合并。
