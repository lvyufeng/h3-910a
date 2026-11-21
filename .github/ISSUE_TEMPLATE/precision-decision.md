---
name: 精度／算法变更决策
about: 数值差异或近似加速必须先报告，再由用户决定
title: '[等待批准] '
labels: h3-roadmap, precision-gate, awaiting-user-decision
---

主路线图：#1；关联候选Issue：

## 当前状态

候选已停用并恢复原基线：
Project状态为 **Awaiting approval**。这个Issue的存在不构成compute或精度变更授权。

## 变更与收益假设

- 数学／归约顺序／dtype／tile窗口／步数／checkpoint具体改了什么：
- 这是实测数值变化，还是尚未执行的可选方案：
- 可复现速度／内存收益，或仅理论预期：
- 为什么严格原计算路线不够／是否还有替代方案：

## 实测精度证据

- 同输入baseline→candidate→restored source／config／input／weight SHA：
- 叶子、完整prequantized tensor、latent差异：
- video／native audio完整结果，绝对／相对误差、未经增益／修饰的媒体：
- native重复噪声另报，不能自动当作candidate允许误差：
- encoded PSNR不是tensor或音频等价证明：

## 请求用户决定

- [ ] 保持原R124＋FA精度，Reject候选。
- [ ] 只批准独立实验，不批准共享基线晋升。
- [ ] 明确接受这个候选的指定数值／算法变化后再晋升。

有效决定必须引用用户原话／时间／范围；不得由代理预先勾选或把未回答当批准。
近似cache、稀疏／量化、少步数、student／LoRA等须逐项许可，不接受“已有噪声所以无损”。
