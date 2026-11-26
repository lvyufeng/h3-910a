# R337 — encoder 计算构成实测（profile）+ AscendC 更正后的待验证清单

## 一、encoder 构成（只读 profile，第一趟 encode）

```
[PROFILE] encode 60.169 s
[PROFILE]   text_model   25.601 s  calls=1
[PROFILE]   vision       23.288 s  calls=1
```

⇒ 60.2 s 首轮 encode 中：**text_model 25.6 s + vision 23.3 s = 48.9 s 计算**，
其余 ≈11 s 为权重 H2D / 预处理。与 R334 的结论一致（常驻只省 0.39 s，因为搬运占比小）。

**对 encoder <2 s 的判定**：需要把 48.9 s 计算压到 <2 s（**24×**），这不是调度/常驻能解决的，
必须换 kernel 实现或降低计算量——而降低计算量（少 token / 改分辨率）属于有损，需你批准。

## 二、AscendC 更正后的待验证清单（R336 已撤回错误结论）

按 R336 的三步走，尚未执行（本会话上下文接近上限）：

1. **步骤 1**：L1→L0A 搬运验证（`LoadData2dParams` 的 `repeatTimes/srcStride` 单位、L1 的 NZ 摆放）；
2. **步骤 2**：16×16 最小 matmul，C 用 `DataCopyCO12DstParams`（nz2ndEn 两种）与手工 NZ 展开两种读法；
3. **步骤 3**：通过后接回 FA 内核重写（S1a-1 → S1 → S2）。

工程现状：`/tmp/h3_r221/faA/proj`（msopgen 工程、`ai_core-Ascend910A`）与 aclnn harness 仍在原处，
可以直接续做，无需重建。

## 三、状态

- 交付不变：**16.34 s/step（−8.9%）**，49 步 ×2 验证，relRMS 在原生带内，`H3_PERF_COMBO=1` 默认 OFF；
- 主机已清理（无我方进程）；远端 master 含本轮记录。
