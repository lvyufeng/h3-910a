# R155 — MLP 的可加分解：激活区确实是 34.7 ms/层，且与 standalone 差 10×

R153b 用 `median(mlp) − median(fc1) − median(fc2)` 做归因，R154 证明那不是可加分解。本轮改用**真正的父子 span 对**：
每个 mlp 调用在同一流上按程序序记录 5 个事件 `E0 mlp-pre → E1 fc1-post → E2 fc2-pre → E3 fc2-post → E4 mlp-post`，
这些区间**按构造即可加**。

## 实测（step 17.902 s，8 rank 完成，150 次调用）

| 区间 | 中位 |
|---|---|
| `fc1`（E0→E1） | 23.274 ms |
| **激活区（E1→E2 = chunk + silu + mul + dispatch）** | **34.717 ms** |
| `fc2`（E2→E3） | 29.999 ms |
| tail（E3→E4） | 0.006 ms |
| **合计** | **88.217 ms** |

**闭合检查**：分项之和 87.996 ms vs 合计 88.217 ms，误差 **0.221 ms**；
逐次求和 median = 88.217 ms（与合计完全一致）。**这次是可加分解，可信。**

每步折算：激活区 **1.736 s**，mlp 合计 4.411 s。

## 关键矛盾

- 模型内激活区：**34.717 ms/层**
- 其中 `silu`：仅 **0.575 ms**
- R154 standalone 同一套算子：**3.24 ms**

即 **mul + chunk + dispatch ≈ 34 ms**，比 standalone 整套慢约 **10×**。
由于本轮已证明该区间是真实时间（可加、闭合、逐次一致），这个差距**不是算术，是环境**：
模型内每层要新分配约 900 MB（600 MB 的 fc1 输出 + 两个 300 MB 临时量），
而 offloader 与其 staging 缓冲正在活动；microbench 在安静进程里跑。

## 下一步

用 **kernel 级 profile**（msprof op summary）看该区间到底是**在算**还是**在等**。
这是唯一能把"分配器/流等待"与"算子慢"分开的手段。

## 本轮失败尝试

`span155a` 失败：新目录没有拷贝 harness 模块（`ModuleNotFoundError: benchmark_actual_attention126`）。
已补齐依赖并用新 label `span155b` 重跑，回执保留。

## 限制

- 未提出候选、未改模型、未装东西；
- 分配器/流行为尚未被直接观测，本轮只证明差距存在且可加。
