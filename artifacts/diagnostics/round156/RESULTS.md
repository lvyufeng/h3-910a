# R156 — 分配器假设**被否定**；并交叉验证了 R125 的融合 MLP 结论

## 为什么要测

R155：MLP 里 `fc1→fc2` 之间是可加的激活区 **34.7 ms/层**，而 standalone 同一套算子只要 3.24 ms。
同时 R125 早已实测 **fused SiluAndMul 融合**：step 18.83 s vs 18.41 s（噪声内），数值更差。
**把 3 个 kernel 合成 1 个却毫无收益**——只可能意味着该区间**在等，不在算**。

最便宜的判别：读 NPU 分配器自己的计数器。

## 实测（step 18.092 s，8 rank，150 次调用）

`mlp` 中位 **88.059 ms**，`silu` 中位 **1.14 ms**（与 R155 一致）。

| 计数器 | 值 |
|---|---|
| `num_alloc_retries` | **0** |
| `num_ooms` | **0** |
| `reserved_bytes` 首调 → 末调 | 5.77 → 11.64 GiB（+5.87 GiB） |

**分配器假设 REJECTED**：无重试、无 OOM，`reserved` 增长只是缓存正常增长。

## 但发现更大的疑点

单次 mlp 调用内分配 **8.51 GiB**、释放 8.30 GiB，
远高于可见的 ~1.2 GiB 临时量。block 内真实分配 churn 比看上去大得多。

## 剩余两个假设

1. 与异步 H2D 权重预取在 copy stream 上的**带宽竞争**；
2. **我的 E2 位于 harness 给 fc2 注册的 pre-hook 之后**，该区间可能**包含了 fc2 权重装载等待**。

第 2 条若成立，那"激活区 34.7 ms"这个名字就是错的，它其实是"fc2 权重等待"。

## 下一步

先读 offloader 与 dispatcher 的注册顺序（不用跑机器），确认 fc2 pre-hook 与 copy stream 的先后，
再决定是否需要 kernel 级 profile。

## 限制

计数器只反映分配行为，不能归因时间；区间边界依赖 hook 顺序。
