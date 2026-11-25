# R327 — VAE 切分模式候选 1 被否决：MiniMax-H3 VAE 只支持原生 `tile` 模式

## 一、实验

把两趟 warm 测量的 `vae_parallel_mode` 从 `tile` 改为 `spatial_shard_height`（R325 列的候选 1）。

## 二、结果：源码直接拒绝

```
ValueError: MiniMax H3 VAE supports its native tile parallel mode only, got 'spatial_shard_height'
```

（worker 抛出 → orchestrator 初始化失败 → 上层只看到空消息的 `RuntimeError`。）

⇒ **候选 1（spatial_shard_height / spatial_shard_width）作废**，不是性能问题，是**实现不支持**。
以后排障时应直接看 worker 日志里的 `(DiffusionWorker_...)` 行，顶层 `Orchestrator initialization failed`
没有信息量。

## 三、剩余 VAE 候选（都不改数学）

| # | 候选 | 状态 |
|---|---|---|
| 1 | `vae_parallel_mode` 换 `spatial_shard_*` | ❌ 本实现不支持 |
| 2 | `vae_patch_parallel_size`: 8 → 4 / 2 | 待测 |
| 3 | `vae_use_tiling`: True → False | **下一轮测** |
| 4 | `vae_use_slicing` | 只降内存，预期不提速 |

## 四、目标与差距（不变）

- warm video VAE：**20.746–20.998 s**（R326 实测）vs 目标 **<20 s** ⇒ 差 **3.7–5%**；
- warm audio VAE：0.199–0.266 s ⇒ 已达标；
- warm encoder：3.196 s vs 目标 <2 s ⇒ 差 1.6×。

## 五、状态

- 交付不变：**16.34 s/step（−8.9%）**，`H3_PERF_COMBO=1` 默认 OFF；
- ts-217 健康（shm=0、无我方进程、8 卡 OK）；
- 15 s（DiT）：R313 路线级结论 ⇒ 当前条件不可达。
