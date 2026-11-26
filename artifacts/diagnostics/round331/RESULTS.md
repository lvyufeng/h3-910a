# R331 — encoder 常驻（代码级）实测：内存不可行，两次触发主机内存耗尽

## 一、用户授权

用户 2026-11-26：「**1 允许代码级优化**」⇒ 获准修改 `vllm_omni` 代码（此前仅 AscendC 自动授权）。

## 二、方案（不改数学）

`resident_patch.py`：在 `MiniMaxH3Pipeline.encode_prompt` 首次调用前，把 encoder 的
逐层 offload 关掉并**常驻**：

1. `_find_offloaded(self)` 深度优先找到带 `_omni_layerwise_enabled` 的子模块 —— **实测 found=True**；
2. 尝试 `layerwise_backend.disable_plan_encoder_layerwise_offload(module, restore_weights=True)`
   （把所有 block 恢复到 device 并移除 hook）。

## 三、两次尝试与结果

| 尝试 | 保护措施 | 结果 |
|---|---|---|
| 1（R330） | 单次 restore + 全局阈值 200 GiB | **看门狗正确拦截**：`free_before=87.4 GiB < 200`，8 rank 全部 ABORT，保持 offload |
| 2（本轮） | 逐 block `restore_next_block()`，每 8 块检查一次，<25 GiB 即中止 | **未及时拦截**：主机内存被迅速吃满，ssh 两次失联（每次约 1–2 分钟），**已 kill 全部进程** |

## 四、机制判断（重要）

- engine init 完成后 host 仅剩 **87.4 GiB**；restore 期间 host 可用内存继续下降直至耗尽；
- 这说明 **offload 设计下每个 rank 都持有可观 host 侧权重副本**，把 77 个 block 一次性恢复到
  device 会额外放大 host 占用（若每 rank 恢复全量 encoder，8×(63 GB) 必然 OOM）；
- ⇒ **在当前 offload 架构下，encoder 常驻不可行**，不是调参问题，是内存模型问题。

## 五、结论

**encoder <2 s 在不再改架构的前提下不可达**：
warm 3.196 s ≈ 63 GB / 3.196 s ≈ **19.7 GB/s**，与 H2D 聚合带宽吻合 ⇒ 瓶颈是**权重搬运**；
唯一解是让权重常驻 device，而这需要**重新设计 offload（按 rank 分片常驻 + 显式释放 host 副本）**，
属于架构级改造，风险与工作量都显著高于本轮授权范围。

## 六、当前环境状态

- 全部我方进程已 kill（`procs=0`）；
- 主机仍有 665 GB 被占用（无我方进程）——疑似被 OOM 打断后未及时回收的共享/固定内存，
  **建议用户重启 ts-217** 后再做任何显存/内存敏感实验；
- 交付不变：**16.34 s/step（−8.9%）**，`H3_PERF_COMBO=1` 默认 OFF。
