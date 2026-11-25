# R281 — encoder / VAE 并行目标：现有 harness **无法度量**，需专用探针

## 一、核对结果

`/tmp/h3_budget149/benchmark_actual_attention126.py`：

- 唯一的计时点是 DiT 步循环（L383–393）：`perf_counter` → `all_reduce(MAX)` → `rank_max_elapsed_s`；
- **没有任何 encoder 或 VAE 的阶段计时**；
- CLI 只暴露 `--checkpoint/--mode/--label/--tp/--usp/--resident/--audit*/--offload/--steps/--warmup/--repeats`，
  **没有分段计时开关**。

⇒ 参考合同里的 `encoder<2 s`、`video VAE<20 s`（冷 142.143 s / 暖 20.565 s + 0.210 s）
**不能用现有 harness 重新测量**，必须新写探针。

## 二、探针需要的最小内容

| 目标 | 探针做法 | 输出 |
|---|---|---|
| encoder<2 s | 包 `text_encoder` 前向（或对应模型入口）计时，单次 + 重复 3 次取中位 | ms、rank_max |
| video VAE<20 s | 包 video VAE decode 计时（区分冷/暖：首调用 vs 后续） | 冷/暖 ms |
| 口径 | 与 R126 一致：`perf_counter` + 8 rank `all_reduce(MAX)`，避免单 rank 偏差 | — |

## 三、为什么现在值得做

1. 这两个是合同里的**并行目标**，之前从未有本机 910A 数据；
2. 探针成本低（只加计时，不改数学、不改模型），可复用现有 8 rank 启动方式；
3. 拿到现状后才能判断差距（例如 encoder 若已 <2 s 则该项可关）。

## 四、下一步（R282）

1. 在 `sw260` 驱动的副本上加 encoder/VAE 计时探针（**进程内包装**，不改部署源码）；
2. 先跑一次 baseline（默认配置，不带性能开关），记录 encoder 与 video VAE 的冷/暖 ms；
3. 结果按合同口径回报 Issue #10（encoder）与 #7（VAE）。

## 五、合规

仅读码核对；未改任何文件；主机无残留进程；部署路径未改。
