# R323 — 阶段计时全套落地：encoder 62.95 s；cold video VAE 80.2 s / audio VAE 59–69 s（与 142.143 s 收据互相印证）

## 一、本轮方法

官方 `generate_full_NPU126.py`（单进程 entry）+ `sitecustomize` 惰性探针（PYTHONPATH 最前 + chain-load
官方那份），默认配置（**不做任何数值/配置改动**），8 卡 TP2+USP4、encoder TP8、49 步、seed 1234、
768×1344、120 帧。

## 二、实测（全部含 `torch.npu.synchronize()` 前后同步）

| 阶段 | 实测 | 分片一致性 |
|---|---|---|
| `encode_prompt`（encoder 方法级） | **62.95 s** | 8 个 TP8 worker：62.950–62.953 s |
| `decode_latent` **MiniMaxH3VideoVAE**（cold） | **80.12–80.32 s** | 8 个 rank 一致 |
| `decode_latent` **MiniMaxH3AudioVAE**（cold） | **59.38 / 60.29 / 60.69 / 69.48 s** | 4 条记录 |

整轮：`engine_initialization_s = 216.0`、`generation_wall_s = 1157.5`（denoise 稳态 **17.03 s/it**）。

## 三、与既有收据的交叉验证（重要）

- 既有：`cold combined VAE 142.143 s`；本轮 cold 实测 video 80.2 + audio 59.4–69.5 = **139.6–149.7 s**
  ⇒ **两份独立测量互相印证**，口径自洽；
- 既有：`warm video 20.565 s / audio 0.210 s` —— 这是**已热**（权重常驻、非首次 decode）的口径；
  本轮是**单次生成内的首次 decode（cold）**，两者**不可互换**。

## 四、对照目标

| 目标 | 现值 | 差距 | 备注 |
|---|---|---|---|
| encoder **< 2 s** | 62.95 s（方法级）/ 108.5–110.3 s（更宽口径） | **31–55×** | 主因：`layerwise offload … 77 blocks` 每次搬 63 GB；TP8 常驻仅 ~7.9 GB/卡（R322 有分步安全方案） |
| video VAE **< 20 s**（warm） | 20.565 s（既有 warm）／本轮 cold 80.2 s | **+2.8%** | 只需 warm 路径再快 3%；候选：tiling/并行粒度 |

## 五、本轮一个已知瑕疵（如实记录）

生成本体跑完、VAE 也完成，但末尾断言失败：

```
assert len(counts)==8 and set(counts.values())=={49}   counts={}
```

即 `worker_alignment_pid*126.jsonl` 里没有 `dit_step_end` 审计事件（审计钩子未在本启动路径生效）。

- 这是**测量轮**、不是验收轮，`quality_accepted` 因此为 False；
- 影响：本轮不能用来做数值验收；但**阶段计时与 wall 数据仍然有效**（探针独立于审计）；
- 后续若要做验收，需要恢复审计路径（官方 launcher 里对应的 env/flags）。

## 六、下一步

1. **warm 口径复测**：同一 engine 连续两次 `generate`，第二次 decode 即 warm —— 直接对照 <20 s 目标；
2. warm 若仍 20.5 s，调 `vae` 的 tile 尺寸/并行粒度（不改数学），逐次报你批准；
3. encoder 常驻按 R322 的**三道门**分步验证（门 1 单 rank 容量 → 门 2 稳态 → 门 3 带内存看门狗的 8 rank）。

## 七、状态

- 交付不变：**16.34 s/step（−8.9%）**，49 步 ×2 验证，relRMS 在原生带内，`H3_PERF_COMBO=1` 默认 OFF；
- ts-217 健康（R320 事故后已自行恢复，OOM killer 清理，我方无残留进程）；
- 15 s：R313 路线级结论 ⇒ 当前条件不可达。
