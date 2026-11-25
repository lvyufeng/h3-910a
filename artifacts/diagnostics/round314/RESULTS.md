# R314 — encoder / VAE 目标：从既有收据得到实测基线（探针路线受阻）

## 一、实测基线（来自仓库既有收据，非新编）

| 阶段 | 实测 | 目标 | 差距 |
|---|---|---|---|
| **encoder** | `live_encoder_elapsed_s` = **108.5 / 109.4 / 109.9 / 110.3 s**（多条收据） | **< 2 s** | **≈55×** |
| **video VAE（warm）** | **20.565 s** | **< 20 s** | **+2.8%** |
| audio VAE（warm） | **0.210 s** | 记录 | — |
| cold combined VAE | 142.143 s（**不同口径，不得替代**） | — | — |

来源：encoder 数字见 `artifacts/` 内早期收据的 `live_encoder_elapsed_s` 字段；
VAE 三个数字见 `AGENTS.md:52`（"cold combined VAE142.143s versus warm video20.565s/audio0.210s"）。

## 二、stage 探针尝试（失败，如实记录）

为拿到**逐阶段**精确分解，写了 `stageprobe290.py`（wrap `encode_prompt` 与两个 `decode_latent`，
实测 `installed wrappers=3` 成功）并复用官方 `generate_full_NPU126.py` 启动。四次失败的根因链：

1. 输出目录固定 `/tmp/h3_budget149/NPU_full_generation126` 已存在（FileExistsError）；
2. 自写最小生成脚本 → `SetPrecisionMode ... 500001`（漏了 `torch.npu.matmul.allow_hf32=False`）；
3. 补上后又因缺少官方启动 patch 模块 → Orchestrator 初始化失败；
4. 复用官方脚本但改标签 → `integration_hooks126.py:20` 断言要求目录名前缀为
   `NPU_full_generation126`；改成合法前缀后再跑仍未通过（最后一次错误未取到）。

⇒ 探针本身可用，但**启动约束链路复杂**，本轮未拿到逐阶段新数据；上述基线来自既有收据，足够支撑决策。

## 三、两个并行目标的可执行候选（均不改数学）

### 1) encoder 109 s → <2 s：**关闭 encoder 的逐层 offload**

- R205 已查明：日志 `Enabled rank-local layerwise offload for encoder text_encoder (77 blocks across 2 stacks)`
  ⇒ encoder **每次 encode 都要流式载入 63 GB 权重**，这才是 109 s 的主因；
- 容量核算：63 GB / 8 卡（TP8）= **约 7.9 GB/卡**，而每卡 32 GB —— **放得下**；
- 候选：让 text encoder 常驻（只关 encoder 侧 offload，DiT 侧保持 layerwise），属部署配置改动，
  **不改任何数值语义**；
- 判据：encoder 阶段 < 2 s，且 155 条 Half 检查与输出位级不变。

### 2) video VAE 20.565 s → <20 s：**tiling / 并行参数微调**

- 现配置 `vae_use_tiling=True, vae_parallel_mode='tile', vae_patch_parallel_size=8`；
- 仅差 2.8%，候选：调整 tile 尺寸/并行粒度，使解码更贴合 8 卡；
- 判据：video VAE < 20 s，且输出与当前 warm 结果位级一致（tiling 仅改变切分，不改数学）。

## 四、状态

- 交付不变：**16.34 s/step（−8.9%）**，49 步 ×2 验证，relRMS 在原生带内，`H3_PERF_COMBO=1` 默认 OFF；
- 15 s：R313 路线级结论——本机 AscendC cube 不可用、TIK 已到 126.19 ms 下限 ⇒ 当前条件下不可达；
- 机器 ts-217 直连正常；远端 master 含本轮记录与探针源码。
