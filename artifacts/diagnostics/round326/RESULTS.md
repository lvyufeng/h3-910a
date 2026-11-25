# R326 — 决定性发现：阶段耗时是**冷启动假象**——warm encoder 仅 3.196 s、warm audio VAE 0.20 s

## 一、实验

`warmvae324.py`：**一个 engine 内连跑两趟** `omni.generate`（同 seed/参数），探针分别记录每趟的
`encode_prompt` 与 `decode_latent`。pass1 = cold（首次），pass2 = warm（权重已在 HBM）。

## 二、结果（全部含 NPU 同步）

| 阶段 | **cold（pass1）** | **warm（pass2）** | 既有收据 | 一致性 |
|---|---|---|---|---|
| `encode_prompt` | **62.15 / 62.95 s** | **3.196 / 3.197 s** | — | — |
| `MiniMaxH3VideoVAE.decode_latent` | **80.12–80.32 s** | **20.746–20.998 s** | warm 20.565 s | ✅ 差 ≤2% |
| `MiniMaxH3AudioVAE.decode_latent` | **56.0–64.7 s** | **0.199–0.266 s** | warm 0.210 s | ✅ 完全吻合 |

整轮：engine_init **327.9 s**；pass1 wall **1176.3 s**、pass2 wall **889.6 s**（差 286.7 s ≈ 冷/热差额）。

## 三、这改变了目标判断

此前 R321/R323 把 encoder 记为「62.95 s，距 <2 s 有 31×」，但那是**每次新 engine 的首次 encode**。
steady-state（第二趟）实测 **3.196 s**：

| 目标 | 旧认知 | 新实测（warm） | 真实差距 |
|---|---|---|---|
| encoder **< 2 s** | 62.95 s（31×） | **3.196 s** | **仅 1.6×** |
| video VAE **< 20 s** | cold 80 s（4×） | **20.75–21.00 s** | **仅 3.7–5%** |
| audio VAE | cold 56–65 s | **0.20 s** | 已达标（记录项）|

⇒ **两个并行目标都已进入「个位数百分比 / 1.6×」的可优化范围**，不再是数量级差距。

## 四、下一步

### 1) video VAE：目标 <20 s（差 0.75–1.0 s，3.7–5%）

按 R325 的 knob 清单做**孤立 VAE 基准**（加载 decoder 9.69 GB，固定随机 latent，连跑 3 次取第 2/3 次）：
- `vae_parallel_mode`: `tile`（当前） vs `spatial_shard_height` / `spatial_shard_width`；
- `vae_patch_parallel_size`: 8（当前） vs 4；
- 判据：warm < 20 s **且** 各配置输出哈希一致（纯切分不变数学）。

### 2) encoder：目标 <2 s（3.196 s → 1.6×）

需要定位 3.196 s 的构成：一次 encode 内 77 个 block 的权重搬运（若 warm 时仍在搬）与计算；
候选：增大 encoder 内部并行度、减少 offload 往返、或让权重常驻（R320 事故后的分步安全方案 R322）。

## 五、状态

- 交付不变：**16.34 s/step（−8.9%）**，49 步 ×2 验证，relRMS 在原生带内，`H3_PERF_COMBO=1` 默认 OFF；
- ts-217 健康，本轮无残留进程；
- 15 s（DiT）：R313 路线级结论 ⇒ 当前条件不可达。
