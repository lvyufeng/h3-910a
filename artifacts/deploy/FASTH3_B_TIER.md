# B 机方案：910A 上的 FastH3 快速档（PDD-8 蒸馏学生）

> 目标：与 ts-217（A 机，原模型 768p/49 步/12.3 min 一条）组成"快速草稿 + 高质量终稿"
> 双档，两档都以 vLLM-Omni `/v1/videos` 契约接入 OpenDirector。

## 0. 硬件结论（2026-11-27 实测，决定 B 机选型）

| 机器 | npu-smi / torch_npu | Board ID | CANN 算子 | 结论 |
|---|---|---|---|---|
| ts-217（A 机） | Ascend910B | 0x21 | ✅ 正常 | 可用（现有服务） |
| ts-87 / ts-88 | Ascend910A | 0x02 | ❌ `GEInitializeV2 failed`（CANN 9.2 不支持 910A 代） | 不可用 |
| ts-188 / ts-190 / ts-194 | Ascend910PremiumA | 0x24 | ❌ `CannKbInit failed` → `TeFusion init failed` → GE init failed；AOE 旁路无效 | 不可用 |
| ts-216 | 910B3（64 GB×8） | — | 期望与 ts-217 同族 | **唯一候选，但测试时 BUSY（8 卡 aicore 100%）** |

结论：B 机必须是 **910B/910B3 代**；910A 与 910PremiumA 在本轮 CANN 9.2 环境下无法初始化计算图。
（注：`npu-smi` 的 910A/910B 标签与 `torch_npu.get_device_name()` 一致，不是显示误差。）

## 5. 已完成的非 NPU 工作

### 5.1 权重下载（ts-87，hf-mirror，直连不经本机）
- `transformer_ref/` 14 分片：已到 7 片（26 GB）
- `text_encoder/` 4/4（16 GB）✓，`audio_vae` ✓，`adaln_table.pt`（499 MB）✓，scheduler/processor/tokenizer ✓
- 剩余：7 个 transformer 分片 + `vae/`（4.2 GB）

### 5.2 FP8→FP16 反量化器（已实现并单分片验证）
- 工具：`artifacts/deploy/fasth3_dequant.py`（运行时在 `/data/h3fast/tools/dequant_fasth3.py`）
- 实测格式：每分片 72 个张量 = **15 个 FP8 线性层**（`weight` + 每输出通道 `weight_scale` [N,1] F32）+ 57 个 BF16；`proj_out.weight`=[3072,5376]、`audio_proj_out.weight`=[1024,5376]（32 头 × 96/32 通道）
- 公式：`w_fp16 = (w_fp8.float() * scale).half()`；roundtrip 校验无 mismatch
- 单分片 CPU 约 5 s（14 分片约 1–2 分钟）

### 5.3 PDD-8 采样契约（依据 FastVideo 源码 `fastvideo/layers/pdd.py` + denoising 测试）
- 细网格：`u = linspace(0.999, 0, 33)`（float64），`pdd_step_indices=[0,4,…,32]`
- 模态 shift：`f_s(u) = s·u·M / (u·(s−1)+M)`，`M=0.999`；video s=12、audio s=3
- 每个 block 的积分权重：`Δσ_j = f_s(u_{j+1}) − f_s(u_j)`（数值稳定式）；采样 `x_end = x_start − Σ_j Δσ_j·m_j`，`m = x0 − eps`
- 前向融合：把 block 内 4 个输出头的权重按 `Δσ_j / ΣΔσ` 加权平均成一个投影，再做一次 matmul（8 次 forward 覆盖 32 个头）
- 片长 124/243/362 帧；guidance 1.0；ref2va 有序参考（图/视频/音频），首尾帧模式用两张有序图

## 1. 实测事实（来自权重与官方文档，非推测）

| 项 | 结论 |
|---|---|
| 模型关系 | FastH3 = **MiniMax-H3 同架构**（50 层、hidden 5376、ffn 14336、56 heads×128、text_dim 5120、time_embed 2688）的 **PDD-8 蒸馏学生** |
| 采样 | **8 次 transformer forward**；PDD 把两个输出投影加宽到 **32 个头**（`proj_out.weight` = [3072,5376]，`audio_proj_out.weight` = [1024,5376]），每次 forward 融合 4 个连续头（integration-weighted mean），再对区块两端 sigma 做一次 Euler |
| 时间网格 | `pdd_steps=32`，`pdd_step_indices=[0,4,8,…,32]`，`grid_max_t=0.999`，guidance 1.0 |
| scheduler | video shift **12**、audio shift **3**（与基座 H3 一致） |
| 片长 | 124 / 243 / 362 帧（5/10/15 s，17n+5），24 fps，原生立体声 |
| 条件输入 | **ref2va**：prompt + 有序参考（最多 9 图 / 3 视频 / 3 音频，共 12，至少 1 图或 1 视频）；文本、Qwen3-VL 视觉 token、参考音频构成附加 conditioning 前缀，叠加在目标行之上 |
| 注意力 | 官方用 **VSA 稀疏注意力**（`VIDEO_SPARSE_ATTN_H3`，sparsity 0.9 / tile 128 / ref keep_rate 0.1）；这是他们的算法近似项 |
| 权重格式 | **`FastVideo/FastH3-Omni-FP8` 是完整 dense 导出**：线性层 `F8_E4M3` + 每输出通道 `F32 weight_scale`，其余 BF16/F32 ⇒ **可离线反量化到 FP16，910A 无需 FP4/FP8 硬件** |
| 陷阱 | `FastH3-Omni-NVFP4-QAD` 的 dense 分片**省略了被量化的线性层**，不能当完整模型用 |
| 其它 | 含 `adaln_table.pt`（0.52GB，预计算调制表，可选用）；text_encoder/tokenizer/processor/vae/audio_vae 为基座 MiniMax-H3 组件 |

## 2. 资源与进度

- **B 机：ts-87**（8×910A 32GB，CANN 9.2.0，/data 825GB 可用；dormant 池，用户已授权）。
- ts-87 无外网直连 HF（`Errno 101`），但 **hf-mirror.com 可用**，下载走镜像。
- 已就绪：uv + Python 3.12 venv（`/data/h3fast/venv`）、huggingface_hub。
- 下载中（`/data/h3fast/models/FastH3-FP8`，约 **73GB**）：`transformer_ref/` 50.89GB +
  `text_encoder/` 16.46GB + `vae/` 4.23GB + `audio_vae/` 0.61GB + `adaln_table.pt` 0.52GB + 配置。
- 待办：Ascend 软件栈复刻（torch_npu + vllm + 我们的 vllm-omni + h3_replay fixtures），
  需要先从 ts-217 取精确版本组合。

## 3. 移植阶段

| 阶段 | 内容 | 验收 |
|---|---|---|
| P1 | B 机 Ascend 栈：torch_npu/vllm/vllm-omni 与 ts-217 同版本，跑通 8 卡通信 | `all_reduce` 微基准、模型可加载 |
| P2 | **反量化工具**：FP8+per-channel scale → FP16 safetensors（线性层），BF16/F32 直转；同时把加宽的 `proj_out/audio_proj_out` 按 32 头保留 | 反量化后张量与 FP8 参考数值一致（相对误差 ≤ 1e-2，逐层校验） |
| P3 | **PDD-8 采样器**：8 次 forward、4 头融合、Euler over block sigmas、video/audio shift 12/3 | 与官方 8 步输出同分布（同 seed 对比参考帧） |
| P4 | **ref2va 条件前缀**：有序参考（图/视频/音频）+ Qwen3-VL 视觉 token 拼接 | 参考图生视频可用；fl2va 用"首帧+尾帧"两张有序图实现 |
| P5 | 服务接入：同一网关、`minimax-h3-fast` + `videoAdapter=minimax_h3`、能力声明 | OpenDirector 双档可切换 |
| P6（可选/后续） | VSA 稀疏注意力移植（先跑 dense 精确版；若速度不够再上） | 448p/768p 速度对比 |

## 4. 预期性能（待 P3 实测）

按 A 机分项折算（token 数 ∝ 像素，FA ∝ N²、GEMM/通信 ∝ N）：

| 分辨率 | 单次 forward 估算 | 8 次 forward + VAE/encoder |
|---|---|---|
| 832×480 | ≈ 4–5 s | **≈ 1–1.5 分钟/条** |
| 1344×768 | ≈ 15 s | ≈ 2.5–3 分钟/条 |

## 5. 风险与对策

| 风险 | 对策 |
|---|---|
| VSA 稀疏注意力是其质量/速度的一部分，dense 重跑输出会与官方不同 | 快速档自成一个质量点（FastH3 自己的产品口径），先 dense 对齐官方示例，再决定是否移植 VSA |
| PDD 融合权重/头间插值细节不明确 | 以 FastVideo 源码（`MiniMaxH3Ref2VAModularPipeline` + `basic_fasth3_omniref_pdd.py`）为准逐行对照 |
| 8 步蒸馏权重对 480p/768p 的表现 | 官方 480p 是主推档；768p 作为可选，验收按视觉对比 |
| B 机软件栈复刻耗时 | 复用 ts-217 的 wheel 源与源码树，先跑通再优化 |

## 6. 910A 实测验证（ts-87，2026-11-27）

补齐 CANN 的 TBE python 依赖后（见 `ASCEND910A_CANN_FIX.md`），真 910A 上的关键链路逐项通过：

| 项 | 结果 |
|---|---|
| torch_npu 算子 | FP16 matmul 2048/5376/8192 = 82.3 / 90.9 / 100.0 TFLOPS |
| 8 卡 HCCL | all_reduce 32MB fp16 = 2.63 ms（22.35 GB/s），求和正确 |
| vllm 平台 | `vllm_ascend` python-only 安装 + SoC 补丁（`soc_version=102` → A2）后，`NPUPlatform` / `Ascend910A` 识别成功 |
| vllm_omni | `diffusion.models.minimax_h3.minimax_h3_transformer` 导入成功 |
| 910A FA 内核 | TIK 内核编译+运行成功（首次 58s，含编译） |
| FastH3 权重 | FP8 导出 68GB 下载完成；FP8→FP16 反量化 14/14 分片完成（proj_out=[3072,5376]、audio_proj_out=[1024,5376]） |
| 基座组件 | text_encoder/vae/audio_vae 从 MiniMaxAI/MiniMax-H3 直连下载中（约 78GB） |

### 权重命名对齐（下一步实现）
FastH3 FP16 张量（688 个）为 diffusers 命名：
`transformer_blocks.N.{adaln_proj.linear, attn.to_q/to_k/to_v/to_out.0/to_gate_compress, attn.norm_q/norm_k, ff.net.0.proj, ff.net.2, norm1/norm2}` +
顶层 `proj_out / audio_proj_out / context_embedder`；其中 `to_gate_compress` 是 VSA 门控（dense 路径先忽略）。
我们的 replay transformer 使用 `blocks.N.*` / `time_embedder.*` 命名，需要一张名字映射表。

### 剩余实现
1. 名字映射 + 32 头 proj_out 的 PDD 融合；
2. PDD-8 采样循环（8 forward、4 头加权、shift 12/3）；
3. ref2va/FL2VA 条件打包（复用现有 keyframe/ref 打包代码）；
4. VAE/audio VAE 解码 + 服务化。
