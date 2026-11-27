# B 机方案：910A 上的 FastH3 快速档（PDD-8 蒸馏学生）

> 目标：与 ts-217（A 机，原模型 768p/49 步/12.3 min 一条）组成"快速草稿 + 高质量终稿"
> 双档，两档都以 vLLM-Omni `/v1/videos` 契约接入 OpenDirector。

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