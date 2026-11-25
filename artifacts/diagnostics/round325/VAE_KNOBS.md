# R325 — VAE 可调参数清单（为 warm <20 s 目标做准备）

## 一、配置项（`vllm_omni/config/omni_config.py`）

| 参数 | 默认 | 当前使用 | 取值域 |
|---|---|---|---|
| `vae_parallel_mode` | `"tile"` | `"tile"` | `{"tile", "spatial_shard_height", "spatial_shard_width"}` |
| `vae_patch_parallel_size` | `1` | `8` | ≥1 整数 |
| `vae_use_tiling` | `False` | `True` | bool |
| `vae_fast_path` | `"lossless"` | 未显式设置（=默认 `lossless`，已是最快无损路径） | `{"off","lossless","channels_last"}` |
| `vae_use_slicing` | `False` | `False` | bool |

## 二、结论：可做的无损候选只有 3 类

1. `vae_parallel_mode` 换成 `spatial_shard_height` / `spatial_shard_width`（纯切分方式变化，数学不变）；
2. `vae_patch_parallel_size` 在 `{4, 8}`（甚至 `{2,4,8}`）间调；
3. `vae_use_tiling` 开/关与分片模式组合。

**不能做**：`vae_fast_path` 已是 `lossless`（再降级就是有损，禁用）；`vae_use_slicing` 只影响内存不改变算法。

## 三、更省的验证方式（推荐，替代整轮生成）

整轮生成一次 20 分钟，只为了测 VAE 的某个配置，代价过高。**应改为孤立 VAE 基准**：

1. 单独加载 video VAE（decoder 参数 9.69 GB，见既有收据 `decoder_parameter_bytes`）；
2. 构造与真实 latent 同形状的固定随机张量（同 dtype/同形状，seed 固定）；
3. 对每个配置：`decode_latent` 连续跑 3 次，取第 2/3 次为 warm；
4. 记录耗时 + 输出哈希（确认不同配置间**输出位级一致**，因为切分不改变数学）。

判据：warm < 20 s，且各配置输出哈希一致（否则说明该配置并非纯切分，需停用上报）。

## 四、状态

- R324 的两趟 warm 测量仍在跑（pass 1 denoise 中，engine_init 327.9 s）；
- 交付不变：**16.34 s/step（−8.9%）**，`H3_PERF_COMBO=1` 默认 OFF。
