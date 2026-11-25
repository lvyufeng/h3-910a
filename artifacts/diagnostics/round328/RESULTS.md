# R328 — VAE `vae_use_tiling=False` 略优但未破 20 s（warm 20.672–20.783 s）

## 一、方法

与 R326 相同的两趟 warm 测量，仅把 `vae_use_tiling` 由 `True` 改为 `False`（仍 `vae_parallel_mode='tile'`、
`vae_patch_parallel_size=8` 不变）。

## 二、结果

| 配置 | warm VideoVAE | cold VideoVAE | pass1 wall | pass2 wall |
|---|---|---|---|---|
| `tiling=True`（R326） | **20.746–20.998 s** | 80.12–80.32 s | 1176.3 s | 889.6 s |
| `tiling=False`（本轮） | **20.672–20.783 s** | 80.71–80.85 s | 1205.5 s | 912.9 s |
| 既有收据（warm） | 20.565 s | — | — | — |

⇒ `tiling=False` 在 warm 下**略优**（最好 20.672 vs 20.746，约 −0.4%），cold 反而略慢；
**仍未达到 <20 s 目标**（最好值差 3.4%）。

## 三、候选耗尽情况

| 候选 | 结论 |
|---|---|
| `vae_parallel_mode` → spatial_shard_* | ❌ 实现不支持（R327） |
| `vae_use_tiling` False | 略优但未达标（本轮） |
| `vae_patch_parallel_size` 8→4 | **下一轮（最后一个数学中性 knob）** |
| `vae_use_slicing` | 只降内存，预期不提速 |
| `vae_fast_path` | 已 `lossless`；再降级属有损，禁用 |

## 四、结论与预期

即使 `patch_parallel_size=4` 也不达标（降低并行度通常更慢），则 **video VAE <20 s 在当前实现下不可达**：
- 现状 20.67 s 已是「warm、lossless、原生 tile」下的最优；
- 20 s 目标等价于再快 3.4%，而可调项已全部用尽；
- 若要继续，只能改 VAE 实现（tile 大小内部常量、通信与计算重叠），属于**代码级改动**，需你批准后另立任务。

## 五、状态

- 交付不变：**16.34 s/step（−8.9%）**，`H3_PERF_COMBO=1` 默认 OFF；
- 阶段计时：encoder warm 3.196 s（<2 s，差 1.6×）、video VAE warm 20.67–21.00 s（<20 s，差 3.4–5%）、audio VAE warm 0.20 s（已达标）；
- ts-217 健康；15 s（DiT）：R313 路线级结论 ⇒ 当前条件不可达。
