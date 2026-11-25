# R329 — VAE 配置空间探尽：`patch_parallel_size` 只能 1 或 8；warm 最优 20.672 s，距目标 3.4%

## 一、被实现拒绝的候选（直接抛错）

```
ValueError: MiniMax H3 native VAE patch parallelism currently requires
            vae_patch_parallel_size=1 or the full DiT group size (8), got 4
```

⇒ `vae_patch_parallel_size` 只允许 **1 或 8**（当前 8 = 最优并行），无中间档可调。

## 二、VAE 配置空间总表（全部实测/源码确认）

| 参数 | 允许值 | 当前 | 结论 |
|---|---|---|---|
| `vae_parallel_mode` | 仅 `tile` | `tile` | 唯一支持（R327） |
| `vae_patch_parallel_size` | 仅 `1` 或 `8` | `8` | 8 为最大并行（本轮） |
| `vae_use_tiling` | True/False | 建议 **False** | warm 20.672–20.783 vs 20.746–20.998（R328） |
| `vae_fast_path` | off/lossless/channels_last | `lossless` | 已最快无损；再降属有损，禁用 |
| `vae_use_slicing` | True/False | False | 仅降内存 |

## 三、结论（VAE）

**在当前实现对 math-neutral 配置下，warm video VAE 最优 = 20.672 s（tiling=False），
距 <20 s 目标差 3.4%，无法通过配置达成。**

要达标只能改实现（tile 内部尺寸常量、通信/计算重叠、或单卡内更细粒度流水）——属于**代码级改动**，
需你批准后另立任务；且没有证据保证能拿到 3.4%。

## 四、encoder 侧同样收敛

- warm `encode_prompt` = **3.196 s**（cold 62.15–62.95 s），目标 <2 s ⇒ 差 **1.6×**；
- TP8 固定（合同要求），`text_encoder_tp_size` 不可调；offload 已是最优（warm 时权重已驻留）；
- 达标同样需要**代码级**优化（更细的重叠或缩小视觉塔开销），非配置可解。

## 五、四个目标的最终状态（本轮定格）

| 目标 | 现状（实测） | 判定 |
|---|---|---|
| DiT ≤10 s/step（用户 11-23 上调） | **16.34 s/step（−8.9%，已交付冻结）** | TIK 结构下限 126.19 ms；AscendC cube 在本机 910A 不可用（R313）⇒ **当前硬件不可达** |
| DiT ≤15 s/step（旧目标） | 同上 | 同上 |
| encoder <2 s | warm 3.196 s | 差 1.6×，**配置不可解**（需代码改动） |
| video VAE <20 s | warm 20.672 s | 差 3.4%，**配置不可解**（需代码改动） |
| audio VAE（记录项） | warm 0.20 s | 达标 |
| A2/A3 对比 | 未开始 | 缺对应机器 |

## 六、状态

- 交付不变：**16.34 s/step（−8.9%）**，49 步 ×2 验证，relRMS 在原生带内，`H3_PERF_COMBO=1` 默认 OFF；
- ts-217 健康；本轮无残留进程；
- 所有失败与拒绝均已留档（R327/R328/R329）。
