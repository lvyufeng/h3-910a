# R321 — 首次精确阶段计时：`encode_prompt` = 62.95 s（TP8 全 rank 一致）

## 一、方法

- 探针：`sitecustomize.py`（PYTHONPATH 最前 + chain-load 官方 sitecustomize）→
  `sys.meta_path` 在 `pipeline_minimax_h3` 被 import 时 patch `encode_prompt` / `decode_latent`；
- 计时：前置/后置各一次 `torch.npu.synchronize()` + `perf_counter`（含同步，非异步估算）；
- 运行：官方 `generate_full_NPU126.py`，单进程 entry，8 卡 TP2+USP4、encoder TP8、49 步、seed 1234、
  768×1344×120 帧，与 R126 同参数；
- 本轮为**默认配置**（encoder 逐层 offload 保持开启），未做任何数值/配置改动。

## 二、结果

```
(DiffusionWorker_SP0_TP0) [STAGE] encoder 62.950s
(DiffusionWorker_SP0_TP1) [STAGE] encoder 62.951s
(DiffusionWorker_SP1_TP0) [STAGE] encoder 62.951s
(DiffusionWorker_SP1_TP1) [STAGE] encoder 62.953s
(DiffusionWorker_SP3_TP1) [STAGE] encoder 62.950s
```

⇒ **`encode_prompt` 阶段 = 62.95 s**（8 个 encoder 分片耗时一致，说明 TP8 负载均衡）。

## 三、与既有收据的关系（口径必须区分）

| 口径 | 数值 | 说明 |
|---|---|---|
| 本轮 `encode_prompt` 包装 | **62.95 s** | 仅 `encode_prompt` 方法本体（含其内部全部 encoder 前向） |
| 既有 `live_encoder_elapsed_s` | 108.5–110.3 s | 更宽口径（含 keyframe/图像预处理、prompt 编码前后的整段 live encoder 流程） |

两者**不可互相替代**；共同结论：encoder 距 **<2 s** 目标仍有 **31–55×** 差距。

## 四、根因仍然成立

`layerwise offload for encoder text_encoder (77 blocks)`：每次 encode 要把 63 GB 权重逐层搬入，
而 TP8 常驻只需 ~7.9 GB/卡 —— 常驻是唯一可能把 encoder 拉进秒级的路径（R320 的事故就是验证它时
host RAM 被撑爆，需改为分步验证）。

## 五、下一步（本轮跑完后）

同一轮跑完 49 步 + VAE 后，`decode_latent` 会给出 **video VAE / audio VAE** 的精确耗时
（目标 video VAE < 20 s），届时一并入库。

## 六、状态

- 交付不变：**16.34 s/step（−8.9%）**，49 步 ×2 验证，relRMS 在原生带内，`H3_PERF_COMBO=1` 默认 OFF；
- ts-217 已从 R320 事故恢复（内存 576/754 GB，OOM killer 清掉我方进程，无残留）；
- 15 s：R313 路线级结论 ⇒ 当前条件不可达。
