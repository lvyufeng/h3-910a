# R314 — encoder / video VAE / audio VAE 阶段计时探针（已启动）

## 一、探针设计

目标：按合同要求给出 **encoder 实际耗时**与**video VAE 实际耗时**（此前 R281 已确认现有 harness 无阶段计时）。

探针模块 `stageprobe290.py`（已入库 `deployed/`）：

1. 导入 `vllm_omni.diffusion.models.minimax_h3.pipeline_minimax_h3`；
2. 在模块命名空间里找**类定义中拥有** `encode_prompt` / `decode_latent` 的类（实测 `installed wrappers=3`）；
3. 用 `torch.npu.synchronize()` + `perf_counter` 包裹（前置 + 后置同步，避免异步算子时间丢失）；
4. `atexit` 把累计耗时 dump 到 `/tmp/h3_r221/stage_times290.json`。

启动脚本 `stage290_gen.py`：用 `Omni(...)` + `OmniDiffusionSamplingParams`（788×1344、120 帧、49 步、seed 1234、
与 R126 生成参数一致）跑一次完整生成，同时记录 `engine_init_s`、`generation_wall_s` 与各阶段耗时。

## 二、踩坑记录

第一次直接 `runpy` 跑官方 `generate_full_NPU126.py`，在 `out.mkdir(exist_ok=False)` 处
`FileExistsError`（固定输出目录 `/tmp/h3_budget149/NPU_full_generation126` 已存在）——
但**同一次运行确认了探针生效**：日志中出现 `[STAGE] installed wrappers=3`。

## 三、当前状态

- 第二次运行（独立最小生成脚本）已启动，8 卡并行，后台 nohup；
- 预计 engine init 2–4 分钟 + 生成约 10 分钟；
- 结果将写入 `/tmp/h3_r221/stage290_result.json`，下一轮读取并归档为 encoder/VAE 首份实测数据。

## 四、后续判据

| 阶段 | 目标 | 判据 |
|---|---|---|
| encoder | **< 2 s** | 探针 `encoder` 累计值（单次生成内） |
| video VAE | **< 20 s** | 探针 `decode_latent:<VideoVAE 类名>` 累计值 |
| audio VAE | 记录即可 | 同上（audio 类） |

若实测距目标有差距，再据此提出候选（不改数值的前提下：并行度、tiling 参数等），逐项报你批准。
