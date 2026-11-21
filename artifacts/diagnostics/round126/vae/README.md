# R126 同 final-latent VAE driver

只测已有 accepted-FA 最终 latent 的完整 video/audio VAE，不执行 DiT 或 encoder forward。默认观察真实 native installer；不会直接 Half 化原始 FP32 参数。只有 lead 在已租用主机审核后运行模型。

## 本地无需 Torch 的检查

```bash
PYTHONDONTWRITEBYTECODE=1 python3 -B -m unittest discover \
  -s artifacts/diagnostics/round126/vae -p 'test_contracts126.py' -v
```

14 个 stdlib tests 已通过。计划模式默认不执行模型；执行需要显式 `--execute` 和完整 SHA manifest。

## Lead 的远端执行接口

先把本目录的小脚本与 manifest 放到 `/tmp/h3_perf12601/vae`。保留当前 CANN Python 路径，不覆盖：

```bash
source /home/mseco/Ascend/cann-9.2.0/set_env.sh
export PYTHONPATH=/tmp/h3_perf12601:${PYTHONPATH:-}
export H3_R126_IPC_KEY_FILE=/tmp/h3_perf12601/private_IPC_key126.bin

/home/mseco/vllm910a/bin/python -B /tmp/h3_perf12601/vae/decode_only126.py \
  --checkpoint /home/mseco/models/MiniMax-H3/FL2VA \
  --latents /tmp/h3_align12401/final_NPU_latents_pid2331981124.pt \
  --root /tmp/h3_perf12601 --label vae_native_observe126 \
  --runtime-manifest /tmp/h3_perf12601/vae/runtime_manifest126.json \
  --validate-only

/home/mseco/vllm910a/bin/python -m torch.distributed.run \
  --standalone --nproc_per_node=8 \
  /tmp/h3_perf12601/vae/decode_only126.py \
  --checkpoint /home/mseco/models/MiniMax-H3/FL2VA \
  --latents /tmp/h3_align12401/final_NPU_latents_pid2331981124.pt \
  --root /tmp/h3_perf12601 --label vae_native_observe126 \
  --runtime-manifest /tmp/h3_perf12601/vae/runtime_manifest126.json \
  --execute --installer-mode observe-native --warm-repeats 3
```

必须新 label，旧目录拒绝重跑覆盖。不要安装 R124 alignment hooks 或开启它的 save-final-latent/媒体 capture；它们依赖完整 DiT 前向。本 driver 自己检查输入、输出与 source SHA。只用现有 R126 auth-only startup，不读取/打印 IPC key。

Manifest 已按 lead 的 source-only preflights 建好。`--validate-only` 核验源文件、原始 component config 和所有 `.py`、完整已存在 latent SHA/字节数，不导入 Torch；不读取 checkpoint 权重内容。

## 证据边界

- `video_clean` / `audio_clean`：cold + 3 warm；模型已 resident，无 per-module hook/counter/clone。root prewarmed events + host wall；D2H/CPU raw-bit compare/hash在 timer 外。
- `video_engagement_profile`：独立 instrumented 调用；记录真实 installer return、dtype、36块 fast binding/function identity、实际 function/operator entry 与 non-None return。预分配 events，无 callback synchronize/CPU transfer。region inclusive，不能求和当端到端。
- `video_tile_capture`：独立 **非 clean** 调用，克隆每个完整36层 FP32 decoder tile 输出，整个 decode 完成后再 CPU hash；GPU capture 上限1GiB，tile payload不落盘、不下载本地。不是完整全局 blend/revert 后 Float 视频证明。
- `video_load/offload`、`audio_load/offload`、D2H分开报告。video wall包括 native tiles、HCCL、blend、temporal与uint8量化，不含权重加载/audio/MP4。
- 原始FP32 parameters、HF32关闭、正常Half video autocast、原生FP32 audio均 fail-closed。native installer若改Half立即在 forward 前停，不能 widen 冒充恢复。
- 当前 native dispatcher 的真实 NPU lookup是 `None`，installer应返回False。driver仍实际观察，不能把R125无命中实验称融合效果。SiluAndMul默认NPU路径是eager；不切 `NO_FUSED_ACT=0`。
- 保存一次完整uint8视频（rank0）和每rank FP32 waveform，run-wide disk cap1GiB；summary和CPU比较覆盖完整raw输出，禁止拉大 `.pt/.npz` 本地。
- 外部 baseline 可用 `--reference-dir`，仅接受未装 fast 且全FP32的完成回执（preserve-FP32 bypass或observed-native-False）；源/输入哈希不能变。任意 candidate完整输出或prequant tile差异会allreduce通知所有rank停下一case并问用户。没有默认数值容差/无损自动验收。

模型完成后只收JSON回执；纯stdlib聚合打印时间和hit evidence，不读激活/权重、不覆写回执：

```bash
python -B /tmp/h3_perf12601/vae/summarize_vae126.py \
  /tmp/h3_perf12601/vae_native_observe126
```

取同一call的rank最大值再取warm median；cold、instrumented及D2H分开。`accepted_lossless` 始终为False，单一latent证据不推广任意输入或BF16。
