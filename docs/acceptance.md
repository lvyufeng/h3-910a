# ts-217 910A MiniMax H3 分级验收

本文给出 ts-217 上两套四卡 ComfyUI 实例从 0.4MP 短片段到 720p 的分级验收流程。工具只编排实例、提交工作流并收集证据；启动前检查、模型哈希和工作流内容都应由环境变量明确指定。

## 1. 被验对象与默认隔离

两套实例运行在同一台主机时，默认分组如下。HTTP、HCCL rendezvous 和 HCCL socket 端口范围必须互不重叠。

| 实例 | 可见 NPU | HTTP | HCCL rendezvous | HCCL socket 范围 | 工作流 `npu_count` |
|---|---|---|---:|---|---:|
| A | `0,1,2,3` | 8801 | 29511 | `60000-60999` | 4 |
| B | `4,5,6,7` | 8802 | 29512 | `61000-61999` | 4 |

每个实例使用独立的运行目录、日志目录、SQLite 数据库、输入/输出/临时目录、工作流副本和模型路径配置副本。这里的“隔离”指这些应用层状态和端口彼此分离；两套实例仍共享同一份模型文件、主机 CPU、主机内存和主机网络。

以下环境按 ts-217 实际情况导出，示例路径需要替换：

```bash
export H3_910A_ROOT=/Users/lvyufeng/Projects/h3-910a   # 真机部署时替换为 910A 工作目录
export COMFY_HOME=/mnt/workspace/ComfyUI
export H3_PYTHON=/mnt/workspace/venv/bin/python
export H3_CANN_ENV=/usr/local/Ascend/ascend-toolkit/latest/set_env.sh
export H3_MODEL_ROOT=/mnt/workspace/ComfyUI/models
export H3_MODEL_MANIFEST=/mnt/workspace/h3-910a/model-manifest.json
export H3_INSTANCE_ROOT=/mnt/workspace/h3-910a/ts217
```

`H3_MODEL_ROOT` 中每个目录名要与 `config/extra_model_paths.yaml` 的逻辑目录对应。若实际目录不同，应在运行前复制并修改配置，然后通过 `H3_MODEL_CONFIG` 指向它；运行脚本会再复制一份到各实例，不触碰输入配置。

## 2. 开始前的只读门禁

在两套实例启动前，至少执行一次完整环境检查。以下示例将 JSON 报告保存到测试证据目录；环境检查脚本本身只读。

```bash
mkdir -p "${H3_INSTANCE_ROOT}/evidence"

H3_REQUIRE_TOPO=1 \
H3_REQUIRE_ATC=1 \
H3_EXPECTED_DEVICES=8 \
H3_EXPECTED_CANN=9.2.0 \
H3_HASH_MODELS=full \
H3_MIN_CPU_CORES=32 \
H3_MIN_RAM_GIB=128 \
H3_MIN_DISK_GIB=120 \
H3_MIN_RUNTIME_DISK_GIB=120 \
bash scripts/check_ts217_env.sh --json \
  > "${H3_INSTANCE_ROOT}/evidence/preflight.json"
```

进入下一阶段的条件：

- JSON 中 `"ok": true`，且不存在 `FAIL`。`WARN` 必须逐项记录原因。
- 能看到 8 个 NPU 和预期 driver；CANN、`atc`、`torch`、`torch_npu` 均可加载。
- 两个四卡子组的拓扑和 HBM 查询结果已人工确认。`H3_REQUIRE_TOPO=1` 时拓扑查询为空会直接记为 `FAIL`。
- CPU、主机 RAM、模型盘和运行盘的可用量满足本次级别。阈值是现场策略，不应把示例值当作硬件规格。
- 有 `H3_MODEL_MANIFEST` 时，每个条目的 `size` 和 SHA256 都一致；没有 manifest 时，`H3_HASH_MODELS=full` 只记录实际摘要，没有期望值可比较。

模型清单是 JSON 数组，`destination` 必须相对模型根，且不能包含 `..`：

```json
[
  {
    "destination": "diffusion_models/minimax_h3_fl2va_int8_convrot.safetensors",
    "size": 123456789,
    "sha256": "0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef"
  }
]
```

建议同时记录端口占用和残留进程：

```bash
ss -ltnp | grep -E ':(8801|8802|29511|29512)\b' || true
npu-smi info
```

验收开始后，目标四卡上不应有其他计算任务；端口已被其他进程占用、可见卡集合与实例不一致，或系统中出现异常掉卡时，本轮样本无效。

## 3. 生成实例工作流

源工作流可使用上游 `fl2va_4step_极速档.json` 或现场确认过的 API 工作流。下面的命令只读取源文件并写出新文件：

```bash
export H3_SOURCE_WORKFLOW=/tmp/ascend-h3-repos-20261109/minimax-h3-ascend/workflows/fl2va_4step_极速档.json
export H3_EVIDENCE_ROOT="${H3_INSTANCE_ROOT}/evidence"
mkdir -p "${H3_EVIDENCE_ROOT}/workflows"

H3_MODEL_DIFFUSION=minimax_h3_fl2va_int8_convrot.safetensors \
H3_MODEL_CLIP=qwen3vl_32b_h3_ultra_uncensored_heretic_int8_convrot.safetensors \
H3_MODEL_VIDEO_VAE=minimax_h3_video_vae_int8_convrot.safetensors \
H3_MODEL_AUDIO_VAE=minimax_h3_audio_vae_fp32.safetensors \
H3_MODEL_LORA=minimax_h3_fl2v_turbo_4step_v1.0_768p_comfyui_bf16.safetensors \
python3 scripts/make_instance_workflow.py \
  --source "${H3_SOURCE_WORKFLOW}" \
  --output "${H3_EVIDENCE_ROOT}/workflows/smoke-864x480-124f-4s.json" \
  --width 864 --height 480 --frames 124 --steps 4 --npu-count 4 \
  --seed 7 --first-image none --last-image none \
  --summary
```

生成器要求宽、高是 32 的倍数，帧数满足 `17k+5`，`npu_count` 只能是 1、2 或 4。用于验收的实例必须是 `4`。生成后应检查：

- `MiniMaxH3ImageToVideo` 的宽、高、`length` 分别是目标值。
- `BasicScheduler.steps` 是目标步数。
- `MultiNPUParallelConfig.npu_count` 是 `4`。
- 模型名与模型目录中的实际文件名完全一致，音频 VAE 是 FP32 文件。
- 首尾帧输入与本次用例一致；文本到音视频用例可显式传入 `none`。

## 4. 启动与冒烟

分别启动 A、B。脚本会在子进程内 source CANN，设置实例环境，并以 `setsid` 启动后台进程。

```bash
H3_INSTANCE_ID=1 bash scripts/run_comfy_instance.sh start
H3_INSTANCE_ID=2 bash scripts/run_comfy_instance.sh start

H3_INSTANCE_ID=1 bash scripts/run_comfy_instance.sh status
H3_INSTANCE_ID=2 bash scripts/run_comfy_instance.sh status
```

两个状态都必须是 `pid_owned=yes`、`healthy=yes`，且实例环境中的设备、HTTP 端口、HCCL rendezvous 和 socket 范围与上表一致。默认启动等待上限是 180 秒；模型首次加载可能更久，可通过 `H3_STARTUP_TIMEOUT` 调整。超时只表示健康检查未通过，不等同于模型加载失败，应结合日志判断。

正式计时前，每个实例各提交一次热身请求。热身结果不计入基线：

```bash
python3 scripts/benchmark_h3_api.py \
  --server a=http://127.0.0.1:8801 \
  --workflow "${H3_EVIDENCE_ROOT}/workflows/smoke-864x480-124f-4s.json" \
  --mode single --width 864 --height 480 --frames 124 --steps 4 \
  --npu-count 4 --run-label smoke-warmup-a \
  --output-root "${H3_EVIDENCE_ROOT}/runs"

python3 scripts/benchmark_h3_api.py \
  --server b=http://127.0.0.1:8802 \
  --workflow "${H3_EVIDENCE_ROOT}/workflows/smoke-864x480-124f-4s.json" \
  --mode single --width 864 --height 480 --frames 124 --steps 4 \
  --npu-count 4 --run-label smoke-warmup-b \
  --output-root "${H3_EVIDENCE_ROOT}/runs"
```

热身请求可以用于发现模型加载、首个图编译、HCCL 初始化和编码器路径错误，但不用于计算中位数。

## 5. 分级矩阵

从短、低分辨率开始。某一级出现失败时，先分类并保存证据，再决定是否降低几何继续定位；不要把低级别成功外推为高级别通过。

| 级别 | 几何 `WxH` | 帧数 | 步数 | 时长约值 | 执行范围 | 主要目的 |
|---|---|---:|---:|---:|---|---|
| L0 环境 | 不生成 | - | - | - | 两套实例启动 | 环境、拓扑、端口、加载 |
| L1 Smoke | 864×480 | 124 | 4 | 5.17 s | A 单路、B 单路、A+B 双路 | 0.4MP 功能与双路隔离 |
| L2 Mid | 1024×576 | 124 | 4 | 5.17 s | A 单路、B 单路、A+B 双路 | 分辨率与中间显存压力 |
| L3 720p Short | 1344×768 | 124 | 4 | 5.17 s | A 单路、B 单路、A+B 双路 | 720p、5 秒级完整性 |
| L4 720p Long | 1344×768 | 345 | 4 | 14.38 s | A 单路；资源稳定后 A+B 双路 | 长序列与注意力/HCCL 压力 |
| L5 Quality | 1344×768 | 124、345 | 8 | 同上 | A/B 分别执行，视时间再做双路 | 质量路线与 LoRA 匹配 |

每级先各跑三次单路，取每个实例成功样本的中位数；再跑三次双路。某级至少完成一次 A 单路、一次 B 单路和一次 A+B 双路，才能判为“功能通过”；要上报延迟中位数，则三次样本都必须有效。

L5 只在以下条件同时满足时执行：基础模型、8 步 LoRA/workflow 和提示词资产明确匹配，且该组合已经在本机通过模型加载；不能把 4 步 LoRA 直接配 8 步工作流当作同一配置。条件不满足时记录 `N/A`，不要用 4 步结果替代质量档。

## 6. 单路与双路命令

每次单路基线和双路测试使用独立 `--run-label`，三个重复建议使用 `r1`、`r2`、`r3`。以下以 L1 为例；后续级别只改工作流几何、帧数和步数。

三次顺序单路基线：

```bash
for repetition in 1 2 3; do
  python3 scripts/benchmark_h3_api.py \
    --server a=http://127.0.0.1:8801 \
    --server b=http://127.0.0.1:8802 \
    --workflow "${H3_EVIDENCE_ROOT}/workflows/smoke-864x480-124f-4s.json" \
    --mode single --width 864 --height 480 --frames 124 --steps 4 \
    --npu-count 4 --seed 7 --run-label "l1-single-r${repetition}" \
    --output-root "${H3_EVIDENCE_ROOT}/runs"
done
```

三次同时提交：

```bash
for repetition in 1 2 3; do
  python3 scripts/benchmark_h3_api.py \
    --server a=http://127.0.0.1:8801 \
    --server b=http://127.0.0.1:8802 \
    --workflow "${H3_EVIDENCE_ROOT}/workflows/smoke-864x480-124f-4s.json" \
    --mode dual --width 864 --height 480 --frames 124 --steps 4 \
    --npu-count 4 --seed 7 --run-label "l1-dual-r${repetition}" \
    --output-root "${H3_EVIDENCE_ROOT}/runs"
done
```

`dual` 模式会在线程中分别准备两个工作流，在准备完成后通过 barrier 同时发起 `/prompt`。两个实例必须使用不同的 URL。每次运行都会保存实际提交的 API 工作流、完整 `result.json`、下载产物及产物 SHA256。

## 7. 每级成功判据

自动判据：

- `/prompt` 返回非空字符串 `prompt_id`，且 HTTP 请求没有协议错误。
- `/history/{prompt_id}` 返回该任务记录，`status.status_str` 不是 `error`、`completed` 不是 `false`，也没有 `execution_interrupted=true`。
- `/history` 中至少有一个可下载产物，且所有产物下载成功、文件非空。
- `result.json` 中每个产物的 `size` 和 `sha256` 已记录。
- `ffprobe` 可读取视频；至少确认一个 video stream。
- 默认确认 audio stream；同时核查 `result.json` 中的 `sample_rate` 为 32000 Hz、`channels` 为 2。音频规格目前需要人工核查，工具默认只强制“存在音频流”。
- 默认确认首条有效视频流的宽、高、帧数和平均帧率与请求一致。H3 输出目标是 24 FPS；工具允许 `24000/1001` 的 ffprobe 表示并换算后比较。
- 时长约等于 `frames / 24`。124 帧约 5.17 秒，345 帧约 14.38 秒；建议允差不超过约 0.10 秒，并在报告中记录实际值。
- 单路和双路两次运行都返回成功，`result.json` 的 `overall_success` 为 `true`。

人工判据：

- 抽看首帧、中间帧和末帧，画面非全黑、无明显撕裂/错帧，内容方向和宽高比符合提示词。
- 音轨可播放，左右声道有声音且无持续削波、静音或明显错位；不能只看文件里“有音频流”。
- 工作流、日志和输出的文件名前缀能对应到本次 `run_label`，没有串到另一实例。
- `submission_skew_seconds` 很小、两次任务都实际执行，且宿主无其他 NPU 计算任务。若一个任务在另一个之后才出队，需记录实际排队情况，不能用 wall time 冒充并发效率。

双路样本除了两条任务各自成功，还要记录：

| 字段 | 含义 |
|---|---|
| `dual_wall_seconds` | 两条 POST 最早开始到两条结果都完成并校验结束的墙钟时间 |
| `submission_skew_seconds` | 两条 `/prompt` 发起时刻的差 |
| 每个实例的 `execution_seconds` | 该实例 POST 开始到进入 `/history` 的时间 |
| 每个实例的 `completion_seconds` | POST 开始到产物下载及媒体校验结束的时间 |

双路的功能判据是两条都成功；性能判据应先建立同一工作流、同一模型、同一精度和同一预热状态下的单路稳定基线，再比较 `dual_wall_seconds` 与各实例 `completion_seconds`。910A 上目前没有可引用的验收门槛，不能把 910B 的 20–25 秒热机数据当作 910A 的 SLA。

## 8. 各阶段 OOM 与错误观测点

每一级都要同时观察 `npu-smi` 的每卡 HBM、主机内存、运行盘/临时盘和 ComfyUI 日志。工具没有内置持续采样器；可在另一个只读终端周期执行 `npu-smi info` 和 `npu-smi info -t memory -i <id>`，并保存时间戳。失败时先保存现场，再决定是否重试。

| 阶段 | 观察点 | 典型信号 | 处置原则 |
|---|---|---|---|
| 实例启动/模型加载 | `/system_stats` 是否在超时内健康；日志是否卡在 `UNETLoader`、`CLIPLoader` 或 HCCL 初始化 | 单卡或全组 HBM 持续上涨；进程消失；`init_collective`/HCCL 超时 | 区分 NPU OOM、HCCL、端口冲突和主机 RAM；不要直接把超时记成 OOM |
| 文本编码 | `CLIPLoader` 后、Prompt 执行日志 | 文本编码阶段 HBM 峰值；`ACL_ERROR_RT_MEMORY_ALLOCATION` 或 OOM | 记录是否因文本编码器、序列长度或双路同时编码触发 |
| DiT 采样 | 每个采样 step 的 HBM 与耗时；720p/长帧最敏感 | NPU OOM、`EZ9999`、算子错误、采样中途被杀 | 保存失败 step；很大可能是 HBM 或特定算子路径 |
| 视频 VAE 解码 | DiT 完成后的 decode 阶段；临时内存是否突升 | HBM 或 host RAM 上涨、临时目录增长、解码中断 | 区分视频 VAE 峰值与 DiT 峰值，记录 124/345 帧差异 |
| 音频 VAE | 音频解码阶段 | 音频 VAE OOM、FP32 精度/算子错误 | 确认音频 VAE 文件为 FP32，不把配置错误归为显存不足 |
| 双路并发 | 两条任务是否同时进入采样；两套 NPU 的 HBM 与 host 内存 | 单路成功、双路 OOM；HCCL 竞争；队列串行 | 若发生串行，记录为并发未生效；若 HBM 双路叠加超限，记录争用而不是冒充双路通过 |
| 主机侧 | `dmesg` 的 OOM killer、meminfo、磁盘满、进程被 SIGKILL | ComfyUI 无任务错误但进程突然退出 | 与 NPU OOM 分开；必要时降低主机侧缓存或清理测试产物 |

如果需要尝试框架级内存策略，Ascend-H3 原生服务提供由常驻到分层流式的候选顺序：`resident`/`none` → 仅文本编码 offload（`text`）→ 整组件 offload（`module`，实验性）→ 分层 offload（`layer`）。这是 Ascend-H3 原生路径的能力；本仓库的 ComfyUI 补丁和启动脚本没有声明这些 offload 模式可用，不能把该顺序写成 ComfyUI 开关。每次改变策略都要重新记录版本、精度、卡数、帧数和是否预热。

## 9. 失败与停止

判定失败时保留以下证据：

```bash
H3_INSTANCE_ID=1 bash scripts/run_comfy_instance.sh logs
H3_INSTANCE_ID=2 bash scripts/run_comfy_instance.sh logs
npu-smi info > "${H3_EVIDENCE_ROOT}/npu-after-failure.txt" 2>&1 || true
dmesg --ctime 2>/dev/null | tail -n 200 > "${H3_EVIDENCE_ROOT}/dmesg-tail.txt" || true
df -h > "${H3_EVIDENCE_ROOT}/disk-after-failure.txt"
free -h > "${H3_EVIDENCE_ROOT}/memory-after-failure.txt" 2>/dev/null || true
```

停止实例：

```bash
H3_INSTANCE_ID=1 bash scripts/run_comfy_instance.sh stop
H3_INSTANCE_ID=2 bash scripts/run_comfy_instance.sh stop
```

控制脚本只停止同时匹配 `main.py` 和该实例 HTTP 端口的 PID；默认等待 60 秒，超时后保留 PID 文件供人工处理，不自动发送 `SIGKILL`。可通过 `H3_STOP_TIMEOUT` 调整等待时间。

## 10. 结果归档

每个级别至少保留：

- 环境检查 JSON 和硬件/端口现场记录。
- 生成后的工作流和每次实际提交的 `submitted-workflow.json`。
- 单路、双路三次运行的 `manifest.json`、`result.json`、下载产物、两个实例的日志。
- 失败时的 NPU、主机内存、磁盘和内核日志。
- 一份人工检查记录：首/中/末帧、音轨、音频采样率/声道、是否有转码或修补。

只有同一模型、同一工作流、同一硬件分组和相同预热条件的结果才可比较。硬件、驱动、CANN、torch_npu、模型、LoRA、节点补丁或精度任一变化，都应新建验收批次。
