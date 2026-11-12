# ts-217 H3 双实例 bring-up 状态

记录时间：2026-11-08T17:06Z（服务器 UTC；工作区日期上下文为 2026-11-09 Asia/Shanghai）

## 结论

基础设施层的只读验收通过：

- `ts-217` 上两个正式 ComfyUI 实例均健康，HTTP `8801` 和 `8802` 分别返回 `system_stats`。
- 两个实例各自只能看到 4 张 NPU：A 为 `0,1,2,3`，B 为 `4,5,6,7`。
- 两个组各自运行的 HCCL 探针都完成了真实 `AllReduce` 和 `AllGather`，未出现探针失败。
- 两组的 HCCL socket 范围、rendezvous 端口、运行时目录、缓存和 Ascend 日志目录互不重叠。
- 验收期间没有停止或重启 `8801/8802`；探针结束后两个主实例的 PID 和监听端口保持不变。
- 模型下载进程仍在运行，验收期间没有停止、重启或改写下载脚本。

这证明的是“双组通信与实例隔离可用”，**不是** H3 已完成一次视频生成。当前尚未加载 H3 完成采样，也尚未完成 A/B 两路同时推理的业务验收。

## 硬件和拓扑

服务器将八张卡显示为 `910B`（无后缀）。按本项目约定，这种无后缀设备按 910A 代际处理，不能据此误用 `910b-ops`。

`npu-smi info -t topo` 的实测结果确认：

- `0,1,2,3` 两两为 HCCS。
- `4,5,6,7` 两两为 HCCS。
- 两组之间存在 `SYS` 或 `PHB` 路径。

因此每个正式实例绑定一个四卡 HCCS 组，避免把实例内部集体通信放到跨组路径上。每张卡的 HBM 总量约为 `32741–32768 MiB`（约 32 GiB）。

## 软件版本

以下版本来自服务器实测，不是本地推测：

| 项目 | 值 |
|---|---|
| CANN 安装根目录 | `/home/mseco/Ascend/cann-9.2.0` |
| Toolkit | `9.2.0` |
| Ops 包 | `Ascend-cann-910-ops` |
| Ops 包版本 | `9.2.0` |
| Ops 内部版本 | `V100R001C12SPC001B237` |
| Ops 安装信息 | `/home/mseco/Ascend/cann-9.2.0/aarch64-linux/ascend_ops_install.info` |
| HCCL 包版本 | `9.2.0` |
| Driver / `npu-smi` | `25.5.2` |
| Python | `3.12.13` |
| PyTorch | `2.10.0+cpu` |
| torch_npu | `2.10.0.post2` |
| ComfyUI | `0.32.0` |

`ascend_ops_install.info` 明确显示 `package_name=Ascend-cann-910-ops` 和 `version=9.2.0`，与目标 CANN 版本和 910A 代际约束一致。HCCL 探针脚本自身没有解析出 `hccl_build_version`，输出为 `unknown`；HCCL 版本以 CANN 包内的 `hccl: 9.2.0` 记录为准。不要把探针中的 `unknown` 解释成 HCCL 缺失。

## 正式实例

| 实例 | PID | 可见 NPU | HTTP | rendezvous `MASTER_PORT` | HCCL socket 范围 | 运行目录 |
|---|---:|---|---:|---:|---|---|
| A | 1227068 | `0,1,2,3` | `0.0.0.0:8801` | `29511` | `60000-60999` | `/home/mseco/h3-910a/run/h3-a` |
| B | 1227610 | `4,5,6,7` | `0.0.0.0:8802` | `29512` | `61000-61999` | `/home/mseco/h3-910a/run/h3-b` |

两个实例的启动方式都是：

```bash
cd /home/mseco/h3-910a
bash runtime/h3-instance-control.sh start a
bash runtime/h3-instance-control.sh start b
```

检查：

```bash
cd /home/mseco/h3-910a
for inst in a b; do
  bash runtime/h3-instance-control.sh status "$inst"
done
curl --noproxy '*' --fail --silent http://127.0.0.1:8801/system_stats
curl --noproxy '*' --fail --silent http://127.0.0.1:8802/system_stats
```

## 只读验收证据

验收时间：2026-11-08T17:05–17:07Z。

### HTTP 和节点可用性

两个实例均满足：

- `/system_stats` 返回 HTTP 200。
- `/queue` 返回 `{"queue_running": [], "queue_pending": []}`。
- `/object_info` 同时包含以下节点：
  - `MultiNPUParallelConfig`
  - `MiniMaxH3ImageToVideo`
  - `MiniMaxH3VAEDecodeParallel`
  - `MiniMaxH3TurboLoRA`
  - `MiniMaxH3TurboSampler`
- `ss` 只显示目标正式端口分别由 A/B 主进程监听：
  - `8801` 由 PID `1227068` 监听。
  - `8802` 由 PID `1227610` 监听。

两个 `system_stats` 都报告 4 个 NPU，单卡 `vram_total=34359738368` 字节（32 GiB）。B 的系统视图没有看到 A 的 `0–3`，A 也没有看到 B 的 `4–7`。

### 进程环境隔离

从 `/proc/<pid>/environ` 读取到的关键环境变量如下：

| 变量 | A | B |
|---|---|---|
| `H3_INSTANCE` | `a` | `b` |
| `ASCEND_RT_VISIBLE_DEVICES` | `0,1,2,3` | `4,5,6,7` |
| `MASTER_PORT` | `29511` | `29512` |
| `HCCL_NPU_SOCKET_PORT_RANGE` | `60000-60999` | `61000-61999` |
| `ASCEND_PROCESS_LOG_PATH` | `run/h3-a/ascend-log` | `run/h3-b/ascend-log` |
| `TMPDIR` | `run/h3-a/tmp` | `run/h3-b/tmp` |
| `XDG_CACHE_HOME` | `run/h3-a/cache` | `run/h3-b/cache` |
| `HF_HOME` | `run/h3-a/cache/huggingface` | `run/h3-b/cache/huggingface` |
| `TORCHINDUCTOR_CACHE_DIR` | `run/h3-a/cache/torchinductor` | `run/h3-b/cache/torchinductor` |
| `TRITON_CACHE_DIR` | `run/h3-a/cache/triton` | `run/h3-b/cache/triton` |

两个运行目录各自包含独立的 `comfyui.db`、`user/`、`input/`、`output/`、`temp/`、`tmp/`、`cache/`、`ascend-log/` 和 `extra_model_paths.yaml`。模型文件可以共享只读 root；可写运行状态不共享。

### HCCL 双组并发探针

执行命令：

```bash
cd /home/mseco/h3-910a
bash runtime/hccl-dual-probe.sh
```

本次探针日志：

- A 组：[`logs/hccl-dual/group-a-20261108T170614Z.log`](/home/mseco/h3-910a/logs/hccl-dual/group-a-20261108T170614Z.log)
- B 组：[`logs/hccl-dual/group-b-20261108T170614Z.log`](/home/mseco/h3-910a/logs/hccl-dual/group-b-20261108T170614Z.log)

探针参数：

| 组 | NPU | rank 数 | rendezvous 端口 | socket 范围 |
|---|---|---:|---:|---|
| A | `0,1,2,3` | 4 | `29611` | `60000-60999` |
| B | `4,5,6,7` | 4 | `29612` | `61000-61999` |

每个 rank 的 `HCCL_PROBE` 都报告：

```json
{"all_gather_ok": true, "all_reduce": [[10.0, 100.0]], "ok": true}
```

实际结果：

- A 的 local rank `0,1,2,3` 全部通过。
- B 的 local rank `0,1,2,3` 全部通过。
- 两组的 `visible_devices` 分别是 `0,1,2,3` 和 `4,5,6,7`。
- 探针日志中没有 `Traceback`、HCCL timeout 或探针失败。

探针运行期间主实例仍保持监听；探针结束后再次检查：

- A PID 仍为 `1227068`，端口 `8801` 正常。
- B PID 仍为 `1227610`，端口 `8802` 正常。
- 没有遗留的 `hccl_allreduce_probe` 或 `torch.distributed.run` 进程。

注意：探针会占用 8 张 NPU 做短时集合通信，应在没有正式生成请求时运行；它能验证通信和端口隔离，但不能替代模型加载或 H3 推理验收。

## 模型下载状态

以下内容保留为 2026-11-08T17:07Z 的下载过程快照；下载已经完成，最终文件、
大小和 SHA256 见文末“收尾复核”章节。

下载进程仍在运行，验收期间未停止：

```text
PID 1155419  bash /tmp/download-h3-models.sh
```

已完成 DiT：

```text
/home/mseco/h3-910a/models/diffusion_models/minimax_h3_fl2va_pruned_int8_convrot.safetensors
20970379616 bytes
```

文本编码器仍在下载：

```text
/home/mseco/h3-910a/models/text_encoders/qwen3vl_32b_minimax_h3_int8_convrot.safetensors.part
11635339264 bytes at 2026-11-08T17:07:07Z
```

下载日志中的 `.part` 文件在验收期间持续增长，说明下载进程仍活跃。不要对 `/home/mseco/h3-910a/models` 执行删除、清理或重新下载。

当前磁盘和内存快照：

- 根文件系统：约 `984 GiB`，已用约 `106 GiB`，可用约 `828 GiB`。
- 主机内存：总约 `754 GiB`，可用约 `729 GiB`。

## 当时未决风险和下一步

以下保留 2026-11-08T17:05–17:07Z 验收时的判断；第 2、3 项已在后续只读复核中
更新，当前重启风险以文末“收尾复核”章节为准。

1. **尚未进行 H3 模型加载与生成。** HTTP 健康、节点存在和 HCCL 通过不能证明 INT8 ConvRot 权重能在 910A 上执行。
2. **日志告警已完成只读复核。** `TBE Subprocess[task_distribute] raised error[]` 只出现在旧 `8189/8190` 失败启动段；正式 `8801/8802` 启动段没有新增 TBE 错误。AIMDO 的自动探测警告没有阻止服务启动，但这不等于推理路径已验证。
3. **模型下载已完成并校验。** 六个文件均已就位，没有残留 `.part`；SHA256 与服务器 `artifacts/manifests/models.sha256` 一致。
4. **双路只完成了基础设施并发验证。** 尚未提交两个独立 H3 工作流并测量采样、VAE、HBM、产物和端到端延迟。
5. **本地目录中曾出现错误的 `Ascend-cann-910b-ops_9.2.0_linux-aarch64.run`。** 服务器的已安装包已核实为 `Ascend-cann-910-ops`；任何部署和提交都必须使用 910-ops，禁止使用 910b-ops。
6. **探针中的 `hccl_build_version=unknown`。** 这是版本探测字段的问题；CANN 安装清单显示 `hccl: 9.2.0`。后续验收可补充更直接的 HCCL 动态库版本读取。

## 验收判定

| 项目 | 判定 | 依据 |
|---|---|---|
| 两个 ComfyUI 进程健康 | PASS | `system_stats` HTTP 200，PID 稳定 |
| 两个实例端口隔离 | PASS | `8801`/`8802` 分别由独立 PID 监听 |
| 两个实例 NPU 隔离 | PASS | 各进程环境为 `0,1,2,3` / `4,5,6,7` |
| 两个实例目录和缓存隔离 | PASS | 独立 `run/h3-a`、`run/h3-b` 路径 |
| 两个四卡组 HCCL | PASS | 8 个 rank 的 AllReduce 和 AllGather 全部成功 |
| 主实例在探针后继续运行 | PASS | 探针后 PID 未变，HTTP 200 |
| 模型文件和 SHA256 | PASS | 六个文件完成校验，和服务器 SHA256 清单一致 |
| H3 单路视频生成 | 未验证 | 尚未加载完整模型并提交工作流 |
| H3 双路并发生成 | 未验证 | 尚未提交两个实际生成任务 |

## 收尾复核

复核时间：2026-11-09T02:40Z。此次只读取状态、日志、进程环境、文件大小与
SHA256；没有重启 A/B，没有修改 `ops.py`，也没有读取或记录任何密码。

### 当前服务和 NPU 视图

| 实例 | PID | 可见 NPU | HTTP | 当前状态 |
|---|---:|---|---:|---|
| A | `1227068` | `0,1,2,3` | `8801` | 进程存活，`system_stats` HTTP 200，4 个 NPU |
| B | `1227610` | `4,5,6,7` | `8802` | 进程存活，`system_stats` HTTP 200，4 个 NPU |

两个实例均报告 ComfyUI `0.32.0`。当前进程环境仍是
`COMFYUI_MULTI_DEVICE_BACKEND=auto` 和
`COMFYUI_TEXT_TP_BACKEND=host`，并启用了：

```text
H3_910A_QUANT_COMPAT=1
H3_910A_QUANT_CHUNK=1024
H3_RESERVED_QUANT_COMPAT=1
```

### 日志告警复核

两个日志当前都是 161 行：

- A：`/home/mseco/h3-910a/logs/h3-a/comfyui.log`
- B：`/home/mseco/h3-910a/logs/h3-b/comfyui.log`

第 1–92 行属于旧 `8189/8190` 启动尝试。`TBE Subprocess[task_distribute]
raised error[], main process disappeared` 只出现在第 83–90 行。正式启动从
第 100 行开始，第 159–161 行分别进入 `Starting server` 并监听 `8801` 和
`8802`；正式启动段没有新的 TBE 错误。

正式启动段仍出现：

```text
Could not autodetect AIMDO implementation, assuming Nvidia
```

但服务随后成功识别 4 张 NPU、加载自定义节点并响应 HTTP。该警告不阻塞当前
ComfyUI 启动；它不能证明 H3 模型加载、INT8 ConvRot 执行或实际生成可用。

### 模型文件与 SHA256

服务器已有权威校验清单
`/home/mseco/h3-910a/artifacts/manifests/models.sha256`，完成标记为
`2026-11-08T17:50:50Z`。本次又对六个文件重新计算 SHA256，结果全部一致，
且没有残留 `.part` 文件。

| 文件 | Bytes | SHA256 |
|---|---:|---|
| `diffusion_models/minimax_h3_fl2va_pruned_int8_convrot.safetensors` | `20970379616` | `e889202c41dafb67b10d67b97f0d8541508036a6090af23425a5c2615d03c47a` |
| `text_encoders/qwen3vl_32b_minimax_h3_int8_convrot.safetensors` | `27141342152` | `bc2ced0fbea64757fa9acddccfc0b3f4819d1dcf1da6c124d690d368be283923` |
| `vae/minimax_h3_video_vae_fp16.safetensors` | `5207808496` | `7c1f131492e7eddacaac9069a61b81bdd39de5cc96561e677c5eab1cdce5e522` |
| `vae/minimax_h3_audio_vae_fp32.safetensors` | `605254808` | `8e505d95dd1561d47abd43d4238fd40d9bb1ae9e147ed0a4cba778d76ae4db48` |
| `loras/minimax_h3_turbo_v4_step600_ema.safetensors` | `779849816` | `5f3a626cd72c93a8b9318d6760c510bc5092d2ab13aaba1f932c5bab07a416d3` |
| `loras/minimax_h3_fl2v_turbo_4step_v1.0_768p_comfyui_bf16.safetensors` | `1956192992` | `c396a9a06f58399e9df9754b18299818d84a2ddd371724ba48fe4a41221437dc` |

正式生成工作流使用前五个文件。最后一个 `fl2v_turbo_4step` LoRA 是旧版本，
只保留用于回滚或对比，不属于正式生产五件套。

### 模型配置路径

服务器现有以下三份配置，内容相同，SHA256 都是
`ac5b7d675b12f8ab6a2377c52608cc7444edb07b80091193a15ed6eab995399b`，
并指向 `base_path: /home/mseco/h3-910a/models`：

```text
/home/mseco/h3-910a/extra_model_paths.yaml
/home/mseco/h3-910a/run/h3-a/extra_model_paths.yaml
/home/mseco/h3-910a/run/h3-b/extra_model_paths.yaml
```

当前 A/B 进程命令显式使用各自的 `run/h3-a`、`run/h3-b` 配置，模型目录扫描
正常。缺口是：

```text
/home/mseco/h3-910a/config/extra_model_paths.yaml
```

服务器上不存在该文件。当前旧启动器默认读取根目录配置，所以 A/B 不受影响；
本地新版启动器默认读取 `config/extra_model_paths.yaml`。如果直接部署新版
启动器而不补该文件，启动会在配置检查阶段失败。

### 重启前必须处理的差异

| 风险 | 影响 | 下次启动前的要求 |
|---|---|---|
| `run/h3-a/instance.json` 和 `run/h3-b/instance.json` 缺失 | 服务器旧控制脚本只读 PID，当前 `status` 仍正常；本地新版控制脚本依赖 token/PID 元数据，直接替换后会把现有实例视为未登记或遇到旧 PID 文件 | 在新控制脚本接管前完成元数据迁移，或在明确停机窗口内规范重建；不要直接覆盖后 `start` |
| 服务器与本地启动脚本哈希不同 | 不能假设本地已验证的启动逻辑已经生效 | 对比并同步 `runtime/start-h3-instance.sh`、`runtime/h3-instance-control.sh`、`scripts/run_comfy_instance.sh`，并在新目录先做一次启动演练 |
| 本地新版 `start-h3-instance.sh` 需要 launcher 预写元数据 | 在 `instance.json` 不存在时绕过控制脚本直接启动，会因 token/instance 身份校验失败 | 统一通过 `scripts/run_comfy_instance.sh` 或兼容控制脚本启动；不要直接裸跑 foreground 脚本 |
| `comfy/ops.py` 的 mtime 为 `2026-11-08T17:50:22Z`，A/B 分别在 `16:57:55Z` 和 `16:58:19Z` 启动 | 当前长驻进程很可能没有导入 17:50 后的 910A compat 改动，不能把它们当作新补丁的验收样本 | 重启后确认实际加载版本和 quant 路径，再运行最小模型 smoke；不要只看 HTTP 200 |
| 当前进程使用 `COMFYUI_MULTI_DEVICE_BACKEND=auto` 和 `COMFYUI_TEXT_TP_BACKEND=host` | 当前健康状态不是 strict HCCL 或新默认后端组合的验收 | 明确选择保留旧组合还是切到新版 strict HCCL，并分别记录；切换后重新跑真实双组验证 |
| `config/extra_model_paths.yaml` 缺失 | 新版启动器默认配置检查失败 | 从现有三份已验证配置同步一份到 `config/`，保持模型路径和权限一致 |

另有一个旧脚本 `runtime/restart-v1.sh`，默认目标是 `8189`，依赖
`start_comfyui-4npu.sh` / `stop_comfyui-4npu.sh`，不是当前 `8801/8802`
正式实例的控制入口。不要用它重启 H3 A/B。

### 判定

| 层级 | 判定 | 说明 |
|---|---|---|
| 基础设施与服务启动 | PASS | A/B 均存活，HTTP 200，4+4 NPU 隔离，HCCL 双组探针通过 |
| 模型文件完整性 | PASS | 六个文件大小和 SHA256 全部匹配权威清单 |
| 当前日志告警 | 不阻塞启动 | TBE 错误属于旧失败启动；AIMDO 警告未阻止当前服务监听 |
| H3 模型加载 | 未验证 | 尚未完成真实 checkpoint 加载与激活 |
| H3 单路视频生成 | 未验证 | 尚无成功产物和端到端指标 |
| H3 双路并发生成 | 未验证 | 尚无两路同时提交、采样、VAE 和资源峰值证据 |

