# MiniMax-H3 FL2VA 服务部署（8× Ascend 910A）与 OpenDirector 对接

## 1. 结论先行：为什么要一层网关

OpenDirector 的自部署视频接入协议 `vllm_omni` 固定调用 **异步视频作业 API**：

| 步骤 | 请求 | 响应 |
|---|---|---|
| 提交 | `POST {base}/videos`（multipart/form-data） | `{"id": "<job>"}` |
| 轮询 | `GET {base}/videos/{id}` | `{"status": "queued\|in_progress\|completed\|failed\|cancelled\|expired", ...}` |
| 取片 | `GET {base}/videos/{id}/content` | MP4（含音轨） |
| 模型 | `GET {base}/models` | 模型列表，`id` 必须命中适配器名字 |

`{base}` 就是填在 OpenDirector 里的 **Base URL**（形如 `http://host:8091/v1`）。

本机的 vllm-omni 是 **0.30.0 定制构建**：它有 `protocol/videos.py`、`video_api_utils.py`、
`video/generation/helpers.py` 这些请求模型与 multipart 解码工具，但**没有注册 `/v1/videos` 路由**
（`vllm serve --omni` 也不在本构建的 CLI 里）。直接升级到 FastH3 部署脚本用的 vllm 0.32 +
vllm-omni 固定 ref，会破坏已经验收的 910A 数值路径（R124/R126 + FA V1）。

因此采用**网关方案**：一个常驻 `Omni` 引擎 + 一层实现 `/v1/videos` 契约的 HTTP 网关
（`h3_service.py`）。OpenDirector 侧零改动，照常选“协议 = vLLM-Omni”。

```
OpenDirector (浏览器)  ──HTTP──▶  h3_service.py :8091
                                  ├─ /v1/videos*       (OpenDirector 契约)
                                  ├─ /v1/models
                                  ├─ /v1/generate      (内部冒烟/脚本)
                                  └─ 常驻 Omni 引擎（8×910A，TP2 DiT + USP4 + encoder TP8）
                                        └─ sitecustomize → 每个 worker 安装 R124/R126 对齐 + 三项加速
```

## 2. 文件

| 文件 | 作用 |
|---|---|
| `serve_h3.sh` | 启动器：CANN 环境、验证过的 R126 运行时 env、三项加速开关、单进程起服务 |
| `h3_service.py` | HTTP 网关 + 常驻引擎 + 串行作业队列 |
| `h3_optimizations.py` | 三项加速（FA V1 / guard fast / fast GEMM），可在任意进程幂等安装 |
| `h3_service_bootstrap.py` | 复刻 R126 的私有 IPC authkey + 选择性 bootstrap，并把加速挂到同一个钩子 |
| `sitecustomize.py` | 部署目录的 startup hook（必须排在 PYTHONPATH 最前） |

## 3. 部署前置（ts-217 已验证环境）

1. CANN 9.2.0：`/home/mseco/Ascend/cann-9.2.0/set_env.sh`
2. Python venv：`/home/mseco/vllm910a`（torch_npu + vllm 0.30.0 + vllm-omni 定制构建）
3. 模型与 fixture（保持 R124/R126 原样，禁止改动数值路径）：
   - checkpoint：`/tmp/h3_align12401/original_metadata124f/FL2VA`
   - bootstrap / 工具模块：`/tmp/h3_budget149`、`/tmp/h3_align12401`
   - 私有 IPC key：`/tmp/h3_budget149/private_IPC_key126.bin`（`serve_h3.sh` 缺失时自动生成）
4. FA V1 内核包：
   - `H3_FA215_PKG=/tmp/fa_pkg_v1`（源码，仓库 `artifacts/kernels/fa_v1/`）
   - `H3_FA215_KERNEL_DIR=/tmp/fa_kernels_v1`（**必须独立于基线的 `/tmp/fa910a_kernels`**，
     缓存身份里含包路径；否则首个请求要把 51 个形状全部重编，约 16 分钟）
   - 缓存缺失时按形状惰性编译（约 19 s/形状），首个请求会明显变慢，建议部署后先跑一次预热
5. R126 运行时要求（启动器已处理）：
   - `H3_R126_GENERATION_OUT` 必须是 `${H3_REF_ROOT}/NPU_full_generation126*` 下的**已存在目录**
     （`integration_hooks126` 的私有捕获根断言；服务自身的 mp4 输出走 `--output-root`）
   - `autocast_operand_cache126` 需在 `${H3_REF_ROOT}/vae/` 下（仓库
     `artifacts/diagnostics/round126/vae/autocast_operand_cache126.py`；本机 /tmp 曾被清理）
5. 依赖：fastapi / uvicorn / pydantic / python-multipart（venv 内已具备前三者）

## 4. 启动与自检

```bash
cd /path/to/artifacts/deploy
H3_PORT=8091 ./serve_h3.sh          # 前台；生产用 systemd/nohup
```

自检：

```bash
curl -s localhost:8091/healthz | jq        # engine_ready=true 之前为 loading（约 5-7 分钟）
curl -s localhost:8091/v1/models | jq
```

一次最小生成（4 步仅用于链路验证，生产用 49 步）：

```bash
curl -s -X POST localhost:8091/v1/videos \
  -F model=minimax-h3 -F prompt='a red ball bouncing' \
  -F size=1344x768 -F fps=24 -F num_frames=120 -F num_inference_steps=4 \
  -F 'extra_params={"task":"fl2va"}' \
  -F input_reference=@first.png -F input_references=@last.png | jq
curl -s localhost:8091/v1/videos/<id> | jq '.status'
curl -s -o out.mp4 localhost:8091/v1/videos/<id>/content
```

## 5. OpenDirector 配置（设置 → 模型接入 → 视频生成）

| 字段 | 值 |
|---|---|
| 接入方式 | 自部署服务 |
| 接口协议 | **vLLM-Omni** |
| Base URL | `http://127.0.0.1:8091/v1`（远端部署时经 SSH 隧道映射到本机端口） |
| Model ID | **`minimax-h3`**（适配器按该名字选中 `vllm-omni-h3` profile；服务端 `H3_SERVED_MODEL_NAME` 可改但两边必须一致） |
| 鉴权 | 无 |

OpenDirector 侧 `vllm-omni-h3` profile 的行为与本网关的对应关系：

| OpenDirector 发送 | 本网关处理 |
|---|---|
| `prompt` | 直接作为生成 prompt |
| `fps` / `num_frames`（按 `round(秒×24)`） | 透传；引擎把 120 对齐成 124（17×7+5），与验收配置一致 |
| `size=1344x768`（768P）/ `832x480`（480P） | 解析 WxH → width/height；1344x768 即验收分辨率 |
| `extra_params.task="fl2va"` | 映射到引擎 `extra_args.task=fl2va`、`frame_indices=[0,-1]` |
| `input_reference`（首帧）、`input_references[0]`（尾帧） | `multi_modal_data.image=[首帧, 尾帧]` |
| 未给 `num_inference_steps` | 默认 **49**（验收配置；FastH3 配方是 8 步，不适用于本无损 FL2VA） |
| `extra_params.task="ref2va"` | **400**：本部署只有 FL2VA checkpoint，参考生视频不可用 |
| `extra_params.continue_from` | **409** 且消息含 `continue_from`，OpenDirector 会转成“服务端无接续缓存”的友好提示 |

## 6. 性能与容量（验收数据）

| 项 | 值 |
|---|---|
| 引擎冷启动 | 约 5–7 分钟（权重 + encoder + VAE 常驻） |
| 单步 DiT | **15.55 s/step**（验收组合；基线 16.32–16.34） |
| 49 步视频 | ≈ 49×15.55 s + encoder/VAE ≈ **13–14 分钟/条**（1344×768、124 帧、含音轨） |
| 并发 | **串行**（8 卡独占）；排队长度见 `/healthz.queue_len` |
| 队列语义 | `POST` 立即返回 job id；OpenDirector 每 3 s 轮询一次 |

> 换算：一条 5 s 视频约 13–14 分钟，因此 OpenDirector 端的超时/重试要按 ≥20 分钟配置；
> 服务端不缓存跨请求 latent，`continue_from` 续写模式不可用。

## 7. 运维

- 端口：默认 `8091`（与 OpenDirector FastH3 部署脚本一致），仅绑定 `127.0.0.1` 时通过 SSH 隧道对外：
  `ssh -N -L 18091:127.0.0.1:8091 <user>@<host>`，Base URL 填 `http://127.0.0.1:18091/v1`。
- 日志：`serve_h3.sh` 标准输出；引擎初始化在 `[h3-service]` 行。
- 停止：`pkill -f h3_service.py`；确认 8 张卡释放后再重启（引擎独占设备）。
- 输出：`--output-root/<job_id>/video.mp4`。
- 冒烟不通过时先看 `/healthz.engine_error`（初始化失败会把完整 traceback 放在那里）。

## 8. 与纯 vLLM-Omni 原生部署的差异（为什么不是 `vllm serve --omni`）

| 项 | FastH3 官方部署（GPU/0.32） | 本部署（910A/0.30 定制） |
|---|---|---|
| 启动 | `vllm serve <model> --omni --config serve.yaml` | `serve_h3.sh`（常驻 Omni + 网关） |
| `/v1/videos` | 上游内置 | 网关实现（同一契约） |
| 精度路径 | NVFP4 量化单卡 | FP16 矩阵 + 原始 FP32 例外，TP2+USP4 |
| 步数 | 8 | 49（无损验收配置） |
| 与 OpenDirector | 协议 vLLM-Omni | **同为协议 vLLM-Omni，零客户端改动** |
## 9. 与 OpenDirector `vllm-omni-h3` 适配器的差异清单

| 项 | OpenDirector 行为 | 本部署 | 处理 |
|---|---|---|---|
| 服务名匹配 | `adapter` 按模型名自动判定：`minimax-h3` / `minimax-h3-ref2va` → `minimax_h3` | 服务端 `H3_SERVED_MODEL_NAME=minimax-h3` | 两边一致即可 |
| 帧角色 | `vllm-omni-h3` 的 `frameRoles` **未声明**，默认只允许 1 张首帧；只有 `fasth3_dual` 适配器才开放尾帧槽 | FL2VA 需要 **首帧+尾帧** | 见下方补丁；未打补丁时网关把单图复制成首/尾两帧（生成会收束回首帧，属语义降级） |
| 帧数 | 发送 `round(秒×24)`（5 s → 120），期望输出 17n+5 | 引擎把 120 对齐成 **124** | 一致 |
| 尺寸 | `size=WxH`，768P 16:9 → `1344x768`，480P → `832x480` | 解析 WxH → width/height | 一致 |
| 步数 | `vllm_omni` 路径不发送步数，服务端默认 | 默认 **49**（无损验收配置） | 一致 |
| 任务 | `extra_params.task` = `fl2va`（有图）/`ref2va`（素材参考）/`t2va` | 只支持 `fl2va` | `ref2va`→400、`t2va`→400 |
| 续写 | `videoMode=continuation` 发 `extra_params.continue_from` | 无跨请求 latent 缓存 | 409 + 消息含 `continue_from`，OpenDirector 会转成"服务端无接续缓存"提示 |
| 结构控制 | `fasth3` 专属 `control_type` | 不支持 | 400 |

### 建议的 OpenDirector 一行补丁（启用真正的首尾帧 FL2VA）

`packages/core/src/opendirector/domain/input_rules.py`，在 `adapter == 'minimax_h3'` 的 `rules.update(...)` 里补上：

```python
rules.update(profile='vllm-omni-h3', allowedKinds=['text', 'image', 'video', 'audio'],
             maxImages=9, maxAudios=3, imageBytes=30 * 1024 * 1024,
             requestBytes=64 * 1024 * 1024,
             videoModes=['first_frame', 'reference', 'continuation'],
             frameRoles=['first_frame', 'last_frame'],   # <- 新增：开放尾帧槽
             outputResolutions=['480P', '768P'])
```

效果：画布 `frame_roles` 变为 `{first_frame, last_frame}`（`creation_canvas.py:136`），
首尾帧最多 2 张，`generation.py:508` 的帧数上限随之放开；网关按 `input_reference`（首帧）+
`input_references[0]`（尾帧）直接映射到引擎的 `fl2va`，语义与验收配置完全一致。

> 该补丁只影响 OpenDirector 前端/校验的输入约束，不改动服务端与数值路径。
> 上游若更愿意保留"H3 自部署=单帧"的语义，也可以继续让网关做首帧复制，两种都能跑通。

## 10. 端到端验收记录

| 项 | 结果 |
|---|---|
| 引擎初始化 | 286.8 s / 321.7 s（两次冷启动） |
| 三项加速装载 | 每个 worker 日志均有 `[h3-opt] applied: fast_gemm,guard_fast,fa_v1_128x384` |
| `GET /v1/models` | 返回 `{"id":"minimax-h3"}` |
| `POST /v1/videos` | 200 + job id；multipart 字段与文件解析正确 |
| `GET /v1/videos/{id}` | `queued → in_progress → completed`，params 已脱敏（不含图像对象） |
| `GET /v1/videos/{id}/content` | 返回 MP4（见下） |

已知引擎行为：**4 步等脱离验收配方的步数会触发 vllm-omni 异步输出打包路径的协议错误**
（`Unexpected response type: AsyncDiffusionOutput`）；服务按验收配方 **49 步** 使用。
