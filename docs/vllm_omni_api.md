# 用 vLLM-Omni 协议对外提供 H3（OpenDirector 直连）

## 结论先说

OpenDirector 是纯 HTTP 客户端：它按固定的协议形状调用模型服务，不加载插件、不改我们的代码。
它内置了 vLLM-Omni 的视频 `/videos` 协议，所以我们只要让 910A 这边"说这套话"，就能直接接入。

本仓库用 `scripts/h3_vllm_omni_server.py` 做这件事：**ComfyUI 只当执行后端**，对外暴露
`/videos` 三件套。

```text
OpenDirector ──POST /videos──▶ h3_vllm_omni_server :9101 ──POST /prompt──▶ ComfyUI :8801 ──▶ 910B ×4（0-3）
             ◀──GET /videos/{id}────┘                      ◀──/history──┘
             ◀──GET /videos/{id}/content──────────────────────────────────┘（/view 取产物）

                                :9102 / :8802 为第二组（4-7），用于双路并发
```

## 启动与管理

```bash
cd /home/mseco/h3-910a

# 第一组（卡 0-3，ComfyUI 8801，服务 9101）
bash runtime/omni-control.sh start smoke     # 短样例工作流
bash runtime/omni-control.sh start target    # 正式 864x480 / 124 帧

# 第二组（卡 4-7）
COMFY_URL=http://127.0.0.1:8802 OMNI_PORT=9102 bash runtime/omni-control.sh start target

bash runtime/omni-control.sh status
bash runtime/omni-control.sh stop
```

日志：`logs/omni/server-<port>.log`；产物：`run/omni/<job_id>.mp4`。

## 接口

| 方法 | 路径 | 说明 |
|---|---|---|
| POST | `/videos`（也接受 `/v1/videos`） | multipart 提交，返回 `{"id","status":"queued"}`，HTTP 202 |
| GET | `/videos/{id}` | `{"id","status"}`，`status ∈ queued / in_progress / completed / failed / cancelled / expired` |
| GET | `/videos/{id}/content` | 成功时返回 `video/mp4` 字节流 |
| DELETE | `/videos/{id}` | 取消排队中的任务（运行中会调用 ComfyUI interrupt） |
| GET | `/models`（或 `/v1/models`） | 能力声明，含 `videoCapabilities`，供 OpenDirector「读取服务声明」使用 |
| GET | `/health` | 存活 + 后端地址 |

## 字段映射

| vLLM-Omni 字段 | 落到工作流的位置 |
|---|---|
| `model` | 服务端忽略（模型由工作流固定） |
| `prompt` | `MiniMaxH3ImageToVideo.prompt` |
| `num_frames` | `MiniMaxH3ImageToVideo.length`，必须 `17n+5`，且落在 4–15 秒 |
| `size`（`宽x高`） | `MiniMaxH3ImageToVideo.width/height`，必须 32 的倍数 |
| `extra_params.aspect_ratio` | 仅在没给 `size` 时用来选默认尺寸（16:9→864×480，9:16→480×864，1:1→640×640，4:3→640×480，3:4→480×640） |
| `extra_params.task` | `t2va` 不连首帧；`fl2va`/`ref2va` 连首帧图片 |
| `fps` | 固定 24（`CreateVideo.fps` 与 H3 帧网格都按 24 设计），传别的值直接 400 |
| `input_reference` / `input_references` | 先 `/upload/image` 上传，再写进 `LoadImage.image` |
| `seed`（可选） | `RandomNoise.noise_seed` |

## OpenDirector 侧配置

1. 模型接入 → 添加视频模型 → 提供商选 **vLLM-Omni 兼容 → 视频接口**。
2. 基础地址填 `http://<ts-217 地址>:9101`（不需要 `/v1`；`endpoint()` 直接拼 `/videos`）。
3. Model ID 填 `minimax-h3`（这样适配方式选"自动"也能识别；自定义名字就手动选适配方式）。
4. 鉴权选"无"（自部署默认 `authMode=none`）。
5. 视频适配方式选 **原生 H3**（`minimax_h3`）：支持首帧、素材参考、480P/768P、4–15 秒。
6. 可选：点「读取服务声明」，`GET /models` 里的 `videoCapabilities` 会自动填分辨率/时长档位。

第二组填 `http://<ts-217 地址>:9102`，即可在 OpenDirector 里看到两个独立的视频模型。

## 当前限制（会返回明确的 4xx，不会静默降级）

- **只支持一张参考图**。多素材 `ref2va` 会返回 `unsupported_reference_count`；参考视频/音频尚未接通。
- **不支持 `continue_from` 接续**。ComfyUI 图没有 latent 缓存，返回 `unsupported_continuation`。
- **固定 24 fps**、**4–15 秒**、**32 像素对齐**——这些是 H3 的帧网格与 patch 对齐要求。
- 每个实例同一时刻只跑一个任务（ComfyUI 队列串行），多个请求会排队而不是并行。

## 本地验证

```bash
~/miniconda3/bin/python -m pytest artifacts/design/tests/test_h3_omni_contract.py \
    artifacts/design/tests/test_h3_workflow_ui_to_api.py -q
```

覆盖帧网格、尺寸对齐、`extra_params` 校验、任务类型与参考图的组合、能力声明结构，以及
UI→API 转换里"widget 输入带 `link: null` 不能被当成断开"这个回归点。

## 已知问题

- `ui_to_api` 早期版本会把所有 widget 输入当成"已断开"而丢弃（`link: null`），导致 `/prompt`
  报 `Required input is missing`。已修复并由上面的回归测试锁定。