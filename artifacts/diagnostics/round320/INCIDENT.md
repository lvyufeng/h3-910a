# R320 — INCIDENT：ts-217 在 encoder 常驻实验中失去响应（未恢复）

## 一、时间线（北京时间 2026-11-26）

| 时间 | 事件 |
|---|---|
| 19:18 | stage320c 基线生成启动（encoder offload 默认）|
| 19:22 | `Orchestrator ready with 1 stages`（TBE 缓存换新后恢复）|
| 19:47 | 49/49 步完成（denoise 23:14，含冷编译；稳态 17.1 s/it）|
| 19:47–20:2x | 进入 VAE 阶段但**长期未完成**（>35 min，正常 warm 20.6 s / cold 142 s）|
| ~20:05 | 启动 stage321（encoder 常驻 + 探针）；两轮**并发抢 8 卡** |
| 20:11 | stage321 `EOFError` + Orchestrator 初始化失败（子进程被拖死）|
| 20:2x | kill 两轮，启动 stage322（单独 encoder 常驻 + 探针）|
| 20:2x | stage322 worker 起来（8 个 worker 均打印 `encoder offload disabled`）；**host `free -g` 显示 750/754 GB 已用、仅 1 GB 可用** |
| 20:34 | **ts-217 ssh 无响应** |
| 20:41–20:55 | 5 次重试均 timeout；`ts_probe` 也是 `OFFLINE` |

## 二、当前状态

- **ts-217 不可达**（ssh `Operation timed out`；fleet bridge 同样 `OFFLINE`）；
- 无法确认进程是否被 OOM killer 清理；**没有任何远程入口**重置；
- 按合同：**不允许** reset device、不允许重启他人环境；因此我**不会**尝试任何硬复位手段。

## 三、根因判断（诚实）

`H3_SKIP_ENCODER_OFFLOAD=1` 让 text encoder（63 GB）**常驻**，在 8 rank 下 host RAM 占用逼近上限
（754 GB 中 750 GB 已用），实验开始后约 10 分钟 host 失去响应。
**极可能是 host 内存耗尽 / swap 抖动导致 ssh 与系统服务失去响应**，而不是 NPU 故障。

## 四、需要用户/运维做的事

1. **重启或物理登录 ts-217**（我无权限、也不应做）；
2. 重启后建议先确认：
   - `free -g` 与 `dmesg | grep -i oom` 是否有 OOM 记录；
   - `npu-smi info` 是否正常；
3. 恢复后如需继续 encoder 常驻实验，必须**限制常驻范围**（例如只让部分 rank 常驻、或先做单卡容量测试），
   不应在 8 rank 上一次性验证。

## 五、本轮之前已确认的成果（未受影响，均已入库）

- **TBE 缓存损坏是 orchestrator 初始化失败的根因**；`mv ~/atc_data` 后管线恢复，
  完整生成可跑通 49 步（stage320c：denoise 23:14，稳态 17.1 s/it）；
- **探针遮蔽问题已修**：PYTHONPATH 顺序 + chain-load 官方 sitecustomize，
  322 轮 8 个 worker 均成功 patch（`wrappers=3`、`encoder offload disabled`）；
- 交付仍是 **16.34 s/step（−8.9%）**，`H3_PERF_COMBO=1` 默认 OFF。

## 六、回退记录

- `~/atc_data` 已被改名为 `~/atc_data.bak_h3_1918`（可回退）；
- 我方实验目录 `/tmp/h3_r221/cann_*`、`NPU_full_generation126_stage*` 已按轮次清理；
- 遗留 `/tmp/h3_r221/oppFA*`（19 个）因权限受限未能删除，均在 /tmp、不影响他人。
