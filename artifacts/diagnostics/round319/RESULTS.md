# R319 — 生成管线恢复：根因是 TBE 编译缓存损坏；探针被官方 sitecustomize 遮蔽

## 一、突破：完整生成管线重新可用

`RuntimeError: Orchestrator initialization failed:` 的真实原因（用 `threading.excepthook` + 更长的等待才暴露）：

```
[ERROR] TBE Subprocess[task_distribute] raised error[], main process disappeared!
OpsManager initialize failed.[FUNC:InnerInitialize][FILE:gelib.cc][LINE:225]
GEInitializeV2 initialize failed ... Init compiler failed
```

⇒ **`~/atc_data`（TBE 编译缓存）损坏**，导致 orchestrator 初始化时的算子编译失败并静默退出。

**修法（可回退）**：`mv ~/atc_data ~/atc_data.bak_h3_1918`。随后同一 launcher：

```
[Orchestrator] Starting event loop
[OmniEngine] Orchestrator ready with 1 stages
```

⇒ 管线恢复，生成跑进 49 步循环。

## 二、第二个发现：探针被官方 `sitecustomize` 遮蔽

- `/tmp/h3_budget149/sitecustomize.py` 与 `/tmp/h3_align12401/sitecustomize.py` 都存在；
- launcher 的 `PYTHONPATH=/tmp/h3_budget149:/tmp/h3_align12401:/tmp/h3_r221` ⇒ **只加载第一个**，
  我们的探针从未执行（`grep -c "[STAGE]"` = 0），解释了此前所有「探针无输出」。

**修法**：
1. 把 `/tmp/h3_r221` 放到 PYTHONPATH **最前**；
2. 我们的 `sitecustomize.py` 增加 **chain-load**：先 `exec` 官方那份（保留 H3 bootstrap 补丁），再 arm 探针。

## 三、本轮同时确认

- 环境侧错误信息会被吞掉：orchestrator 在后台线程，必须 `threading.excepthook` 才能看到栈；
- TBE 缓存损坏时，GE 初始化（`OpsManager`）失败会表现为「空消息」的 orchestrator 失败；
- 单进程 entry（`python generate_full_NPU126.py --output-label …`）才是正确启动方式
  （`torch_npu_run` 会让 8 个进程同时 `mkdir(exist_ok=False)` 竞态）。

## 四、下一步（已在跑）

当前 stage320c 基线生成（encoder offload 保持默认）跑完 49 步后，用修好的探针 +
`H3_SKIP_ENCODER_OFFLOAD=1` 跑 stage321，直接读 encoder / VAE 阶段耗时。

## 五、状态

- 交付不变：**16.34 s/step（−8.9%）**，49 步 ×2 验证，relRMS 在原生带内，`H3_PERF_COMBO=1` 默认 OFF；
- 15 s：R313 路线级结论 ⇒ 当前条件下不可达；
- 机器 ts-217 直连正常；远端 master 含本轮记录。
