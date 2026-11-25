# R315 — encoder offload 没有开关；实现「encoder 常驻」需 monkeypatch

## 一、源码结论

- 只有 `enable_layerwise_offload` / `enable_distributed_layerwise_offload`（`omni_config.py:845-846`），
  **无单独 encoder offload 开关**；
- encoder 逐层 offload 在 `layerwise_backend.py` 的 `prepare_pipeline_components(...)` 中
  **无条件**安装（`enable_encoder_blocks → enable_plan_encoder_layerwise_offload`，L404-423）；
- ⇒ 要测「encoder 常驻」（63 GB / TP8 ≈ **7.9 GB/卡** < 32 GB），只能 monkeypatch：
  引擎初始化前把 `layerwise_backend.enable_plan_encoder_layerwise_offload` 换成返回 `False` 的 no-op。

## 二、探针启动链（4 次尝试全部失败）

| 尝试 | 失败点 |
|---|---|
| 直接 runpy 官方脚本 | 输出目录已存在 `FileExistsError` |
| 自写最小脚本 | 缺 `allow_hf32=False` → `SetPrecisionMode 500001` |
| 补 allow_hf32 | 缺官方 bootstrap patch → Orchestrator 初始化失败 |
| 复用官方脚本改标签 | `integration_hooks126.py:20` 断言目录前缀；改合法前缀后仍未通过（时间窗耗尽，最后错误未取到） |

## 三、下轮执行方案（最优）

用 `sitecustomize.py` 注入（PYTHONPATH 上自动加载，**官方脚本零修改**，绕开启动断言）：

```python
# /tmp/h3_r221/sitecustomize.py
import os
if os.environ.get('H3_STAGE_PROBE') == '1':
    import stageprobe290
    stageprobe290.install()
    if os.environ.get('H3_SKIP_ENCODER_OFFLOAD') == '1':
        from vllm_omni.diffusion.offloader import layerwise_backend as lb
        lb.enable_plan_encoder_layerwise_offload = lambda *a, **k: False
```

运行：`H3_STAGE_PROBE=1 H3_SKIP_ENCODER_OFFLOAD=1 torch_npu_run ... generate_full_NPU126.py --output-label <合法且不存在>`

## 四、判据

1. encoder 阶段耗时（目标 < 2 s）；
2. 若常驻后显存不足或数值改变 ⇒ 立即回退并如实记录；
3. video VAE 仍 20.565 s（目标 < 20 s），单独调 tiling 验证。

## 五、状态

- 交付不变：**16.34 s/step（−8.9%）**，49 步 ×2 验证，relRMS 在原生带内，`H3_PERF_COMBO=1` 默认 OFF；
- 15 s：R313 路线级结论 ⇒ 当前条件下不可达；
- 机器 ts-217 直连正常；远端 master 含 R314/R315 记录。
