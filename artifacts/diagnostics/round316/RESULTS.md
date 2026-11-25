# R316 — encoder 常驻实验：启动链继续受环境前提阻断（未拿到数据）

## 一、本轮进展（基础设施层）

把「只关 encoder offload」做成可执行的包装器 `run_stage292.py`（零修改官方脚本）：

```python
import gen_stage292 as g                       # 官方脚本副本（仅 mkdir exist_ok=True）
import stageprobe290; stageprobe290.install()  # wrap encode_prompt / decode_latent
from vllm_omni.diffusion.offloader import layerwise_backend as lb
lb.enable_plan_encoder_layerwise_offload = lambda *a, **k: False   # encoder 常驻
g.main()
```

依次踩掉的坑（每项都已定位到源码行）：

| # | 失败 | 根因与修法 |
|---|---|---|
| 1 | `FileExistsError` | `out.mkdir(exist_ok=False)` 在 8 rank 下竞态 ⇒ 副本改 `exist_ok=True` |
| 2 | `assert _DLO_DP_WAVE_TIMEOUT_S>=7200` | 需 `VLLM_OMNI_DLO_DP_WAVE_TIMEOUT=7200`（`multiproc_executor.py:46`） |
| 3 | `assert GENERATION_ROOT.name.startswith('NPU_full_generation126')` | 标签必须保留该前缀 |
| 4 | `SetPrecisionMode ... 500001` | **探针在 `main()` 之前 import pipeline**，与官方 `allow_hf32=False` 的时序冲突 |

## 二、当前卡点与下一步

第 4 条是本轮新卡点：探针必须在 **`main()` 设完 `allow_hf32=False` 之后、引擎构造之前**安装，
但又不能改动官方脚本。可行做法：

- 用 `sys.meta_path` 钩子**延迟**到 `pipeline_minimax_h3` 被 import 时再打补丁（不在进程启动时 import）；
- 或包装 `Omni.__init__`，在首次构造引擎时安装探针（需确认早 import `omni` 不触发 acl 初始化）。

## 三、状态

- **encoder 基线仍是 109 s 量级**（既有收据 `live_encoder_elapsed_s`），目标 < 2 s，差距 ≈55×；
  结构原因已确认：encoder 每次 encode 都要流式载入 63 GB（`layerwise offload ... 77 blocks`），
  而 TP8 常驻只需 **~7.9 GB/卡**（每卡 32 GB），**容量上完全可行**；
- **video VAE 20.565 s** vs 目标 < 20 s（差 2.8%），tiling 微调候选待测；
- 交付不变：**16.34 s/step（−8.9%）**，49 步 ×2 验证，relRMS 在原生带内，`H3_PERF_COMBO=1` 默认 OFF；
- 15 s：R313 路线级结论 ⇒ 当前条件下不可达；
- 机器 ts-217 直连正常；远端 master 含本轮记录。
