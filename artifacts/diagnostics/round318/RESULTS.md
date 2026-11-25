# R318 — 完整生成管线在当前环境无法 bring-up（encoder/VAE 实测被阻塞）

## 一、按 R317 配方做的官方 launcher（已入库）

`deployed_launch_stage318.py`：完全照 `launch_budget149.py` 结构——

```python
cmd = ['/bin/bash','-c',
       'source .../set_env.sh; '
       'export PYTHONPATH=/tmp/h3_budget149:/tmp/h3_align12401:/tmp/h3_r221:$PYTHONPATH; '
       'export <全部 R126 环境变量 + H3_STAGE_PROBE=1>; '
       'exec /home/mseco/vllm910a/bin/python -u .../generate_full_NPU126.py --output-label <LABEL>']
Popen(cmd, cwd='/tmp/h3_budget149', start_new_session=True)
```

## 二、六种启动变体，全部止于同一处

| # | 变体 | 结果 |
|---|---|---|
| 1 | `torch_npu_run` + 官方脚本（runpy 包裹） | `FileExistsError`（mkdirdir 竞态） |
| 2 | 自写最小生成脚本 | `SetPrecisionMode 500001` |
| 3 | 补 `allow_hf32=False` 后 | Orchestrator 初始化失败 |
| 4 | `torch_npu_run` + 官方脚本副本（mkdir 竞态修掉） | Orchestrator 初始化失败 |
| 5 | `torch_npu_run` + 官方原脚本（外注 probe） | Orchestrator 初始化失败 |
| 6 | **单进程 python + 官方原脚本 + 官方 env（本轮）** | **Orchestrator 初始化失败**（内部异常为空） |

⇒ **不是探针问题**（去掉 probe 也失败），也不是 encoder-offload monkeypatch 问题；
**当前环境无法把完整生成管线拉起来**。`RuntimeError: Orchestrator initialization failed:` 内部异常为空，
说明 orchestrator 子进程在初始化早期退出（无 traceback 传回）。

## 三、对比：同一台机器上，**benchmark harness（49 步）此前是可跑的**

R259–R262 的 49 步长跑与 16.34 s 交付均在本机完成 ⇒ 差异在**完整生成路径**（Omni + text encoder +
VAE + 文件输出），而非 NPU/驱动本身。可能诱因：本会话自 R309 起做了大量 AscendC 算子包安装/卸载与
多次中断运行，导致共享状态不一致；按合同**不允许** reset device，故未做硬复位。

## 四、清理（按合同执行）

- 我方实验进程：`remaining_all=0`、`npu_procs=0`；
- 已删除：`/tmp/h3_r221/cann_*`、`/tmp/h3_r221/NPU_full_generation126_stage*`、`/tmp/h3_budget149/NPU_full_generation126_stage*`;
- **残留**：`/tmp/h3_r221/oppFA*` 共 19 个安装目录，其中 `.../scripts/uninstall.sh` 权限受限（非本用户可删），
  无法彻底删除；均为 `/tmp` 下我方私有文件、不占用 NPU、不影响其他人。磁盘 `984G` 用 `560G`、余 `374G`。

## 五、结论与建议

- **encoder/VAE 目标在本会话无法继续实测**：需要先把完整生成管线恢复；
- 恢复建议（下一会话第一件事）：在**干净会话**里先用官方 launcher 跑一次 `generate_full_NPU126.py`
  （不加任何探针）确认基线可用，再注入 `sitecustomize` 探针；
- 已实测基线（既有收据）：encoder **~109 s**（目标 <2 s，结构原因是 63 GB 逐层流式载入；
  TP8 常驻仅 ~7.9 GB/卡，**容量可行**）；video VAE **20.565 s**（目标 <20 s，差 2.8%）。

## 六、交付状态（不变）

**16.34 s/step（−8.9%）**，49 步 ×2 验证，video relRMS 全在原生自重复带内，
`H3_PERF_COMBO=1` 默认 OFF，远端 `master` 含全部记录。
