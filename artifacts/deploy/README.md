# 部署开关：DiT 三合一性能组合（默认 OFF）

`h3_perf_switches.py` 提供**单开关、可逐项、可一键回退**的三项已验证性能改动。
它**不修改共享安装的 vllm-omni/vllm 源码**，只在启动进程内 monkeypatch。

## 用法

```bash
export H3_PERF_COMBO=1          # 总开关，默认 0 = 完全等同部署行为
export H3_PERF_FA_TILES=1       # FA tiles 64/912
export H3_PERF_FA_TRIM=1        # R218 padding-task trim
export H3_PERF_GUARD_FAST=1     # guard scale==1 快路径（逐位一致）
# PYTHONPATH 需含本目录与 fa218trim.py 所在目录
```

启动脚本在模型构建前调用：

```python
import h3_perf_switches as perf
perf.install(torch, label="...")   # 返回 applied/skipped 报告，可写入回执
```

## 已验证结果（49 步全量）

| 记录步 | rank_max 步时 | vs 基线 17.928 s | video relRMS vs R124 |
|---|---|---|---|
| 0 | 16.3149 s | −9.00% | 9.76e-04 |
| 10 | 16.3202 s | −8.97% | 5.13e-04 |
| 25 | 16.3405 s | −8.86% | 2.59e-04 |
| 40 | 16.3405 s | −8.86% | 3.28e-04 |
| 48 | 16.3255 s | −8.94% | 5.30e-04 |

- 中位 **16.32 s/step**，全程无漂移；
- 数值全程在原生自重复带（9.84e-04–1.077e-03）内；
- 距 15 s 目标仍差约 1.32 s（8%），后续在 #17（AscendC FA / 整图）继续。

## 回退

取消 `H3_PERF_COMBO`（或任一子开关）后重新启动即恢复部署行为；
也可以用 `perf.install` 返回的 `applied`/`skipped` 列表确认本次到底改了什么。

## 约束

- 不改 50 块 / 49 步 / FP16 矩阵 / 原 FP32 参数 / 155 条 Half 检查 / TP2+USP4；
- guard 快路径在 peak>32752 时原样回落部署实现（49 步实测触发 34 次）；
- FA 两项为用户 2026-11-25 批准的数值变更；guard 快路径在 scale==1 分支逐位一致。
