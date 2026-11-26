# R375 — FA V1 整模验证通过：16.113 s/step（交付基线 16.32–16.34），relRMS 8.34e-04

## 一、候选（用户已批准 1-ulp 数值变化）

- **内核改动**：`compute_in_each_kv_block` 中把 `PV(j)` 提到下一块 `QK(j+1)` 之前（纯调度重排，diff 6 行）；
- **tile**：64/912 → **128/384**；
- 源码固化：`artifacts/kernels/fa_v1/flash_attention_fwd.py`（+README）；
- 运行时开关：`H3_PERF_COMBO=1`，并设
  `H3_FA215_PKG=<fa_v1 包根>`、`H3_FA215_KERNEL_DIR=<内核缓存>`、`H3_FA_BR_CAP=128`、`H3_FA_BC_CAP=384`。

## 二、整模 49 步口径测量（8 rank，torchrun 8 进程，TP2+USP4）

| 项 | 值 |
|---|---|
| 测量脚本 | `run_fa374.py`（= `run_combo259.py` + V1 内核环境），steps 0,10,25,40,48 |
| **step time（rank 一致）** | **16.1125 s/step** |
| 交付基线（R126 combo） | 16.32–16.34 s/step |
| 增益 | **−0.22 s（−1.3%）** |
| video relRMS vs R124 | **8.3435e-04**（原生自重复带 9.84e-04–1.077e-03 内） |
| 数值裁决 | 用户 2026-11-27 允许（FA 单体内核 relRMS 1.29e-04 / 1 fp16 ulp） |

## 三、FA 单体与模型的关系

| 口径 | 基线 | V1 128/384 | 差值 |
|---|---|---|---|
| FA 单次（N=41792,B=1,H=7,D=128，事件计时） | 137.79 ms | 128.81 ms | −8.98 ms |
| 内核自身（msprof op_summary） | 135.4 ms | 126.5 ms | −8.9 ms |
| 模型每步（49 次 FA） | — | — | 实测 −0.22 s（≈理论 0.44 s 的一半，其余被调度/负载波动吸收） |

## 四、profiling 方法（可复用，后续整模优化的基础）

```bash
source $CANN/set_env.sh
msprof --output=<dir> --task-time=l1 \        # l1 才带 op 名称；--aic-metrics=PipeUtilization 带 pipe 占比
  --application="/bin/bash <wrapper.sh>"      # wrapper 内补 PYTHONPATH=<vllm-omni>，否则 editable 安装丢失
```

- 产物：`PROF_*/device_N/sqlite/{ascend_task,ai_core_op_summary}.db`（原始）+（导出后）`op_summary_*.csv`；
- 单进程 app 会自动导出 CSV；**torchrun 8 进程** 目前只留原始 db，需自行按 `ascend_task` 聚合或用 `--task-time=l1` 重跑。

## 五、下一步

用 `--task-time=l1` 的整模 profile 做 **op 级端到端归因**，把 16.11 s 拆成
FA / GEMM / 集合通信 / copy / elementwise / 调度，再按占比挑下一批优化对象。
