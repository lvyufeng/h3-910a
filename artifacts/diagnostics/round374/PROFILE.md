# R374 — 首次拿到真实 profiling（msprof）：FA 内核是「多 pipe 混合受限」，并由此找到 −6.5% 候选

## 一、profiling 是怎么通的（此前判定"不可用"是错的）

- 早前结论 `msprof` "Operation not permitted" **是误判**：真实原因有两层
  1. `--application="python -c '...'"` 的**引号嵌套**把脚本写坏（SyntaxError）；
  2. 本机 venv 的 `vllm_omni` 是 editable 安装，`msprof` 启动 app 时丢了该路径。
- 正确做法：**wrapper 脚本**（`/tmp/wrap_fa.sh`）里补 `PYTHONPATH=/home/mseco/vllm-omni:$PYTHONPATH`，
  再 `msprof --output=... --aic-metrics=PipeUtilization --application="/bin/bash /tmp/wrap_fa.sh script.py"`。
- 产物：`PROF_*/mindstudio_profiler_output/op_summary_*.csv`，含每 op 的
  `vec_ratio / mac_ratio / scalar_ratio / mte1_ratio / mte2_ratio / mte3_ratio / memory_bound / cube_utilization`。

## 二、FA 内核真实 pipe 占用（N=41792, B=1, H=7, D=128, block=30）

| 配置 | 内核时间 | vec | mac(cube) | scalar | mte1 | mte2 | mte3 | memory_bound |
|---|---|---|---|---|---|---|---|---|
| **基线 64/912** | 135.4 ms | 36.9% | 28.4% | 12.4% | 22.7% | 20.1% | 7.7% | **54.5%** |
| V1 128/384 | **126.5 ms** | 49.1% | 27.1% | 21.6% | 20.1% | 11.7% | 10.5% | 23.8% |
| 双缓冲流水 64/384 | 203.0 ms | 31.9% | 21.1% | 16.3% | 18.7% | 14.1% | 6.7% | 44.2% |

结论：
1. 基线**没有任何单一 pipe 饱和**（cube 仅 28%），`memory_bound=54.5%` —— 是**访存+依赖混合受限**，
   这解释了此前"去掉 exp 不掉时间""重排/双缓冲无收益"的全部现象；
2. **减小 Bc、增大 Br（128/384）把 GM 流量减半**：mte2 20.1%→11.7%，memory_bound 54.5%→23.8%，
   内核 135.4→126.5 ms；代价是 vec/scalar 上升（块数变多，output_update 重标定变多）；
3. 双缓冲流水版 mac 21%、vec 32% 都掉下来了，但总时间反而 203 ms —— 同步/迭代开销压过收益，与 R371 一致。

## 三、由此得到的性能候选

关键改动是 **V1：把 `PV(j)` 的 cube matmul 提到 `QK(j+1)` 之前**（纯调度重排）。
单独在 64/912 下它是负优化（139.95），但在 **Br=128** 下是正优化：

| 配置 | 事件计时中位 | vs 基线 |
|---|---|---|
| 基线 pkg 128/384（无 V1） | 141.99 ms | +3.1% |
| **V1 pkg 128/384** | **128.81 ms** | **−6.5%** |
| V1 pkg 128/448 | 128.83 ms | −6.5% |
| V1 pkg 128/480 | 128.83 ms | −6.5% |
| 基线 64/912（交付配置） | 137.79 ms | — |
| V1 256/* | UB 溢出（`Oi_ub_zN` 131072B > 10880B），不可行 |

## 四、数值验证（模型实际走的 trim 路径，`bb218trim`）

| 对比 | relRMS | max_abs | 差异元素占比 |
|---|---|---|---|
| 基线自重复 | 0 | 0 | 0 |
| 候选自重复 | 0 | 0 | 0 |
| **候选 vs 基线** | **1.290e-04** | **4.768e-07**（1 个 fp16 ulp） | 6.86% |

⇒ 属 **1 ulp 级** tile 累加顺序差异，不是算法改变；但按契约仍属"可测数值变化"，需用户批准。

## 五、模型级影响（估算）

FA 每次调用省 8.98 ms × 49 次/步 ≈ **0.44 s/step**；即 16.34 s → **≈15.90 s/step**（待整模验证）。

## 六、下一步

1. 等用户对数值的裁决（relRMS 1.29e-4 / 1 ulp）；
2. 若批准：把 V1 重排固化进 `fa910a` 构建源 + 把交付 tile 改为 128/384，跑整模 49 步验证；
3. 继续压 vec(49%)/scalar(22%)：output_update 重标定是块数敏感项，可考虑 O 常驻 UB 或合并重标定。
