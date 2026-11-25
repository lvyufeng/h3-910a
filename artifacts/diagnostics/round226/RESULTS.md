# R226 — 把 R223 的 `rest` 4.121 s/step 拆开：88% 在 attention 与 MLP 的父区间里

## 一、做了什么

R223 把 8.118 s/step 的固定开销列为最大未知项，其中 `rest` 4.121 s/step 是减法残项、从未实测。
本轮在 ts-217 上跑**一个**测量步（`--steps 0 --warmup 1 --repeats 1`，原 `baseline` 模式、
TP2+USP4、50 块、`native` offload），在每个 block 及其子模块上挂**设备事件**，
forward 内零同步，只在步后 synchronize 一次。

- 驱动 `run_rest226.py` + 探针 `budget_probe226.py`（本仓库 `round226/deployed/`）
- 输入 pin 与 R149 记录逐一核对：`NPU_encoder_FP16_124.py` `0959895b…`、
  `h3_replay123/transformer.py` `5d94ba09…`、`same_initial_sampling123.pt` `8a560c43…`
- 8 个 rank 全部完成，receipt 在 `round226/receipts/candidate_rank*.json`
- 契约自检（来自回执）：`mode=baseline`、`full_transformer_layers=50`、
  `parallel_config={tp:2, ulysses:4, ring:1}`、`safety.matrix_checks=155`、
  `nonfinite_count=0`、`boundary_count=0`、`host_weight_bytes=32,279,142,400`

## 二、实测（8 rank 的中位数，ms/次）

| 标签 | 中位 | 范围 |
|---|---|---|
| `block`（父） | **345.775** | 345.539–345.937 |
| `attn`（父） | **243.397** | 243.089–243.576 |
| `mlp`（父） | **87.904** | 87.599–88.042 |
| `qkv` | 17.066 | 17.062–17.071 |
| `out_proj` | 18.361 | 18.073–18.658 |
| `fc1` | 23.220 | 23.078–23.480 |
| `fc2` | 29.978 | 29.817–30.291 |
| `adaln_proj` | 1.885 | 1.752–2.087 |
| `norm1` | 1.770 | 1.727–1.794 |
| `norm2` | 1.575 | 1.568–1.580 |

跨 rank 极差 ≤0.5 ms（除 `out_proj` 0.585），说明这是稳定的结构量而非抖动。

## 三、`rest` 的构成（每步 = ×50 块）

| 组成 | ms/block | s/step |
|---|---|---|
| `attn` 父区间 − qkv − out_proj − FA(149.08) − USP(21.54) − qknorm(4.60) | **32.55** | **1.63** |
| `mlp` 父区间 − fc1 − fc2 | **34.71** | **1.74** |
| `adaln_proj + norm1 + norm2` | 5.23 | 0.26 |
| `block` 父区间 − attn − mlp − adaln − norms（modulate/gate/residual） | 9.24 | 0.46 |
| **合计** | | **4.09** |

与 R223 的 `rest` 4.121 s/step 相差 **0.03 s**（0.7%），分解闭合。

## 四、结论

1. **`rest` 不是“一堆小算子”**：其中 **82%（3.37 s/step）落在 attention 与 MLP 的父区间内**
   （attn 1.63 + mlp 1.74），norms/adaln 只有 0.26 s，块级 modulate/gate/residual 0.46 s。
2. 这两个残项都约 **33–35 ms/block**，与“每块 646 MB 权重预取在 21.3 GB/s 下需 30.3 ms”
   **量级吻合**。因此首要假设是：**这部分主要是被暴露的权重预取等待**，
   而不是 silu/rope/reshape 这些每块只值 1–2 ms 的元素级算子。
3. 这条假设**尚未证实**：需要对照实验（权重常驻，或直接测 prefetch 的暴露等待）。
   在证实前不能把它算作“可回收”。
4. `mlp` 残项里确定包含 `silu(gate.float())*up.float()`（产生两个 300 MB FP32 临时量）；
   `attn` 残项包含 rope/reshape/mask 与 attention wrapper。它们的单独代价仍未分离。

## 五、对 10 s 判定的影响

R223 判定不变：固定开销 8.118 s/step 里，`rest` 4.09 s 现已定位到 attn/mlp 父区间。
若假设 2 成立（约 2.4 s 是权重等待），则**权重流重叠（#4）的暴露量远大于此前估计的“已基本重叠”**，
这会让 #4 从第 4 位升到第 2 位，与 all-reduce 重叠（#9）并列。
若假设 2 不成立，则这 3.37 s 是元素级/调度开销，需要不同的方向（算子融合属改数学路径，需另议）。

## 六、合规

- 未加载新权重（复用已核主机缓存）、未下载、未回传张量；回执只含标量计时与契约字段。
- 未改数学/dtype/块数/offload 策略/调度；未安装候选；FA trim 仍 OFF。
- 单步测量，不是 49 步回归；`warm_step_rank_max` 本次为 17.928 s，与 R147 的 17.919 s 一致，
  说明测量环境未漂移。

## 七、下一步（R227）

1. **证实/否证权重等待假设**：用 `--mode resident`（权重常驻）跑同一步，对比
   `attn`/`mlp` 残项是否塌缩；同时测 prefetch hook 的暴露等待。
2. 若证实：把 #4（权重流重叠）作为 10 s 路径的主要候选，先给暴露量与理论上限，再请用户决定。
3. 分离 `attn` 残项内的 rope/reshape 与 wrapper 开销（挂更细的钩子），避免把权重等待
   误记成算子成本。