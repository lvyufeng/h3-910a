# R141 — roofline 追加：GEMM 与 TP all-reduce 拆分（measure-only）

R140 只给出了区域级分解，`attn` 里混着 qkv/o GEMM，另有一个 2.545 s 未归因桶。
本轮在同一 replay harness 上增加 5 个区域 hook（block / qkv_proj / out_proj / fc1 / fc2）
并对 `tensor_model_parallel_all_reduce` 做纯包装计时；**未改数学、未安装候选**。

> **口径警告**：本轮 warm step **20.798 s** 比 R140 的 19.906 s 慢，是因为多了 5 个
> `synchronize` 区域 hook。**绝对步时请用 R140，本轮只看拆分**，不可当作性能回退。

## GEMM 真实达成率（每 rank / 每 step）

| 区域 | device 时间 | 占步比 | FLOPs | 达成 | 占 310 TF 峰值 |
|---|---|---|---|---|---|
| `qkv_proj`（column） | 2.087 s | 10.0% | 20.1 TF | 9.6 TFLOPS/卡 | **3.1%** |
| `out_proj`（row + all-reduce） | 0.951 s | 4.6% | 20.1 TF | 21.2 | 6.8% |
| `mlp.fc1`（merged column） | 2.730 s | 13.1% | 80.5 TF | 29.5 | 9.5% |
| `mlp.fc2`（row + all-reduce） | 1.515 s | 7.3% | 40.3 TF | 26.6 | 8.6% |
| **GEMM 合计** | **7.283 s** | **35.0%** | 161.0 TF | **22.1 TFLOPS/卡** | **7.1%** |

**这是本轮最重要的新事实**：GEMM 与 FA 同量级（7.28 s vs FA 7.45 s），而且只跑在 **7.1% peak**。
`qkv_proj` 只有 **3.1% peak**（9.6 TFLOPS/卡），异常低，指向小 batch/布局问题而非算力上限。

## 区域与残差

| 区域 | 时间 | 占步比 |
|---|---|---|
| block（整块，父区间） | 20.501 s | 98.6% |
| attention（FA+USP+q/k norm+qkv+out，子区间） | 12.242 s | 58.9% |
| mlp（fc1+fc2，子区间） | 4.452 s | 21.4% |
| adaln | 1.365 s | 6.6% |
| **block 残差** = block − 子区间 | **2.442 s** | 11.7% |
| misc（refiner/patch/final/time） | 0.157 s | 0.8% |
| step 残差 | 0.140 s | 0.7% |

block 残差 2.442 s 现在有了明确归属：
TP all-reduce 未被计算吸收的部分 + 残差加法 + prefetch 等待。

## TP all-reduce

- **420 次调用**（50 块 × `out_proj`/`fc2`，另有 refiner/USP 路径），
  device 驻留 **7.09 s**，平均 16.89 ms/次。
- **不可相加**：HCCL 在独立 stream 上，逐次 span 在 device 时间上重叠；这个总量是"驻留"而非墙钟，
  且已经落在 `out_proj`/`fc2` 与 block 区间内部。未被吸收的部分才进 block 残差。
- 权重 prefetch：201 次、copy-stream 6.87 s；
  纯 H2D 带宽 21.74 GB/s（与 R140 的 21.81 一致）。

## 对 10 s 预算的修订

两条腿现在都清楚了，且都只有 ~7–13% peak：

| 目标项 | 现状 | 2× 后 | 可省 |
|---|---|---|---|
| FA kernel | 7.454 s（R140） | 3.73 s | **−3.73 s** |
| GEMM 合计 | 7.283 s | 3.64 s | **−3.64 s** |
| 合计 | 19.906 s（R140 口径） | — | **−7.37 s → ≈12.5 s** |

剩余 ~2.5 s 需要从 block 残差（2.442 s）、
adaln（1.365 s，本轮偏高疑含 hook 开销）与 H2D 重叠里拿。
**结论：10 s 的路径 = FA 2× + GEMM 2× + collective/staging 收尾，与 R140 判断一致，但 GEMM 已升为一级目标。**

## 限制

- 区域计时用 `synchronize`，属 coarse 归因；父子区间包含、**不相加**；5 个额外 hook 使本步慢约 0.9 s。
- 310 TFLOPS 是数量级假设，非厂商报价；MFU 仅用于比较排序。
- qkv/out_proj/fc1/fc2 的 FLOPs 是按公开形状（N=41792、TP2、d=5376、ffn=14336、50 层）建模，非 profiler 计数。
- 本探测不改数学、未安装候选；结论**不构成任何数值批准**。
