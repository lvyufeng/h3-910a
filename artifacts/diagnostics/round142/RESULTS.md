# R142 — in-model GEMM 慢在哪：staging 等待，不是 kernel 效率

**这轮修正了 R141 的一个判断。** R141 报"GEMM 合计 7.283 s、只有 7.1% peak"，
我当时的解读是"GEMM layout 有 2× 机会"。实测同形状独立 GEMM 后，结论不同了。

方法：纯测量、**不加载任何模型权重**、不改数学。在 8 卡上用真实形状
（M=10448 = 41792/USP4、K=5376、N=7168/28672、TP2）单独跑 `torch.mm`，
分别测"权重常驻"与"每次调用都做一次整权重 H2D 拷贝"两种情形。

## 同形状独立 GEMM（权重常驻，8 rank 全部完成）

| case | 时间 | 达成 | 占 310 TF peak |
|---|---|---|---|
| qkv_proj（FP16 输出） | 7.468 ms | **107.8 TFLOPS/卡** | 34.8% |
| mlp.fc1（FP16 输出） | 29.754 ms | **108.3 TFLOPS/卡** | 34.9% |
| out_proj（FP32 输出） | 11.767 ms | 68.4 TFLOPS/卡 | 22.1% |
| mlp.fc2（FP32 输出） | 22.717 ms | 70.9 TFLOPS/卡 | 22.9% |

**重要**：这些形状的 `torch.mm` 实际上限是 **~108.3 TFLOPS/卡（≈35% peak）**，
不是 310 TF。所以"GEMM 有 2× 空间"如果理解成"打败 torch.mm"是错的。

## in-model vs 独立（每层，in-model = R141 区域时间 ÷ 50 块）

| 层 | in-model | 独立（权重常驻） | 差距 |
|---|---|---|---|
| `qkv_proj` | 41.74 ms | 7.47 ms | **5.59×** |
| `mlp.fc1` | 54.6 ms | 29.75 ms | 1.84× |
| `attn.out_proj` | 19.02 ms | 11.77 ms | 1.62× |
| `mlp.fc2` | 30.3 ms | 22.72 ms | 1.33× |

**决定性对照**：qkv 带"每次调用都整权重 H2D 拷贝"的版本是
11.948 ms，
仍比 in-model 的 41.74 ms **快 3.5×**。
即 in-model qkv 连"每次都重新搬权重"都不如 —— 说明瓶颈不是带宽本身，而是
**staging 没有被计算重叠、块与块之间在等权重落地**。

## 由此得到的可回收时间

按"in-model 达到独立 GEMM 速率"算：**7.283 s → 3.585 s，
可回收 3.698 s/step**。这正好等于我此前假设的"GEMM 2×"，
但机制是 **staging/lifecycle 重叠（#4）**，不是 GEMM layout（#6）。

## 候选排序因此改变

1. **#4 staging 重叠 —— 优先**：潜在 −3.698 s，且**可以做到逐位不变**
   （只改生命周期/流/双缓冲，不改数值），不需要数值批准，只需逐位验证。
2. **#5 FA**：7.454 s @13.5%，仍是单步最大单项；但 FA 已是自研 kernel，
   2× 需要**新 kernel**（tiling/核内流水），会改舍入 → 需要数值批准。
3. **#6 GEMM layout —— 降级**：独立 `torch.mm` 只有 35% peak，layout 空间比想象小；
   应先摘掉 staging 的 3.7 s，再看剩余 GEMM 时间是否还值得动。

## 未完成项（如实记录）

- FP32 范围保护（`out_proj`/`fc2` 用的 `_MiniMaxH3HalfInputFP32OutputLinearMethod`）的
  prologue/mm/epilogue 拆分**未测出**：其整张量 abs-max 归约在这些形状上抛 `ReduceMax`。
  已记录为 unresolved，没有给猜测数字。注意该 prologue 含 FP32 upcast 与主机同步，
  是下一步值得单独测的对象（若 `peak ≤ 32752` 则 `scale=1`，理论上存在逐位等价快路径）。
- FA 独立计时未做：FA 本来就是定制 kernel，重测它不能发现更快实现，故无信息量。

## 限制

- 微基准是纯 `torch.mm`，不是模型里的 exact op 序列（模型里 out_proj/fc2 还含 row-parallel
  all-reduce 与 range guard），因此"独立"是**下界参照**，不是等价替换。
- in-model 数字取自 R141，含 5 个额外同步 hook 的 ~0.9 s 开销；比例结论不受影响。
- 310 TF 为数量级假设；本轮未加载权重、未改数学、未安装候选，**不构成任何数值批准**。
