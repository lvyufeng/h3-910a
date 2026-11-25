# R239 — 差额在 wrapper（不是 GEMM 内核）；但暴露出更严重的问题：`M` 口径错了

## 一、方法

patch `UnquantizedLinearMethod.apply`（qkv/fc1 使用的类），每次调用记录：
`apply` 的设备事件 span、host 墙钟、`x`/`weight` 的 shape/dtype/连续性/步长。

步时不变性：18.0251 s vs 基准 17.928 s = **+0.097 s**（阈值内），8 rank 完成。

## 二、结果（rank 0，ms）

| 层 | `apply` 中位 | p10–p90 | host 中位 | R237 模块调用 | **差额（wrapper）** |
|---|---|---|---|---|---|
| `qkv`（QKVParallelLinear，weight 10752×5376） | **17.071** | 17.064–17.075 | **0.359** | 41.561 | **24.49** |
| `fc1`（MergedColumnParallelLinear，weight 14336×5376） | **23.306** | 23.150–23.501 | **0.305** | 54.430 | **31.12** |

- `apply` 的 host 墙钟只有 0.3–0.36 ms ⇒ 派发极快，不是 host 瓶颈。
- 每个标签 104 次调用（2 步 × 50 块 + 4 次额外）⇒ 每块每层约 1 次。

**结论（分层）**：R237 的 3× 差额**不在 GEMM 内核里**，而在
`QKVParallelLinear.forward` / `MergedColumnParallelLinear.forward` 的 **wrapper**：
qkv 多 24.5 ms/块、fc1 多 31.1 ms/块，合计 **2.78 s/step** 的 wrapper 开销。

## 三、但发现了更严重的问题：`x` 的行数不是 10448

`apply` 记录的输入形状是 **`x = [2043, 5376]`**（FP16、连续、stride `[5376,1]`），
而 R146 的独立微基准一直按 **M = 10448** 复现“真实形状”，
R223/R237 的 FLOPs、TFLOPS 与 MFU 也全部建立在 M=10448 上。

- 若真实每调用 M=2043，则 `qkv` 的 FLOPs 只有我此前假设的 **1/5.11**：
  17.07 ms 对应 **2*2043*5376*10752/0.01707 = 13.8 TFLOPS/卡**，
  而 R146 在 M=10448 下测得 12.47 ms ≈ **92.6 TFLOPS/卡** —— 两者不可直接比较。
- 也就是说 **R223/R237 里所有“in-model vs 独立 mm”的比值与 MFU 都可能是错的**，
  因为比的是**不同 M** 的 GEMM。
- 2043 与 10448 不是简单整除关系（10448/2043≈5.11），
  说明模型的实际 token 打包/分块方式与我此前的假设不同。

## 四、结论

1. **wrapper 是新的、明确的 2.78 s/step 开销来源**（qkv 24.5 + fc1 31.1 ms/块），
   且 `apply` 本身很快（host 0.3 ms）、内核按 M=2043 计也不慢。这条属 #6，可继续归因。
2. **在修正 `M` 口径之前，不得再引用任何 in-model vs standalone 的 TFLOPS/MFU 数字**。
   R223 的 roofline 表与 R237 的“3.11×”需要按真实 M 重算或标注作废。
3. 下一步必须先拿到模型真实的 token 布局：attention 输入行数、`packed_total`、
   `max_seqlen`、`cu_seqlens`、`sp_seq_lens`，以及每块 attention/MLP 的调用次数。

## 五、下一步（R240）

1. 在探针里记录 attention forward 的 `x.shape`、`packed_total`、`max_seqlen`、
   `cu_seqlens.tolist()[:8]`、`sp_seq_lens`，以及每块 attention/MLP 调用次数；
2. 据此重算四个 linear 的真实 FLOPs 与达成率（并更正 R223/R237 的表述）；
3. 再剖 wrapper：`QKVParallelLinear.forward` / `MergedColumnParallelLinear.forward`
   在 `apply` 之外还做了什么（切分、gather、reshape、cast），是否与数值语义耦合。

## 六、合规与诚实边界

- 未改盘上源码、未安装候选、FA trim 仍 OFF；未下载/回传张量；回执只含标量与形状。
- 本轮**主动报告**了此前 M 口径可能错误，并明确暂停引用相关 TFLOPS/MFU 结论，
  直到 R240 用实测形状重算。
- 步时 +0.097 s 在阈值内。