# R378 — GEMM 加速（bit-identical）+ 集合通信达到硬件带宽上限：**15.5489 s/step**

## 一、GEMM：定位到两个 column-parallel linear 的慢路径

profiler 显示同 dtype 下 `MatMulV2`(aclnn, fp32 出) 可达 147 TFLOPS，而两个 aten `MatMul`(fp16 出) 只有 68–70。
映射形状后确认：

| GEMM | 形状（每 rank） | 实现 | 基线耗时 | 优化后 |
|---|---|---|---|---|
| fc1 (`MergedColumnParallelLinear`) | [10448,5376]×[14336,5376] | vLLM 默认 UnquantizedLinearMethod | 24.33 ms | **16.78 ms** |
| qkv (`QKVParallelLinear`) | [10448,5376]×[10752,5376] | 同上 | 17.19 ms | **13.71 ms** |
| o_proj / fc2 (`RowParallelLinear`) | — | 已是自定义 fp32-out 方法 | — | — |

优化形式：`torch.mm(x, W_nz_or_T, out_dtype=torch.float32).half()`（fc1 额外用 NZ 权重缓存 `npu_format_cast(w.t().contiguous(), 2)`）。
**`torch.equal(out, baseline_out) == True`**——fp32 累加顺序未变，只是输出转换/权重布局不同，**无数值变化**。

模型级：GEMM patch 使 step 15.9717 → **15.4663 s**（−0.51 s，与微基准预估 0.55 s 一致）。

## 二、集合通信：已达到 HCCL 带宽上限，且 0% 重叠

稳态 forward（15.20 s）内：集合通信 2.235 s（14.7%），**并发计算重叠 = 0%**（完全串行暴露）。

| collective | 次数 | 总时长 | 单次 |
|---|---|---|---|
| allReduce | 98 | 1.256 s | 12.82 ms |
| alltoall | 196 | 0.978 s | 4.99 ms |
| allGather | 49 | 0.001 s | 0.03 ms |

HCCL 微基准（8 rank，TP2 组，224.7 MB fp32）：

| payload | 耗时 | 算法带宽 |
|---|---|---|
| fp32 224.7 MB | 11.95 ms | **37.6 GB/s** |
| fp16 112.3 MB | 6.20 ms | 36.4 GB/s |
| fp32 56.2 MB | 3.15 ms | 35.7 GB/s |

⇒ 线性缩放 = **带宽受限**，模型内 12.82 ms vs 微基准 11.95 ms ⇒ 已接近硬件上限（差距 7%）。

## 三、整模验证（8 rank，steps 0,10,25,40,48）

| 配置 | step time | relRMS |
|---|---|---|
| 交付基线 R126 combo | 16.32–16.34 s | 带内 |
| + FA V1 | 16.1125 s | 8.34e-04 |
| + 快速审计 | 15.9717 s | 9.92e-04 |
| **+ 快速 GEMM（全部开启）** | **15.5489 s** | **9.27e-04（带内）** |

⇒ 相对交付基线 **−0.79 s（−4.8%）**。三轮验证均 0 审计失败。

## 四、集合通信的剩余选项（需决策）

1. **fp16 allreduce**：224→112 MB ⇒ 单次 12.82→~6.5 ms ⇒ 省 ~0.58 s/step（3.8%）。
   但这是**真实的数值改变**（fp32 归约和 → fp16 归约和），按契约需用户批准；
2. **按行切分流水**：把 allreduce 拆成行段，与下一段的 norm/MLP 计算重叠（数学不变、无数值变化），
   但需要改 block 前向与 linear 封装，工程量中等；
3. alltoall（0.978 s）：4 次/block，合并同类调用或与 attention 计算重叠，收益有限。
