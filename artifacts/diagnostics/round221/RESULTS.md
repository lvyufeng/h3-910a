# R221 — 两个受保护 linear 的精确语义分解：guard 只占 0.52 s/step，另 1.1 s/step 在别处

## 一、做了什么

R147 用设备事件测出 in-model 四个 linear 的时间，并推断“FP32 范围保护 prologue”是主要浪费。
本轮的意图是把这个推断变成**可分解的实测**：不加载权重、不 import 模型、不改源码，
按部署副本 `transformer.py`（sha256 `5d94ba09…`）第 495–517 行
`_MiniMaxH3HalfInputFP32OutputLinearMethod.apply` 的**逐 op 序列**，在真实形状上用合成张量复现，
用设备事件把每个分量切开，每个样本只在最后 synchronize 一次。

真实形状来自 R146（M=10448 tokens/rank）：

| case | K | N | 输入 dtype | in-model R147 中位 |
|---|---|---|---|---|
| `attn.out_proj` | 3584 | 5376 | FP16 | 18.306 ms |
| `mlp.fc2` | 7168 | 5376 | FP32 | 29.934 ms |

受测脚本 `gemm_guard221.py`（本仓库），receipt `receipts221.json`
（sha256 `875f6780…`）。设备：ts-217 卡 0，`Ascend910B`，torch 2.10.0+cpu，
torch-npu 2.10.0.post7，CANN 9.2.0 env。

## 二、实测（ms，30 样本中位，单卡）

| 变体 | `out_proj` | `fc2` |
|---|---|---|
| `mm_only`（含 x→FP16 cast，FP32 out） | **2.952** | **11.613** |
| `amax_cast_mm`（upcast→abs/amax→host sync→cast→mm，无 /scale、无 ×scale） | 6.556 | 17.491 |
| `full_guard`（部署序列原样，scale==1 分支） | **7.080** | **18.429** |
| `full_guard_big`（强制 peak>32752，exponent>0 分支） | 7.145 | 18.401 |
| R147 in-model 中位 | 18.306 | 29.934 |

`full_guard` 分量（ms）：

| 分量 | `out_proj` | `fc2` |
|---|---|---|
| `x.float()` upcast | 0.474 | 0.067（**别名，无操作**） |
| `abs()` + `amax()` | 2.345 | 4.753 |
| `float(peak)` host sync | 0.065 | 0.063 |
| `/scale` + `.to(float16)` | 0.595 | 1.114 |
| `mm` | 2.783 | 11.605 |
| `*scale` | 0.436 | 0.436 |
| 合计（分量和） | 6.698 | 18.038 |

GEMM 达成率（含 x cast 的 `mm_only` 口径，310 TF 为 R146 沿用的**假设**峰值）：

| case | FLOPs | ms | TFLOPS/卡 | 假设峰值占比 |
|---|---|---|---|---|
| `out_proj` | 401.7 G | 2.952 | 136.4 | 44.0% |
| `fc2` | 803.4 G | 11.613 | 69.3 | 22.4% |

## 三、结论（三条，其中一条推翻 R147 的归因）

1. **guard 的净开销约 0.52 s/step，不是 R147 推的上界 1.665 s。**
   逐调用 guard−mm：`out_proj` 4.13 ms、`fc2` 6.82 ms；×50 块 ×2 层
   ⇒ `50×4.13 + 50×6.82 = 547 ms/step`。
2. **R147 的“可回收 2.099 s/step”高估了。** in-model 比 standalone 多出的
   11.2 ms（`out_proj`）与 11.5 ms（`fc2`）**既不在 guard 也不在 mm 里**：
   两项 ×50 块合计 **≈1.14 s/step** 仍待归属，候选是 RowParallelLinear 的 TP all-reduce
   或 guard 内 `float(peak)` 把权重预取流一并排空（R147 自己踩过的坑）。
3. **scale==1 分支才是常态，且该分支里有两次恒等运算。**
   两个 case 的观测 peak 都是 0.274，`scale=1.0`；此时 `product/1.0` 与 `result*1.0`
   在 IEEE 下精确。`full_guard` 与 `amax_cast_mm` 在 **112,336,896 个 word 上逐位相同、
   差异 0**（两个 case 各一次），即“跳过两次恒等运算”这件事本身不改变输出。
   但这只是**一个分支上的证据，不是数值批准**，也**没有**被安装。
   其可回收量仅 `50×0.524 + 50×0.938 = 73 ms/step` —— **不值得作为候选推进**。
4. guard 中真正的大头是 `abs()+amax()`（`out_proj` 2.345、`fc2` 4.753 ms）：
   它把 `abs` 物化成整张 FP32 张量再归约，读放大一倍。融合式 abs-max 理论可省一半，
   但仍是 ~0.15 s/step 量级。

## 四、对 10 s 目标的意义

按本轮实测，GEMM 区域里“靠 guard 微调”能拿到的量级是 **0.05–0.2 s/step**，
不足以改变 10 s 结论。真正待攻的是：

| 区域 | 量级 | 状态 |
|---|---|---|
| FA 实际 kernel | **7.454 s/step（41.5%）** | #5，等待数值批准；最高杠杆 |
| in-model vs standalone linear 差 | **≈1.14 s/step** | 本轮新暴露，需 R222 归属（all-reduce vs host sync 排空） |
| guard 净开销 | 0.52 s/step | 本轮实测；可优化空间 ≤0.2 s |
| USP pre/post + q/k norm | 1.307 s/step | 未测 |
| 其余（AdaLN、权重流、155 检查等） | ~4.5 s/step | 未测 |

**10 s 仍不可达**：即使 GEMM 区域全部拿掉 1.14 s、guard 拿掉 0.2 s，仍需 FA 从 7.45 s
降到 ~2.2 s 才够。本轮不宣布任何进度完成。

## 五、合规与限制

- 未加载权重、未 import 模型、未改源码、未安装候选、FA trim 保持关闭。
- 单卡测量：不含 TP all-reduce、RowParallelLinear wrapper、rope/silu/residual/AdaLN、
  权重流邻居，故比 in-model 乐观；这正说明差值是**待归属**而非“已知浪费”。
- 310 TF 是沿用的假设峰值，不是本轮实测；达成率只用于数量级比较。
- `full_guard_big` 用预放大输入触发 exponent>0 分支，只测时间，不采样其数值。
- 逐位比较只发生在 scale==1 分支的合成输入上，不外推到全模型。

## 六、下一步（R222，仍需用户数值决定才可上线）

1. **归属那 1.14 s/step**：单进程内并发跑一条拷贝流（模拟 32.3 GB 权重预取），
   分别测“guard 带 `float(peak)` 同步”与“设备侧算 scale、零 host 同步”的耗时差；
   同时用 TP2 组实测 10448×5376 FP32 all-reduce 的耗时。二者对照即可判定主因。
2. 若主因是 host sync 排空拷贝流，候选是**设备侧算 scale 的无同步 guard**：
   数学等价（2 的幂、无 clip），但属精度保护路径改动，**必须先做逐位验证再请示**。
3. 若主因是 all-reduce 串行，候选是**把 row-parallel 归约与下一段计算重叠**；
   不改归约树、不改 dtype，需实测重叠率。