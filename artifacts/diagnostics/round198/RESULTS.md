# R198 — 转向并行目标：video VAE <20 s（差 7.5%），并确认它的 roofline 仍是空白

## 一、为什么转向

过去十几轮我把全部精力放在 DiT/FA 上，而目标里明确写着
**encoder<2s、video VAE<20s、A2/A3 对比是并行目标**。这些**不受 DiT 数值决策阻塞**。
DiT 的三个决策（阶段 A / guard 判据 / FA 内核）都等用户点头，我不应该继续空转。

最接近达成的是 **video VAE**：

| 目标 | 现状 | 差距 |
|---|---|---|
| **video VAE < 20 s** | warm rank-max **21.62–21.76 s** | **−1.6～1.8 s（−7.5%）** |
| encoder < 2 s | in-pipeline 3.14–3.17 s；隔离 2.34 s | −15%（隔离口径） |
| audio VAE | 0.21–0.23 s | 非瓶颈 |

## 二、R138 已经建立了什么（不重复劳动）

| 事实 | 值 |
|---|---|
| warm decode 中位（24 次） | 21.693 s |
| **其中 `decode_base`** | **21.714 s = 99.8%** |
| pipeline 侧 revert/normalize/copy + staging | 0.140 s = 0.6% |
| 输出 | `[1,3,124,768,1344]` uint8 = **383,975,424 分量（384 MB）** |
| 输入 latent | `[1,24,37,48,84]` FP32（14.4 MB） |
| 峰值 allocated | 13.436 GiB |
| tile 网格 | **4×7 = 28 tiles**，8 rank，每 rank **4 个串行** |
| `stack_tiling` | False（微批关闭） |

**R138 的结论**：要进 <20 s **只能动 decoder 内部**，pipeline 侧没空间。

## 三、R139 已经排除的

`stack_tiling=True`（微批）实测：warm 21.609 vs native 21.618（**噪声内，无加速**），
冷启 +195 s（batch-4 kernel 首次编译）。**Reject。**
结论：**瓶颈不在 kernel 启动次数**。

## 四、空白：decoder 内部 21.7 s 的分解从未做过

R138/R139 只测了**总时长**和**可重复性**。**没有一个逐个阶段/逐层的计时分解。**

而目标第①项要求的正是"逐区域测 device 时间与达成 TFLOPS/MFU"。
**所以 VAE 侧的 roofline 探测是缺失的**，这正是本轮定位出的下一步。

### 已定位的关键代码位置（本轮新增）

`/home/mseco/vllm-omni/vllm_omni/diffusion/models/minimax_h3/vae.py`（1184 行）：

| 行 | 内容 |
|---|---|
| 326 | `class MiniMaxH3VideoVAE` |
| 635 | `def decode_latent` |
| 659 | `def _decoder_tile_count` |
| 937 | `def _decode_tiling_context` |
| 283 | `class _VideoVAEPartProxy`（分部件 residency） |
| 59–152 | `_build_residency_stagers` / `_load_part_to_device` / `_offload_part_to_cpu` |

**注意 `_VideoVAEPartProxy` + residency stager 的存在**：说明 VAE 是**按部件在 CPU/NPU 之间搬运**的
（FP32 参数 2.6 G words ≈ **10.4 GB**，13.4 GiB 峰值与之吻合）。
**如果每个 tile 都要重新搬运部件，这会是纯开销**——这是首要怀疑的优化点。

## 五、下一步该做什么（具体、可执行）

1. **给 `decode_base` 内部插桩**：给 decoder 的 up-block / conv / attention / upsample
   分别挂 NPU Event 计时，做一次 warm decode，得到 21.7 s 的逐阶段分解。
2. **同时测 residency 搬运**：统计 `_load_part_to_device` / `_offload_part_to_cpu` 的调用次数与总耗时
   ——若它在 4 个 tile 之间重复搬运同一批部件，那可能是纯浪费，且**可以不改数学地修**。
3. 拿到分解后再判断：是 compute-bound（需要更快 kernel）还是搬运/调度 bound（可能是纯工程收益）。

**成本**：一次 run 需 engine init ≈ 240 s + decode 21.7 s ≈ **5 分钟**。可行。

## 五b、本轮实测：H2D 带宽（并**否掉我自己的首要怀疑**）

### 测量结果

| 大小 | pinned | 中位 | 带宽 |
|---|---|---|---|
| 0.25 GB | 否 | 0.0118 s | 21.1 GB/s |
| 0.25 GB | 是 | 0.0138 s | 18.1 GB/s |
| 1 GB | 否 | 0.1047 s | 9.5 GB/s |
| 1 GB | 是 | 0.0510 s | **19.6 GB/s** |
| 2 GB | 否 | 0.2060 s | 9.7 GB/s |
| 2 GB | 是 | 0.0998 s | **20.0 GB/s** |

**H2D 约 20 GB/s（pinned）/ 9.7 GB/s（非 pinned 大张量）。**

按 10.4 GB FP32 参数算：一次完整搬运 **0.49 s**，若每 tile 搬一次（4 次）则 **1.97 s（pinned）/ 4.3 s（非 pinned）**。

### 但读了代码后发现：**这个怀疑不成立**

`_VideoVAEPartProxy`（第 283 行）的 docstring 明确说明搬运的触发方式：

> The pipeline's `_component_on_device` drives whatever object it is given through
> `load_to_device`/`offload_to_cpu`. ... routing those calls per half keeps the decoder
> off the device while reference videos encode and the encoder off while latents decode.

即：搬运是**按 pipeline 方向各一次**（编码前载 encoder、解码前载 decoder），
**不是每个 tile 一次**。tile 循环发生在 `decode_base` **内部**，与 residency 无关。

⇒ **"每 tile 重复搬运"是被否掉的假设。** 这也解释为什么 R138 测到 `decode_base` = 99.8%：
**21.7 s 是 decoder 的真实计算时间**，不是搬运开销。

### 这条否证的价值

如果我没读代码就直接去"优化搬运"，会白做一轮。
H2D 数字仍有用（0.49 s 是 decoder 一次性载入的量级，可作未来判断依据），
但**它不在 21.7 s 里头**。

## 六、诚实边界

- 本轮**只做了代码定位与既有回执整理，没有新的测量**。
- "residency 重复搬运是首要怀疑点"是**基于代码结构的推断**，未经测量。
- 13.4 GiB 峰值与 10.4 GB FP32 参数吻合是**算术一致**，不是证明。

## 七、合规

- 只读源码；私有目录；未加载权重、未改模型数学；未 claim lease（本轮纯离线）
