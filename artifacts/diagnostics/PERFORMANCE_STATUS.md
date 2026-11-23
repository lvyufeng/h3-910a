# 当前性能状态与结论（截至 R139，用户暂停）

单一入口，避免再从聊天记录里翻数字。所有数字都带口径；嵌套 span 不相加。
更细的证据在各 `round*/RESULTS.md`；本文件只做"现在在哪、差多少、下一步该打哪"。

## 1. 目标 vs 现状（全部热态 warm）

| 目标 | 现状 | 差距 | 证据 |
|---|---|---|---|
| DiT **<15 s/step** | 实际 whole-step **19.37 s**（R126 hook）；R136 逐 step rank-max 中位 **20.03/20.11/20.03 s** | 需 **−23% ~ −25%** | round126 / round136 |
| encoder **<2 s** | in-pipeline `encode_prompt` rank-max **3.14–3.17 s**；隔离态 live FP16 **2.34 s** | 按 pipeline 口径 −36% | round136 / round126 |
| video VAE **<20 s** | `decode_latent` warm rank-max **21.62–21.76 s** | 差 **1.6–1.8 s（−7.5%）** | round138 / round139 |
| audio VAE | **0.21–0.23 s** | 非瓶颈 | round136 |
| A2/A3 真实硬件同口径 | 未做 | — | #13 open |

整段生成（49 步，热态）**1018.8–1023.1 s**；冷态 **1327.5 s**；engine 初始化 **240–286 s**。
历史：隔离态 DiT 40.47 → 18.41 s（R125），整段生成 2316.96 → 1329.96 s（R125→R126）。

## 2. 暖态墙钟的代价构成（这是"下一步打哪"的依据）

以 R136 热态 1018.765 s 为口径：

| 项 | 耗时 | 占比 |
|---|---|---|
| **DiT 49 步**（20.034 s × 49） | **≈981.7 s** | **≈96.4%** |
| video VAE decode | 21.52 s | 2.11% |
| `encode_prompt` | 3.14 s | 0.31% |
| audio VAE | 0.23 s | 0.02% |
| 其余（stream/gather/收尾） | ≈12 s | ≈1.2% |

结论：**优化窗口基本只剩 DiT denoising**；VAE/encoder 即使做到 0 也只省 2.4%。
DiT 单步内部已知分解（R126）：FA kernel ≈7.45 s/step、USP 前后 ≈1.08 s（含 packing/等待，非纯 HCCL），
其余 ≈11.5 s 为 GEMM/norm/elementwise/offload 与通信。

## 3. R137–R139 建立了什么（本阶段主要收获是"边界"，不是新速度）

- **R137**：原数学组件**不可逐位重复**。固定输入下 encoder 8 次全不同（≈96% 输出词不同，relRMS 0.88–2.32%，
  首变化在 `encoder.text.layers.0` post）；audio decoder 8 次全不同（maxabs 6e-5，首变化在
  `resblocks.9.convs1.0.parametrizations.weight.0` post）。
  → R136 的 encoder 级差（candidate 2.53%、restored 3.70%）落在原生抖动内，**不能作候选/恢复的因果证据**。
  用户决定：**只记录，不做数值归因**。
- **R138**：video VAE decode 计时分解 + 重复性。warm **21.62–21.76 s** 中 **99.8% 在 `decode_base`**；
  pipeline 侧（revert/normalize/copy 0.031 s + staging 0.109 s）总计 ≈0.14 s。
  几何：VAE 并行 8、tile **4×7=28**、每 rank 4 个、`stack_tiling=False`；
  时间维 `clip_length=17 / tokens_chunk_size=5 / token_overlap=2`。
  重复性：**不可逐位重复**，253k/384M（**0.0660%**）uint8 分量差 **恰好 1 LSB**。
  用户选择判据"用原生噪声带做上界"。
- **R139**：候选 `stack_tiling=True`（上游原生微批）**Reject**：warm 只差 0.010 s，冷启 **+195 s**。
  副产品：开启微批后 decode **逐位可重复**（0 差异），说明原生非确定性来自"逐 tile 顺序 forward"路径。
  且证明用户判据的字面形式**不可满足**——纯原生 restored 对照自身就 0.066412% > 0.066253%，需改统计形式。

## 4. 现在的阻塞与下一步候选（按性价比排序）

1. **DiT #5 FA（≈7.45 s/step，单步最大单项）**：910A 真实长序列 FA 的 IO/tiling/30 核工作分配。
   状态：Todo。这是唯一可能吃掉数秒的单项。
2. **DiT #6 GEMM 布局**：真实 QKV/MLP/out 形状的静态布局、预排布与后处理。状态：Todo。
3. **#4 worker-local copy stream / 安全 prefetch**：仍是 P0，但 **Blocked by #15**；
   现在又有 R139 的新线索（批量路径可逐位重复）可能帮助建立可复现基准。
4. **video VAE 剩余 1.7 s**：只剩时间 chunk 流水（`clip_length/tokens_chunk_size/overlap`）与 tile 分配；
   kernel 启动次数这条路已被 R139 证伪。**用户决定暂停这一路**。
5. 判据：#7 的噪声带判据需改成统计形式（见 R139 建议）；否则任何 VAE 候选都会被同一问题卡住。

## 5. 暂停时的资源状态

- ts-217 lease 已释放；无遗留 private 进程/IPC key；无共享 CI/dormant 借用；未做 device reset。
- 权重/输入/激活/原始媒体全部留在主机；客户端只有 source、JSON、文本。
- GitHub：Issue #7 本轮结论已回写（Reject + 判据建议），#15 保持 open，#4 仍 Blocked，#1 持续 In Progress。