# R187 — FA 可达上限测量：**未完成**，但排除了三条路

## 目的

10 s 需要 FA 从当前 **41.1 TFLOPS/卡** 提到约 **69 TFLOPS/卡**。
在投入写新 FA 内核之前，先要知道这台机器上因果注意力的可达上限是多少。
若是库实现也只能到 ~41，那天花板是真的；若能到 ~80，则 69 可达、值得投。

## 确切形状（从 R159 回执恢复）

```
input_N=41792  kernel_N=43776  key_limit=41769
core_budget=30 blockDim=30 B=1 H=7 N=43776 D=128 Br=128 Bc=304
每步 313 TFLOP/rank（dense 口径），实测 7.454 s
```

FLOPs 口径校核：`4·B·H·N²·D = 6.26 TFLOP`（单 block），× 50 block = **313 TFLOP** ✓
——所以 313 TFLOP 是 **dense（非因果）口径**。

## 三条路都失败（都是环境/API 问题，不是结论）

| 尝试 | 结果 |
|---|---|
| `torch_npu.npu_fusion_attention(pre_tockens=N, next_tockens=0)` | **161001 参数无效** |
| `torch_npu.npu_fusion_attention(sparse_mode=2)` 因果 | **161001 参数无效** |
| `F.scaled_dot_product_attention(7 heads, is_causal=True)` | **OOM 45.55 GiB**（32 GiB 卡） |
| sdpa 逐 head（切片张量） | `can not cast format when output is input` |

### 这两条失败本身有价值

1. **`npu_fusion_attention` 在 B=1,H=7,N=41792,D=128 上直接报参数无效** ——
   说明模型用的不是这个通用入口，而是 910A 专用插件（`VLLM_OMNI_910A_FLASH_ATTN=1`
   对应主机上的 `prompt_flash_attention`）。**所以拿通用库去比不是同口径**。
2. **sdpa 在 7 head 上要 45.55 GiB** —— 它把 N×N 分数矩阵真的物化了
   （41792² × 7 × 2 B ≈ 45.5 GiB）。这从反面确认了**flash 结构是硬性要求**，
   也说明任何"先算分数矩阵"的替代方案在 32 GiB 卡上都不可行。

## 结论：**尚未回答"可达上限是多少"**

本轮没有拿到 `可达 TFLOPS/卡` 这个数字。**不得据此推断 69 不可达。**

## 下一轮的具体做法（已想清楚）

正确口径是**测模型实际用的那个 FA 算子**，而不是通用入口：

1. 在主机上定位 910A FA 插件（`prompt_flash_attention` / MindIE 实现）的 Python 入口；
2. 用与 R159 相同的 plan 参数（`Br=128 Bc=304 core_budget=30`）在单卡上直接调它，
   测单 block 耗时 → 与 `7.454/50 = 149 ms` 对比；
3. 再用不同的 `core_budget`（30 → 全部 AIV 核数）扫一遍，看**核占用是否是瓶颈**；
4. 若能找到 tile 参数入口，扫 `Br/Bc` 组合，看当前配置离该算子的最优差多少。

**关键判断点**：如果模型自己的 FA 算子在全核数下也只能到 ~41 TFLOPS/卡，
那 69 就是该算子实现的上限问题，需要新内核；如果全核数下明显更高，
说明当前 30 核预算就是主要瓶颈，**那可能不需要新内核，只要改核预算**——
这会是比写新 FA 内核便宜得多的路径。

## 合规

- 合成张量，未加载权重，未改模型数学，未采用任何东西
- 共享 CANN 未改；未做 driver/device reset；lease 已释放
