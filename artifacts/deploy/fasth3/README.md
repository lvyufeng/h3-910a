# FastH3 PDD-8 移植补丁（B 机 / ts-87，Ascend910A）

FastH3 = MiniMax-H3 同架构的 **PDD-8 蒸馏学生**。权重是完整 dense 导出
（`FastVideo/FastH3-Omni-FP8`，已离线反量化为 FP16），不需要 LoRA 融合。

本目录是把 PDD 采样接入 `vllm_omni` 原生 H3 栈的最小补丁：

| 文件 | 作用 |
|---|---|
| `pdd.py` | PDD 数学核心：细网格、shift（video 12 / audio 3）、每步积分权重、块融合权重 |
| `patch_model.py` | 给 native transformer 加 `pdd_steps` 架构字段与 `MiniMaxH3PDDHead`（32 头加宽 final projection，按步输出块融合结果）+ `prepare_pdd()` |
| `patch_denoise_loop.py` | 去噪循环加 PDD 路径：用计划的 node sigmas 替换调度器 sigma，逐步传 `pdd_step`，状态更新 `x += (σ_next − σ_curr)·v` |
| `patch_pipeline.py` | 管线在 `arch.pdd_steps > 0` 时建 PDD 计划、把计划传进循环（request 模式；step 模式暂不支持） |

## 契约（来自 FastVideo 源码 `fastvideo/layers/pdd.py`）

- 细网格 `u = linspace(0.999, 0, 33)` float64，`pdd_steps=32`，8 个运行时步取 `[0,4,…,32]`
- 模态 shift `f_s(u) = s·u·M/(u·(s−1)+M)`，`M=0.999`；video s=12、audio s=3
- 每段权重 `Δσ_j = f_s(u_{j+1}) − f_s(u_j)`；块内归一化后与 4 个头做加权平均
- 状态推进 `x += (σ_next − σ_curr)·v`，`v = eps − x0`（native H3 的 raw 方向）
- 模型时间输入 `t = 1 − σ`；guidance 1.0

## 已做的验证（CPU）

- 网格/节点 sigma：video `0.999, 0.987247, …, 0.630947, 0`；audio `0.999, …, 0.2997, 0`
- 融合权重与逐头加权平均：最大误差 ~5e-7（fp32）
- 更新恒等式 `x + ΣΔσ_j·v_j == x + block_total·(fused·h)`：误差 ~1e-7
- `MiniMaxH3PDDHead` 与手工 4 头加权：误差 ~3e-4（fp32 大矩阵累积，相对 ~4e-6）

## 应用方式

```bash
python artifacts/deploy/fasth3/patch_model.py        <vllm_omni>/.../minimax_h3_transformer.py
python artifacts/deploy/fasth3/patch_denoise_loop.py <vllm_omni>/.../denoise_loop.py
python artifacts/deploy/fasth3/patch_pipeline.py     <vllm_omni>/.../pipeline_minimax_h3.py
```

补丁均带幂等检查（重复应用会拒绝），并保留 `.pre_pdd` 备份。
