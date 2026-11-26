# R333 — encoder 常驻机制已跑通（显存仅 8.4 GiB/rank、host 安全），差最后一步验证

## 一、机制验证成功（关键突破）

改法（`resident_patch.py`，in-repo）：

1. 首轮 `encode_prompt` 正常跑完（prefetch→forward→offload）；
2. 之后对 77 个 block 调 **`hook.prefetch_layer(non_blocking=False)`** 把权重物化到 device；
3. 把 **非 block 状态**（embeddings/norm）搬到 device：`move_non_block_state_to_device(target, groups, dev)`，
   其中 `dev = torch.device('npu', torch.npu.current_device())`
   —— **不能**用 `next(target.parameters()).device`（此时可能是 CPU，会把权重搬去 CPU）；
4. 之后 `prefetch_layer` / `offload_layer` **全局短路** → 后续 encode 不再 H2D。

实测（8 rank 一致）：

```
[RESIDENT4] resident=True blocks=77/77 restore=0.46–1.65 s
            free_after=104–136 GiB   hbm=8.4 GiB
```

⇒ **encoder 分片只占 8.4 GiB 显存/rank，host 内存安全（≥104 GiB 余量）**，
**常驻在资源上完全可行**——推翻了 R331「内存不可行」的结论（当时是用的 CPU 版 restore API）。

## 二、本轮修掉的三个 bug

| # | 现象 | 根因 / 修法 |
|---|---|---|
| 1 | `Expected all tensors to be on the same device … mat1 npu, other cpu` | 只物化了 block；**非 block 权重仍在 CPU** ⇒ 补 `move_non_block_state_to_device` |
| 2 | 同上，仍报错 | `move_non_block_state_to_device(..., dev)` 的 dev 取自 `next(params).device` = **cpu** ⇒ 改用 `torch.device('npu', torch.npu.current_device())` |
| 3 | `IndexError: Dimension out of range` | 残余 CPU 权重扫描用了 `self.named_parameters()`（**整个 pipeline**，含 DiT/VAE）⇒ 误搬 DiT/VAE，hbm 涨到 18.7 GiB；**改为只扫 encoder 模块 `target`** |

## 三、当前状态

- warm337（over-broad sweep）已做到 **0 个 device-mismatch**，但随后因误搬 DiT/VAE 报 `IndexError`；
- warm338（encoder-scoped sweep）**未进入 encode**：orchestrator 初始化失败（疑似上一轮 kill 后残留/瞬时问题），需重跑；
- 待验证目标：**第 2 趟 encode < 2 s**（首轮仍 ~61 s），且 155 条 Half 检查与输出位级不变。

## 四、环境

- 全部我方进程已 kill（`procs=0`）；host 干净。
- 交付不变：**16.34 s/step（−8.9%）**，`H3_PERF_COMBO=1` 默认 OFF。

## 五、下一步（一条命令即可复现）

用 `warmvae338.py` 的同一套环境变量把 `H3_ENCODER_RESIDENT=3` 打开重跑一次
（label 换新），观察 `[RESIDENT4] warm encode` 行是否 < 2 s。
