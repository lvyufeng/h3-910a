# R138 — video VAE 解码计时分解与逐位重复性判定

测量型实验（Issue #7 的第一步）：固定 R136 原 True final latent、原 FP32 VAE 参数、
原 `decode_latent` 路径，**不改数学、不加候选、不采用 tolerance**。host `ts-217`（8×910B）。

## 输入与门

- 固定 `video_latent [1,24,37,48,84]` FP32，file SHA `bf9daec8…f0f5d02e0`（R136 `r01_original_false`），
  原 CPU 与冻结 NPU 副本完整词域 **0/3,580,416** 差异。
- 原 FP32 VAE 参数 2,603,868,984 words，未改写；结束状态校验通过、`decode_base` 计时旁路已移除、
  runtime 与 initial 一致。
- engine init **239.811 s**；8 rank 全部完成 32 次解码。

## 几何与调度事实（实测）

- 输出 `[1, 3, 124, 768, 1344]` `torch.uint8`，共 383,975,424 个分量。
- VAE 并行 **8**；tile 网格 **4×7 = 28 tiles**，
  每 rank 轮转持有 **4** 个；**`stack_tiling=False`（微批当前关闭）**。
- 时间维：`clip_length=17`、`tokens_chunk_size=5`、
  `token_overlap=2`、`vae_ratio_t=4`、`token_drop=3`。

## 计时分解（rank-max）

| 项 | 值 |
|---|---|
| 冷启首次 decode | **80.293 s** |
| warm decode（rank-max, rep1/2/3） | **21.737 / 21.760 / 21.678 s** |
| warm decode 中位（24 次） | 21.693 s |
| 其中 `decode_base` | **21.714 / 21.729 / 21.658 s** |
| pipeline 侧 revert/normalize/copy | 0.031 s |
| staging load | 0.109 s |
| 峰值 allocated | 13.436 GiB |

结论：warm decode 的 **99.8% 在 `decode_base`**（tiled decoder）内；pipeline 侧全部开销约
0.140 s（≈0.6%）。因此要进 <20 s
只能动 decoder 内部的 tile/时间调度，不可能靠 pipeline 侧拷贝优化。

## 逐位重复性（关键）

同一 latent、同一参数、同一路径，8 rank 输出**逐 rank 完全一致**，但**跨次不等**：

| 对比 | 不同分量 | 占比 | maxabs | relRMS |
|---|---|---|---|---|
| rep2 vs rep1 | 253,384 | 0.065990% | **1 LSB (uint8)** | 0.013473% |
| rep3 vs rep1 | 253,471 | 0.066012% | 1 LSB | 0.013476% |
| rep4 vs rep1 | 253,135 | 0.065925% | 1 LSB | 0.013467% |

即：**video VAE decode 也不可逐位重复**，差异是 uint8 量化边界上的 1 级翻转（0.066% 分量），
与 R137 在 encoder / audio decoder 上观察到的原生非确定性一致。

## 对 #7 的影响

1. Issue #7 原本要求「逐位门禁」。本测量证明该门在跨次比对上**不可能满足**——不是候选造成的，
   而是原生就抖 1 个 uint8 级。因此必须先由用户给出可接受判据（例如"不比原生 1 LSB / 0.066% 更差"这类经验带，
   或某类按构造必然位相等的改法），否则任何调度候选都会被同一问题卡住。
2. 若允许继续，最小改动候选已明确：`stack_tiling=False` 意味着 28 tiles / 8 rank、每 rank 4 个 tile 串行；
   开启微批（2/4 tiles 每 forward）或改时间 chunk 流水，是能吃到那 1.7 s 的方向，但会改变累加/舍入顺序。

## 失败尝试（均保留，未 replay）

- 138a：launcher 门禁键名不一致，spawn 前 fail-closed；随后 driver 误读 `approval137.json`，`FileNotFoundError`，
  未加载权重、未占 NPU。
- 138b：engine init 成功并进入 RPC，tiling 元数据误用 `ProcessGroup.ranks`（该版本不存在）→ `AttributeError`，
  仍在任何 decode 之前；atomic fatal 清理正常。
- 138c：修正后完整成功，上表数据出自此轮。三次尝试回执分别留存。
