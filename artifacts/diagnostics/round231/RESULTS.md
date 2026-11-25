# R231 — attention/MLP 内部算子级归属：FA+USP 占 51%，仍有 3.43 s/step 未归属

## 一、先纠正一个框架错误

R230 结束时我把 `rest` 描述成“设备空闲/调度间隙”。本轮用数据纠正：`block` span 345.96 ms ×50
= **17.30 s**，步时 **18.02 s** ⇒ 设备忙碌率 **96%**。那 4 s 不是空闲，而是**尚未归属的设备执行**。

## 二、方法（不重写前向）

保留部署前向原样，只加设备事件（默认流，无流切换——R227 证明切流会扰动 15%）：

- 模块钩子：`attn`、`mlp`、`qkv_proj`、`out_proj`、`q_norm`、`k_norm`、`fc1`、`fc2`、`adaln_proj`、`block`
- monkeypatch 模块级 `_apply_rope`（每块 2 次）
- monkeypatch `MiniMaxH3Attention._run_packed_attention`（把 **USP all-to-all + FA kernel** 包在一起）
- monkeypatch `F.silu`（全局聚合）

单步测量（`--steps 0 --warmup 1`，baseline、TP2+USP4、50 块、native offload）。

**步时不变性**：18.020 s vs 基准 17.928 s，**+0.092 s**，在 ±0.1 s 阈值内（临界，已记录风险）。
8 rank 全部完成。

## 三、实测（8 rank 中位，ms/次；`rope` 为单次、2 次/块）

| 标签 | 中位 | 说明 |
|---|---|---|
| `block` | 345.958 | 父区间 |
| `attn` | 243.529 | 父区间 |
| `mlp` | 87.957 | 父区间 |
| **`packed_attention`（USP+FA）** | **176.747** | 最大单项 |
| `out_proj` | 18.245 | 含 TP all-reduce |
| `qkv` | 17.066 | |
| `fc2` | 30.005 | 含 TP all-reduce |
| `fc1` | 23.199 | |
| `q_norm` / `k_norm` | 2.340 / 2.251 | |
| `rope` | 1.101（2 次/块 = 2.20） | |
| `adaln_proj` | 1.932 | |
| `silu` | 0.339 | 全局聚合 |

## 四、归属结果（每块 / 每步 ×50）

| 区段 | ms/block | s/step |
|---|---|---|
| **FA + USP**（`packed_attention`） | 176.75 | **8.84** |
| 四个 linear（qkv+out_proj+fc1+fc2） | 88.52 | 4.43 |
| q/k norm + rope + adaln | 8.72 | 0.44 |
| **`attn` 未归属**（split/view/reshape/metadata/dispatch） | **24.7** | **1.23** |
| **`mlp` 未归属**（chunk + 2×cast + multiply + dispatch） | **34.8** | **1.74** |
| `block` 未归属（modulate/gate/residual/norms） | 9.2 | 0.46 |
| 合计 | 342.7 | 17.14（≈ block span ×50 = 17.30） |

**关键数字**：FA+USP 单项占 `block` 的 **51.1%**、占步时 **49.3%** —— 再次确认 R223 的
“FA 是第一杠杆”判定。

## 五、仍未解释的 3.43 s/step

`attn` 未归属 1.23 s + `mlp` 未归属 1.74 s + `block` 未归属 0.46 s = **3.43 s/step**。

已经排除的解释（R228/R229/R230）：权重预取等待、allocator/拷贝争用、strict 审计的 host 同步。
已实测的对照：MLP 元素链独立代价仅 **3.6 ms**（含 chunk 非连续视图），`silu` 仅 0.34 ms，
`rope` 2.2 ms。所以 `mlp` 的 34.8 ms 里约 **31 ms** 既不是这些算子、也不是搬运或同步。

**剩下最强的假设是 host 派发在父区间内造成设备气泡**（事件 span 会包含设备队列排空后的等待）。
这必须直接判定，不能猜：在同样的模块钩子里同时记录 **host `perf_counter`** 与设备事件，
对比 host 间隙与设备间隙的大小。

## 六、合规与限制

- 未重写前向、未改数学/dtype/块数/offload/调度；未安装候选；FA trim 仍 OFF。
- 未加载新权重（复用已核缓存）、未下载、未回传张量；回执仅含标量计时。
- **步时扰动 +0.092 s 属临界**，本轮的绝对时间只用于结构比例，不用于宣称提速。
- `packed_attention` 是 USP+FA 合体，不是纯 kernel 时间；R126 的 FA 149.08 + USP 21.54 与之相差
  约 6 ms，差额在 wrapper/metadata 内。

## 七、下一步（R232）

1. 在 `attn`/`mlp`/`block` 钩子里同时记录 host 墙钟与设备事件，直接判定 3.43 s 是否为 host 气泡；
   判定后再决定是“派发/脚本层优化”（可能完全不触碰数值）还是“算子内部机制”（需更细探针）。
2. 若确认 host 气泡：量化 host 侧单次 op 派发代价（例如固定循环里的 `torch.add` 派发吞吐），
   并评估脚本层手段（如减少 Python 级 op、提升派发吞吐配置）；任何改变执行顺序的方案都需先请示。
3. FA（#5）仍是唯一能在量级上改变 10 s 判定的杠杆，等待用户数值决定。