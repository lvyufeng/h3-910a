# R139 — video VAE 原生 `stack_tiling` 微批候选（按用户噪声带判据）

用户选择"用原生噪声带做上界"（Issue #7 选项一）后，对上游原生 `stack_tiling` 开关做单项评估。
固定 R136 原 True latent、原 FP32 VAE 参数、8 卡；同进程三段各 4 次：native(False) → candidate(True) → restored(原值)。

## 结论：Reject（不进性能，且判据本身需要修正）

**1) 没有任何加速。** warm 中位几乎完全相同：

| 阶段 | warm 中位 | warm rank-max | 首次（冷） | 峰值 |
|---|---|---|---|---|
| native | 21.618 s | 21.696 s | **85.347 s** | 13.44 GiB |
| candidate | 21.609 s | 21.732 s | **280.774 s** | 13.43 GiB |
| restored | 21.619 s | 21.806 s | 22.118 s | 13.43 GiB |

warm 差 0.010 s（噪声内），**冷启反而多 195.4 s**（batch-4 kernel 首次编译）。28 tiles / 8 rank / 每 rank 4 个的
微批没有带来吞吐收益，说明当前瓶颈不在 kernel 启动次数。

**2) 一个有意义的新事实：开启 stack_tiling 后 decode 变成逐位可重复。**
`candidate_vs_candidate` = **0 个分量不同**（4 次完全一致）；而 native 自身 4 次之间差 0.0663%、
restored 对照内部同样差 0.0663%。即原生非确定性来自"逐 tile 顺序 forward"这条路径，
批量前向后消失（但没变快）。

**3) 用户判据在字面形式下不可满足——纯原生对照自己就过不了。**

| 比较 | 最大不同分量占比 | maxabs |
|---|---|---|
| native vs native | 0.066253% | 1 LSB |
| **restored（纯原生）vs native** | **0.066412%** | 1 LSB |
| candidate vs native | 0.067496% | 1 LSB |

`restored_vs_native` 只是"原生对原生、跨阶段"，就已超出 `native_vs_native` 上界 0.000159 个百分点。
这说明按"单次取 max 的跨阶段对比"来卡，噪声本身就会越界；候选 0.067496% 与它的差距（+0.00108pp）
也在同一量级。因此**不能**据此宣布候选数值不合格，也不能据此宣布合格——需要把判据改成统计形式。

## 建议的判据修正（待用户确认）

把"不劣于原生"从单点 max 改成带对照的分布判定，例如同时满足：

1. `maxabs ≤ 1 LSB`（与原生同）；
2. `candidate vs native` 的不同占比 ≤ `max(native internal, restored-control vs native) × 1.05`
   —— 即允许跨阶段再测一次的固有漂移，用**本轮实测对照**（0.066412%）而不是 R138 的单阶段值；
3. 候选内部重复散布 ≤ 原生内部散布（本条 stack_tiling 反而更优：0 vs 0.0663%）。

按此形式，本候选在数值上会通过（0.067496% ≤ 0.066412%×1.05=0.069733%），但仍因**无加速 + 冷启 +195 s**
被拒；也就是说数值判据与性能判据分开，避免互相掩盖。

## 本轮其它事实

- 8 rank 全部完成（native/candidate/restored 各 4 次，共 96 次 decode），各 rank verdict 完全一致。
- 参数/buffer/CPU master、`stack_tiling` 原值、runtime flags 全部还原；未安装任何数学补丁；无采样/DiT。
- driver 的事后校验用了 R138 的字段名（`decode_base_timer_removed`），在**工作完成之后**报 `KeyError`
  导致进程 exit 1；worker 回执完整，故数据有效。该 driver 缺陷已在 evidence 中标注，后续轮次修正。
- 两次前置失败均在 spawn 前或模型加载前 fail-closed，回执保留。
- 私有进程/key 已清理，ts-217 lease 已释放。