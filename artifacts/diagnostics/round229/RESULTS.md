# R229/R230 — `rest` 4.09 s 的三个候选全部否证：不是分配、不是元素链、不是审计同步

R228 已否证权重等待。本轮依次检验剩下三个最可能解释 `rest`（`attn` 残项 32.7 ms/block、
`mlp` 残项 34.6 ms/block）的机制，全部用**免权重微基准**或**同 run 配对 A/B**，
不改数学、不装候选、FA trim 保持 OFF。

## 一、R229：元素链本身只有 3.6 ms，分配争用完全无影响

在 ts-217 卡 0 上复现 MLP 的真实元素链
`silu(gate.float()) * up.float()`（(10448, 7168) FP16 输入，三个 300 MB FP32 临时量），
并分别叠加「每次新建 646 MB 预取缓冲并 H2D」「复用缓冲」「只分配不拷贝」三种条件：

| 条件 | 中位 | 相对 idle |
|---|---|---|
| idle，连续张量 | **2.994 ms** | 1.00× |
| idle，**`chunk` 出来的非连续视图**（in-model 真实布局） | **3.636 ms** | 1.21× |
| 非连续 + 复用 646 MB 缓冲并发拷贝 | 3.617 ms | 1.20× |
| 非连续 + 每轮新建 646 MB 并拷贝 | 3.612 ms | 1.20× |
| 非连续 + 只新建分配不拷贝 | 3.639 ms | 1.21× |

- in-model 的 `mlp` 残项是 **34.6 ms/block**，是元素链自身代价（3.6 ms）的 **9.5×**。
- **allocator/拷贝争用对元素链没有任何影响**（three variants 与 idle 差 ≤0.03 ms）。
  ⇒ 分配争用假设（R226 的一个备选）**否证**。

## 二、R230：`strict` vs `deferred` 审计配对 A/B —— 只差 0.036 s/step

`MatrixSafetyAudit126` 在 strict 模式下对每条检查做 `int(bad)`、`int(boundary)`
两次 host 同步；155 条/步 ⇒ 310 次同步/步，钩子挂在 `qkv_proj`/`mlp.fc1`/`adaln_proj.linear`，
落点正好在 `attn`/`mlp` 父区间内。仓库自带 `deferred` 模式：同样 155 条检查、同样在
sampler 更新前判定，但设备侧归约、前向结束一次性读回。

同一次运行内 `--audit-sequence strict,deferred`，同一 step 0、同一快照、8 rank：

| 审计模式 | rank_max 步时 | matrix_checks | 结论 |
|---|---|---|---|
| `strict` | **17.955 s** | 155 | 基准 |
| `deferred` | **17.919 s** | 155 | 差 **0.036 s** |

- 8 个 rank 数值完全一致（17.955 / 17.919）。
- ⇒ **审计的 310 次 host 同步只值 0.036 s/step，不是 4 s 的来源。同步假设否证。**
- 该 A/B 本身不改模型数值路径（检查不喂入推理），但**未采用**；是否切换审计模式属流程决定，
  且收益仅 0.036 s，不具备推进价值。

## 三、`rest` 现在的状态：三个候选全灭，只剩“host/调度侧”

已否证：权重等待（R228，自然对照差 0.0042 s/step）、分配争用（R229，≤0.03 ms）、
审计同步（R230，0.036 s/step）。已知元素链仅 3.6 ms。因此 `mlp` 残项里约 **31 ms/block**、
`attn` 残项里约 **33 ms/block** 是**设备空闲或调度间隙**，不是已识别的算子或搬运成本。

下一步必须直接判据“设备空闲 vs host 派发”：
- 在 `MiniMaxH3MLP.forward` / `MiniMaxH3Attention.forward` 内部记录 host `perf_counter`
  与默认流设备事件，对比 host 墙钟间隙与设备事件间隙；
- 空探针先验证步时不变性（±0.1 s）；
- 若 host 间隙 ≈ 设备间隙 ⇒ 流水线 host-bound，方向是派发/脚本层（可能不触碰数值）；
  若设备间隙 >> host ⇒ 是算子内部/同步机制，需要更细的算子级证据。

## 四、合规

- 微基准免权重、免模型；A/B 复用已核主机缓存，未下载、未回传张量。
- 未改数学/dtype/块数/offload/调度；未安装候选；FA trim 仍 OFF。
- R230 的 `all_steps_bit_identical_to_R124_baseline=false` 与 native 自重复噪声（issue #15）一致，
  本轮不作为数值判定依据。

## 五、证据

- R229 脚本 `round229/mlp_alloc229.py`，receipts 在 ts-217 `/tmp/h3_r221/receipts229.json`、
  `receipts229b.json`
- R230 回执 `round230/receipts/candidate_rank*.json`（8 rank）