# R263 — 读现役 TIK FA 内核源码：结构是「分阶段串行」，这是 mac_ratio 22.7% 的直接原因

## 一、源码事实

路径：`/tmp/fa_pkg/flash_attention/ops/flash_attention/flash_attention_fwd.py`（**538 行**）。

函数结构（每个 KV 块迭代内依次调用）：

```
compute_in_each_kv_block(...)
  ├─ rowmax_compute(...)        # 行最大值（向量）
  ├─ rowsum_compute(...)        # 行和（向量）
  ├─ softmax_compute(...)       # exp / 归一（向量）
  ├─ output_update(...)         # P·V 累加（cube）+ 缩放（向量）
  ├─ output_compute(...)
  └─ output_norm(...) / output_norm_and_write(...)
```

- `set_flag` / `wait_flag` 在源码中**出现 0 次** ⇒ 同步由 TIK 的高层队列语义隐式完成；
- 迭代内是**顺序调用的阶段序列**，没有显式的跨迭代双缓冲工作区。

## 二、与 S1 指标对上

| S1 指标 | 值 | 与结构的关系 |
|---|---|---|
| `vec_ratio` | 0.548 | rowmax/rowsum/softmax/output_update 的向量部分占主导 |
| `mac_ratio` | 0.227 | cube 只在 `output_update`/QK^T 段工作，且被向量阶段阻断 |
| `cube_utilization` | 99.4% | cube 被“占用”但大量时间在等向量阶段产出 P |

⇒ **单核内先用向量算 softmax、再让 cube 算 PV，两者串行**，这就是 15% MFU 的来源。

## 三、M1 的具体改法（据此确定，不再猜测）

1. **跨迭代软件流水**：为 P 与 PV 累加器各准备两份工作区，
   让「第 i 块的 rowmax/softmax（向量）」与「第 i−1 块的 P·V（cube）」并行；
   TIK 里通过显式双缓冲 tensor + 手动 sync 实现，或在 AscendC 用 TQue 双缓冲直接写；
2. **削减向量工作量**：把 `output_update` 里的缩放并入 exp 的指数因子（数学等价，需数值闸门）；
   rowmax 与 rowsum 可合并为一次 reduction（当前分两趟）；
3. **尾块均衡**：trim 后 327×138 网格，末行/末列任务量小，考虑把尾块并入相邻核。

预期：若 cube 与向量真正重叠，内核时间由 `max(vec, cube)` 决定 ⇒ 从 152 ms 降到约 90–100 ms，
即 **1.5–1.7×**，足以覆盖 15 s 所需的 1.22×。

## 四、下一步（R264）

1. 复制 `/tmp/fa_pkg` 到私有工作目录，改 `compute_in_each_kv_block` 为双缓冲版本（TIK 先验证可行性）；
2. 独立微基准（B1 H7 N41792 D128、seed 20261124）测 ms/块，与 152.19 对比；
3. 数值闸门①：输出 max|diff|/relRMS（预期改变归约顺序，需用户批准后才进模型）。
