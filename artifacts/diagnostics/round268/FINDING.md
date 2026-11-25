# R268 — 找到串行的**具体原因**：`Sij` 工作区被原地复用（WAR 依赖）

## 一、代码事实（`/tmp/fa_pkg_r266/.../flash_attention_fwd.py`）

`compute_in_each_q_block`（L401–465）只为整个 KV 循环分配**一个** `Sij` UB 缓冲：

```python
Sij_ub_zN = self.tik_instance.Tensor(FP16, (self.Bc // 16, q_blk_h_aligned, 16),
                                     scope=UB, name="Sij_ub_zN")
Sij_ub_zN = self.tik_ops_utils.matmul_compute(Qi_l1_zN, KjT_l1_zN, ..., workspace=Sij_ub_zN)
with self.tik_instance.for_range(0, self.Tc - 1) as kv_blk_idx:
    ... = self.compute_in_each_kv_block(..., workspace=Sij_ub_zN)
```

而 `compute_in_each_kv_block` 内部（L315–400）：

```python
Pij_l1_zN, ... = self.softmax_compute(Sij_ub_zN, ...)      # 向量：读 Sij
...
with self.tik_instance.if_scope(kv_blk_idx < self.Tc - 2):
    ... KjT for kv_blk_idx+1 ...
    Sij_ub_zN = self.tik_ops_utils.matmul_compute(Qi_l1_zN, KjT_l1_zN, ...,
                                                  workspace=Sij_ub_zN)   # cube：写同一个 Sij
```

## 二、结论：这是 **WAR 依赖**，不是“编译器没优化”

- 第 j 块的 `softmax` **读** `Sij`；
- 第 j+1 块的 `QK^T` **写**同一个 `Sij`（原地复用）；
⇒ 硬件/编译器**必须**等 softmax 读完才能启动下一个 QK^T。
这就是「向量阶段期间 cube 空转」（S1：`mac_ratio 22.7%`）的直接机制，也解释了为什么
已有的 K 预取 + 下一块 S 计算（第 5 步）没能与 softmax 重叠——它们落回同一个缓冲。

## 三、最小改动（R269 实施，比“双 P 缓冲”更小更准）

1. 在 `compute_in_each_q_block` 里分配**两个** S 缓冲 `Sij_a` / `Sij_b`；
2. 首次 QK^T 写入 `Sij_a`；
3. `compute_in_each_kv_block` 增加参数 `workspace_cur` / `workspace_next`：
   - `softmax_compute(workspace_cur, ...)`；
   - 下一块的 `matmul_compute(..., workspace=workspace_next)`；
4. 循环调用处按 `kv_blk_idx % 2` 交换两个缓冲（Python 侧生成展开代码，TIK 无需动态索引）；
5. 首/末块与非掩码分支保持原路径；
6. 用 `new_stmt_scope` 保持同步范围不变，必要时加显式 set/wait flag。

**预期**：QK^T(j+1) 与 softmax(j) 真正并行 ⇒ cube 占用上升；
R266 估计的 1.45–1.6× 有望达到，甚至更高（因为这是唯一的硬串行点）。

## 四、数值影响

- 计算顺序**不变**（仍是逐块累加、同样的 FP32 累加），只是缓冲交替；
- 理论上输出应**逐位一致**（除非 TIK 编译器的调度改变了 CUBE 累加顺序）；
- 实测按闸门记录：`max|diff|`、`relRMS`、非有限计数；若逐位一致则无需动用已获批的数值容差。

## 五、下一步（R269）

按第三节改 `/tmp/fa_pkg_r266`，用 R267 的替换/恢复回路跑微基准（同口径基线 **154.73 ms**），
产出 ms 与数值四项；若 ≥1.2× ⇒ 进 8 rank 单步（叠加 `H3_PERF_COMBO=1`）。
