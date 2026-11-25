# R271 — 方案 A 的**代码级**补丁规格（可机械执行）

目标：让 `S_{j+1}` 的 cube matmul 与 `softmax_j` 的向量工作真正重叠。
现状顺序（R266 已读证）：`[cube S_j] → [vec softmax_j] → [cube S_{j+1}] → [cube PV_j]`。
只换缓冲**不会**带来重叠——必须**把 `S_{j+1}` 的 matmul 提前到 `softmax_j` 之前**，再用双缓冲消除依赖。

## 补丁 1：`compute_in_each_kv_block` 增加两个参数

```python
def compute_in_each_kv_block(self, Qi_l1_zN, Oi_l1_zN, li_ub, mi_ub, batch_start,
                             batch_idx, q_blk_idx, kv_blk_idx, q_blk_h, kv_blk_h,
                             workspace=None, workspace_next=None):
```

## 补丁 2：把「预取 K_{j+1} + 算 S_{j+1}」整块**上移**到 `softmax_compute` 之前

从原位置删除：

```python
        with self.tik_instance.if_scope(kv_blk_idx < self.Tc - 2):
            kv_gm_offset = self.get_gm_offset(..., kv_blk_idx + 1)
            KjT_l1_zN = self.tik_instance.Tensor(FP16, (self.d // self.N0, self.Bc, self.N0), ...)
            self.unroll_data_move(...)
            Sij_ub_zN = self.tik_ops_utils.matmul_compute(Qi_l1_zN, KjT_l1_zN, q_blk_h,
                                                          self.actual_d, self.Bc,
                                                          N1MN0_to_MN=False, workspace=Sij_ub_zN)
```

插入到 `Pij_l1_zN, ... = self.softmax_compute(...)` **之前**，并且：

- 落点改为 `workspace=workspace_next`（当 `workspace_next is not None` 时），否则保持原 workspace；
- 其余参数（`kv_blk_idx + 1`、`self.Bc`、`if_scope(kv_blk_idx < self.Tc - 2)`）不变。

顺序变为：`[mask] → [cube S_{j+1} → B]` 然后 `[vec softmax_j ← A]` 然后 `[cube PV_j]`。
编译器即可把 softmax_j（读 A）与 S_{j+1}（写 B）排到同一时间窗。

## 补丁 3：`compute_in_each_q_block` 分配两个 S 缓冲并按奇偶交替

因为 TIK `for_range` 体只 trace 一次（R269），必须**循环体展开 2 块**：

```python
        Sij_a = self.tik_instance.Tensor(FP16, (self.Bc // 16, q_blk_h_aligned, 16), scope=UB, name="Sij_a")
        Sij_b = self.tik_instance.Tensor(FP16, (self.Bc // 16, q_blk_h_aligned, 16), scope=UB, name="Sij_b")
        Sij_a = self.tik_ops_utils.matmul_compute(Qi_l1_zN, KjT_l1_zN, q_blk_h, self.actual_d,
                                                  self.Bc, N1MN0_to_MN=False, workspace=Sij_a)
        n_kv = self.Tc - 1
        with self.tik_instance.for_range(0, (n_kv + 1) // 2, name="kv_pair") as kv_pair:
            j0 = kv_pair * 2
            with self.tik_instance.if_scope(j0 < n_kv):
                with self.tik_instance.if_scope(tik.all(j0 - self.next_block_num <= q_blk_idx,
                                                        q_blk_idx <= j0 + self.prev_block_num)):
                    Oi_l1_zN, li_ub, mi_ub = self.compute_in_each_kv_block(
                        Qi_l1_zN, Oi_l1_zN, li_ub, mi_ub, batch_start, batch_idx, q_blk_idx,
                        j0, q_blk_h, self.Bc, workspace=Sij_a, workspace_next=Sij_b)
            with self.tik_instance.if_scope(j0 + 1 < n_kv):
                with self.tik_instance.if_scope(tik.all(j0 + 1 - self.next_block_num <= q_blk_idx,
                                                        q_blk_idx <= j0 + 1 + self.prev_block_num)):
                    Oi_l1_zN, li_ub, mi_ub = self.compute_in_each_kv_block(
                        Qi_l1_zN, Oi_l1_zN, li_ub, mi_ub, batch_start, batch_idx, q_blk_idx,
                        j0 + 1, q_blk_h, self.Bc, workspace=Sij_b, workspace_next=Sij_a)
```

- 若 `n_kv` 为奇数，最后一次只处理 `j0`（`j0+1 < n_kv` 自动跳过）；
- **末块**（`kv_blk_idx == Tc-1`）仍走循环外的 `workspace=None` 路径，保持不变；
- `next_block_num / prev_block_num` 的 `if_scope` 过滤按**实际块号**保留（上面已按 j0 / j0+1 分别写）。

## 补丁 4（可选）：显式同步

若 TIK 自动同步不足以保证「softmax 读 A 与 matmul 写 B 并行」，在 S 计算与 softmax 之间加
`with self.tik_instance.new_stmt_scope()` 并在两缓冲间加 set/wait flag（先不加，靠 auto-sync 试）。

## 验证与风险

- 先 `py_compile`；再按 R267 回路替换 `/tmp/fa_pkg` 跑微基准（基线 **154.73 ms**）；
- 若编译报 UB 超限：把 `q_blk_h_aligned` 对应的 S 缓冲 dtype 保持 FP16、检查总 UB 占用；
- 若数值不逐位一致：按已批准闸门记录 `max|diff|`、`relRMS`、非有限计数；
- 若 <1.05×：判为该方向收益不足，回到 16.32 s 如实上报。
