# R269 — 实施前发现的关键约束：TIK 的循环体只 trace 一次，Python 侧缓冲交换**不会**生效

## 一、问题

R268 的方案是「双 S 缓冲 + 调用处按 `kv_blk_idx % 2` 交替」。但该文件用的是 TIK DSL：

```python
with self.tik_instance.for_range(0, self.Tc - 1, name="kv_blk_idx") as kv_blk_idx:
    ... = self.compute_in_each_kv_block(..., workspace=Sij_ub_zN)
```

`for_range` 的函数体在**生成期只执行一次**（trace 一次、由 TIK 生成设备循环），因此：

- Python 层的 `if kv_blk_idx % 2: swap` **在生成期就固定了**，不会形成“奇偶交替”的设备行为；
- TIK tensor 对象也不能用运行时下标动态选择（无 `Sij[kv_blk_idx%2]` 语义）。

⇒ 直接把 R268 的“调用处交换”写上去，**会产生错误结果或编译失败**，必须先改实现形态。

## 二、可行的三种实现（按改动量排序）

| 方案 | 做法 | 代价 | 风险 |
|---|---|---|---|
| **A. 循环体展开 2 块** | `for kk in range((Tc-1)//2)`：体内处理 2kk（A→B）与 2kk+1（B→A），再处理奇偶余块 | 中等：一个循环体写两遍（可抽函数） | 低（仍是同一算法） |
| B. TIK 条件分支 | 用 `if_scope(kv_blk_idx % 2 == 0)` 两支分别引用 A/B 缓冲 | 编码生成两份体，UB 占用上升 | 中（两份体重复，易失步） |
| C. 改数据布局 | 把两个 S 缓冲拼成一个 `(2, Bc//16, q_blk_h_aligned, 16)` tensor，用 offset 切换 | 需要 TIK 支持带 offset 的 matmul workspace | 高（要改 matmul_compute 接口） |

**选择 A**：改动可控、语义最清晰。抽一个 `_kv_block_step(kv_blk_idx, workspace_cur, workspace_next, ...)`
包装现有 `compute_in_each_kv_block` 的调用，循环体里调用两次（第二次时 A/B 对调）。

## 三、附带的两个必须处理的边界

1. **末块**：`compute_in_each_kv_block` 在 `kv_blk_idx == Tc-1` 时走 `workspace=None` 的本地路径
   （KV 循环外单独调用）——展开时保持该调用在循环外，不参与双缓冲；
2. **`kv_blk_idx < Tc-2` 的内部预取条件**：展开后要按**实际块号**判断（2kk 与 2kk+1 各自判断），
   不能沿用单一 `kv_blk_idx`；
3. **`next_block_num` / `prev_block_num` 的 if_scope 过滤**：仅在因果/滑窗时生效（dense 下恒真），
   展开时要保留同样的过滤语义。

## 四、数值影响（复核）

计算顺序不变（仍逐块累加、FP32），仅缓冲地址不同 ⇒ 预期逐位一致；仍按批准的四项闸门记录。

## 五、下一步（R270）

1. 在 `/tmp/fa_pkg_r266` 用方案 A 实现（不替换 `/tmp/fa_pkg`，先 `py_compile` 校验）；
2. 用 R267 的替换/恢复回路跑微基准（同口径基线 **154.73 ms**）；
3. 产出：ms/块、`max|diff|`、`relRMS`、非有限计数；≥1.2× 且数值可接受 ⇒ 8 rank 单步。
