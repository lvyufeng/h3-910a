# R266 — 读透 `compute_in_each_kv_block`：内核**已有一层流水**，缺的是「softmax 与 P·V 重叠」

## 一、实际循环体（私有副本 `/tmp/fa_pkg_r266`，未改安装目录）

`flash_attention_fwd.py:315-400`，每个 KV 块的执行顺序是：

| 步 | 操作 | 引擎 |
|---|---|---|
| 1 | load `V_j`（L1）、load `K_j`（L1） | MTE |
| 2 | `S_ij = Qi · K_jᵀ` | **cube** |
| 3 | attn mask（仅 when has_attn_mask） | 向量 |
| 4 | `softmax_compute(S_ij)` → `P_ij`、li、mi、update_scale | **向量** |
| 5 | **预取 `K_{j+1}` 并算 `S_i(j+1)`**（`kv_blk_idx < Tc-2` 时） | MTE + **cube** |
| 6 | `output_compute(P_ij, V_j)` → `O_i`（P·V 累加 + 缩放） | **cube** + 向量 |

## 二、这修正了 R263 的说法（更准确）

- ❌ R263 说“完全没有双缓冲/流水”——**不准确**：内核**已经**做了 K 预取 + 下一块 `S` 的 matmul（第 5 步），
  这是 TIK 侧已有的一层流水；
- ✅ 真正的缺口：**第 4 步 softmax（向量）与第 6 步 P·V（cube）仍是串行**，
  而第 5、6 步都是 cube，中间夹着一次纯向量阶段 ⇒ cube 在 softmax 期间空转（对应 S1 的 mac_ratio 22.7%）。

## 三、据此确定的最小改动（R267 实施）

目标顺序（2 深度流水，双 P 缓冲）：

```
load K_{j+1};  S_{j+1} = Qi·K_{j+1}ᵀ        (cube)
PV_j = P_j·V_j                              (cube, 与下一块的 softmax 并行发出)
softmax(S_{j+1}) -> P_{j+1}                 (向量, 与 PV_j 重叠)
```

具体：
1. `Pij` 改为 **两份 UB 缓冲**（`Pij_a` / `Pij_b`，按 `kv_blk_idx % 2` 轮换），使 `P_{j+1}` 的 softmax 写入
   与 `P_j` 的 P·V 读取互不冲突；
2. 把第 6 步 `output_compute` 的调用**提前**到第 4 步之前（用上一轮的 P、V），
   即循环体改为「先发 PV，再 softmax 下一块」；
3. 用显式 `set_flag/wait_flag`（当前源码 0 处显式同步）保证跨引擎顺序；
4. 首/末块（`j==0`、`j==Tc-1`）保留标量分支。

## 四、预期

若 cube 与向量真重叠，单块时间由 `max(向量, cube)` 决定：
152.19 → 约 95–105 ms（1.45–1.6×），足以覆盖 15 s 所需的 1.22×。

## 五、下一步（R267）

1. 在 `/tmp/fa_pkg_r266` 内改 `compute_in_each_kv_block` 与 `softmax_compute`/`output_compute` 的缓冲签名；
2. 编译并在微基准（同形状、seed 20261124、tiles 64/912）对比 154.73 ms；
3. 记数值：输出 `max|diff|`、`relRMS`、非有限计数（**数值闸门已获用户批准**）。
