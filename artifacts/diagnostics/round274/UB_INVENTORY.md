# R274 — UB 占用清单与 U1 的具体腾挪方案

## 一、关键缓冲（按源码实测尺寸，d=128、N0=16、q_blk_h_aligned=M）

| 缓冲 | 位置 | 形状/类型 | 尺寸（M=128） | 生命周期 |
|---|---|---|---|---|
| `Sij_ub_zN` | UB | `(Bc//16, M, 16)` FP16 | Bc=304: 19×128×16×2 = **76 KB** | softmax 读完后**即死**，只用于下一块 QK^T |
| `Oij_ub_zN`（`output_compute` 内） | UB | `M×d` FP32（mmad 输出） | 128×128×4 = **64 KB** | 仅本次 `output_compute` 的 stmt scope 内 |
| `Oi_ub_zN`（`output_update` 内） | UB | `(d/16, M, 16)` FP32 | 8×128×16×4 = **64 KB** | 仅本次 `output_update` 内 |
| `Oi_ub_zN_fp16`（`output_norm` 内） | UB | 同形状 FP16 | 32 KB | 仅输出归一化时 |
| `Pij_l1_zN` | **L1** | — | — | 不占 UB |

R272 的报错正是 `Oi_ub_zN`（64 KB）申请不到（仅剩 29 KB）。

## 二、结论：UB 里**同时存在**两块 64 KB 级临时量

`output_update` 与 `output_compute` 在相邻调用中各自申请 64 KB 临时缓冲，
而 `Sij`（76 KB）在 softmax 之后到下一块 matmul 之前**是死区**。
⇒ 如果能让 `Oij`/`Oi` 临时量**复用 Sij 的死区**，可一次性腾出 ≥64 KB，
正好够第二个 S 缓冲（64 KB 级）做双缓冲。

## 三、U1 的三种具体做法（按可控性排序）

| 方案 | 做法 | 预期腾出 |
|---|---|---|
| **U1a（推荐）** | 把 `Sij` 的生命周期**明确收缩**：软 max 后立刻 `free`（用 `new_stmt_scope` 包住 softmax 段），让后续 64 KB 申请能复用其空间 | 76 KB |
| U1b | `output_update` 的 `Oi_ub_zN` 改为**两段搬运**（每次只搬 n1_size/2），把 64 KB 降到 32 KB | 32 KB |
| U1c | `output_compute` 的 `Oij` 改为 FP16 累加（**改变数值**，需闸门；不优先） | 32 KB |

## 四、下一步（R275）

1. 试 **U1a**：给 `softmax_compute` 调用包 `new_stmt_scope`，观察 `Oi_ub_zN` 是否能在其后分配到空间；
2. 若 U1a 生效 → 重跑 R272 的双缓冲补丁（私有副本已是修补版），走 R267 回路测 ms；
3. 若 U1a 无效 → 试 U1b（分两段搬运，纯搬运拆分、不改数值）；
4. 全程用 `trap restore`，部署路径必须恢复。

## 五、合规

仅阅读源码与尺寸推算；未改任何文件；部署路径为原始状态。
