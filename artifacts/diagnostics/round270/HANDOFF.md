# R270 — 交接：M1（FA 双缓冲）实施规格 + 当前全量状态

## 一、当前交付（可复现，勿再改动）

| 项 | 值 | 证据 |
|---|---|---|
| 基线 | 17.928 s/step | R235/R240 |
| **交付组合（49 步 ×2 验证）** | **16.32 s/step（中位），−9.0%** | `round259/`、`round261/` |
| 数值 | 全程在原生自重复带 9.84e-04–1.077e-03 | 同上 |
| 形态 | `artifacts/deploy/h3_perf_switches.py`，`H3_PERF_COMBO=1`，**默认 OFF** | `artifacts/deploy/README.md` |
| 组件 | FA padding trim（已批准）+ tiles 64/912（已批准）+ guard scale==1 快路径（逐位一致） | — |

## 二、M1 剩余目标：15 s（需 FA 152→125 ms，1.22×）

### 已确认的事实链

1. R254：整步设备 **100% 忙碌、空闲 0.1 ms**，多引擎重叠 2.21× ⇒ 只能靠内核更快；
2. R255 S1：FA `vec 54.8%` / `cube MAC 22.7%`（向量受限）；
3. R266/R268：`compute_in_each_kv_block` 的真实顺序为
   **[cube S_j] → [vec softmax_j] → [cube S_{j+1}（预取）] → [cube PV_j]**；
   向量段期间 cube 空转，因为**下一块的 S 计算被放在 softmax 之后**、且 S 只有一份缓冲；
4. R269：TIK `for_range` 体只 trace 一次 ⇒ Python 侧“按奇偶换缓冲”不生效，**必须循环体展开 2 块**。

### 实施规格（方案 A：展开 2 块 + 双 S 缓冲）

目标顺序（每块）：
```
[cube S_j] → [cube S_{j+1}] → [vec softmax_j] → [vec softmax_{j+1}] → [cube PV_j] → [cube PV_{j+1}]
```
即：先发两块 matmul，再发两块 softmax，再发两块 PV ⇒ 向量与 cube 在指令流上交错。

具体步骤（在 `/tmp/fa_pkg_r266`，**不替换** `/tmp/fa_pkg`）：
1. `compute_in_each_q_block` 分配 `Sij_a` / `Sij_b` 两个 UB 缓冲；
2. 抽出 `_kv_step(kv_idx, s_cur, s_next, P_prev...)`，函数体即现有 `compute_in_each_kv_block`，
   但把「下一块 S 的 matmul」提前到函数**开头**（用 `s_next` 写）；
3. 主循环 `for kk in range((Tc-1)//2)`：依次调
   `_kv_step(2kk, a, b)`、`_kv_step(2kk+1, b, a)`；
4. 余块（`Tc-1` 为奇数）单独调一次；
5. 末块（`kv_blk_idx == Tc-1`）保持循环外的 `workspace=None` 路径；
6. `next_block_num/prev_block_num` 的 `if_scope` 过滤按**实际块号**保留；
7. 完成后 `py_compile` 校验。

### 验证回路（R267 已固化）

```bash
mv /tmp/fa_pkg /tmp/fa_pkg_baseline_hold
cp -a /tmp/fa_pkg_r266 /tmp/fa_pkg
# 跑微基准（同口径基线 154.73 ms，tiles 64/912，seed 20261124）
rm -rf /tmp/fa_pkg && mv /tmp/fa_pkg_baseline_hold /tmp/fa_pkg   # 强制恢复
```

产物必须有四个数：**ms/块、`max|diff|`、`relRMS`、非有限计数**。
≥1.2× 且数值可接受 ⇒ 8 rank 单步（叠加 `H3_PERF_COMBO=1`，对照 16.32 s）⇒ 49 步。

### 风险与备选

- 展开后若 TIK 编译报 UB/tiling 错误 ⇒ 回退单缓冲，改试「softmax 与 PV 换序」（更小改动）；
- 若微基准 <1.05× ⇒ 该方向收益不足，回到 16.32 s 并如实上报；
- 数值若超出原生带（已获批可接受差异，但仍需按闸门记录）。

## 三、未动的并行目标

encoder<2 s、video VAE<20 s（冷 142.143 s / 暖 20.565 s + 0.210 s）、A2/A3 对比 —— 均未开始。

## 四、清理与租约

- 主机无残留进程；`/tmp/msprof*` 已删；`/tmp/h3_budget149` 1.2 G、`/tmp/h3_fa245b` 76 M、
  `/tmp/fa_pkg`（原始）+ `/tmp/fa_pkg_r266`（候选工作区）保留供继续；
- 会话结束时：删除 `/tmp/fa_pkg_r266`、`/tmp/h3_r221` 临时件，并 `ts_release ts-217`。
