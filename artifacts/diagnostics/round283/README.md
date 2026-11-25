# R283 — FA 内核变体的通用数值/性能闸门（两进程，供 W1 及后续使用）

## 为什么需要

`fa910a.py:217` 硬编码 `sys.path.insert(0, "/tmp/fa_pkg")` ⇒ 参考内核与候选内核**不能在同一进程**加载。
本脚本用「两进程 + 一个 `.pt` 中转」解决：

```bash
# 1) 参考：在当前 /tmp/fa_pkg 上测并保存输出
python numeric_gate283.py --mode baseline  --store /tmp/fa_gate.pt --out /tmp/gate_base.json

# 2) 换上候选包后测，与保存的参考逐元素对比
python numeric_gate283.py --mode candidate --store /tmp/fa_gate.pt --out /tmp/gate_cand.json
```

## 输出字段（candidate）

`median_ms`、`delta_ms`、`speedup`、`identical`、`mismatched_words`、`max_abs_diff`、`relRMS`、`nan`、`inf`、`sum`。

## 与合同的关系

- 该脚本**只测量、不采纳**；
- 任何 `identical=false` 的候选必须先在 Issue 上报 `max_abs_diff`/`relRMS` 并取得用户决定；
- 建议判据：`relRMS` 与 in-model 49 步的原生自重复带（9.84e-04–1.077e-03）同量级，再谈是否进模型。

## 使用前置

- 形状 B1 H7 N41792 D128、seed 20261124、tiles 由 env `H3_FA_BR_CAP/BC_CAP` 控制
  **仅在加载 trim 模块时生效**（R279 已记录该坑；普通路径固定 128/304）；
- 换包必须走 R267 的「替换 → 测量 → **强制恢复**」流程，并核对部署 md5。
