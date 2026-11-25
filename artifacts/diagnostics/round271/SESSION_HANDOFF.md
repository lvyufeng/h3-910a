## 会话交接（2026-11-25 20:xx）：可交付成果已固化，M1 内核实现待接手

### 已交付（可复现，勿动）

- **DiT 16.32 s/step（−9.0% vs 17.928 s 基线）**，49 步 ×2 独立验证，数值全程在原生自重复带；
- 形态：`artifacts/deploy/h3_perf_switches.py` + `H3_PERF_COMBO=1`，**默认 OFF、可逐项、可回退、可审计**；
- 口令：`H3_PERF_FA_TILES=1 H3_PERF_FA_TRIM=1 H3_PERF_GUARD_FAST=1`。

### 待接手：M1（15 s 硬目标，只差 FA 152→125 ms，1.22×）

**上下文完备，无未决设计问题：**

1. 事实链：R254（设备 100% 忙碌、空闲 0.1 ms）→ R255（FA 占 44%，vec 54.8%/cube MAC 22.7%）
   → R266/R268（顺序 `[cube S_j]→[vec softmax_j]→[cube S_{j+1}]→[cube PV_j]`，串行根因）
   → R269（TIK 循环体只 trace 一次 ⇒ 必须展开 2 块）；
2. 代码级补丁：`round271/PATCH_SPEC.md`（4 个补丁：加参数、把 S_{j+1} matmul 上移到 softmax 之前、
   分配 Sij_a/Sij_b 并展开 2 块、必要时补显式同步）；
3. 验证回路：`round267/TESTLOOP.md`（替换 `/tmp/fa_pkg` → 微基准 → **强制恢复**）；
   同口径基线 **154.73 ms**（tiles 64/912，seed 20261124）；
4. 数值闸门：用户已批准（2026-11-25），按四项记录 ms / `max|diff|` / `relRMS` / 非有限计数；
5. 判定：≥1.2× 且数值可接受 ⇒ 8 rank 单步（叠加组合，对照 16.32 s）⇒ 49 步 ⇒ 15 s 即达成。

### 10 s 的结论（已定档）

需 **FA ~2.2× 且非 FA −3.7 s** 同时成立，概率 10–20%；仅在允许下调数值精度时重新立项。
见 `artifacts/diagnostics/FEASIBILITY_10S.md`。

### 未动的并行目标

encoder<2 s、video VAE<20 s（冷 142.143 s / 暖 20.565 s + 0.210 s）、A2/A3 对比。

### 环境与清理

- 主机 ts-217 无残留进程；`/tmp/fa_pkg`（原始）与 `/tmp/fa_pkg_r266`（候选工作区）完好；
- 会话收尾：删 `/tmp/fa_pkg_r266`、`/tmp/h3_r221`，并 `ts_release ts-217`。
