# R363 — TIK 在 Ascend910 上的 L0C 写回三条路全被拒；raw builtin 是最后一种可能

## 一、本轮实测（TIK 已通过 SoC/版本检查，问题集中在 L0C 写回）

| 尝试 | 结果 |
|---|---|
| `data_move(c_gm, c_l0c, ...)`（L0C→GM） | `Error: data_move doesn't support L0C to OUT` |
| `fixpipe(c_gm, c_l0c, 16, 2, 0, 0)` | 生成 `copy_matrix_cc_to_ubuf` → **v200 架构不支持该指令** |
| `fixpipe(c_l1, c_l0c, ...)`（L0C→L1） | `Error: dst's scope must be scope_gm` |
| `data_move(c_l1, c_l0c, ...)`（L0C→L1） | `Error: data_move doesn't support L0C to L1` |

TIK 内部映射表（`tik_data_move_common.py:172`）只把 L0C 写回接到 **`copy_matrix_cc_to_ubuf`**
（L0C→UB），而该指令在 Ascend910（v200）上不被支持。

## 二、结论

- TIK **能**在 Ascend910 目标上通过全部 SoC/版本检查并进入代码生成（R362，重大进展）；
- 但 TIK 暴露的 **L0C 写回 API 只有 UB 中转这一条**，在 910A 上不可用 ⇒
  **纯 TIK 高层 API 仍不足以写出可用的 matmul**。

## 三、剩下唯一可能：直接调 raw builtin

编译器内建（R354 已确认存在）里有 **`copy_matrix_cc_to_cbuf`**（L0C→L1），
这正是 CANN 自家 910A 算子（Conv2D，144 条 MAD）使用的方式。
TIK 生成代码用的是 `cce_aicore_intrinsics.h` 里的 builtin，所以可以：

1. 在 TIK 生成的 `.cce` 中间代码里**手工把 `copy_matrix_cc_to_ubuf(...)` 换成
   `copy_matrix_cc_to_cbuf(dst=l1, src=l0c, ...)`**，再 `data_move(l1→gm)`；
   或
2. 直接在 AscendC kernel 里调用 raw builtin：
   `load_cbuf_to_ca/cb(...)` + `mmad(...)` + `copy_matrix_cc_to_cbuf(...)`，
   绕开 AscendC 与 TIK 两层封装。

两条都需要查 `cce_aicore_intrinsics.h` 的参数顺序（TIK 源码里 `tik_data_move_operation_.py:379`
有 `copy_matrix_cc_to_ubuf` 的完整参数列表，可据此推断同族 builtin 的格式）。

## 四、状态

- 交付不变：**16.34 s/step（−8.9%）**，`H3_PERF_COMBO=1` 默认 OFF；
- AscendC 高层/底层：已关闭（R356/R359）；
- **TIK：SoC 打通、代码生成可达，卡在 L0C 写回指令族**；
- 下一步：raw builtin（`copy_matrix_cc_to_cbuf`）——这是与 CANN 自家 910A 算子一致的写回路径。
