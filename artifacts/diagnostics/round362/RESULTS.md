# R362 — TIK 已打通到**代码生成**阶段（Ascend910 目标）；只剩 L0C 写回方式

## 一、本轮突破（关键）

### 1. 正确的 TIK SoC 名字是 **`Ascend910`**（不是 `Ascend910A`）
`te/tik/common/tik_api_map.py:598` 的 `arch_version_mode` 键为：
`Ascend910 / Ascend910B / Ascend910_93 / Ascend950 / Ascend610 / Ascend310P / BS9SX1A / ...`
之前的 `KeyError` 全是因为用了 `Ascend910A`/`Ascend910A1` 这类**不在映射表里**的名字。

### 2. 拦截方式（合规，不改共享 SDK）
`/tmp/h3_r221/tik_socfix.py`，在 `import te.tik` **之前**：
- patch `tbe.common.platform.get_soc_spec` 与 `tbe.common.platform.platform_info.get_soc_spec`，
  令 SOC 类 key 返回 `"Ascend910"`；
- patch `te.tik.common.tik_get_soc_name.get_soc_spec` 并清掉 `TIK_SOC_INFO` 缓存。

效果：`get_soc_name() -> Ascend910`，`BuildCCE` 通过 SoC/版本模式检查，进入真正的代码生成。

### 3. TIK API 细节（已全部摸清）
| API | 用法 |
|---|---|
| `Tensor(dtype, shape, scope, name)` | name 必填；scope: `global` / `local.L1` / `local.L0A/B/C` / `local.UB` |
| `data_move(dst, src, sid, nburst, burst, src_stride, dst_stride)` | GM↔L1/UB |
| `load2dv1(dst, src, index, repeat_times, src_stride, sid, if_transpose)` | L1→L0A/L0B |
| `mmad(dst, src_fm, src_filter, m, k, n, is_bias, ...)` | cube 乘加 |
| `fixpipe(dst, src, cburst_num, burst_len, dst_stride, src_stride)` | L0C→（GM/UB） |

## 二、当前唯一阻塞点

`fixpipe(c_gm, c_l0c, 1, 32, 0, 0)` 时 TIK 生成的中间代码走 **L0C→UB** 路径：

```
error: function type ... of 'copy_matrix_cc_to_ubuf' does not support the given target feature
```

⇒ 在 **Ascend910（v200 架构）上不支持 L0C→UB 的写法**（那是 310P/910B 的路径）。
需要在 `fixpipe` 上用 **L0C→GM 的直通**（`copy_matrix_cc_to_gm`）或设置 `extend_params`
指定目标为 GM/ND，避免走 UB 中转。

## 三、下一步（1–2 轮即可判定）

1. 给 `fixpipe` 传 `extend_params`（查 `te/tik` 里 `fixpipe_api` 源码看合法键，可能有
   `"is_nd"`/`"src_scope"`/`"dst_scope"` 等）；
2. 或改用 `mmad` 后直接 `fixpipe` 到 `local.L1` 再 `data_move` L1→GM（v200 支持的组合）；
3. 编译成功后**立刻反汇编确认 `MAD.` 存在**，再跑数值判据（A=I ⇒ C=B）。

## 四、状态

- 交付不变：**16.34 s/step（−8.9%）**，`H3_PERF_COMBO=1` 默认 OFF；
- **TIK 路线已从"能不能用"推进到"最后一步写回方式"**，且 Ascend910 目标确认受支持
  （`arch_version_mode` 里有 `Ascend910`，平台配置有 `Ascend910A.ini`）；
- 脚本：远端 `/tmp/h3_r221/tik_mm_fixed.py`、`tik_socfix.py`。
