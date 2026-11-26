# R360 — TIK 可用性实测：API 全部就绪，但 SOC 被解析成 Ascend310B（下一步要解决的选择器问题）

## 一、TIK 可用性（正面）

主机上 TIK 完整可用：

```python
from te import tik
tik.Tik()                      # OK
dir(tik.Tik)                   # 600 个方法，含 mmad / load2dv1..v3 / data_move / BuildCCE / matmul
```

关键 API 签名（实测）：

| API | 签名 |
|---|---|
| `Tensor` | `(dtype, shape, scope, name, ...)` —— **name 必填**（local/gm 都是） |
| `load2dv1` | `(dst, src, index, repeat_times, src_stride, sid, if_transpose=False, addr_mode=None)` |
| `load2dv2` | `(dst, src, start_index, repeat_times, dst_gap, src_stride, sid, ...)` |
| `mmad` | `(dst_fm, src_fm, src_filter, matrix_m, matrix_k, matrix_n, is_bias, fm_offset=0, ...)` |
| `data_move` | `(dst, src, sid, nburst, burst, src_stride, dst_stride, *args)` |
| `BuildCCE` | 构建入口 |

**scope 取值**（`te/tik/api/tik_tensor.py` 的常量实测值）：

```
gm = 'global'   ca = 'local.L0A'   cb = 'local.L0B'
cc = 'local.L0C'   ub = 'local.UB'   cbuf = 'local.L1'
```

## 二、阻塞点：SOC 解析成 Ascend310B

最小 16×16×16 TIK matmul 脚本已写好（`/tmp/h3_r221/tik_mm.py`），构建时报：

```
Error: current chip: Ascend310B AiCore doesn't support float16 for Instruction load2dv1
```

TIK 的 SOC 来自 `te/tik/common/tik_get_soc_name.py` 的
`get_soc_spec("SHORT_SOC_VERSION")`，本机解析结果为 **Ascend310B**（与真实设备 910A/910B 不符）。

已尝试（均无效）：
- `tik.Tik(soc_version=...)` —— 该参数不存在；
- `SOC_VERSION` / `ASCEND_SOC_VERSION` 环境变量；
- 直接改 `TIK_SOC_INFO.soc_name`；
- 替换 `tik_get_soc_name.get_soc_spec`。

⇒ SOC 选择在更上游（`te.platform` / `tbe.common.platform` 的缓存或 `ASCEND_PLATFORM` 类配置），
**下一会话第一步应定位并覆盖它**（候选：`tbe.common.platform.get_soc_spec`、
`te.platform.platform_info`、`ASCEND_PLATFORM` 环境变量、或 CANN 的 `platform_config` 文件）。

## 三、下一会话的行动清单

1. 找到 SOC 解析的真实来源并覆盖为 `Ascend910A`（可用 `python -c "from tbe.common.platform import get_soc_spec; print(get_soc_spec('SOC_VERSION'))"` 逐步排查）；
2. 编译最小 TIK matmul → **反汇编确认 `MAD` 出现**；
3. 通过后跑数值（A=I ⇒ C=B）；
4. 移植 64×512×128 FA，接回部署开关。

## 四、状态

- 交付不变：**16.34 s/step（−8.9%）**，`H3_PERF_COMBO=1` 默认 OFF；
- AscendC 路线已证据完整关闭（R356/R359）；TIK 路线**已可运行到构建阶段**，仅差 SOC 选择；
- TIK 脚本与工程：远端 `/tmp/h3_r221/tik_mm.py`、`tikbuild/`。
