# R361 — TIK 的 SOC 覆盖仍未成功：`te_update_version` 返回 success 但 `get_soc_spec` 不变

## 一、本轮实测

```python
from tbe.common.platform import get_soc_spec
print('before:', get_soc_spec('SHORT_SOC_VERSION'))       # Ascend310B
te_update_version(soc_version="Ascend910A", core_type="AiCore")   # → 'success'
print('after:', get_soc_spec('SHORT_SOC_VERSION'))        # 仍是 Ascend310B
```

⇒ `te_update_version` **不改动** `cce.get_soc_spec` 的返回值（该 global func 由编译期平台数据驱动，
在普通 Python 进程里没有生效路径）。

## 二、已排除的覆盖手段

| 手段 | 结果 |
|---|---|
| `tik.Tik(soc_version=...)` | 参数不存在 |
| `SOC_VERSION` / `ASCEND_SOC_VERSION` 环境变量 | 无效 |
| `TIK_SOC_INFO.soc_name = "Ascend910A"` | 无效（其它缓存仍 310B） |
| 替换 `tik_get_soc_name.get_soc_spec` | 无效（检查走的是 `tbe.common.platform.get_soc_spec`） |
| `te_update_version(soc_version=..., core_type=...)` | 返回 success，但值不变 |

## 三、平台数据是齐的

`$ASCEND_HOME_PATH/aarch64-linux/data/platform_config/` 下 **`Ascend910A.ini` 存在**
（同目录共 73 个 ini，含 910B/910_93x 等）——说明不是"不支持 910A"，而是**默认 SOC 选择没指到它**。

## 四、下一会话的可行方案（按合规性排序）

> 注意：AGENTS 约束"不修改共享 SDK"，因此**不能**替换/软链 `platform_config` 里的 ini。

1. **Python 层拦截**：monkeypatch `tbe.common.platform.platform_info.get_soc_spec`
   （以及 `tbe.common.platform.get_soc_spec`）使其对 SOC 类 key 返回 `Ascend910A`，
   并在 `import te.tik` **之前**完成；同时清空 `TIK_SOC_INFO` 的所有缓存字段（不止 `soc_name`）。
2. **走 op-compile 上下文**：把 TIK kernel 放进 `tbe.dsl.build`/opc 流程，让 SOC 由 op 的
   soc_info 传入（这是 CANN 官方构建 910A 算子的方式）。
3. 若 1、2 都不行 ⇒ 结论：本机 CANN 9.2.0 的 TIK **默认目标不是 910A 且无公开开关**，
   FA 内核重写在当前软件栈上不可行，应把结论写入 Issue 并等待 CANN 升级或换机器。

## 五、状态

- 交付不变：**16.34 s/step（−8.9%）**，`H3_PERF_COMBO=1` 默认 OFF；
- TIK：API 全部就绪、最小 matmul 脚本可运行到构建、**仅卡在 SOC=310B 的选择**；
- 证据与脚本：远端 `/tmp/h3_r221/tik_mm.py`、`tik_soc.py`、`tikbuild/`。
