# R176 — soc 配置修复成功；但 `custom_compile_options` 不是 cstdint 失败的那个变量

## 做成的

### 1. soc 配置真正修好了（这轮确定的两处）

| 位置 | 原值 | 新值 |
|---|---|---|
| `CMakePresets.json` → `ASCEND_COMPUTE_UNIT` | `ascend910` | **`ascend910b`** |
| `op_host/h3_range_stats.cpp` → `AddConfig(...)` | `"ascend910"` | **`"ascend910b"`** |

验证：构建目标目录变为 `H3RangeStats_ascend910b`，编译日志前缀为 `[ascend910b]`。
R175 只改 op_host 是不够的（会卡在 `gen_ops_config`），**工程级 soc 在 CMakePresets 里**。

### 2. 补丁打对了

`dynamic/H3RangeStats.py:211` 现在是：

```python
custom_compile_options = {"__ALLSOC__": ['-isystem', '<hcc>/include', '-isystem', '<hcc>/include/c++/14.3.0', ...]},
```

（R175 的幂等检查写错了——文件别处含 `__ALLSOC__` 字样导致误判"已打补丁"；本轮改成精确匹配
`custom_compile_options = {},` 并验证只有一处替换。）

## 没做成的（重要负面结论）

**补丁到位后重跑构建，仍然 10 处 `cstdint` fatal error。**

⇒ `dynamic/H3RangeStats.py` 里的 `custom_compile_options` **不是**驱动失败那次 bisheng 调用的变量。
R175 观察到"补丁在 asc_opc 后存活"是真的，但它喂的是另一个（后续）阶段。

失败发生在 `fwk_modules/util/ascendc_compile_kernel.py:196` 的 `kernel_builder.ascendc_build()`，
而 `ascendc_build()` 的编译参数由 `ascendc_compile_base.py` / `ascendc_compile_v200.py` 组装
（后者我们已看过：拼 `--cce-aicore-arch=... --cce-aicore-only ...` 与 `compile_option_tuple.compile_options`）。

## 下一轮该试的（按风险从低到高）

1. **私有 PYTHONPATH overlay**（我认为这是最短路径）：
   把 `compiler/tikcpp/ascendc_kernel_cmake/` 复制到 `/tmp/.../overlay/`，在组装 compile_options 的地方
   追加三个 `-isystem`，然后 `PYTHONPATH=$OVERLAY:$PYTHONPATH bash build.sh`。
   **不修改共享 CANN 安装**，且属于"换一份脚本"，符合私有实例原则。
2. 找 `compile_option_tuple.compile_options` 的上游来源（可能来自 `bin_param/*_param.json`，
   可 dump 出来看），若可控则直接加 flag。
3. 兜底：绕开 msopgen 构建，用已验证可用的手工 bisheng 命令编出 `.o`，
   再自行构造 op 包的 kernel 目录（工作量大，且 kernel meta json 需 asc_opc 生成）。

## 已排除

- `build_out/autogen/custom_compile_options.ini` / `custom_opc_options.ini`：空且全 CANN 无读取方（R175 grep 过）
- `dynamic/*.py` 的 `custom_compile_options`：本轮实测不是失败路径的参数来源

## 合规

未修改共享 CANN 安装（只改 `/tmp/h3_kern173` 下构建产物与工程副本）；
未下载权重；未做 driver/device reset；lease 已释放。
