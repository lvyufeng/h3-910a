# R175 — 编译参数注入：机制打通

## 本轮结论

**注入点有效且补丁存活。** 上一轮手工 bisheng 已证明内核代码可编译；本轮解决"如何把
`-isystem` 交给官方构建流程"。

## 找到的注入点

`build_out/op_kernel/H3RangeStats_ascend910/dynamic/H3RangeStats.py` 第 211 行：

```python
custom_compile_options = {},          # 原始
custom_compile_options = {"__ALLSOC__": ['-isystem', '<hcc>/include', ...]},   # 补丁后
```

`get_shortsoc_compile_option()` 会把它并入 bisheng 的 `options`。

**关键验证：打补丁后跑一次 `asc_opc`，补丁仍然存活**（`grep -c __ALLSOC__` = 5）——
说明该 py 由 cmake 阶段生成一次、之后被复用，不是每次编译重写。
这意味着"生成后打补丁 → 重跑构建"这条路可行。

## 但这一轮的 asc_opc 直接调用没有产出产物

按 `bin_param/H3RangeStats-...-0.sh` 里的形式手写调用：

```
asc_opc <kernel.cpp> --main_func=h3_range_stats --input_param=<param.json> \
  --soc_version=Ascend910A --output=<kernel_0> \
  --impl_mode=high_performance,optional --op_mode=dynamic --simplified_key_mode=0
```

只打印了 "Opc tool start working now"，`kernel_0/` 为空，无错误输出。
`asc_opc` 的 positional 参数语义与 `.sh` 包装里的 `$1/$2` 不一致（`.sh` 已经把 `--kernel-src=*` 吃掉并 shift），
所以不该手写复刻——**下一轮直接重跑 `bash build.sh` 即可**（补丁会保留）。

## 另外两个已确认的待修项

1. **soc 配置错误**：编译命令用 `--soc_version=Ascend910A`，但设备是 **910B**。
   骨架 `op_host` 里 `AddConfig("ascend910")` 是硬编码的，需改 `ascend910b`；
   否则即使编译成功，kernel 也不能在本机加载。
2. `build_out/autogen/custom_compile_options.ini` 与 `custom_opc_options.ini` 都是空文件，
   且全 CANN 源码里**没有读取方**（已 grep 确认），所以它们不是官方注入口——已排除。

## 下一轮（明确步骤）

1. `cp` 补丁脚本进构建流程：`build.sh` 跑完生成阶段 → 执行 `patch175.sh` → 重跑 `build.sh`
   （或直接重跑 `build.sh` 看补丁是否仍在）；
2. 改 `op_host` 的 `AddConfig("ascend910")` → `ascend910b`，确认 `--soc_version=Ascend910B*`；
3. 成功后 `bash build.sh --install`（或按 CANN 方式装到 `opp/vendors`）；
4. aclnn 单算子调用 + 计时，与原生位模式计数做判定等价验证。

## 合规

未改共享 CANN（补丁只在 `/tmp/h3_kern173` 构建产物里）；未下载权重；未做 driver/device reset；lease 已释放。
