# R174 — 内核**编译通过**；缺的只是几个 `-isystem`

## 结论先行

内核代码本身**是正确的**。用手工命令行（把 CANN 自带的 hcc C++ 头显式传入）编译：

```
bisheng -fsyntax-only -x cce --cce-aicore-only --cce-aicore-arch=dav-c220 \
  -isystem $R/toolkit/toolchain/hcc/aarch64-target-linux-gnu/include \
  -isystem $R/toolkit/toolchain/hcc/aarch64-target-linux-gnu/include/c++/14.3.0 \
  -isystem $R/toolkit/toolchain/hcc/aarch64-target-linux-gnu/include/c++/14.3.0/aarch64-target-linux-gnu \
  -I $R/aarch64-linux/asc/include -I $R/aarch64-linux/tikcpp/tikcfw -I $R/aarch64-linux/tikcpp/tikcfw/interface \
  h3_range_stats.cpp
→ EXIT=0（仅 CANN 自己头文件里的 C++17 扩展告警）
```

## R173 的诊断需要更正

R173 我判断是"`toolkit/toolchain/hcc` 与 `tools/hcc` 结构不一致导致路径解析失败"。**这是错的**：

```
cann-9.2.0/toolkit/toolchain/hcc -> ../../tools/hcc      （符号链接，存在）
toolkit/toolchain/hcc/.../include/c++/14.3.0/cstdint      （存在）
```

路径其实是通的。真正的状况是：**`asc_opc` 编译设备代码时根本没有传这些 `-isystem`**。
所以 `<cstdint>` 找不到与目录结构无关，是编译命令缺少 include 参数。

## 已确认的事实

| 项 | 状态 |
|---|---|
| `msopgen` 生成工程 | ✅ |
| `op_host`（tiling）编译 | ✅ 产出 `h3_range_stats.cpp.o`、`cust_optiling` 构建成功 |
| 设备内核代码正确性 | ✅ 手工 bisheng 语法检查通过（exit 0） |
| 设备内核经 `asc_opc` 编译 | ❌ 缺 `-isystem` |
| 根因 | `asc_opc` 的 `compile_options` 里没有 hcc C++ 头路径 |

## 注入点（已定位）

`build_out/op_kernel/H3RangeStats_ascend910/dynamic/H3RangeStats.py`：

```python
custom_compile_options = {},        # ← 可填 {"__ALLSOC__": ["-isystem", "<path>", ...]}
custom_all_compile_options = {},
```

`get_shortsoc_compile_option()` 会把它并入 `options`。但该文件由构建过程**重新生成**，
所以要么在生成后、调用 `asc_opc` 前打补丁，要么绕过 msopgen 的构建直接用自己的方式产包。

另一个已观察到的配置问题：生成的编译命令用 `--soc_version=Ascend910A`，
而实际设备是 **910B**（`Ascend910B*` 平台配置存在）。骨架里 `AddConfig("ascend910")` 是硬编码的，
需要改成 `ascend910b`，否则编出来的 kernel 不能在本机跑。

## 下一轮计划

1. 把 op_host 的 `AddConfig("ascend910")` 改为 `ascend910b`，确认编译命令变为 `--soc_version=Ascend910B*`；
2. 在 `asc_opc` 调用前给生成的 `dynamic/H3RangeStats.py` 打补丁注入 `-isystem`（只影响本工程、不碰共享 SDK）；
3. 编译成功 → 安装到 `opp/vendors` → 用 aclnn 单算子测试 + 计时；
4. 与原生位模式计数做判定等价验证（R166 的 37 个对抗用例 + 真实张量）。

## 产物

`src/` 下保存了可编译的内核与 host 源码，避免 `/tmp` 丢失。

## 合规

未下载权重、未修改共享 CANN 安装、未做 driver/device reset；`/tmp/h3_kern173` mode 700；lease 已释放。
