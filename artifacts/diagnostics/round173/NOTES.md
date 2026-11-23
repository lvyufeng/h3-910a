# R173 — 自定义算子链路：工程生成成功，编译被工具链路径挡住

## 本轮进展

| 步骤 | 结果 |
|---|---|
| msopgen 生成工程 | ✅ 成功（`-c ai_core-ascend910b -lan cpp -f aclnn`） |
| op_kernel 实现 | ✅ 已写（单趟幅度位归约） |
| 编译 | ❌ 卡在 CANN 自带头文件的 include 路径 |

生成物：`op_host/h3_range_stats.cpp`、`op_kernel/h3_range_stats.cpp`、
`op_kernel/h3_range_stats_tiling.h`、`build.sh`、`CMakeLists.txt`、`CMakePresets.json`。

## 内核设计（比原计划更小、更可证）

不做精确计数，只做**幅度位最大值**归约。依据：

- FP16 幅度位 `bits & 0x7fff` 取值 0x0000–0x7fff，**随 |x| 单调**；
- 因此 `maxMag >= 0x7c00` ⟺ 存在 NaN/Inf；`maxMag == 0x7bff` ⟺ 存在 |x|==65504（最大有限幅值）；
- R164 已确定**审计只消费"是否为零"这个判定**（strict 报错、deferred 求 any），所以判定等价足够；
- 一趟 `Max` 归约即完成，无需 compare + sum，实现风险最低；
- 全程只用**位模式**，不使用浮点判据（R165 已证本设备它们漏 NaN）。

每核处理自己那段连续切片，写 `y[coreIdx]`（int32 存储最大幅度位），主机侧取 max 得两个判定。
这样避免跨核原子累加，也不需要 workspace。

## 编译失败的精确原因

```
/home/mseco/Ascend/cann-9.2.0/aarch64-linux/asc/impl/utils/sys_macros_impl.h:18:10:
fatal error: 'cstdint' file not found
```

注意：这是 **CANN 自己的头**（`kernel_operator.h` → `sys_macros_impl.h`）include 的 `<cstdint>`，
**不是本次内核代码的问题**。我先把 tiling 头里的 `#include <cstdint>` 去掉（改用 `unsigned int`/`short`），
错误依旧——证明根因在工具链，不在代码。

工具链里其实**有**这套头：

```
/home/mseco/Ascend/cann-9.2.0/tools/hcc/aarch64-target-linux-gnu/include/c++/14.3.0/cstdint
```

但构建体系期望的另一条路径：

```
$ENV{ASCEND_HOME_PATH}/toolkit/toolchain/hcc/aarch64-target-linux-gnu/include/c++/14.3.0
```

**`toolkit/toolchain/hcc`（期望）与实际 `tools/hcc`（安装位置）结构不一致**，所以 include 找不到。

（见 `compiler/tikcpp/ascendc_kernel_cmake/aicpu_modules/CMakeAICPUInformation.cmake:48-53`，
以及 `fwk_modules/device_task.cmake:16-17` 对 `${ASCEND_CANN_PACKAGE_PATH}/toolkit/toolchain/hcc/bin/...` 的引用。）

## 下一轮的修复方向（按"是否触碰共享 SDK"排序）

1. **私有 overlay**（首选）：建私有目录，内含指向真实 `tools/hcc` 的 `toolkit/toolchain` 符号链接，
   把 `ASCEND_HOME_PATH` 指过去——**不改动共享 CANN 安装**；
2. 查 TBE 编译路径（`op_compilation.py`）用哪个环境变量拼 include，
   若有官方变量（如 `ASCEND_CXX_INCLUDE` 之类）直接设；
3. 在 `op_kernel/CMakeLists.txt` 里追加 `-isystem <真实 hcc c++ 路径>`（改动限于本工程，安全）；
4. 最后才考虑在 CANN 安装内建符号链接——**这属于修改共享 SDK，需要用户批准**。

## 合规状态

- 未下载权重、未中转数据；实验目录 `/tmp/h3_kern173`（mode 700）
- 未改动共享 CANN 安装、未做 driver/device reset、无他人进程
- lease 已释放
