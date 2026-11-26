# R351 — `KERNEL_TASK_TYPE_DEFAULT` 在此 CANN 版本存在但无法直接用；task type 由构建脚本的 CodeMode 决定

## 一、本轮事实

1. 宏确实存在：
   `asc/impl/basic_api/utils/kernel_utils_macros.h:349-352`
   ```cpp
   #define KERNEL_TASK_TYPE(key, value) ENABLE_FEATURE_FOR_COMPILE(key, value)
   #define KERNEL_TASK_TYPE_DEFAULT(value) ENABLE_FEATURE_FOR_COMPILE(default, value)
   ```
   `ENABLE_FEATURE_FOR_COMPILE` 只在 `__CHECK_FEATURE_AT_PRECOMPILE` 展开，否则空。
2. 我按样例写法加 `KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIC_ONLY);` → **编译失败**
   （`KERNEL_TYPE_AIC_ONLY` 在头文件搜索范围内没有定义，只有构建脚本里的 `CodeMode` 枚举用同名常量）。
3. 关键发现：**构建脚本 `ascendc_kernel_cmake/legacy_modules/util/extract_host_stub.py`
   用 `CodeMode` 枚举决定 kernel 是 AIC/AIV/MIX**（`CodeMode.AIC` / `CodeMode.KERNEL_TYPE_AIV_ONLY` / `CodeMode.KERNEL_TYPE_MIX_AIC_1_2` …）。
   ⇒ 我们的 kernel 被编成什么模式，是这套脚本判定的，不是源码里一行宏能改。

## 二、与现象的契合

如果该脚本把我们的 kernel 判定为 **AIV**（例如它按某些启发式扫描 cube 指令但与我们的写法不匹配），
那么 `Mmad` 全部无效、`DataCopy` 正常、L0C 恒为 0——**与观测完全一致**，
也能解释为什么换 LoadData/DataCopy/CO1 参数都毫无变化。

## 三、下一步（精确）

1. 读 `extract_host_stub.py` 里 **CodeMode 的判定函数**（搜索 `CodeMode.AIV` 的赋值处 /
   `def .*code_mode` / `get_code_mode`），确认它依据什么（源码扫描？op json 字段？CMake 变量？）；
2. 按该依据在我们的 kernel/op json 里**显式声明** AIC 或 MIX_AIC_1_2；
3. 重建 → 跑 `test_mmad16`，判据 A=I 时 `bad=0`；
4. 同时可对照：把生成的 `kernel_meta/*.json` 里 `code_mode`/`kernel_task_type` 字段打印出来确认。

## 四、状态

- 交付不变：**16.34 s/step（−8.9%）**，`H3_PERF_COMBO=1` 默认 OFF；
- AscendC：cube 可用已确证；**当前最强假设 = 我们 kernel 的编译模式不含 AIC**，且有源码依据（CodeMode 判定在构建脚本里）；
- 工程：`/tmp/h3_r221/faA/proj` + `test_mmad16`；注意 kernel 源里那行宏已加但编译失败，需先移除或改为正确常量。
