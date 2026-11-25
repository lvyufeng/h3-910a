# R292 / S0 — AscendC 工具链冒烟：**通过**（附关键 include 路径配方）

## 一、结论

`msopgen` 生成的 AddCustom 算子项目，在 ts-217（Ascend910B4）上**完整编译并打包成功**：

```
Compile op info cfg successfully.
Self-extractable archive "custom_opp_ubuntu_aarch64.run" successfully created.
```

⇒ AscendC 自定义算子链路在本机可用，S1（FA 等价实现）具备前提。

## 二、踩过的三个坑（同一根因：C++ 标准头路径）

| 尝试 | 报错 | 原因 |
|---|---|---|
| 默认环境 | `fatal error: 'cstdint' file not found` | bisheng 编译器没有系统 C++ 头路径 |
| 加 CANN 自带 14.3.0 头 + 系统 11 头 | `explicit template argument list not allowed`、`_S_nothrow_construct not declared` | 14.3 与 11 两套 libstdc++ 头混用，宿主侧 opbuild 失败 |
| 只用系统 11 头（少 multiarch） | `fatal error: 'bits/c++config.h' file not found` | 缺 `/usr/include/aarch64-linux-gnu/c++/11` |
| **正解** | ✅ 构建成功 | 见下 |

**可用配方**（每次构建 AscendC 算子前 export）：

```bash
source /home/mseco/Ascend/cann-9.2.0/set_env.sh
export CPLUS_INCLUDE_PATH=/usr/include/c++/11:/usr/include/aarch64-linux-gnu/c++/11:/usr/include/aarch64-linux-gnu:/usr/include
export CPATH=$CPLUS_INCLUDE_PATH
```

要点：**只用系统 g++ 11 这一套头**（含 multiarch 目录），不要掺 CANN 的 14.3.0 头。

## 三、骨架结构（供 S1 使用）

```
/tmp/h3_r221/asc0/proj/
├── build.sh                 # 构建入口
├── op_host/add_custom.cpp   # 算子原型/推断/tiling
├── op_kernel/add_custom.cpp # AscendC kernel 实现（S1 在这里写 FA）
├── op_kernel/add_custom_tiling.h
└── framework/               # 各框架插件（暂时不用）
```

构建产物：`build_out/custom_opp_ubuntu_aarch64.run`（可安装的自定义算子包）。

## 四、下一步（S1）

1. 用 `msopgen gen` 生成 **FA 算子原型**（6 输入 q/k/v/…、2 输出 o/l，attr: scale/valid_keys 等）；
2. 在 `op_kernel/` 写**等价实现（不做流水优化）**：QK^T → 行 max → exp → 行 sum → PV；
3. 与 TIK 参考对比：`relRMS ≤ 1e-04`、记录 ms（对比 126.19 ms）；
4. 通过后进入 S2（双缓冲/流水，目标 ≤105 ms）。
