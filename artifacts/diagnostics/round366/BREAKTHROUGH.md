# R366 — 突破＋更正：正确架构是 `--cce-aicore-arch=dav-c100`；cube 完全可用

## 一、必须更正此前结论

**R356 / R359 / R365 的"本机不生成 cube 指令 / target feature 缺失"结论作废**，原因是两个方法错误：

1. **架构参数错**：应传 `--cce-aicore-arch=dav-c100`（Ascend910A = v100 架构）。
   不传时 ccec 用默认架构，导致所有 cube intrinsic 报
   `does not support the given target feature`（我误读成硬件不支持）。
2. **反汇编对象错**：`ccec -c` 产出的是 **host aarch64 ELF + `.aicore_binary` 段内嵌设备 ELF**。
   我之前直接反汇编 `.text`（host stub，里面只有 `LD./ST./ADD.`），所以数到 0 条 `MMAD`。
   正确做法：先 `llvm-objcopy --dump-section .aicore_binary=dev.bin x.o`，
   再 `llvm-objdump -d dev.bin`（设备 ELF 自动识别为 `elf64-hiipu`）。

## 二、新证据（本轮实测）

拿 TIK 生成的 `.cce`（含 `load_cbuf_to_ca` / `mmad` / `copy_matrix_cc_to_gm`）：

```bash
ccec -x cce --cce-aicore-arch=dav-c100 -c k.cce -o k_dav-c100.o   # 0 errors
llvm-objcopy --dump-section .aicore_binary=dev.bin k_dav-c100.o
llvm-objdump -d dev.bin | grep -oE "\b(MMAD|LOAD_L1_TO_L0A[A-Z0-9_]*|MOV_L0C[A-Z0-9_]*)\b" | sort | uniq -c
```

结果：

```
   1215 MMAD
   1200 LOAD_L1_TO_L0A_2D
     15 MOV_L0C32_TO_UB
```

⇒ **本机（910A / dav-c100）完全支持 cube 指令生成与 L0C 写回**。

## 三、对既有事实的重新解释

| 原结论 | 更正 |
|---|---|
| AscendC 不发射 cube | 很可能是 AscendC 构建系统传了错误 arch；需查 msopgen/CMake 里 arch 来源并覆盖为 `dav-c100` |
| TIK 无法写回 L0C | 错误——`copy_matrix_cc_to_gm` 在 dav-c100 下正常生成 |
| target feature 缺失 | 错误——只是 arch 参数缺失 |
| CANN 自带 Conv2D 有 144 MMAD 而我们的没有 | 因为它们是按正确 arch 编译的；我们的不是 |

## 四、下一步（路线重新打开）

1. **找出 AscendC/msopgen 构建里 arch 的来源**，覆盖为 `dav-c100`
   （候选：`ASCENDC_ARCH` 环境变量、CMake 变量、opc/atc 的 `--cce-aicore-arch` 透传）；
2. 或直接走「手写 `.cce` + ccec dav-c100 编译 + 打包成 op」的路径（TIK 生成的 `.cce` 可作模板）；
3. 判据（这次是正确的判据）：
   - 构建后用 `llvm-objcopy --dump-section .aicore_binary` 提取设备 ELF；
   - `llvm-objdump -d` 里 **`MMAD` 计数 > 0**；
   - 再跑 A=I ⇒ C=B 的数值判据。

## 五、状态

- 交付不变：**16.34 s/step（−8.9%）**，`H3_PERF_COMBO=1` 默认 OFF；
- **自定义 cube 内核路线重新打开**（此前"不可行"结论作废）；
- 证据文件：远端 `/tmp/ccetest/k_dav-c100.o`、`dev.dis`（107,701 行）。
