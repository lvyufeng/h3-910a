# R356 — **决定性**：kernel 二进制里根本没有 cube 指令（`Mmad` 被编译器丢弃）

## 一、静态检查结果

对生成的算子目标文件（`elf64-hiipu`）做反汇编：

```
obj = .../op_kernel/ascendc_kernels/binary/ascend910/flash_attention_custom/
      FlashAttentionCustom_<hash>.o
反汇编：llvm-objdump -d → 686 行，section .text 只有 1 个函数
```

关键字统计（全文件、大小写不敏感）：

| 关键字 | 命中数 |
|---|---|
| `MAD` / `MMAD` | **0** |
| `CBUF` / `L0A` / `L0B` / `L0C` | **0** |
| `LOAD` / `DATACOPY` | **0** |

反汇编内容是 `MOV/MOVK/SBITSET/MOVEMASK` 等标量启动代码。

## 二、结论（有静态证据）

**编译器没有把 `Mmad`（以及 cube 相关搬运）发射进二进制**——整条 cube 路径被丢弃。
这**同时解释**了本季所有现象：

- 输出全 0（cube 从未执行）；
- 无论 `LoadData`/`DataCopy`/`nz2ndEn`/tiling/OpDef 怎么改，结果**逐位相同**；
- `GM→L1→UB` 正常（MTE/标量路径保留）；
- 框架 `Matmul` 能算（它走 TBE 编译路径，不是 AscendC 源码路径）。

⇒ 与本季前序结论一致并升级为**定论**：
**AscendC C++ 高级 API 的 cube 原语在本机（910A/v200）不会生成任何指令**，
不是"参数没调对"，而是**代码被丢弃**。

## 三、对应的可用路径（不变）

- **TIK（TBE DSL）**：CANN 自带 910A 算子即此实现；主机上有完整源码
  `python/site-packages/te/tik/api/cube/matmul.py`；
- 或直接调编译器 builtin（`load_cbuf_to_ca` 等），但需确认它们在此 v200 目标下是否同样被丢弃——
  **下一步应先做同样的静态检查**（写一个只调 builtin 的 kernel，反汇编确认指令是否出现）。

## 四、状态

- 交付不变：**16.34 s/step（−8.9%）**，`H3_PERF_COMBO=1` 默认 OFF；
- DiT ≤10/15 s 未达成；本季结论：**AscendC 路线不成立（代码被丢弃，有静态证据）**，
  应转 TIK；
- 反汇编文件留在远端 `/tmp/h3_r221/dis.txt`。
