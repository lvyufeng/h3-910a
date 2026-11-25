# R256 — Triton-CANN 原型：通了架构与 UB 两关，但编译时间超时（尚未拿到基准数）

## 一、已解决的问题（两个）

1. **target 名不匹配**：Triton-CANN 把 `torch_npu` 报告的 `Ascend910B` 直接传给 `bishengir-compile`，
   而 bishengir 只接受 `Ascend910B1/B2/B3/B4` ⇒ 报 `Cannot find option named 'Ascend910B'`。
   **解法**：驱动内覆盖 `triton.runtime.driver.active.utils.get_arch = lambda: "Ascend910B4"`
   （本机 32 GiB ⇒ 910B4），编译命令行随后出现 `--target=Ascend910B4`，该关通过。
2. **UB 溢出**：BM=128/BN=128 时 bishengir 报
   `ub overflow, requires 2633728 bits while 1572864 bits available`（UB≈192 KB），
   关掉 multi-buffer/code-motion 后仍失败 ⇒ 该 tile 组合不可行，必须缩小 tile。

## 二、尚未解决的问题

- 改用 BM=64/BN=64、64/128、32/64、32/128 四组，**每组编译都超过 240 s 的等待窗口**，
  四组串行时被外层 700 s 预算截断，**尚未拿到任何 median_ms**；
- 之后一次单独复跑因为漏 `source set_env.sh` 导致 `torch_npu` 加载失败（环境问题，非内核问题）。

⇒ 结论：**Triton 路线的可行性仍未证明**，目前的阻塞是**编译时间**（不是运行性能）。

## 三、下一步（S2 续，R257）

1. 提高编译等待（单组 600–900 s）、设置持久 `TRITON_CACHE_DIR` 让编译只做一次；
2. 先用**缩小形状**（如 N=4096）验证 Triton 代码正确性并测吞吐，再放大到 N=41792；
3. 若 Triton 编译仍不可接受（>10 min/形状），则**转 AscendC 双缓冲实现**（编译更可控），
   这符合用户“AscendC / TBE / Triton / 整图 都可尝试”的指示；
4. 并行准备**整图**方向：用 `npugraph_ex` 捕获单个 DiT block，测能否减少 R251 里的
   `N/A` 小算子群与重复 cast（该方向不依赖 Triton 编译链）。

## 四、合规

- 仅原型与工具链试验；未改盘上源码、未装候选、无数值采用；
- 所有失败路径（target 名、UB 溢出、编译超时、环境漏 source）均如实记录，未重放不确定操作。
