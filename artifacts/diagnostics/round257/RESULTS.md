# R257 — Triton-CANN：编译已通（arch+UB 都解决），但运行时与 CANN/驱动不兼容 ⇒ 转 AscendC

## 一、进展（编译侧全部打通）

在 R256 两个关口（target 名、UB 溢出）之后，用 **N=4096、BM=64、BN=64、600 s 等待、持久 TRITON_CACHE_DIR**：

- **编译成功**（不再有 `Cannot find option named 'Ascend910B'`、不再有 `ub overflow`）；
- 内核被正常 launch。

## 二、新的阻塞（运行时，非我们的代码问题）

```
aclrtAllocatorGetByStream failed. Parameter stream is invalid.
Reason: The stream is not registered with any allocator.
...
The current system or device does not support rtsGetHardwareSyncAddr.
[ERROR] ERR00100 PTA call acl api failed
```

判读：这是 **triton-ascend 运行时与当前 CANN 9.2.0 / 驱动 25.5.2 的流与 allocator 约定不一致**，
不是 kernel 语义或 tile 问题（编译产物已生成、launch 已发出）。
我们这边可尝试的覆盖（自定义 stream、`torch.npu.stream` 包裹）属于猜测性适配，
按合同不做不确定的重放式试错。

## 三、决策：转 AscendC（用户已授权 AscendC/TBE/Triton/整图 任一）

理由：
1. AscendC 走 CANN 官方工具链（`msopgen` + `ascendc` 头文件，已确认存在），与当前驱动版本天然匹配；
2. S1 已明确瓶颈（vec 54.8% / cube MAC 22.7%），AscendC 可显式控制 cube/vector 流水与 UB 分配；
3. Triton 路线的编译时间本身也偏长（小形状单次编译即达数分钟），即便修好运行时也不适合快速迭代。

## 四、下一步（R258）

1. **整图（便宜、可先做）**：`npugraph_ex` 捕获单个 DiT block，测能否削减 R251 的 `N/A` 小算子群与重复 cast；
2. **AscendC FA**：从 `msopgen` 生成算子骨架 → 实现双缓冲 QK^T/PV 与 FP32 online softmax →
   独立微基准对比 152.19 ms/块与 `mac_ratio`；
3. Triton 路线保留为「待 triton-ascend 与 CANN 版本对齐后再试」，不再投入时间。

## 五、合规

- 未改盘上源码、未装候选、无数值采用；两个失败（运行时 allocator、环境漏 source）均如实记录；
- 只在 ts-217 运行，未下载/回传权重。
