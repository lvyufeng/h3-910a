# R349 — `LoadData` 与 `DataCopy` 两种 L1→L0A 写法都全 0；手写 Mmad 路线需要新环境继续

## 一、本轮实验

用源码读到的 `DataCopyParams{blockCount, blockLen=16, srcStride=0, dstStride=0}`（blocks:512B=16×32B）
替换 `LoadData`，重建后：

```
MMAD16 maxdiff=192 bad=256/256  row0=0 0 0 0...
```

⇒ 与 `LoadData` 版**同样全 0**。

## 二、累计判定表

| 环节 | LoadData 版 | DataCopy 版 |
|---|---|---|
| GM→L1→UB | ✅ 精确 | ✅ 精确 |
| L1→L0A/L0B | 全 0 结果 | 全 0 结果 |
| Mmad（+PIPE_M，按源码） | 全 0 | 全 0 |
| L0C→UB（nz2ndEn 两种） | 全 0 | 全 0 |
| 框架 Matmul（同数据） | **行 0 精确** | **行 0 精确** |

⇒ **cube 硬件与 Mmad 指令是好的**（框架能算出行 0），我们手写路径仍有未知前置条件。

## 三、未验证的剩余变量（按可能性）

1. **L0C 的分配方式**：框架用自己的 L0C 管理（`TBuf<CO1>` + 显式 512B 对齐/双缓冲），我们用的 `TQue<CO1>`
   可能拿到的不是 fixpipe 期望的地址；
2. **`DataCopyCO12DstParams` 的 `nSize/mSize` 单位**（元素 vs 32B 块）；
3. **`Mmad` 在 910A 是否有额外要求**（如必须由 `MatmulImpl` 在特定 tiling 下调用）。

## 四、建议（诚实结论）

- 手写 Mmad 这条线**值得做，但已不适合在当前（多次压缩的）会话里继续**；
- 建议**新会话**继续，并从下面这条最短路径切入：
  1. 用 `TBuf<TPosition::CO1>` 手工分配 L0C（512B 对齐、双缓冲），而不是 `TQue<CO1>`；
  2. `DataCopyCO12DstParams` 的 `nSize/mSize` 试「元素」与「32B 块」两种单位；
  3. 若仍 0，则直接**把框架 `Matmul` 的 `CopyCubeIn`/`MmadCompute` 用 `-E` 展开后的 PTX 级中间代码**（`atc --op_debug_level`）当作唯一依据，逐行复刻。

## 五、状态

- 交付不变：**16.34 s/step（−8.9%）**，`H3_PERF_COMBO=1` 默认 OFF；
- AscendC：**可用已确证**（撤回 R313 结论，见 R336）；手写 path 未通；
- 工程与三个判据 harness 全部保留在 `/tmp/h3_r221/`；
- 远端 master 含 R332→R349 全部记录。
