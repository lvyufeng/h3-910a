# R348 — 框架 ND 路径用 DataCopy 做 ND→NZ 转换（非裸 LoadData）；我方的全 0 仍未定位

## 一、源码事实

`copy_cube_in/copy_tile_to_cube/` 里，ND 格式的 GM→L1 转换是用 **`DataCopy` + `DataCopyParams`** 完成：

```
srcStride = (alignedGRow - alignedHeight) * c0Size * sizeof(T) / ONE_BLK_SIZE
DataCopy(dst, src[srcOffset], {nburst, blockLen, srcStride, dstStride});
```

即：**ND→NZ 的排布转换在 DataCopy 的参数里做**，不是靠裸 `LoadData`。
（对 16×16 fp16 而言 ND 线性 == NZ 分形，所以这一点**不能解释**我们的全 0。）

## 二、本轮结论

- 已按源码抄了 `MmadParams`（m/k/n/unitFlag/cmatrixInitVal）与小块必需的 `PipeBarrier<PIPE_M>()`；
- 已确认 GM→L1→UB 与 UB→GM 两条链路精确正确；
- 但 `Mmad` 路径（手写同步 / TQue 自动同步 / 加 PIPE_M）**三条都全 0**；
- 框架 `Matmul` 在同样数据上能算出**精确的第 0 行**。

⇒ 剩下唯一未验证的是 **L1→L0A/L0B 的 `LoadData` 在这台 910A 上的实际语义**（参数单位/是否需要 `LoadData2dv2`），
以及 `Mmad` 在 910A 上是否要求额外的前置（例如 L0C 初始化）。

## 三、下一步（明确的验证顺序）

1. 在 kernel 里**先 DataCopy L1→UB 确认数据在 L1**（已做，正确）；
2. 用 `LoadData2dv2`（若 910A 支持）替换 `LoadData`，参数按其文档填写；
3. 若仍 0：把 `Mmad` 换成框架的 `Matmul` 但**只跑 16×16 单块并检查是否只有行 0 错**
   （已完成：行 0 对）——说明框架的 L0A 加载是好的；
4. 最后由 `Matmul` 的 `CopyCubeIn` 实现反推：它对 **A 的每次 LoadData 究竟传什么**
   （需要继续读 `copy_tile_to_cube_intf.h` 的 A 分支）。

## 四、状态与建议

- 交付不变：**16.34 s/step（−8.9%）**，`H3_PERF_COMBO=1` 默认 OFF；
- AscendC：cube 可用已确证；**手写 Mmad 路径仍卡在 L0A 加载语义**；
- 建议：这条线仍值得继续，但更适合**在新会话**里做（本会话上下文已多次压缩，读源码+试错的效率在下降）；
- 工程：`/tmp/h3_r221/faA/proj` + `test_mmad16` 判据全部保留，可直接续做。
