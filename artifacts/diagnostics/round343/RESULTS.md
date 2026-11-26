# R343 — NZ 原型重建成功（aclnn 层已接受）；tiling 暴露并修正了 SetSingleShape 错配；matmul 仍只有行 0 正确

## 一、本轮完成的两件事（都是实打实的进展）

### 1) 算子原型可改，NZ 路线已打通到 aclnn
在 `op_host/flash_attention_custom.cpp` 的 OpDef 里把 q/k/v 的 `.Format({ge::FORMAT_ND})`
与 `.UnknownShapeFormat(...)` 改为 `FORMAT_FRACTAL_NZ`（共 6 处），重建后：

- 之前 `rc=161002 ... inconsistent with OpDef` **消失**；
- aclnn 正常接受 NZ 输入、ND 输出。

⇒ **原型不是障碍**（我上一轮说"必须重建工程"过于悲观，实际改 OpDef 即可）。

### 2) `TILINGDUMP` 终于打出来了，并暴露一个真实 bug

修 `SetSingleShape` 前：

```
[TILINGDUMP] M=16 N=32 Ka=32 ... scM=32 baseM=32 baseN=32    ← M=16 却 baseM=32
ACL_FAIL 507015                                              ← AICore 异常
```

根因：R339 遗留的 `t.SetSingleShape(32,32,32)` 覆盖了 `SetShape(16,32,32)`。
修正为 `SetSingleShape(16,32,32)` 后：

```
[TILINGDUMP] usedCore=1 M=16 N=32 Ka=32 Kb=32 scM=16 scN=32 scK=32 baseM=16 baseN=32 baseK=32 stepM=1 stepN=1
```

⇒ tiling 内部现在**完全自洽**（M=scM=baseM=16）。

## 二、但结果仍然一样

```
NZ16 maxdiff=96 bad=480/512
row0 = 1 2 3 4 5 6   ← 正确
row1 = 1 2 3 4 5 6   ← 应为 33 34 35 36 37 38
```

**在 ND/NZ 两种格式、tiling 自洽、单 M 块（M=16）的情况下，matmul 仍然只正确算出第 0 行。**

## 三、这意味着什么

已排除：kernel 缓存、C 读取链路、IterateAll/形状参数、块迭代、OpDef 格式、tiling 自洽性。
仍未排除的只剩 **A 的 L1→L0A 加载**（框架内部 `LoadData` 的 `srcStride`/分形索引）
或 **fxpipe 的 C 写出 M 步长**。

两者都必须脱离框架 `Matmul`、改手写 `LoadData` + `Mmad` 才能判定 —— 这就是路线 B。

## 四、下一步（唯一剩余路径：手写 Mmad）

1. `DataCopy` GM→L1（ND 线性，16×32 = 1 个 16×32 区域）；
2. `LoadData2dParams{startIndex=fractal_idx, repeatTimes=1, srcStride=1, dstGap=0}` 逐分形 → L0A/L0B；
3. `MmadParams{m=16,n=16,k=16, cmatrixInitVal=true}`，K 方向 2 个分形累加；
4. L0C→UB 用 `DataCopyCO12DstParams{nSize=16,mSize=16}`，再按块拼回 ND；
5. 判据：A=I 时 `bad=0`。

## 五、状态

- 交付不变：**16.34 s/step（−8.9%）**，`H3_PERF_COMBO=1` 默认 OFF；
- AscendC：cube 可算（行 0 精确）、OpDef 可改（NZ 已通、tiling 自洽）；缺陷只剩加载/写出二选一；
- 工程：`/tmp/h3_r221/faA/proj`（已含 NZ OpDef 与修正后的 tiling）+ `test_nz.cpp`；
- 远端 master 含本轮全部记录。
