# R347 — 读到框架源码：`Mmad` 需 `PipeBarrier<PIPE_M>`（已加，仍 0）；LoadData 走 `dataCopyInfo` 包装层

## 一、源码事实（新增，可复用）

### `MmadCompute`（`asc/impl/adv_api/detail/matmul/stage/compute/mmad_compute.h`）

```cpp
MmadParams mmadParams;
mmadParams.m = mmadM; mmadParams.k = mmadK; mmadParams.n = mmadN;
if (IsUnitFlagEnabled()) mmadParams.unitFlag = unitFlag;
if (!IsSupportCmatrixInitVal()) {                 // 910A 分支
    mmadParams.cmatrixSource = cmatrixSource;
    mmadParams.cmatrixInitVal = cmatrixInitVal;
} else {
    mmadParams.isBias = isBias;
}
Mmad(cMatrix, l0A, l0B, mmadParams);
if ((mmadM/16) * (mmadN/16) < 10) PipeBarrier<PIPE_M>();   // ← 小块的强制要求
```

按此在 `Mmad` 后加了 `PipeBarrier<PIPE_M>()` —— **结果仍全 0**（MMAD16 bad=256/256）。

### LoadData 侧（新线索）
框架不走裸 `LoadData`，而是 `copy_cube_in/copy_tile_to_cube/` 下的一组包装：
- `data_copy_wrapper_nd.h`、`data_copy_wrapper_vec.h`、`data_copy_wrapper_using_ub_nd.h`
- 参数结构是 `dataCopyInfo`（含 `srcStride` 等），再转成底层的 K 参数。

## 二、当前完整判定表

| 步骤 | 状态 |
|---|---|
| GM→L1 | ✅ 精确 |
| L1→L0A/L0B（裸 LoadData） | ❓ 仍可能是全 0 的根源 |
| Mmad + PipeBarrier | 已按源码加，仍 0 |
| L0C→UB（两种 nz2ndEn） | 全 0 |
| UB→GM | ✅ 精确 |
| 框架 Matmul | 行 0 精确（cube 本身好） |

## 三、下一步（精确，不再盲试）

1. 打开 `data_copy_wrapper_nd.h`，找到 16×16/ND 情况下最终调用的底层指令
   （`LoadData2D`/`LoadData2dTranspose`/`LoadData3dv2`）与**全部参数**；
2. 照抄到我们的 kernel（这一步做完就能判定 L0A 是否真被装填）；
3. 若 L0A 已正确而 Mmad 仍 0 ⇒ 再抄 `cmatrixSource/cmatrixInitVal/isBias` 的 910A 取值组合
   （4 种，小规模可穷举）；
4. 判据：A=I 时 `bad=0`。

## 四、状态

- 交付不变：**16.34 s/step（−8.9%）**，`H3_PERF_COMBO=1` 默认 OFF；
- AscendC：已从"猜 API"转为**读框架源码照抄**；Mmad 参数与 PIPE_M 要求已抄到；
- 工程：`/tmp/h3_r221/faA/proj`（TQue + PIPE_M 版）+ `test_mmad16`；
- 远端 master 含本轮记录。
