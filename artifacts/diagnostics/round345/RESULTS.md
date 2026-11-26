# R345 — `nz2ndEn` 两种取值都全 0 ⇒ 嫌疑移到 `LoadData` 或 `Mmad` 本身

## 一、本轮排除

| 试验 | 结果 |
|---|---|
| `DataCopyCO12DstParams.nz2ndEn = true` | 全 0（R344） |
| `DataCopyCO12DstParams.nz2ndEn = false` | **仍全 0**（本轮） |
| 顺带修正 | R344 的诊断块（带 `return`）此前残留，导致一次实验其实跑的还是"输出 A"；本轮已移除，确认现在跑的是真 Mmad 路径 |

```
MMAD16 maxdiff=192 bad=256/256 row0=0 0 0 0 0 0  row1=0 0 0 0 0 0
```

## 二、结论

L0C→UB 的两种拷贝语义都取不到数据，因此**要么 Mmad 没写 L0C，要么 LoadData 没把数据装进 L0A/L0B**。
UB→GM 与 GM→L1 已分别验证正确（R344），所以问题被夹在 `LoadData`/`Mmad` 这两步之内。

## 三、下一步（三条，按成本排序）

1. **改用 `TQue` 管理 L0A/L0B/L0C**（官方推荐写法）：
   ```cpp
   TQue<TPosition::A2, 1> qa; TQue<TPosition::B2, 1> qb; TQue<TPosition::CO1, 1> qc;
   // AllocTensor -> LoadData -> EnQue -> DeQue -> Mmad -> qc.EnQue/DeQue -> DataCopy
   ```
   TQue 会自动插入正确的事件同步，直接排除我们手写 `SetFlag/WaitFlag` 错配的可能；
2. **单独验证 LoadData**：LoadData 后立刻用 `LoadData` 反向（`LoadData3dv2`?）或在 `Mmad` 前
   把 L0A 内容通过 `DataCopy` 拷回 UB（若 L0A→UB 不可用，改用 `TQue` 路径后自然跳过此步）；
3. 若 TQue 版仍 0 ⇒ 检查 `MmadParams.unitFlag`（试 1）与 `LoadData2dParams.ifTranspose`（B 试 true）。

## 四、状态

- 交付不变：**16.34 s/step（−8.9%）**，`H3_PERF_COMBO=1` 默认 OFF；
- AscendC：GM→L1→UB 已验证；剩 `LoadData`/`Mmad` 两步；
- 工程：`/tmp/h3_r221/faA/proj`（当前手写 Mmad 版、OpDef 已回 ND）+ `test_mmad16`；
- 远端 master 含本轮记录。
