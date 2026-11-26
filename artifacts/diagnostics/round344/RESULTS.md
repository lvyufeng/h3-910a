# R344 — 手写 Mmad 全 0；但 GM→L1→UB 已证明**完全正确**（隔离成功）

## 一、隔离实验结果（关键）

把 kernel 改成"只做 GM→L1→UB→GM，输出 A 本身"，测得：

```
out row0 = 1 0 0 0 0 0 ...     ← A 是单位阵
out row1 = 0 1 0 0 0 0 ...
```

⇒ **GM→L1→UB 搬运链路逐元素正确**。这一层已确认无误。

## 二、手写 Mmad 路径的实测

M=N=K=16，A=I、B=arange，期望 C=B：

```
MMAD16 maxdiff=192  bad=256/256
row0 = 0 0 0 0 0 0
row1 = 0 0 0 0 0 0
```

⇒ 输出**全 0**，说明 `Mmad` 没有产生有效结果，或 `L0C→UB` 拷贝没取到数据。

## 三、因此剩余嫌疑只有两处（都在 cube 内部）

| 环节 | 状态 |
|---|---|
| GM→L1 | ✅ 已验证正确（本轮） |
| L1→L0A/L0B（`LoadData2dParams`） | ❓ `startIndex` 单位（512B/fractal?）、`srcStride`、`ifTranspose` |
| `Mmad`（`MmadParams{m,n,k,cmatrixInitVal}`） | ❓ 是否要求 `PipeBarrier<PIPE_M>`/特定 unitFlag |
| L0C→UB（`DataCopyCO12DstParams`） | ❓ `nSize/mSize` 单位（fractal? 元素?）、`nz2ndEn` |
| UB→GM | ✅ 已验证正确（本轮） |

## 四、下一步（最省时的判定顺序）

1. **先验 L0C→UB**：用 `Mmad` 之前先手工把 L0C 填一个已知值不可行，改为——
   在 `Mmad` 后同时用两种参数拷贝 L0C→UB（`nz2ndEn` true/false，`mSize/nSize` 试 16 与 256），
   哪个非 0 就说明拷贝参数语义；
2. 若两种都 0 ⇒ 问题在 `LoadData` 或 `Mmad`，用 `TQue` 版（EnQue/DeQue 自动插同步）重写这段，
   避免手写 `SetFlag/WaitFlag` 错配；
3. 判据仍是 A=I 时 `bad=0`。

## 五、状态

- 交付不变：**16.34 s/step（−8.9%）**，`H3_PERF_COMBO=1` 默认 OFF；
- AscendC：cube 可算（框架路径行 0 精确）；手写路径的 GM→L1→UB 已正确，剩 cube 内部三步待定；
- 工程：`/tmp/h3_r221/faA/proj`（当前为手写 Mmad 版）+ `test_mmad16`；
- 远端 master 含本轮记录。
