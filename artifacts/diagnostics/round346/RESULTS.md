# R346 — TQue 版仍全 0：同步不是原因；下一步改为**读框架源码照抄调用序列**

## 一、本轮实验

用官方推荐写法重写（`TQue<A1/B1/A2/B2/CO1/VECCALC>` + `AllocTensor/EnQue/DeQue`，
同步由队列自动插入），结果：

```
MMAD16 maxdiff=192 bad=256/256
row0 = 0 0 0 0 0 0
row1 = 0 0 0 0 0 0
```

⇒ 与手写 `SetFlag/WaitFlag` 版**完全相同（全 0）**，因此**同步错配的假设被排除**。

## 二、现在的对照关系（重要）

| 路径 | 结果 |
|---|---|
| 框架 `Matmul`（ND/NZ，M=16/32） | **第 0 行精确正确**，其余行 = 第 0 行 |
| 手写 `Mmad`（手写同步） | 全 0 |
| 手写 `Mmad`（TQue 自动同步） | 全 0 |
| GM→L1→UB | 精确正确（R344） |
| L0C→UB（nz2ndEn 两种） | 全 0 |

框架能算出"第一行"，说明 **cube 硬件和 Mmad 本身是好的**；
我们的手写调用缺少了框架里的某些**必要前置步骤**（例如 L0C 初始化/atomic 设置/
`Mmad` 前对 L0A/L0B 的 layout 声明）。

## 三、下一步（不再猜，直接读实现）

CANN 里框架 `Matmul` 的实现就在安装目录的 AscendC 头文件/source 中：

1. 定位 `MatmulImpl` 的 `Iterate()`/`IterateAll()` 实现（`lib/matmul/matmul_impl*.h`）；
2. 抄出它在调用 `Mmad` 前后**逐行做了什么**：
   - L0C 是否需要先 `InitL0C`/`SetAtomic`；
   - `MmadParams` 的 `m/n/k/unitFlag/cmatrixInitVal` 具体取值；
   - A/B 的 `LoadData` 参数（`srcStride`/`ifTranspose`/`startIndex` 单位）；
3. 按抄出的序列 1:1 复现，判据仍是 A=I 时 `bad=0`。

## 四、状态

- 交付不变：**16.34 s/step（−8.9%）**，`H3_PERF_COMBO=1` 默认 OFF；
- AscendC：cube 好（框架行 0 精确）；手写路径缺前置步骤，下一步以**源码为准**照抄；
- 工程：`/tmp/h3_r221/faA/proj`（TQue 版）+ `test_mmad16` 判据；
- 远端 master 含本轮记录。
