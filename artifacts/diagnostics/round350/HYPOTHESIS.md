# R350 — 新假设（强）：kernel task type 可能默认 AIV-only，导致 `Mmad`（cube）被静默忽略

## 一、为什么这个假设能解释全部现象

| 观察 | AIV-only 解释 |
|---|---|
| `Mmad` 输出全 0 | cube 指令在 AIV-only 任务里不执行 → L0C 从未被写 |
| GM→L1→UB 精确正确 | 都是 MTE/vector 路径，AIV 上正常 |
| 哨兵实验正常 | 纯 DataCopy，AIV 正常 |
| L0C→UB 全 0 | 因为 L0C 里本来就是 0 |
| `LoadData`/`DataCopy` 两种写法都 0 | 传输本身可能成功，但 Mmad 没算 |
| 框架 `Matmul` 行 0 精确 | 框架 kernel 有自己的 task type（cube/mix），所以能算 |

也就是说：**我们试的所有参数都没问题，问题可能是整个 kernel 根本没被编译成含 cube 的任务。**
CANN 的 AscendC 用 `KERNEL_TASK_TYPE_DEFAULT(...)` 宏声明任务类型（AIC_ONLY / AIV_ONLY / MIX_AIC_1_2 等），
默认值在部分版本是 **AIV_ONLY**——与"DataCopy 正常、Mmad 无效"完全吻合。

## 二、下一步（一个宏 + 一次重建即可判定）

1. 在 kernel 源文件顶部加：
   ```cpp
   KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIC_ONLY);      // 先试纯 cube
   ```
   若编译报错，改 `KERNEL_TYPE_MIX_AIC_1_2`（cube:vector = 1:2）；
2. 重建 → 安装 → 跑 `test_mmad16`；
3. 判据：A=I 时 `bad=0`；
4. 若 `KERNEL_TASK_TYPE_DEFAULT` 在此 CANN 版本不可用，则查生成物里的
   `kernel_meta/*.json` 中 `"kernel_task_type"` 字段确认当前值，再在 msopgen 的 op json 里显式指定。

## 三、注意

- 这个假设**不需要改任何参数**，只加一行宏声明，成本极低；
- 若成立，则此前 R344–R349 的所有"参数排查"其实都是在错误的 task type 下做的，
  那些结论（LoadData/DataCopy/CO1 参数）需要在新 task type 下重新确认一遍；
- 仍不改变"cube 可用"这一结论（框架 Matmul 已证明）。

## 四、状态

- 交付不变：**16.34 s/step（−8.9%）**，`H3_PERF_COMBO=1` 默认 OFF；
- 工程：`/tmp/h3_r221/faA/proj`（当前 TQue + PIPE_M + DataCopy 版）+ `test_mmad16`；
- 远端 master 含 R332→R350 全部记录。
