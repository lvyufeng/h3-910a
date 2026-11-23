# R188 — 核预算不是瓶颈：**FA 的便宜路不存在**

## 问题（R187 留下的决策点）

我上一轮提出：如果 FA 的 `core_budget=30` 是人为限制而非硬件上限，
那"改核数"可能比写新 FA 内核便宜得多。

## 答案：**30 就是硬件可用核数**

证据一 —— `fa910a.py` 的 `_runtime_core_budget()` 直接向 ACL 查询：

```python
rc = query(device_id, 101, ctypes.byref(value))   # aclrtGetDeviceInfo
if rc != 0 or not 1 <= value.value <= 64: raise ...
return int(value.value)
```

实测返回 **30**（`fa910a_core_budget: 30`）。

证据二 —— 该函数的 docstring 写明了全部来龙去脉：

> This 910A toolchain reports CORE_NUM=32, while ACL and DCMI report **30 available cores**.
> A 32-task static equal-work grid on 30 cores leaves two tasks for a second wave (558 ms);
> **repartitioning across 30 gives 317 ms**.

即：
- 工具链编译默认 `CORE_NUM=32`，但 **ACL/DCMI 报告实际可用 30 核**
- 之前的作者**已经**踩过"32 任务排到 30 核"的坑：会多出一个只有 2 个任务的第二波，558 ms
- 改成按 30 核切分后 **317 ms**（1.76× 改善）
- 所以 `instance.core_num = min(instance.core_num, core_budget)` 取的是**物理上限**，不是保守设限

## 结论

| 假设 | 判定 |
|---|---|
| 30 是人为保守限制 | ❌ **否，是硬件可用核数** |
| 提高核数能提速 | ❌ **否，已经在全部 30 核上运行** |
| 存在"改核预算"的便宜路 | ❌ **不存在** |

**41.1 TFLOPS/卡 是 FA 内核实现的效率上限，不是核饥饿造成的。**
要提升只能改内核或 tiling（即 #5，必然改舍入，需用户批准）。

## 这条负面结论的价值

它避免了走一条看起来便宜、实际走不通的路。如果没有这轮确认，
下一步很可能去尝试"调 blockDim / 加核"，写完才发现核数已经到顶。

同时确认了两件正面事实：
1. FA 现有的核切分**已经是优化过的**（32→30 重切分的 1.76× 已拿到）
2. 当前 30 核下每步 FA 7.454 s，对应 41.1 TFLOPS/卡；要达到 10 s 需要的 69 TFLOPS/卡
   **必须**来自内核/分块效率提升，没有捷径

## 尚未做（R187 原计划的后半）

- **仍未测**"这个算子在当前 30 核下的理论可达上限"——
  即用同样的 30 核、同样的 tile 参数，一个理想实现能到多少 TFLOPS/卡。
  若理想值远高于 69，则新内核值得写；若理想值也就在 45 左右，则 10 s 的 FA 部分不成立。
- 做法：用 `Br/Bc` 与 30 核的分块约束算 roofline（不需要跑机器），
  再与 41.1 对比得出"实现效率占理论的比例"。

## 合规

- 只读：查询驱动属性、阅读主机上的源码；未分配显著显存、未加载权重、未改模型数学
- 共享 CANN 未改；未做 driver/device reset；lease 已释放
