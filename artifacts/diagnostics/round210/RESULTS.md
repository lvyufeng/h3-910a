# R210 — 实测：**HCCL all-reduce 逐位可重复 ⇒ R209 的假设被否掉**

## 一、这轮做了什么

R209 提出假设：**HCCL 集合通信归约顺序不确定**，并设计了决定性测试。
本轮把测试真跑了——**而且不需要模型**：直接在 8 rank 上用随机数据做重复 all-reduce，逐位比较。

### 先确认变量存在

```
strings libhccl.so | grep HCCL_DETERMINISTIC
  → HCCL_DETERMINISTIC set by default to [false]
  → ErrToWarn: HCCL_DETERMINISTIC is set to [%s], which is incorrect
```

**该环境变量确实存在、默认 false。** 所以测试前提成立。

### 测试设计

- 8 rank（`torch_npu_run --standalone --nproc-per-node 8`），HCCL backend
- 每个 rank 用 `manual_seed(1234+n)` 构造**完全相同的输入**
- 同一输入重复 all_reduce **12 次**，每次 `clone()` 出新 buffer、`barrier()` 后取 SHA256
- 三种形状：4096 / **5120（hidden）** / **25600（intermediate）** —— 都是模型真实宽度
- 先跑 `HCCL_DETERMINISTIC` **不设**（默认 false）的情形

## 二、结果：**全部可重复**

| 形状 | n | distinct_results | reproducible |
|---|---|---|---|
| small | 4096 | **1** | ✅ |
| **hidden5120** | 5120 | **1** | ✅ |
| **inter25600** | 25600 | **1** | ✅ |

所有 8 个 rank、所有 12 次重复，哈希完全一致
（例如 `hidden5120` 恒为 `23c6a4722df1e0f4`，`inter25600` 恒为 `968bf6d020f341bc`）。

**⇒ HCCL all-reduce 在默认配置下就是逐位可重复的，连 `HCCL_DETERMINISTIC=1` 都不需要。**

## 三、结论：**R209 的假设被否掉**

| 项 | 判定 |
|---|---|
| R209 假设"集合通信归约顺序不确定" | ❌ **实证否掉** |
| 归约顺序是不是 #15 的根因 | **不是** |

**这是本轮第二个被我自己否掉的假设**（第一个是 `timesteps` 常驻 buffer，R209 内否掉）。

## 四、那么 #15 的根因范围被收窄到哪里

既然集合通信可重复，`relRMS 0.026%` 的抖动只能来自**算子层面**：

| 候选 | 说明 | 如何继续验证 |
|---|---|---|
| **matmul 分块 / split-K 顺序** | cube 累加顺序随 tiling 或调度变化 | 单算子重复：同输入跑同一 matmul 多次比位 |
| **FA 内核内部归约** | TIK kernel 内 vector/cube 交错时序 | 单算子重复：只跑 FA |
| **atomic 累加** | 某些 op 用原子加 | 查算子实现 |
| **AICPU/异步流时序** | 多个 stream 交错 | 单流对照 |

**⇒ 下一步应该是"单算子逐位重复"探针**：对 matmul / FA / norm 各跑 N 次同输入，看哪个先变。
这是**比跑整个模型便宜得多**的定位方式，而且完全不需要权重。

## 五、这个否证的价值

1. **排除了一整类候选**（集合通信），#15 的搜索空间显著缩小
2. 证明了**不需要模型就能测**这类假设 —— 成本从 16 分钟降到 1 分钟
3. 建立了一个可复用的探针（`hccl_determinism210.py`），后续算子级探针可以照抄结构
4. **对 #4 的意义**：#4 的"安全 prefetch 生命周期"关切的是**拷贝/同步**的正确性，
   若能证明算子本身逐位可重复，那么 #4 就不必为"数值抖动"背锅，可以专注于
   prefetch 的**正确性**（不覆盖在用缓冲），这反而让 #4 更简单

## 六、诚实边界

- 输入是"所有 rank 相同"的值。真实模型里各 rank 值不同，归约路径的**数值**不同，
  理论上仍可能有不同舍入行为。但"同输入同结果"这一条已足以否掉"顺序随机"的假设。
- 我只跑了 `HCCL_DETERMINISTIC` **不设**的模式（因为已经可重复，再跑 `=1` 无信息增量）。
- 没有测 reduce-scatter / all-gather，只测了 all-reduce。

## 七、合规

合成随机数据，无权重、无模型、无 pipeline；未改模型数学；未采用任何东西；
实验私有目录 `/tmp/h3_hccl210`（0700）；lease 已释放。
