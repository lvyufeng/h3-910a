# R332 — encoder 常驻可行性关键实测：**每 rank 分片仅 8.7 GiB HBM**，但 prefetch 侧 host 内存会爆

## 一、代码级改动（不改数学）

只读源码后定位到真正的搬运点：`LayerwiseOffloadHook.post_forward()` 每个 block 前向后调用
`offload_layer()` 把权重搬回 host。于是做了最小改动：

```python
lb.LayerwiseOffloadHook.offload_layer = lambda self: None   # 只关“搬回”，不关“搬入”
```

目的：让 block 权重逐块累积常驻 device，而不额外分配 host 内存。

## 二、实测结果（8 rank 一致）

```
[RESIDENT3] encode done 70.758–70.788 s   hbm_max=8.7GiB  reserved=8.6GiB  skipped_offloads=231
[STAGE] encoder 70.773 s / 70.780 s
free -g: used 743 / 754   ← host 内存被吃满
```

三个关键事实：

1. **`skipped_offloads=231`** ⇒ 改动确实生效（77 blocks × 3 组/栈 ≈ 231）；
2. **`hbm_max = 8.7 GiB`** ⇒ **encoder 每 rank 分片只有 ~8.7 GB**（≈63 GB/8 + 开销），
   **device 侧常驻完全可行**（每卡 32 GB HBM）；
3. 但 **host 内存涨到 743/754 GiB**，且首轮 encode 反而变慢（70.76 s vs 基线 62.95 s）
   ⇒ 说明 `prefetch_layer()` **每次仍从 host 重新拷贝**（没有"已常驻则跳过"的短路），
   且拷贝过程在 host 侧产生了大量未释放的固定（pinned）缓冲。

⇒ 已立即按 worker 进程名 kill 保护主机（`vLLM-Omni` 匹配），host 回到 **58 GB used / 694 GB free**，
剩余 3 个 `defunct` 僵尸进程（不占内存，PID 1 未回收）。

## 三、结论与下一步设计（已明确）

**encoder 常驻的 device 成本已验证为 8.7 GiB/rank，不是障碍；障碍是 prefetch 的 host 侧记账。**

正确的代码级改法（下一轮实现）：

1. 在 hook 上实现真正的“materialize once”语义：
   - `prefetch_layer()` 开头短路：若 `self.is_materialized` 为真则直接 return（不再 H2D）；
   - 首次 encode 后，对已 materialize 的 block 保持 `is_materialized=True`，并**不再调用**
     `offload_layer()`（或让 `offload_layer` 更新 state 但不搬运）；
2. 为 host 侧加监控：每次 prefetch 打印 host 可用内存，低于 150 GiB 即中止（R331 教训）；
3. 验收：第二趟 encode < 2 s，且 155 条 Half 检查与输出位级不变（常驻不改变数值）。

## 四、状态

- 交付不变：**16.34 s/step（−8.9%）**，`H3_PERF_COMBO=1` 默认 OFF；
- 主机干净（无我方运行进程；僵尸不占资源）；
- 远端 master 含本轮全部实测与改动。
