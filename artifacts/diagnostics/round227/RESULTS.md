# R227 — 探针扰动过大，本轮判定失败；但排除了一条候选

## 一、目的

R226 把 `rest` 4.09 s/step 定位到 `attn` 父区间内 1.63 s 与 `mlp` 父区间内 1.74 s（合计 3.37 s），
并提出假设：这部分是被暴露的权重预取等待。

读了 offloader 源码后，机制确认如下（`vllm_omni/diffusion/offloader/`）：

- `LayerwiseOffloadHook.pre_forward` 在**私有 copy stream** 上预取**下一块**权重（non-blocking）。
- `post_forward → offload_layer → clear_block_storage(..., ready_event)` 在**计算流**上
  `wait_event(下一块拷贝完成)` 后才释放本块存储 —— 所以暴露等待发生在 **block 边界**，
  不在 R226 的 pre/post span 之内。
- `pre_forward` 还有一条**同步补拷**兜底（`prefetch_layer(non_blocking=False)`）。

因此本轮目标：直接测 copy stream 上每次预取的时长、同步兜底的次数与耗时。

## 二、结果：**探针本身把步时拖慢了 15%，数据不可用**

| 指标 | R226（干净） | R227（挂探针） | 变化 |
|---|---|---|---|
| 测量步 `local_elapsed` | 17.875 s | **20.621 s** | **+2.75 s** |
| rank_max | 17.928 s | 20.622 s | +2.69 s |
| `block` 中位 | 345.775 ms | **407.96 ms** | **+62.2 ms/block** |
| `attn` 中位 | 243.397 ms | 256.96 ms | +13.6 ms |
| `qkv`/`fc1`/`fc2` | 17.07 / 23.22 / 29.98 | 17.06 / 23.1 / 30.3 | 不变 |

逐 rank 完全一致（8 rank 全为 20.62 s），说明是**系统性探针开销**，不是环境漂移。

**原因**：每个 prefetch 里用 `with torch.npu.stream(self.copy_stream)` 切流并创建 2 个计时事件
（50 次/步 = 100 个事件 + 50 次流切换）。NPU 上每次流上下文切换与事件记录都会插入排队开销，
累积成每块 +62 ms。

**被污染的字段**：`copy_stream.copy_ms_sum_per_step` 在 8 rank 全部为 `null`
（跨流 `elapsed_time` 抛异常被吞掉），**没有任何有效拷贝时长数据**。
因此本轮的权重等待假设**既未证实也未否证**，R226 的数据仍是当前有效值。

## 三、本轮唯一有效结论：同步兜底不是成本

8 个 rank 的 `synchronous_fallback.count` **都恰好是 1**（每次 1 次/步），
不是 50 次/步。⇒ `pre_forward` 里那条“块未 materialize 就同步补拷”的路径
**只在步首触发一次**，不承担 3.37 s 的可疑量。这条候选可以划掉。

## 四、失败留档与修正方案

这是本目标下第一次“探针扰动使测量无效”的失败，按合同如实记录，不重放。

下一版探针必须满足**步时不变性自检**（与 17.93 s 相差 ≤0.1 s）才允许采信：
1. 不再用 `with torch.npu.stream(...)` 切流，改用 `event.record(stream)` 直接指定流；
2. 每个 prefetch 只记 2 个事件，且**只对每 5 个块采样一次**（10 次/步，预计扰动 <0.3 s）；
3. 先跑一次“空探针”（只挂事件不切流）验证步时不变性，通过后再采数；
4. 若仍不可控，改用不侵入的替代判据：`--mode resident`（权重常驻，若可容下）与
   `--offload legacy`（同步加载）两端对照，用差值界定权重流的总代价上界。

## 五、合规

- 未改数学/dtype/块数/offload 策略/调度；未安装候选；FA trim 仍 OFF。
- 回执仅含标量计时与契约字段；未回传张量；未下载权重。
- 主机已清空（无残留进程），R227 目录保留 15 MB 级回执与日志。