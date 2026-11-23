# R192 — 无法用 profiler 确认气泡假设（芯片不支持设备侧 profiling）

## 一、想做的事

R190 用算术推断：FA 每 tile 6.00 µs 对 1.44 µs cube 下限，
而 cube 24% / vector ≤21% / HBM 23% **三个单元同时都没跑满** ⇒ 推断瓶颈是**流水线气泡/依赖链串行**。

这个假设是整条 10 s 路径的关键支点：如果它错了，FA 的 1.50× 可能不可达；
如果它对，方向就明确（加大重叠、减少 tile 数）。

所以本轮想用 profiler 直接读 cube/vector 占用时间线来确认。

## 二、结果：工具不支持

找到了正确的工具 —— `$CANN/tools/msopprof/bin/msopprof`，它支持正是我要的指标：

```
--aic-metrics PipeUtilization,ArithmeticUtilization
  (ArithmeticUtilization | MemoryUB | Memory | MemoryL0 | L2Cache |
   PipeUtilization | ResourceConflictRatio | Default | BasicInfo | Roofline)
```

但运行报错：

```
[ERROR] Device profiling is not supported on current chip.
```

**这台芯片（910A/910B + 当前 CANN 26.2 工具链）不支持设备侧 profiling。**
同目录的 `msprof` 也因缺 `libascendalog.so` 起不来，且它给的粒度更粗（无 cube/vector 分项）。

⇒ **占用率时间线这条路在现有工具下走不通。**

## 三、气泡假设目前的证据强度（必须说清）

| 证据 | 类型 | 强度 |
|---|---|---|
| cube 24% / vector ≤21% / HBM 23%，三者同时低 | **算术推断** | 中（"三者都不满 = 没重叠"是合理但非唯一解释） |
| `Bc=384` 最快、`Bc=128`（2.4× tiles）慢 1.65× | **实测** | **较强**——符合"每 tile 有可观固定成本" |
| `Bc=304`（304=2×128+48）只比 384 慢 8.9% | **实测** | 强——**排除**了 cube 碎片化是主因 |
| softmax 12 op/元素也只占向量峰值 21% | 算术推断 | 中强——排除向量算力受限 |

**结论：气泡假设有间接支持（尤其 R191 的固定成本特征），但没有直接观测证据。**
必须这样表述，不能声称已确认。

## 四、不依赖 profiler 的替代验证思路（下一轮可选）

1. **空转成本测量**：用同样的 `blockDim=30`、同样的波数（829）跑一个极轻的内核，
   测出"每波固定开销"的量级。若 829 波 × 该开销可解释 4.56 µs/tile 中的显著部分，
   气泡假设得到独立支持。
2. **改变波数**：在同一总工作量下改变 tile 数（已部分由 R191 覆盖），
   拟合 `时间 = a × tiles + b × flops`，若 `a` 显著大于 0 即证明固定成本存在。
   —— **这个拟合 R191 的数据已经可以做**，见下。
3. 若两者都指向固定成本，则优化方向确定为"减少 tile 数 + 提高重叠"。

## 五、从 R191 数据直接拟合（现在就能算）

R191 的五个配置给了不同的 (tiles, 时间)，可以用来分离"每 tile 固定成本"与"每 flop 成本"。
这是**不依赖 profiler** 的直接证据，下一轮做这个拟合即可。

## 六、合规

- 只读：未改 fa910a 原文件（用 `/tmp` 私有副本）、未加载权重、未改模型数学
- 未做 driver/device reset；profiler 输出目录在 `/tmp`；lease 已释放
