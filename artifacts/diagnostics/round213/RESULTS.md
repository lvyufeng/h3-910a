# R213 — 决定性实验：**关掉 prefetch 后仍然非位同** ⇒ 假设也被否掉

## 一、做了什么

R212 收窄后我提出："prefetch 是 #15 的根因"。本轮直接测它。

benchmark 的 `--offload` 正好就是这个开关：

| 模式 | 机制 | 有异步 prefetch？ |
|---|---|---|
| `native` | `_install_layerwise_hook_group(..., copy_stream, True)` | **有** |
| **`legacy`** | `block.register_forward_pre_hook(onload)` + `register_forward_hook(offload)`，逐块 `module.to()` | **无（同步）** |

**跑法**：`--offload legacy --mode baseline --steps 25 --warmup 1 --repeats 3 --audit strict`

## 二、结果：**仍然非位同**

9 次重复（3 case × 3 repeat），全部：

| 指标 | 值 |
|---|---|
| `bit_identical` | **9/9 全部 false** |
| `candidate_prediction_sha256` | **9 个全不相同** |
| `relative_RMS` 范围 | 0.0002626 – 0.0002726 |
| `differing_elements`（video） | 3 579 840 – 3 579 893 / 3 580 416 |
| audio `differing_elements` | 13 245 – 13 248 / 13 248 |
| **`all_steps_bit_identical`** | **false** |
| `warm_step_median_s` | **26.923 s** |

**⇒ 关掉 prefetch、走完全同步的逐块 `module.to()`，输出依然每次不同。**

## 三、假设被否掉

| 假设 | 提出 | 判定 |
|---|---|---|
| `timesteps` 常驻 buffer 被改写 | R209 | ❌ 否（每次新分配） |
| **HCCL 归约顺序不确定** | R209 | ❌ **实测否**（R210） |
| **单算子非确定** | R210→R211 | ❌ **实测否**（9 个算子全可重复） |
| **FA 内核非确定** | R211 | ❌ **实测否**（R212） |
| **prefetch 异步拷贝竞争** | R212 | ❌ **实测否（本轮）** |

**五轮、四个实测否证。** 这是诚实的结果：**我连续提出错误假设，但每个都用实验干净地排除了。**

## 四、还剩什么（按可疑度排序）

| 候选 | 为什么还没被排除 |
|---|---|
| **USP 的 all-to-all** | 我只测了 **all-reduce**，没测 all-to-all / reduce-scatter。**一个真实的漏洞。** |
| AICPU / host 侧调度 | 无从观察（profiler 不可用） |
| 模型自有的非参数状态 | 我只看了 `timesteps` 一例，没系统排查 |
| allocator 地址影响 tiling | 未验证 |
| 多卡 collective 与计算交错顺序 | 单算子测试都是单卡单流 |

**最该先补的是 all-to-all 测试** —— 它是我 R210 明确漏掉的通信原语，
而 USP（`--usp 4`）恰恰重度依赖它。

## 五、本轮附带发现

| 项 | native | legacy |
|---|---|---|
| `warm_step_median_s` | R136 约 20.0 s | **26.923 s** |
| 峰值 allocated | — | **7.03 GB** |

**⇒ legacy 比 native 慢约 35%**（同步搬运没有重叠）。这也印证了 native 的 prefetch
**确实在起性能作用**（虽然它不是非位同的根因）。

## 六、诚实边界

- 只跑了 **step 25** 一个点、每 case 3 次重复。样本小。
- 我**没有**对照跑一次同配置的 `native`（R128 的历史数据是不同代码状态），
  所以"legacy 非位同"这个结论本身是自足的（9/9 不同），但**与 native 的严格对照缺失**。
- `peak_allocated_bytes` 7.03 GB 是在 encoder 未常驻的前提下。

## 七、合规

合成无关；复用已核 host 权重；未改模型数学；未采用任何东西；
实验输出在 `/tmp/h3_perf12601/legacy213`；日志 `/tmp/h3_r213`；lease 已释放。
