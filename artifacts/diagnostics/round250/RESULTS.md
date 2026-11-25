# R250 — msprof 首个零扰动 op-level profile：FA 完全吻合，残余指向「大量小算子 + 拷贝 + Cast」

## 一、做法

用 CANN 原生 `/home/mseco/Ascend/cann-9.2.0/bin/msprof` 包裹标准 8 rank 运行
（`--warmup 1 --steps 0`，baseline 配置，无任何 Python 钩子/包装），
采集 `task_time`（每 rank 一个 PROF 目录），聚合成下表。
原始汇总：`round250/receipts/msprof250b_summary.json`（365 种内核、**404,935** 个 task）。

## 二、实测（7 个 rank 目录 × 2 步 = 14 个 rank-step 归一）

| 项 | calls/rank-step | ms/rank-step | **ms/block** |
|---|---|---|---|
| **FA kernel**（`b1_h7_n43776_d128_br128_bc304_c30_k41769`） | 50.0 | 7609.5 | **152.19** |
| **未解析小算子**（`N/A`） | 16027.5 | 6324.8 | **126.50** |
| **MemcopyAsync** | 1600.0 | 2163.2 | **43.26** |
| **Cast** | 206.0 | 1536.4 | **30.73** |
| `te_matmul` A / B | 50 / 50 | 1166 / 853 | 23.33 / 17.06 |
| `MatMulV2` A / B | 50 / 50 | 584 / 133 | 11.67 / 2.66 |
| `te_cast` | 800.0 | 538.7 | 10.77 |
| `te_reducemaxd` | 100.0 | 272.0 | 5.44 |
| `te_mul` / `te_add` | 349 / 199 | 227 / 136 | 4.54 / 2.72 |
| 合计（列出的） | — | 21,543 | 430.87 |

## 三、三点关键判读

1. **FA 内核 152.19 ms/次，与独立微基准 154.57 ms 和我们 in-model span（176.9 含 USP ~22）完全一致**
   ⇒ 之前的钩子/事件测量是准确的，FA 没有隐藏开销。
2. **每块设备内核时间合计约 431 ms > 块 span 347 ms** ⇒ 有约 84 ms 的重叠（拷贝/HCCL 在其他流），
   同时说明**设备几乎被铺满**，不存在“大段空闲”。
3. **此前完全看不见的三块时间**：未解析小算子 **126.5 ms/块（16,027 次/rank-step，约 320 次/块）**、
   `MemcopyAsync` **43.3 ms/块（1,600 次/rank-step）**、`Cast` **30.7 ms/块（206 次/rank-step）**。
   这些数量级与 R235 的 3.457 s/step（69 ms/块）残余同一量级，**很可能就是残余的实体**。

## 四、必须说明的局限

- 上表按 14 个 rank-step 平均，**混入了 encoder TP8 阶段的算子**（每次运行跑一次 encoder），
  因此 `Cast`/`MemcopyAsync` 的绝对值偏高，不能直接当作 DiT 每步的量；
- 要得到干净的 DiT 预算，必须按 **task 时间戳**把「encoder 窗口」和「DiT 步窗口」切开；
- `N/A` 组是 msprof 未能解析名字的算子（多为小 vector 算子），需要按 task_id 顺序与其相邻算子归组。

## 五、下一步（R251）

1. 用 `task_start(us)` 时间戳切出 DiT 单步窗口（最后一次 encoder 之后到结束），
   在该窗口内重算上表 ⇒ 得到**零扰动的干净单步算子预算**；
2. 对 `N/A` 小算子按时间邻接聚类，识别它们属于哪个模块（预期是 norm/rope/elementwise/cast 群）；
3. 据此给出 15 s 的可达性重估：若 `Cast`+`MemcopyAsync`+小算子中有 1.5 s 可消（纯布局/转换、不改数值），
   则 15 s 可达；否则需重回 FA/内核层面。

## 六、合规

- 只读采集（msprof），未改盘上源码、未装候选、无数值变更；
- 临时数据在 `/tmp/msprof250b`（会话结束清理）；主机无残留进程。