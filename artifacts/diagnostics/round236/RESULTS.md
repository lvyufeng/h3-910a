# R236 — MLP 缺口破解：`fc1` 真实耗时 54.7 ms/block，模块钩子少算了 31.3 ms

## 一、方法

在 `MiniMaxH3MLP.forward` 内按**原语句顺序**插入设备事件（不改数学、不改 dtype、
不改分配顺序）：

```
ev0 → hidden = self.fc1(x)                    → ev1
ev1 → gate, up = hidden.chunk(2, -1)          → ev2
ev2 → gate_f = gate.float()                   → ev3
ev3 → silu = F.silu(gate_f)                   → ev4
ev4 → up_f = up.float()                       → ev5
ev5 → product = silu * up_f                   → ev6
ev6 → out = self.fc2(product)                 → ev7
```

默认流、无流切换、forward 内零同步；8 rank 完成。
**步时不变性**：17.9145 s vs 基准 17.928 s = **−0.014 s**（最好的一次）。

中途两次失败按合同留档：第一次 `NameError: names`（探针退出分支引用了局部变量），
第二次 `KeyError: 'mlp_chunk'`（事件字典在编译/追踪路径下丢键）；改为**显式 8 个事件变量**后通过。

## 二、实测（rank 0，ms/block，中位）

| 段 | 中位 ms | 说明 |
|---|---|---|
| **`fc1`** | **54.703** | MergedColumnParallelLinear |
| `chunk` | 0.100 | 视图 |
| `gate.float()` | 0.862 | FP32 临时量 |
| `silu` | 1.139 | |
| `up.float()` | 0.743 | |
| `mul` | 0.913 | |
| **`fc2`** | **30.099** | RowParallelLinear（含 TP all-reduce） |
| **MLP 合计** | **88.277** | 与父区间 88.250 吻合 |

8 rank 高度一致（fc1 54.36–54.70、fc2 29.84–30.17、合计 88.27–88.30）。

## 三、关键结论：钩子对 `fc1` 少算 31.3 ms，正好是缺口

- 模块钩子（R226/R231/R235）测到 `fc1` = **23.4 ms**；本轮 MLP 内部直接取事件 = **54.7 ms**。
- 差值 **31.3 ms**，加上元素链（chunk+两cast+silu+mul ≈ 3.76 ms）≈ **35.1 ms**
  = R235 报告的 `mlp` 缺口 **34.8 ms**。**缺口被完全解释。**
- 也就是说，此前 R147/R226/R231/R235 对 `fc1` 的归因**系统性偏低**；
  原因待查（可能是该 module 的 forward 被编译/内联后，模块级 pre/post 钩子只框住了图的一部分）。

## 四、对 roofline 的修正（重要）

`fc1` 真实 54.703 ms/block ⇒ **2.735 s/step**。
FLOPs = 2×10448×5376×14336 = 1.609 TFLOP ⇒ 达成 **29.4 TFLOPS/卡（假设峰值的 9.5%）**；
而 R146 独立测量同形状 `mm` 为 **19.04 ms（84.6 TFLOPS/卡）**。

- **in-model 的 fc1 是独立 mm 的 2.87× 慢**，量级可回收 **约 1.8 s/step**（若能让 fc1 接近独立 mm 的效率）。
- 修正后的 four-linear in-model 合计 = qkv 17.07 + out_proj 18.59 + fc1 54.70 + fc2 30.10
  = **120.5 ms/block = 6.02 s/step**（此前 R147 记为 4.435 s，低估约 1.59 s）。
- R223 的 roofline 表需要按此修正：GEMM 区而不是“只剩 ≤0.2 s 可回收”，
  仅 fc1 一项就有约 1.8 s/step 的 in-model 效率缺口。

## 五、下一步（R237）

1. **同样用“前向内直接事件”复测 `qkv`、`out_proj`**（attn 前向里直接插事件），
   确认它们的模块钩子值是否也被低估——`attn` 仍有 25.0 ms/block 缺口，很可能同因。
2. 复测后更新 R223 roofline 表与 10 s 判定（GEMM 份额变大，FA 份额相应下降）。
3. 若确认 fc1 的 in-model 低效（2.87× vs 独立 mm），针对该差异做归因：
   是权重流竞争、tile 选择、还是 `MergedColumnParallelLinear` 的双 GEMM 布局问题——
   这项属于 #6 GEMM 布局，需先测后提候选。

## 六、合规

- 未改盘上源码；未安装候选；未改数学/dtype/块数/offload/调度；FA trim 仍 OFF。
- 未加载新权重、未下载、未回传张量；回执只含标量统计与时间分布。
- 两次失败（NameError / KeyError）已如实记录，未重放不确定操作。