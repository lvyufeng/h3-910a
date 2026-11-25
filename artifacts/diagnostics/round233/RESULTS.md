# R233 — 否证 H3：去掉 guard 的 100 次 host 同步并不更快（18.027 vs 17.947 s）

## 一、假设

R232 发现受保护层 `out_proj` 设备只用 18.2 ms、host 却阻塞 179.6 ms（`float(peak)` 同步），
`adaln_proj` 也是 1.9 ms 设备 / 27.0 ms host。H3 主张：这些同步让 host 落后、追赶期设备空转，
构成仍未归属的 3.43 s/step。

## 二、方法（仅测量，不改盘上源码）

在同一步内把 `_MiniMaxH3HalfInputFP32OutputLinearMethod.apply` 换成**数学等价、无 host 同步**版本：
`peak_t = amax(...)`（同一归约）→ `ratio = clamp(peak/32752)` → `exponent = clamp(ceil(log2 ratio),0)`
→ `scale_t = 2**exponent`（2 的幂，FP32 精确）→ `(product/scale_t).to(fp16)` → mm → `*scale_t`。
非有限值门限改为设备侧掩码累积、步末读取一次（不再每次调用同步）。

**逐位依据**：R221 已在合成输入上证明该分支与部署路径逐位相同（112,336,896 word、差异 0）。
本轮是 in-model 计时验证，**不是**数值批准；跨 run 逐位比较不可行（native 自重复非确定，issue #15）。

## 三、结果：更慢，假设否证

| 变体 | 步时（rank_max） | 差异 |
|---|---|---|
| A：部署 guard（含 `float(peak)` host 同步） | **17.9465 s**（R232 同配置） | — |
| B：设备侧 scale，无 host 同步 | **18.0274 s** | **+0.081 s（更慢）** |

- 受保护调用被拦截 **208 次**（warmup+测量两步 × 50 块 × 2 层），`nonfinite_peak_seen=false`。
- 8 rank 步时完全一致（18.0274 s），说明是系统性差异而非抖动。
- 设备侧算 scale 引入的额外小 kernel（`clamp`/`log2`/`ceil`/`pow`）抵消并超过了去同步的收益。

## 四、结论：四个候选机制全部否证

| 机制 | 结论 |
|---|---|
| 权重预取等待 | ❌ R228（慢 H2D 两卡与快卡差 0.0042 s/step） |
| allocator/拷贝争用 | ❌ R229（≤0.03 ms） |
| strict 审计 host 同步 | ❌ R230（0.036 s/step） |
| **guard host 同步（H3）** | ❌ **R233（−0.081 s，更慢）** |

⇒ R232 里 179.6 ms 的 host 阻塞**没有转化为步时损失**：host 被同步按住时设备仍在干活，
同步后 host 也能追上。**“同步换流水线”这条路径在本模型上不成立**，且去同步的替代实现反而更慢
（因此它既不该作为性能候选，也不必再作为数值候选提交）。

## 五、3.43 s 未归属的现状与下一步

仍未解释：`attn` 26.7 ms/block（1.34 s）、`mlp` 34.9 ms/block（1.75 s）、`block` 9.2 ms/block（0.46 s）。
四个“外部机制”候选已全部排除，因此最可能是**分解本身不完整**：父区间里还有我们没挂到的设备工作，
或事件之间存在的设备侧间隙（例如大块临时量分配触发的流内同步）。

下一步（R234）必须先验证“分解完备性”，而不是继续猜机制：
1. 用**计数器式**探针（不开事件）统计每个 block 实际发生的关键 op 次数与类型，
   与实际设备时间对齐，找出未被挂钩子的 op；
2. 若确认为分配触发的流内同步，则测 **allocator 配置**（例如大块释放策略）这一**环境级、
   非数学**的变量：同一步内对比不同 allocator 配置的步时，只接受输出逐位一致的配置。

## 六、合规

- 未改盘上源码；未安装候选；FA trim 仍 OFF；未加载新权重、未下载、未回传张量。
- B 变体仅进程内 monkeypatch，进程结束即消失；回执只含标量。
- 交叉 run 的数值一致性不在本轮结论范围内。

## 七、证据

- 驱动 `round233/deployed/run_guard233.py`
- 8 rank 回执 `round233/receipts/candidate_rank*.json`（含 `guarded_calls=208`、`nonfinite_peak_seen=false`）