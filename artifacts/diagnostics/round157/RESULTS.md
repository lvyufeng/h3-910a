# R157 — 整模型常驻对照 OOM（不可用）；并更正一个我自己的事实错误

## 1. 更正：所谓"hook 被安装 3 次"是**错的**

我在 R149 / R152 / R153b 里把 3 倍的调用计数（`fa` 150、`mlp` 150、`silu` 312）解释成
"`_install_layerwise_hook_group` 被调用 3 次、hook 被重复登记"。

**真相**：harness 对每个选定 step 先做 `--warmup` 次预热、再做 `--repeats` 次测量。
`--warmup 2 --repeats 1` → **模型前向 3 次**。
- `mlp` 150 次 = **3 步 × 50 块**
- `silu` 312 次 = 3 步 × 104
- `fa` 150 次 = 3 步 × 50

**没有任何 hook 被安装过多次。**

### 为什么数值没错

"因为以为装 3 次而除以 3" 与 "因为有 3 个 step 而除以 3" 在算术上相同，
所以推导出的每步数值仍然正确（FA 7736 ms/步 与 R150 直接实测的 7736.2 ms 吻合）。**错的是解释，不是数据。**

## 2. 整模型常驻对照：**不可用（OOM）**

想用 `--resident` 去掉逐层 offload（从而**完全没有 H2D copy stream**）来做对照，结果：

```
NPU out of memory. Tried to allocate 404.00 MiB
(NPU 31.06 GiB already allocated; 32.00 GiB total capacity)
```

这正好印证 R125 的记录："提升并行度让 DiT 常驻 —— 在基线精度配置下**不可用**"。

**后果**：copy-stream 竞争**不能用常驻法检验**，必须改用预取时序对照。

## 3. 顺带确认 MLP 源码

`MiniMaxH3MLP.forward` 精确为 `fc1 → chunk(2,-1) → F.silu(gate)*up → fc2`，
**没有额外的 `.float()` 调用**。所以 E1→E2 区间确实只覆盖那三个逐元素步骤。

## 下一步对照（仍为诊断，不作候选）

把 offloader 的预取改成**同步**（`non_blocking=False`），使 block 执行期间**没有拷贝在飞**。
- 若激活区塌回 ~3 ms → **竞争假设成立**；
- 若仍是 ~35 ms → **竞争假设排除**。

## 失败尝试

`res157a`：`--resident` OOM，8 rank 全失败，回执保留。
