# R275 — U1a 的隐藏风险：`Pij` 是 L1 张量，不能简单用 scope 包住 softmax

## 一、原本的 U1a 设想

给 `softmax_compute` 调用包一层 `new_stmt_scope`，让 `Sij`（76 KB UB）在 softmax 结束后立即释放，
从而让后续 `Oi_ub_zN`(64 KB) / 第二个 S 缓冲拿到空间。

## 二、读源码后的新发现（这层不能直接包）

`softmax_compute` 的返回值是 `Pij_l1_zN`，供**后面**的 `output_compute` 使用：

```
Pij_l1_zN, li_ub, mi_ub, update_scale_ub = self.softmax_compute(Sij_ub_zN, ...)
...
Oi_l1_zN = self.output_compute(Oi_l1_zN, Pij_l1_zN, Vj_l1_zN, ...)
```

而 TIK 的 `new_stmt_scope` 会**同时回收该 scope 内分配的 UB 与 L1 缓冲**。
若把 `softmax_compute` 包进 scope，`Pij_l1_zN`（在 scope 内分配）会在出 scope 时被回收，
随后 `output_compute` 读到的是已回收的 L1 区域 ⇒ **静默错误或编译报错**。

## 三、修正后的 U1 路线（按安全度）

| 方案 | 做法 | 安全性 | 腾出 |
|---|---|---|---|
| **U1a′（推荐）** | 把 `Pij_l1_zN` 的分配**提到 scope 外**（显式建 L1 tensor 并作为参数传给 `softmax_compute`），再只把 `Sij` 的生命周期收缩 | 高（L1 常驻不变） | 76 KB |
| U1b | `Oi_ub_zN` 分两段搬运，64 KB → 32 KB | 高（纯搬运拆分、不改数值） | 32 KB |
| U1c | `Oij` 改 FP16 累加 | 低（改数值，不优先） | 32 KB |

U1a′ 需要：①在 `compute_in_each_kv_block` 里显式 `Pij_l1_zN = Tensor(FP16, (Bc//16, M, 16), scope=L1)`；
②`softmax_compute` 增加 `out_pij` 参数并写入该 tensor（而非内部新建）；
③`Sij` 仍由调用方持有，softmax 后即可被后续 matmul 复用。

## 四、下一步（R276）

1. 读 `softmax_compute`（L117–161）确认 `Pij_l1_zN` 的分配位置与是否支持外部传入；
2. 若支持 → 实施 U1a′ + 重试双缓冲；若不支持 → 先做 U1b（纯搬运拆分，风险最低）；
3. 每次实验 `trap restore`，部署路径 md5 必须回到 `ec8d48c22878429fe74acb07f3cbf038`。

## 五、合规

仅读码与分析；未改任何文件；部署路径为原始状态（md5 已核对）。
