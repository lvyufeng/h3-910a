# R277 — U1a′ 已具备完整实施条件（`Pij` 分配点已确认）

## 一、读码结果

`softmax_compute`（L117）签名与关键行：

```
def softmax_compute(self, Sij_ub_zN, li_ub, mi_ub, kv_blk_idx, drop_mask_gm_offset):
    ...
    mi_ub, update_scale_ub = self.rowmax_compute(Sij_ub_zN, mi_ub, kv_blk_idx)   # 向量
    ... h_sub / 掩码 / exp ...
    Pij_l1_zN = self.tik_instance.Tensor(FP16, (n1, m, n0), scope=L1, name="Pij_l1_zN")   # ← 分配点
    self.cont_data_mv_1_bust(dst=Pij_l1_zN, src=Sij_ub_zN, burst=n1*m*n0//16)
    li_ub = self.rowsum_compute(Pij_l1_zN, li_ub, update_scale_ub, kv_blk_idx)
    if self.has_drop_mask: ...  # 若启用会重新写 Pij
    return Pij_l1_zN, li_ub, mi_ub, update_scale_ub
```

## 二、U1a′ 的三处改动（最小、可回退）

1. **签名**：`def softmax_compute(self, Sij_ub_zN, li_ub, mi_ub, kv_blk_idx, drop_mask_gm_offset, out_pij=None):`
2. **分配点**：`Pij_l1_zN = out_pij if out_pij is not None else self.tik_instance.Tensor(...)`
   （`out_pij` 由 `compute_in_each_kv_block` 在 **softmax 之前**、以 `scope=L1` 常驻分配）
3. **收缩 `Sij` 生命周期**：在 `compute_in_each_kv_block` 中，
   `Sij`（UB 76 KB）在 softmax 调用完成后不再被本块使用（下一块 matmul 若用 `workspace_next`，
   则不碰当前 S 缓冲）⇒ 可用 `new_stmt_scope` 把「softmax 调用 + 其内部临时量」包住，
   使 76 KB 在出 scope 后归还。

> 注意：`Pij` 已是 scope 外 L1，不受回收影响（R275 的风险点已被这一步消除）。

## 三、预期效果

- 出 scope 后归还 76 KB ⇒ `Oi_ub_zN`(64 KB) 可正常分配，**R272 的 UB 报错消失**；
- 同时第二个 S 缓冲（用于 `workspace_next` 的 cube 写入）也有了空间 ⇒ 双缓冲可行；
- 之后按 R267 回路测 ms（同口径基线 **154.73 ms**，tiles 64/912）+ 四项数值。

## 四、下一步（R278）

1. 用带断言的补丁脚本实现上述三处（与 R272 的脚本同风格：锚点唯一、失败即不动文件）；
2. `py_compile` → 替换/微基准/强制恢复（部署路径 md5 必须回到 `ec8d48c22878429fe74acb07f3cbf038`）；
3. 记录：ms/块、`max|diff|`、`relRMS`、非有限计数；≥1.2× 且数值可接受 ⇒ 8 rank 单步。

## 五、合规

仅读码；未改任何文件；部署路径保持原始。
