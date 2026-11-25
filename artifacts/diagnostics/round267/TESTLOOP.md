# R267 前置：内核改动的编译/测试回路（已确认）

## 关键事实

`/home/mseco/vllm-omni/vllm_omni/diffusion/attention/backends/fa910a.py:217` **硬编码**：

```python
sys.path.insert(0, "/tmp/fa_pkg")
```

⇒ 部署后端只会从 `/tmp/fa_pkg` 导入 TIK 内核源码，**没有环境变量可切换**。

## 因此的测试回路（不改共享安装）

1. `/tmp/fa_pkg` 是我们自己的**临时目录**（非安装目录），原始内容已复制到 `/tmp/fa_pkg_r266`（pristine 备份另存）；
2. 修改在 `/tmp/fa_pkg_r266` 中进行；
3. 测微基准时：
   ```bash
   mv /tmp/fa_pkg /tmp/fa_pkg_baseline_hold      # 让出硬编码路径
   cp -a /tmp/fa_pkg_r266 /tmp/fa_pkg            # 放上候选版本
   ... 运行 fa910a_sweep/point 微基准 ...
   rm -rf /tmp/fa_pkg && mv /tmp/fa_pkg_baseline_hold /tmp/fa_pkg   # 无论成败都恢复
   ```
4. **恢复是强制步骤**：任何异常退出都必须先恢复 `/tmp/fa_pkg`，否则后续所有运行会静默使用候选内核；
5. 内核缓存以形状+核数为 key（`_kernel_cache_key`），候选与基线内核不冲突，但**缓存目录要区分**
   （`H3_FA215_KERNEL_DIR` 指向不同目录），避免把候选 `.o` 误当基线。

## 已有基线（供对比）

| 配置 | ms/块 | 来源 |
|---|---|---|
| c30, tiles 128/304 | 152.19 | R251/R255（in-model 与独立一致） |
| c30, tiles 64/912 | **154.73** | R265 独立微基准（最新一次，含 memcpy 开销） |
| c30, tiles 64/912（历史最优） | 137.81 | R245 扫描（纯 kernel 时间口径） |

⚠️ 两个 64/912 数字口径不同（R245 只计 kernel，R265 计整次调用含 to_nz 拷贝），对比时须同口径。
R267 起统一用 **R265 口径**（`fa_dual265.py` 的单流分支）作为基线 154.73 ms。

## 下一步（R268）

1. 在 `/tmp/fa_pkg_r266` 实施 R266 的双 P 缓冲 + PV 提前 + 显式同步；
2. 按上述回路运行微基准，产出：ms/块、输出 `max|diff|`、`relRMS`、非有限计数；
3. 与 154.73 ms（同口径）对比；若 ≥1.2× 且数值可接受 ⇒ 进 8 rank 单步（叠加 `H3_PERF_COMBO=1`）。
