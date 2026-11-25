# R264 前置探测：双流实验的两个必要接口都在

`/home/mseco/vllm-omni/vllm_omni/diffusion/attention/backends/fa910a.py`：

| 需要的能力 | 现状 | 结论 |
|---|---|---|
| 调用时可指定 stream | `def flash_attention_910a(q, k, v, scale=None, stream=None, n_pad=0)`（L305） | ✅ **已支持**，双流并发无需改后端 |
| 每个内核可指定核数 | `core_budget = _runtime_core_budget()`（L200），并进入 cache key `..._c%d`（L186） | ✅ **可实现**：构建第二内核前临时替换 `_runtime_core_budget` 的返回值，即生成 c15 版本；`instance.core_num = min(instance.core_num, core_budget)`（L239）也会生效 |
| 内核按形状+核数缓存 | `_kernel_cache_key(... core_budget ...)`（L183-186） | ✅ 两套核数的内核可共存，互不覆盖 |

⇒ R264 的双流实验**不需要改盘上源码**：在探针脚本里
①monkeypatch `_runtime_core_budget` 为 15 → 构建 h=4 与 h=3 两个 c15 内核；
②恢复为 30；
③用 `flash_attention_910a(..., stream=s1/s2)` 分组并发；
④`torch.npu.synchronize()` 后比较 ms 与输出是否逐位一致。

## 待验证的风险

- 两个 c15 内核并发时，向量与 cube 是否真能错峰（而非各自拿到一半核后更慢）；
- 每个块 2 次 launch + 2 次 sync 的开销（50 块/步）；
- 输出逐位一致是否成立（head 独立计算，应当成立）。

## 下一步

按上述四步写 `round264/deployed/fa_dual264.py`，在真实形状（B1 H7 N41792 D128，seed 20261124）上跑：
单流 c30 基线 152.19 ms 对照双流 (c15 h4 + c15 h3)，同输入，报告 ms 与 bitwise。
