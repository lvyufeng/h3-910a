# R249 — USP 侧只读勘察：存在 permute-free all-to-all 开关，但收益上限约 0.1–0.2 s

## 一、发现（只读，未运行任何模型）

`vllm_omni/diffusion/attention/parallel/ulysses.py`（721 行，与 `.prebak` **完全一致**，说明未被本地改动）：

- `UlyssesParallelAttention(..., ulysses_a2a_permute: bool = False)` —— 已有**融合 permute-free all-to-all** 路径
  （`_a2a_permute_enabled`，要求 `a2a_permute` 扩展可用：`vllm_omni/diffusion/distributed/a2a_permute.py` 存在）；
- `pre_attention()` 里按 `get_ulysses_mode()` 分支：`"strict"`（当前默认）与 `"advanced_uaa"`；
- 两次 all-to-all 的实现：
  - QKV 方向：`reshape → permute(2,1,0,3,4) → .contiguous()` 后 `dist.all_to_all_single`，再 `permute → contiguous`；
  - O 方向：`permute(1,0,2,3) → contiguous → all_to_all_single`，再 reshape/permute/contiguous。
- 我们的 benchmark 不经过 engine，因此 `ForwardContext` 未设置 ⇒ `get_ulysses_mode(default="strict")` 恒为 strict，
  `ulysses_a2a_permute` 恒为默认 False。

## 二、收益上限估算（先用算术，不跑模型）

- 每块 6 次 `.permute().contiguous()`（Q/K/V 各一次入、出方向再各若干），
  每次约 10448×3584 FP16 ≈ 75 MB ⇒ 每块约 450 MB 纯拷贝；
- 以 ~800 GB/s 计 ≈ 0.56 ms/次 ⇒ 每块约 3.4 ms ⇒ **每步约 0.17 s**（乐观上限 0.2–0.3 s）；
- 真正的 all-to-all 通信量（packed 内 ~22 ms/块 ≈ 1.1 s/step）不会因此减少。

⇒ **即使用上该开关，也只值约 0.1–0.2 s**，不足以补上 15 s 目标剩余的 1.5 s。

## 三、当前 15 s 缺口的现实构成（基于实测）

组合候选 16.5034 s；FA+USP 中 kernel 已降到 ~137.8 ms/块（trim+tile）。

| 项 | 估计 s/step | 可动性 |
|---|---|---|
| FA kernel（trim+tile 后） | ~6.9 | 只能靠内核算法级改造 |
| USP all-to-all | ~1.1 | 通信量不可减；permute 融合只值 ~0.2 |
| 四个 linear | ~4.44 | in-model 额外开销 ~0.9 s 成因未定 |
| norm/rope/adaln/元素链 | ~0.9 | 小 |
| 残余（零扰动不可观测） | ~3.4 | 手段用尽（钩子/包装/替换全部证伪） |

**结论**：靠现有已识别杠杆（USP ~0.2 + GEMM ~0.9）只能再省约 1.1 s，**仍差约 0.4 s**；
要稳达 15 s 必须解决那 3.4 s 残余中的一部分。

## 四、下一步（R250）：换观测手段，而非继续猜

本项目所有钩子/包装/替换手段已证伪（R240/R243/R244）。剩下唯一**零扰动**办法是
**CANN 自身的算子级 profiler（`msprof`）**：在单步上采集 op-level timeline，
不插入任何 Python 钩子、不改执行路径。

- 先确认 `msprof` 可用（`which msprof` / CANN 目录）；
- 用 `--steps 0` 跑单步并采集，得到每类算子的设备时间与间隙；
- 目标：判定 3.4 s 残余是真实算子（可优化）还是事件对结构造成的测量假象；
- 若为真实算子 ⇒ 有明确优化目标；若为假象 ⇒ 修正 step 预算后重新评估 15 s 可达性。

## 五、合规

- 本轮**纯只读源码**，未在主机跑模型、未改任何文件、未下载/回传数据；
- 未装候选；FA trim/tile 仍未成为默认（组合候选仅单步测量）。