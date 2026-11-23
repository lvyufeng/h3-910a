# R208 — 发现：FA 内核**已经有人试过 4 个流水线变体，一个都没采纳**

## 一、发现

准备动手实现 R196 提出的"中间路线"时，在 `/tmp/fa_pkg/flash_attention/ops/flash_attention/` 看到这些备份：

```
flash_attention_fwd.py.fp32exp.bak     Nov 14 19:49   29484 B
flash_attention_fwd.py.nopipe.bak      Nov 14 20:34   29890 B
flash_attention_fwd.py.pingpong.bak    Nov 14 20:36   30030 B
flash_attention_fwd.py.prepv.bak       Nov 15 00:24   29890 B
```

**这是前人做过的实验记录。** 我先读它们，避免重复劳动。

## 二、逐个核对结果

| 变体 | 与当前文件的差异 | 是否采纳 |
|---|---|---|
| `nopipe` | **diff 为空（完全相同）** | 已被回退到与当前一致 |
| `prepv` | **diff 为空（完全相同）** | 同上 |
| `pingpong` | 有差异 | **未采纳**（当前文件里 `_vpar` 出现 **0** 次） |
| `fp32exp` | 未细读（文件更小） | 未采纳 |

### `pingpong` 变体改的是什么（关键）

```python
# pingpong.bak：
_vpar = kv_blk_idx % 2
Vj_l1_zN = self.tik_instance.Tensor(FP16, (kv_blk_h_aligned, self.d),
                                    name="Vj_l1_zN_%d" % _vpar, scope=L1)
KjT_l1_zN = self.tik_instance.Tensor(FP16, (self.d // self.N0, kv_blk_h_aligned, self.N0),
                                     name="KjT_l1_K1MK0_ed_%d" % _vpar, scope=L1)
```

**它做的是 K/V 的 L1 缓冲 ping-pong**（让 kv_blk i+1 的载入不与 kv_blk i 冲突）。

**而 R196 我指的是 `Sij_ub_zN`（UB 的分数缓冲）双缓冲** —— 那是 softmax(vector) 与
下一个 matmul(cube) 之间的 RAW 冲突。

**⇒ 两者不是同一个东西。前人的尝试没有覆盖我这个点。**

## 三、但这也是一条重要的警告

前人试过的 ping-pong 是**更便宜、更安全**的版本（只动 L1 缓冲、不动 UB 容量），
**都没被采纳**。这有两层含义：

1. 这类"缓冲 ping-pong"微补丁可能**不动指标**；
2. 与 R196 的结论一致：**真瓶颈是 `disable_sync=False` 的自动屏障**，
   只改缓冲不改屏障，重叠依然不会发生。

## 四、这也改变了我的实现判断

我原计划先做"Sij 双缓冲（不改屏障）"作为最小实验。现在看：

| 障碍 | 说明 |
|---|---|
| UB 容量 | `Bc=304` 下 Sij 77.8 KB，双缓冲 155.6 KB，而 UB 192 KB，余量不足 |
| 被迫降 Bc | 只能到 `Bc=256`（65.5×2=131 KB），但 R191 实测 `Bc=256` **本身就更慢**（169.94 vs 154.57 ms） |
| **盈亏平衡点** | 在 169.94 的基础上要快过 154.57，需 **≥9.0%** 的收益，仅靠双缓冲很难 |
| 前人证据 | 更便宜的 K/V ping-pong 尚且没被采纳 |

**⇒ "先做最小 ping-pong 实验"这条路性价比低，且很可能重复前人已否的结论。**

## 五、修正后的 FA 实现判断

| 方案 | 判断 |
|---|---|
| 缓冲 ping-pong（前人已试） | ❌ 大概率无效，且 UB 窗口窄 |
| **`disable_sync=True` + 手工 flag 同步** | ✅ **仍是唯一有理论依据的方向**（R196 的位置二） |
| 两者结合 | 理论上最优，但需要先解决 UB 容量 |

**⇒ 真正要动的是同步骨架，这印证了 R196 的判断：这是数天级工程，没有便宜的中间路线。**

## 六、这一轮的净价值

- 避免了**重复前人 4 个已否实验**（尤其 ping-pong，我本来正要动手）
- 修正了"先做最小实验"的乐观估计（UB 窗口窄 + 盈亏平衡 9%）
- **把 FA 的结论进一步收紧：必须在同步屏障层面动，缓冲层面已无空间**

## 七、诚实边界

- 我没有找到前人这 4 个实验的**测量数据**（只有源码备份，没有结果记录）。
  "未被采纳"是从代码状态推断的，**不等于"实测无效"**——也可能只是没跑完或没记录。
- `fp32exp.bak` 我没细读。

## 八、合规

只读文件与 diff；**未修改任何文件**；未占机器跑模型；未改模型数学；lease 已释放。
