# R203 — encoder <2 s 在"每次载入"前提下**物理不可能**；唯一出路是常驻

## 一、查询结果：pinning 这条杠杆已经用尽

`encoder.py:1369 load_to_device()`：

```python
def load_to_device(self) -> None:
    if getattr(self, "_omni_layerwise_enabled", False):
        stager = PinnedModuleStager(
            self._omni_non_block_modules(), self.device_target,
            pin_memory=pin_memory,                    # ← 默认 True，pinned
            cache_retention=getattr(self, "_omni_component_cache", None))
        stager.load()
        return
    self.vision.to(self.device_target)               # ← 普通 .to()，pageable
    self.text_model.to(self.device_target)
```

两条路：逐层 offload 时用 **`PinnedModuleStager`（pinned）**，否则用普通 `.to()`（pageable）。

**而实测 `63 GB ÷ 3.05 s = 20.7 GB/s` ≈ pinned 的 20.0 GB/s**
⇒ **当前已经在 pinned 路径上。R202 提出的"换 pinned"杠杆不存在了。**

## 二、决定性算术：<2 s 物理不可能

| 项 | 值 |
|---|---|
| 实测最大 H2D 带宽（pinned 大张量，R198） | **20.0 GB/s** |
| encoder 权重体积 | **63 GB** |
| **最短搬运时间** | **63 / 20.0 = 3.15 s** |

| 目标 | 20 GB/s 下最多能搬 | 需要搬 | 判定 |
|---|---|---|---|
| **2.0 s** | **40 GB** | 63 GB | ❌ **物理不可能** |
| 3.0 s | 60 GB | 63 GB | ❌ 也差一点 |
| 实测 | — | — | **3.05–3.14 s（已贴极限）** |

**⇒ 只要每次 `encode_prompt` 都要载入 63 GB，3.1 s 就是硬下限，<2 s 不可能。**

这与"算力只占 0.17 s"完全一致：**encoder 的时间 ~99% 是搬运。**

## 三、唯一出路：权重常驻

| 项 | 值 |
|---|---|
| TP8 下每卡 encoder 份额 | `63 / 8` = **7.88 GB** |
| 单卡 HBM | 32 GB |
| **留给其余部分** | **24.12 GB** |

### 若 DiT 也常驻则放不下

| 组合 | 每卡占用 |
|---|---|
| DiT 全常驻（TP2，62 GB / 2） | 31 GB |
| + encoder | **38.9 GB > 32 GB** ❌ |

**但 DiT 用的是逐块 offload（`--offload native`），稳态只驻留少数块。**
⇒ **如果 DiT 稳态占用 < 24.12 GB/rank，encoder 就能常驻。**

### 若常驻成功，encoder 变成纯算力受限

| 项 | 值 |
|---|---|
| 算力下限 | **≈ 0.17 s** |
| 加实际效率折扣（取 3×） | **≈ 0.5 s** |
| 相对目标 2 s | **大幅达标** |

## 四、于是问题收敛成**一个可测的数**

> **DiT 阶段的实际稳态 device 显存是多少？是否 ≤ 24.12 GB/rank？**

- 若是 → encoder 常驻可行 → encoder 降到约 0.5 s → **<2 s 达成**
- 若否 → **encoder <2 s 不可达**，除非改搬运方式（而 20 GB/s 已是链路极限，
  更高只能换硬件或改数值 dtype，后者违反"保持原数学"）

**这个数可以从现有回执里找，也可能需要一次测量。**

## 四b、**答案：内存够，常驻可行**（用现有回执实测数据）

R203 第三节把问题收敛成"DiT 稳态显存是否 ≤ 24.12 GB/rank"。**这个数已经在回执里了。**

R170 实测（DiT block 区间）：

| 项 | 值 |
|---|---|
| **allocated 峰值** | **7.13 GiB** |
| **reserved 峰值** | **11.64 GiB** |
| 余量 | 20.36 GiB（= 32 − 11.64 ✓ 自洽） |

encoder 每卡份额（TP8）：`63 GB / 8 = 7.88 GB` = **7.33 GiB**

### 共存核算

| 口径 | DiT | + encoder | 合计 | 32 GiB 下 |
|---|---|---|---|---|
| allocated | 7.13 | 7.33 | **14.46 GiB** | ✅ 剩 **17.54 GiB** |
| reserved | 11.64 | 7.33 | **18.97 GiB** | ✅ 剩 **13.03 GiB** |

**R203 第三节的条件（DiT ≤ 24.12 GiB/rank）满足：11.64 ≤ 24.12 ✓**

⇒ **encoder 常驻在显存上完全可行，且留有 13 GiB 以上余量。**

### 于是 encoder 的时间变成纯算力

| 项 | 值 |
|---|---|
| 算力 roofline | **0.17 s** |
| 加 3× 效率折扣 | **≈0.5 s** |
| 目标 | **<2 s** |
| 判定 | ✅ **大幅达标** |

**⇒ encoder <2 s 目标可达，路径是"常驻而非每次载入"，不需要改数学。**

## 四c、但出现一个新的、真实的约束：**host RAM**

| 项 | 值 |
|---|---|
| DiT 权重（host 侧 pinned masters） | 62 GB |
| encoder 权重（host 侧） | 63 GB |
| **合计 host 侧需求** | **125 GB** |

**这需要在 host 上有 ~125 GB 的 pinned 内存可用。** 若主机 RAM 不足，
"两者都常驻 host、轮流上卡"这个前提就不成立。

**这是下一步必须先查的数**（`free -g` + pinned 上限），
比继续推算更重要——**如果 host RAM 不够，整条常驻路就断了。**

## 四d、host RAM 查了：**RAM 够，但 pinned 上限不够**

| 项 | 实测 |
|---|---|
| 主机 RAM 总/可用 | **754 GB / 741 GB** |
| **`ulimit -l`（memlock 上限）** | **98,842,564 KB = 94.3 GiB** |
| NUMA | 8 节点（4 socket × 48 核） |
| HugePages | 0（未启用） |

### 对照需求

| 项 | 值 |
|---|---|
| DiT 权重（host 侧 pinned masters） | 62 GB |
| encoder 权重（host 侧） | 63 GB |
| **需要同时 pinned** | **125 GB** |
| **实际上限** | **94.3 GiB** |
| **差额** | **约 −31 GB** ❌ |

**⇒ 瓶颈不是 RAM（754 GB 很充裕），而是可锁页内存上限 94.3 GiB。**

### 三条可能的解法（按代价排序）

| # | 解法 | 代价 | 是否改数学 |
|---|---|---|---|
| 1 | **提高 `ulimit -l`**（需 root 或 CAP_IPC_LOCK） | 运维动作 | ❌ 不改 |
| 2 | **encoder 只 pin 需要的部分**：Qwen3-VL 的 **vision 塔**（27 层、hidden 1152）在纯文本 prompt 下不需要 —— 若它能留在 pageable 或干脆不加载，pinned 需求显著下降 | 代码改动 | ❌ 不改 |
| 3 | 让 DiT 与 encoder **分时 pin**（进入 encoder 阶段前 unpin DiT） | 代码改动 + 仍要搬一次 | ❌ 不改 |

**解法 2 值得优先算**：vision 塔参数量远小于 text 塔（depth 27、hidden 1152 vs 64 层、hidden 5120），
若它占 63 GB 中的一部分，省下来可能刚好填平 31 GB 的缺口。

### 另一个被忽略的隐患：NUMA

**8 个 NUMA 节点**。pinned 内存若跨 NUMA 节点，H2D 带宽会明显低于同节点。
R198 测得的 20 GB/s 是在什么 NUMA 布局下测的**没有记录**。
若 encoder/DiT 的 pinned masters 落在远端 NUMA，实际带宽可能低于 20 GB/s，
**那连 3.15 s 的下限都会被推高**。

## 五、诚实的结论

| 结论 | 强度 |
|---|---|
| encoder 时间 ~99% 是 63 GB 搬运 | **强**（算力 roofline 0.17 s vs 实测 3.05 s；速率 20.7 vs 实测 20.0 GB/s 吻合） |
| 已在 pinned 路径、pinning 无更多收益 | **强**（源码 + 实测速率吻合） |
| **每次载入前提下 <2 s 物理不可能** | **确定**（63 GB > 40 GB 上限） |
| 常驻可让 encoder 降到 ~0.5 s | **条件成立才真**（取决于 DiT 稳态显存） |

## 六、与前几轮的区别

- R201b：推断"搬运受限"，标为未验证
- R202：源码确认机制，但提出的修法**同轮被自己推翻**（N=1）
- **R203（本轮）：用算术给出硬边界（<2 s 不可能），并把问题收敛到一个可测的数**

**每轮都在收窄，而不是在错误上加码。**

## 七、合规

只读源码 + 本地算术；**本轮未占机器**；未加载权重；未改模型数学。
