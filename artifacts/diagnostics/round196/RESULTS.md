# R196 — 找到每 tile 串行的**具体位置**：`Sij` 单缓冲阻断了预取

## 一、背景

R193 实测出 FA 有 **47% 的时间是每 tile 固定开销**（`T = 83.5 ms + 2.922 µs × tiles`，
残差 1.1%）。但"固定开销"只是个标签，不知道具体卡在哪。

本轮读了 MindIE TIK 内核源码（`/tmp/fa_pkg/flash_attention/ops/flash_attention/flash_attention_fwd.py`，538 行），
定位到了具体位置。

## 二、内核结构

`compute_in_each_q_block` 里有**双层循环**：

```
for q_blk_idx:                    # 342 个 q 块
    load Qi
    Sij = Qi × K0.T               # 第一块单独处理
    for kv_blk_idx in 0..Tc-2:    # 每个 q 块扫 144 个 k 块（causal 用 if_scope 跳过）
        compute_in_each_kv_block(...)
    output_norm_and_write(...)
```

`compute_in_each_kv_block` 内部的**关键执行顺序**：

| 步骤 | 内容 | 硬件单元 | 涉及缓冲 |
|---|---|---|---|
| 1 | `Sij = Qi × Kj.T` | cube | 写 `Sij_ub_zN` |
| 2 | `Sij += attn_mask`（仅对角块） | vector | 读改 `Sij_ub_zN` |
| 3 | **`Pij = softmax(Sij)`** | **vector** | **读 `Sij_ub_zN`** |
| 4 | **`load K(j+1)` → `Sij = Qi × K(j+1).T`** | **cube** | **写 `Sij_ub_zN`** ← **同一缓冲区！** |
| 5 | `Oi = output_compute(Oi, Pij, Vj)` | cube | 读 `Pij_l1_zN` |

## 三、**冒烟枪：第 3 步与第 4 步共用 `Sij_ub_zN`**

代码的**意图**是软件流水——第 4 步提前算好下一块的 S，供下一轮直接用：

```python
Sij_ub_zN = self.tik_ops_utils.matmul_compute(Qi_l1_zN, KjT_l1_zN, ...,
                                              workspace=Sij_ub_zN)   # 写回同一个 workspace
```

但正因为写回的是**同一个** `Sij_ub_zN`，就产生了写后读（RAW/WAR）依赖：

> **softmax 必须完全读完 Sij，matmul(j+1) 才能开始覆盖写它。**

而 softmax(第 3 步) 和 matmul(第 4 步) 本该是最能重叠的一对（vector 与 cube）。
**单缓冲把这对重叠彻底堵死了**，于是每次 tile 都要付一次 cube↔vector 的往返等待。

这与 R193 测到的 **2.922 µs/tile** 固定开销在量级和机理上都吻合：
- softmax 的向量时间（R190 估算 ≲21% 向量峰值占用）
- 加上 cube 等向量的空转
- 两者本可重叠，实际却串行

## 四、具体修法：**Sij ping-pong 双缓冲**

把 `Sij_ub_zN` 换成两个交替的缓冲（`Sij_ub_zN[0]`、`Sij_ub_zN[1]`）：

```
迭代 j:   softmax 读 buf[j%2]
          matmul(j+1) 写 buf[(j+1)%2]     ← 不再冲突，可与 softmax 重叠
```

### UB 容量核算（UB 每核 192 KB）

`Sij` 大小 = `Bc × Br_aligned × 2 字节`（`Br=128`）：

| Bc | 单个 Sij | 双缓冲 | 192 KB 下是否可行 |
|---|---|---|---|
| 256 | 65.5 KB | **131.1 KB** | ✅ 宽裕（余 61 KB） |
| 304 | 77.8 KB | **155.6 KB** | ⚠️ 勉强（余 36 KB，但 Pij 等还需空间） |
| 384 | 98.3 KB | **196.6 KB** | ❌ **超出** |

**所以双缓冲的可行配置是 `Bc=256`**（且 256 整除 43776，tile 数 = 29412）。

注意：`Bc=256` 单独用时比 `Bc=304` **慢**（R191：169.94 vs 154.57 ms，0.910×），
但双缓冲若能把 2.922 µs 的开销压掉大半，净效果可能大幅超过。

### 预期收益估算

`Bc=256` 下 tiles = 29412：
- 单缓冲现测：169.94 ms（其中固定开销 = 2.922 × 29412 = **85.9 ms = 51%**）
- 若双缓冲消掉 50% 开销 → 127.0 ms（相比 `Bc=304` 基准 154.57 ms，**1.22×**）
- 若消掉 80% 开销 → 101.2 ms（**1.53×**）
- 若完全消除 → 84.0 ms（**1.84×**）

**1.53× 就已超过 10 s 所需的 1.50×。** 所以这个改动理论上足以达成目标。

## 五、另一个发现的次要点

外层循环 `for kv_blk_idx in range(Tc-1)` 对 causal 用 `if_scope` **运行时跳块**，
而不是编译期按 q 块裁剪循环上界。理论上不同 q 块需要不同数量的 k 块，
若能编译期/标量期算出上界，可省掉 828 波里那些空转的 `if_scope` 判断。
但这个开销应该远小于 2.922 µs/tile 的缓冲等待，优先级低于双缓冲。

## 六、诚实边界

- **本轮是源码阅读，不是实现，也不是测量。** 上面所有收益都是**估算**，
  实际能否拿到取决于 TIK 的同步语义是否真能重叠（`new_stmt_scope` 的 barrier 行为需验证）。
- TIK 的 `Tensor`/`DataMove` 是否允许在同一个 `new_stmt_scope` 内真正异步重叠，
  需要看 `tik_ops_utils` 里 sync 的插入位置——**这是实现前必须先确认的**。
- 改 tile/缓冲 ⇒ 累加顺序可能变 ⇒ **数值非等价**，必须仅作测量、报数后请用户决定。
- 未验证 `Bc=256` 双缓冲能否编译通过。

## 七、合规

- 只读源码；私有目录在 `/tmp`；未改任何文件
- 未加载权重、未改模型数学、未做 driver/device reset；lease 已释放
