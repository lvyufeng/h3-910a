# ⚠️ R199 的候选建议**是错的**，这是修正

## 一、错在哪

R199 把 `tokens_chunk_size` 当成**自由参数**，并且假设 `token_overlap` 固定为 2，
据此算出"`chunk_size=18` → 工作量 0.816×、drop 到 17.7 s"。

**两个前提都错了。** 读了 `klvae.py` 的赋值代码后发现：

```python
line 104:  self.tokens_chunk_size = math.ceil(self.clip_length / self.vae_ratio_t)   # 派生值！
line 105:  self.token_overlap     = (-self.token_drop) % self.tokens_chunk_size      # 随 chunk 增长！
```

- **`tokens_chunk_size` 不是可调参数**，它 = `ceil(clip_length / 4)`。
  想要 chunk=9 就必须把 `clip_length` 改成 36。
- **`token_overlap` 随 chunk_size 一起变**（`(-3) % chunk_size`）：
  chunk=5→2、chunk=4→1、chunk=3→**0**、chunk=9→6。
  所以 chunk 越大，重叠浪费越大，不是越小。

## 二、用真实公式重算

`num_chunks` 的真实算法（`decode_temporal` 开头）：

```python
pseudo_total_tokens = z.shape[2] - isolated + self.token_drop      # 37 - 0 + 3 = 40
pad_tokens          = 补到 tokens_chunk_size 的整数倍
pseudo_num_chunks   = pseudo_total_tokens // tokens_chunk_size     # 40 // 5 = 8
num_chunks          = pseudo_num_chunks - int(token_drop > 0)      # 8 - 1 = 7
```

**这一步恰好验证了 R199 的 `num_chunks=7` 和 196 帧 / 1.58× 是对的。**
（我原来写的 `ceil((37-2)/5)` 是错的算式，但碰巧给出同样的 7。）

| clip_length | chunk_sz | overlap | chunks | 帧/chunk | 总帧 | vs 124 | 相对当前 |
|---|---|---|---|---|---|---|---|
| **12** | 3 | **0** | 13 | 12 | **156** | 1.26× | **0.796** |
| **16** | 4 | **1** | 9 | 20 | **180** | 1.45× | **0.918** |
| **17（当前）** | 5 | 2 | 7 | 28 | 196 | 1.58× | **1.000** |
| 20 | 5 | 2 | 7 | 28 | 196 | 1.58× | 1.000 |
| 24 | 6 | 3 | 6 | 36 | 216 | 1.74× | 1.102 |
| 32 | 8 | 5 | 4 | 52 | 208 | 1.68× | 1.061 |
| **36** | 9 | 6 | 4 | 60 | **240** | 1.94× | **1.224 ← R199 说这是 0.816，实际更差 22%** |

**⇒ R199 的"chunk=18 省 18.4%"是完全错的，方向反了。**

## 三、修正后的真实候选

| 候选 | 相对工作量 | 预计步时 | 风险 |
|---|---|---|---|
| `clip_length=16`（chunk=4, overlap=1, 9 chunks） | **0.918** | **≈19.9 s** | 中：仍有一次重叠混合，改动温和 |
| `clip_length=12`（chunk=3, overlap=**0**, 13 chunks） | **0.796** | **≈17.3 s** | **高：overlap=0 意味着没有边界混合** |

### 关于 `overlap=0` 的风险（比 R199 说的严重）

`token_overlap` 的存在是为了在 chunk 边界做 `blend()`，让时间上连续。
`overlap=0` ⇒ **没有混合窗口** ⇒ 可能出现**可见的时域接缝**。

这**不是"舍入差异"**，而是**输出质量的实质改变**。按合同这远超"可测数值变化"，
必须明确上报为**可能影响画质**的改动。**我不建议走这条。**

### 所以唯一现实的候选是 `clip_length=16`

- 收益：0.918× → **≈19.9 s**，**刚刚好压线达标**
- 风险：`chunk=4, overlap=1`，保留了边界混合，改动相对温和
- 内存：每 chunk 20 帧 vs 当前 28 帧 ⇒ **内存反而下降**（比 R199 担心的情况好得多）

## 四、教训

**R199 我在没有核对赋值代码的情况下，把一个派生量当成了自由参数，并据此给出了方向相反的建议。**
和 R189（漏了 H=7）是同一类错误：**先算，后核对**。

正确顺序应该是：**先在源码里确认"这个量是谁算的、能不能改"，再算收益。**

## 五、R199 中仍然成立的部分

- ✅ `num_chunks=7`、196 帧、**1.58× 重叠浪费**这一**观测**是对的（虽然算式写错）
- ✅ 浪费发生在 `decode_temporal` 的分块上、pipeline 侧无空间
- ❌ "调 `tokens_chunk_size` 到 9/18 可省 10–18%"——**错的，方向反了**
- ❌ "内存可能不够"——对 `clip_length=16` 而言**内存反而下降**，风险解除
