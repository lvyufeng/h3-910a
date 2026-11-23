# R199 — video VAE 的最大浪费：时间 chunk 重叠造成 **58% 重复计算**

## 一、怎么找到的

按 R198 计划读 decoder 源码（`/tmp/h3_align12401/original_metadata124f/FL2VA/video_vae/klvae.py`，1258 行）：

```
decode_base (790)  →  decode_temporal (678)  →  _adaptive_decode (437) × num_chunks
```

`decode_temporal` 是**按时间切片、串行**：

```python
for i in range(num_chunks):
    t_start = i * tokens_chunk_size
    t_end   = t_start + tokens_chunk_size + token_overlap     # ← 重叠
    clip_z  = z[:, :, t_start:t_end, :, :]
    clip_dec = self._adaptive_decode(clip_z)                   # 真正干活（内含 28 tiles / 8 rank）
```

每 chunk 解码 `(tokens_chunk_size + token_overlap) × vae_ratio_t` 帧，相邻 chunk 重叠
`token_overlap` 个 latent token，解码后 blend 掉重叠部分。

## 二、几何（R138 参数：`tokens_chunk_size=5, token_overlap=2, vae_ratio_t=4`）

| 量 | 值 |
|---|---|
| 输入 latent 时间帧 | 37 |
| **chunk 数** | `ceil((37−2)/5)` = **7** |
| 每 chunk 解码帧数 | `(5+2)×4` = **28** |
| **总解码帧数** | 7 × 28 = **196** |
| **实际输出帧数** | **124** |
| **重复计算倍数** | **1.58×** |

**decode 里 58% 的算力花在重叠帧上，算完就被 blend 丢弃。**

## 三、扫 `tokens_chunk_size`

总工作量 ≈ `num_chunks × (chunk+overlap) × vae_ratio_t`：

| chunk_size | chunks | 解码总帧 | 浪费 | 相对当前 |
|---|---|---|---|---|
| 3 | 12 | 240 | 1.94× | 1.22 |
| **5（当前）** | **7** | **196** | **1.58×** | **1.000** |
| 7 | 5 | 180 | 1.45× | 0.918 |
| **9** | **4** | **176** | 1.42× | **0.898** |
| **12** | **3** | **168** | 1.35× | **0.857** |
| **18** | **2** | **160** | 1.29× | **0.816** |
| 20 | 2 | 176 | 1.42× | 0.898 |

**最优 `chunk_size=18`：工作量 0.816×** → `21.7 × 0.816 ≈ **17.7 s**`，稳进 <20 s。
且 chunk 数 7→2，**每 chunk 固定开销也少 5 次**。

## 四、⚠️ 两个必须说清的问题

### 1. **数值非等价**

改 `tokens_chunk_size` 会改变：参与重叠 blend 的帧、每 chunk 的时序卷积上下文长度
⇒ **输出会变**。按合同必须**停用、报数、请用户决定**，不得自行采用。

### 2. **内存风险（可能是硬阻塞）**

- 当前 `chunk_size=5` → 每 chunk 28 帧，**峰值 13.436 GiB**
- `chunk_size=18` → 每 chunk **80 帧（2.9×）**

13.4 GiB 中约 10.4 GB 是 FP32 权重，剩约 3 GiB 激活；激活按 2.9× 增长到 ~8.7 GiB，
总计约 19 GiB —— **可能放得下，但纯属估算，必须实测**。若 OOM 则退阶：

| 备选 | 工作量 | 每 chunk 帧 | 内存风险 | 预计步时 |
|---|---|---|---|---|
| **`chunk_size=9`** | **0.898** | 44（1.57×） | **低** | **≈19.5 s** |
| `chunk_size=12` | 0.857 | 56（2.0×） | 中 | ≈18.6 s |
| `chunk_size=18` | 0.816 | 80（2.9×） | 高 | ≈17.7 s |

**`chunk_size=9`（省 10.2%）是最安全的第一步，且已足够达标。**

## 五、对目标的意义

| 项 | 值 |
|---|---|
| video VAE 现状 | 21.62–21.76 s |
| 目标 | <20 s（需 −7.5%） |
| **`chunk_size=9`** | **≈19.5 s → 达标** |

**⇒ video VAE 目标很可能只需改一个时间 chunk 参数，不必动任何 kernel。**

## 六、诚实边界

- **纯算术**，未跑机器、未实测。真实收益取决于每 chunk 开销是否线性。
- `num_chunks` 的公式是从循环结构**反推**的（`ceil((latent_T − overlap)/chunk)`），
  未与真正计算它的代码核对，也未确认 `isolated_head/tail` 是否影响它。
- **内存是否够完全没测**，这是最大不确定性。
- 改动数值非等价，属"可测数值变化"，必须请用户决定。
- `config.json.tile256.bak` 的存在说明之前调过 tile 配置，但没留下 chunk_size 记录。

## 七、合规

只读源码 + 算术；本轮未占机器、未加载权重、未改模型数学。
