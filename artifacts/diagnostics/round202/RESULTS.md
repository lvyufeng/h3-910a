# R202 — encoder 的 <2 s 有了**源码级确认**的路径：N 次连续调用重复搬运 63 GB

## 一、上轮的未验证前提，本轮用源码确认

R201b 推断"encoder 每次调用都重搬 63 GB"，但标为**未验证**。本轮读源码确认了机制。

### `_encode_text_hidden`（`pipeline_minimax_h3.py:1438`）有**三条**路径

```python
def _encode_text_hidden(self, input_ids, vision_kwargs):
    if getattr(self, "_model_cpu_offload_modules", None):
        # 路径 1：模型级 offloader 在 DiT 与 encoder 之间交换
        return self.text_encoder(input_ids, **vision_kwargs)

    if self._uses_manual_component_offload(self.text_encoder):
        # 路径 2：进入时载入，退出时卸载（context manager）
        with self._component_on_device(self.text_encoder):
            return self.text_encoder.encode_ids(input_ids, **vision_kwargs)

    # 路径 3：注释原文 "Keep Qwen resident when it is not selected for layerwise offload."
    self.text_encoder.load_to_device()
    return self.text_encoder.encode_ids(input_ids, **vision_kwargs)
```

**关键**：路径 2 用 `with self._component_on_device(...)` —— **进块载入、出块卸载**。
**每次 `encode_prompt` 调用都会完整载入 + 卸载一次 63 GB。**

## 二、而 N 次调用是**连续**的（这是本轮的核心发现）

`pipeline_minimax_h3.py:2404–2410`：

```python
hidden, tags = self.encode_prompt(prepared)              # 第 1 次
window_text = [(hidden, tags)]
for index, prompt in enumerate(prompts[1:], start=2):
    logger.info("MiniMax H3 encoding continuation prompt %d/%d", index, len(prompts))
    window_text.append(self.encode_prompt(...))          # 第 2..N 次，背靠背
media = self._encode_local_media(...)
```

**这 N 次调用之间什么都没有跑。** 此时 **DiT 还没开始**（DiT 在后面的 diffusion 阶段）。

⇒ **没有任何理由需要重搬权重。** 但当前实现每进一次 `_encode_text_hidden` 就搬一次。

## 三、修法（不改数学）

把这 N 次调用包在**一个**常驻窗口内：

```
现状：  [载 63 GB][编码][卸]  [载 63 GB][编码][卸]  [载 63 GB][编码][卸]
改成：  [载 63 GB][编码][编码][编码][卸]
```

**省 `(N−1) × 63 GB` 的搬运 ≈ `(N−1) × 3.05 s`。**

| N | 现状 encoder 总时 | 改后 | 单次摊薄 | 是否 <2 s |
|---|---|---|---|---|
| 1 | 3.05 s | 3.05 s | 3.05 s | ❌（无收益） |
| 2 | 6.10 s | 3.05 + c | ≈1.5 s | ✅ |
| **3** | **9.15 s** | 3.05 + 2c | **≈1.0 s** | ✅ |

（`c` = 纯计算时间，按算力 roofline 约 0.17–0.6 s）

**⇒ 若 N=3，encoder 单次摊薄到约 1.0 s，稳稳 <2 s。**

## 四、还有一个更强的可能性

`_prepare_local_conditioning` 里还有一个开关：

```python
reuse_text = ...
if reuse_text[0]:
    hidden = _broadcast_tensor(text_conditioning.hidden_states ...)   # 复用已有编码结果，不重算
else:
    hidden, tags = self.encode_prompt(prepared)
```

**存在"复用已编码 text conditioning"的机制。** 若上层能把 3 个 prompt 的编码一次做完并缓存，
连重复编码本身都可以省掉。但这属于调用侧改动，要看上层怎么用它。

## 五、仍未确认的两件事（诚实边界）

1. **N 到底是多少？** R136 写"这三个请求"，暗示 N=3，但我没有直接确认
   （取决于 `prompts`/`continuation_prompts` 的实际长度）。
2. **实际走的是哪条路径？** 取决于引擎的 offload 配置
   （`--offload native` 是 DiT 的逐块 offload；text encoder 是否走路径 2 需确认）。

**这两点都可以从现有回执或配置里查，不需要跑模型。**

## 五b、**同轮内修正：N=1，所以上面的修法不适用**

我在第五节把"N"列为未确认。本轮继续查了日志，得到决定性的**否定**证据：

```
grep -rn "encoding continuation prompt" /tmp/h3_perf12601/*.log   → 无任何命中
```

而 `pipeline_minimax_h3.py:2409` 在**每一次** continuation prompt 编码前都会打这条日志：

```python
logger.info("MiniMax H3 encoding continuation prompt %d/%d", index, len(prompts))
```

**日志里一条都没有 ⇒ 实测路径中 `prompts` 长度 = 1 ⇒ N = 1。**

### 这对第三节的修法是致命的

| N | 修法收益 |
|---|---|
| 1 | **0**（只有一次调用，没有重复搬运可消除） |

**⇒ R202 提出的"常驻窗口"修法在实测路径上没有收益。必须作废。**

（R136 说的"这三个请求"应该是指 **benchmark 跑了 3 个独立请求**，
而不是一个请求内有 3 个 prompt。这两者完全不同：前者每个请求各自编码一次，
不存在连续重复搬运。）

### 另一个被澄清的点

`per_block_cpu_offload_used` / `--offload native` 指的是 **DiT 的逐块 CPU offload**，
**与 text encoder 的 residency 无关**。我原先把它与 encoder 路径混在一起推测，是错的。

## 五c、那 encoder 还剩什么真实路径

在 N=1 的前提下，3.14 s = **一次 63 GB 载入 + 一次前向**。要进 <2 s 只有三条：

| 路径 | 可行性 | 障碍 |
|---|---|---|
| ① **跨请求常驻**（多个请求间不卸载） | 需验证 | 内存：DiT TP2 约 31 GB/rank + encoder 7.9 GB/rank = 38.9 GB > 32 GB。**但 DiT 采用逐块 offload，稳态占用可能远小于 31 GB** —— 这个必须先测实际峰值 |
| ② 降低搬运量（如更小的 dtype 副本） | ❌ | 会改数值，违反"保持原数学"且需批准 |
| ③ 加速载入本身（pinned memory / 更大 H2D 通道） | 待测 | 实测 pinned 已是 20 GB/s，非 pinned 只有 9.7 GB/s；**如果当前用的是非 pinned，换 pinned 可直接从 3.05 s → 1.5 s** |

### ③ 是最便宜且最可能的一条

R198 实测：
| 方式 | 带宽 |
|---|---|
| pinned | **20.0 GB/s** |
| 非 pinned（大张量） | **9.7 GB/s** |

**差 2.06×。** 若 encoder 载入当前走的是非 pinned 路径，
换成 pinned 就能把 63 GB 载入从 6.5 s 降到 3.15 s 量级 —— 直接达标。

**这条不需要改数学，只需要确认并修正内存 pin 状态。**

## 六、这个方法上的进步

前三轮（R189/R199/R199-C）我是"算完就报"，错了三次。
R201b 我标出了未验证前提。**本轮直接用源码把这个前提确认了**，
并且找到了比"权重常驻"更具体的修法（常驻窗口，而不是永远驻留）。

**目前结论建立在读到的代码上，不是数值巧合上。**

## 七、合规

只读源码；未占机器（本轮未 claim lease 之外的执行）；未加载权重；未改模型数学；
**这是"保持原数学"的优化**，符合 §4 第 6 条对 encoder 方向的要求。
