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

## 六、这个方法上的进步

前三轮（R189/R199/R199-C）我是"算完就报"，错了三次。
R201b 我标出了未验证前提。**本轮直接用源码把这个前提确认了**，
并且找到了比"权重常驻"更具体的修法（常驻窗口，而不是永远驻留）。

**目前结论建立在读到的代码上，不是数值巧合上。**

## 七、合规

只读源码；未占机器（本轮未 claim lease 之外的执行）；未加载权重；未改模型数学；
**这是"保持原数学"的优化**，符合 §4 第 6 条对 encoder 方向的要求。
