# R334 — encoder 常驻成功并量出真实构成：H2D 只占 0.39 s，**2.8 s 是计算**

## 一、最终可用的常驻实现（代码级、不改数学）

`resident_patch.py`（已入库，opt-in `H3_ENCODER_RESIDENT=3`）：

1. 首轮 `encode_prompt` 正常执行；
2. 之后对 77 个 block 逐个 `hook.prefetch_layer(non_blocking=False)` → 真正物化到 device；
3. 非 block 状态用 `move_non_block_state_to_device(target, groups, torch.device('npu', torch.npu.current_device()))`；
4. 仅对这些 hook 实例（`hook._h3_resident=True`）短路 prefetch/offload —— **DiT 的 hooks 不受影响**（这是最后一个 bug 的修法）。

## 二、实测（8 rank 一致）

```
[RESIDENT4] resident=True blocks=77/77 restore=0.41–0.55 s
            free_after=85.3 GiB   hbm=8.4 GiB   leftover cpu params=0
[STAGE] 首轮 encoder  = 63.16 s
[RESIDENT4] warm encode = 2.805 s   hbm=8.8 GiB     ← 第二趟同 engine 内
```

- device-mismatch 错误：**0**；
- 常驻显存：**8.4–8.8 GiB/rank**（每卡 32 GB，余量充足）；
- host 余量：**85.3 GiB**（安全，未再出现 R331/R332 的内存事故）。

## 三、结论：encoder <2 s 的真实障碍是**计算**，不是搬运

| 版本 | warm encode | 说明 |
|---|---|---|
| 基线（offload 每次重流） | **3.196 s** | R326 两趟实测 |
| 常驻（本轮） | **2.805 s** | 权重已在 device，零 H2D |

⇒ **H2D 只占 0.39 s（12%）**；剩下 **2.805 s 是纯计算**（视觉塔 + 文本塔 77 个 block 前向）。

也就是说：
- 之前"63 GB ÷ 3.196 s ≈ 19.7 GB/s"的带宽推断**不成立**（那只是巧合）；
- 要 <2 s，需要 **1.4× 计算加速**（更优 kernel/更小 padding/更好 batch），
  已属于模型计算优化范畴，**不是内存或部署配置能解决的**。

## 四、交付判断

- 常驻补丁作为**可选开关**保留（`H3_ENCODER_RESIDENT=3`），不默认启用；
- 收益：warm encoder **−0.39 s（−12%）**，代价 **+8.4 GiB HBM/rank**；
- 是否采纳需要你决定（数值上应位级一致，但采纳前必须跑一遍输出比对）。

## 五、状态

- 交付不变：**16.34 s/step（−8.9%）**，`H3_PERF_COMBO=1` 默认 OFF；
- 目标：encoder <2 s **仍未达成**（现 2.805 s，差 1.4×，需计算优化）；
- video VAE warm 20.672 s（差 3.4%，需数值批准改 tile）；
- 本轮生成仍在跑第二趟（用于观察整轮 wall 收益）。
