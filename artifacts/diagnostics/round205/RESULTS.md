# R205 — 找到 encoder 的真正开关（**一个配置项**），并推翻我 R203/R204 的 pinned 担忧

## 一、日志给了决定性证据

`/tmp/h3_perf12601/NPU_full_generation126b.log`：

```
[layerwise_backend.py:363] Enabled rank-local layerwise offload for encoder text_encoder
                           (77 blocks across 2 stacks)
[layerwise_backend.py:454] Applying layer-wise offloading on ['transformer']
```

**⇒ text_encoder 和 transformer（DiT）都开了逐层 offload。**

这**证实**了 R203 的诊断：encoder 走的是 `_uses_manual_component_offload` 为真的路径
（进入时载入、退出时卸载），所以每次 `encode_prompt` 都要重新流式载入 63 GB。

## 二、但这**推翻**了我 R203/R204 的核心担忧

我之前算"DiT 62 GB + encoder 63 GB 都要 pinned = 125 GB > 94.3 GiB 上限，差 22 GiB"。

**这个模型是错的。** 逐层 offload 的含义是：**任何时刻只有一个 block 的权重在 pinned 缓冲里**，
不是全部 125 GB。

| 项 | 我之前（错） | 实际 |
|---|---|---|
| 每 block 大小 | — | `63 GB / 77 ≈ 0.82 GB` |
| pinned 需求 | 125 GB | **约 1–2 GB（单个 block + 预取缓冲）** |
| 对 94.3 GiB 上限 | ❌ 差 22 GiB | ✅ **远远够** |

**⇒ pinned 上限从来不是阻塞点。R203 第四d节与 R204 的"方案 B 才勉强可行"整套推演
建立在错误前提上，应作废。**

（保留的部分：encoder 时间 ~99% 是搬运、算力 roofline 0.17 s、每次载入 3.15 s 硬下限——
这几条仍然成立，且与日志一致。）

## 三、真正的开关：**一个配置项**

`offloader/config.py`：

```python
DIT_COMPONENT = "dit"
TEXT_ENCODER_COMPONENT = "text_encoder"
OFFLOAD_COMPONENTS = frozenset({DIT_COMPONENT, TEXT_ENCODER_COMPONENT})
DEFAULT_OFFLOAD_COMPONENTS = frozenset({DIT_COMPONENT})      # ← 默认只 offload dit！

def should_offload_component(config, component):
    resolved = resolve_offload(config)
    if resolved.strategy not in {LAYER_WISE, DISTRIBUTED_LAYER_WISE}:
        return False
    return resolved.offloads(component)                       # ← 看 components 集合
```

而 `pipeline_minimax_h3.py:1438`：

```python
if self._uses_manual_component_offload(self.text_encoder):    # = should_offload_component(...)
    with self._component_on_device(self.text_encoder):        # 载入 → 跑 → 卸载
        ...
# 否则走这里：
self.text_encoder.load_to_device()                            # 注释原文 "Keep Qwen resident"
```

**⇒ `diffusion_offload_config.components` 从 `["dit","text_encoder"]` 改成 `["dit"]`，
encoder 就变成常驻，不再逐层流式搬运。**

**这是配置改动，不是代码改动，更不是内核改动。**

## 四、改后的预期

| 项 | 现状 | 改后 |
|---|---|---|
| encoder 模式 | 逐层 offload，每请求流式载入 63 GB | 常驻（`.to()` 一次） |
| device 占用 | 流式，峰值低 | **7.33 GiB/rank**（需确认，见下） |
| 每请求 encoder | **3.14 s**（≈63 GB / 20 GB/s） | **≈0.5 s**（纯算力） |
| 目标 | <2 s | ✅ **达成** |

## 五、必须核对的三个点（不能再假设）

1. **device 内存够不够**：encoder 常驻 7.33 GiB/rank + DiT 逐层 offload 的块。
   此前用 R170 的 11.64 GiB 峰值算过合计 18.97 GiB < 32，但那是 DiT **不带** encoder 常驻时的峰值，
   **需要实测确认**。
2. **`.to()` 是否真的释放 host 副本**：PyTorch 的 `Module.to()` 是原地移动参数，
   理论上不留 CPU 副本；但若实现里另存了 state_dict 或 pinned master，就不一定。
   **这决定 host RAM 是否真的省下来**（不过现在知道 pinned 本来就不是瓶颈，这条重要性下降）。
3. **DiT 是否依赖 encoder 的 offload 腾出的内存**：若 DiT 逐层 offload 的峰值 +
   encoder 常驻超出 32 GiB，就会 OOM。**这是唯一真正的风险。**

## 六、这是本轮最有价值的产出

**从"需要写代码/改内核"降级为"改一个配置项"，而且推翻了一个我自己造出来的伪瓶颈（pinned 上限）。**

顺着这条，encoder 的 <2 s 变成一个**低风险、不改数学的配置实验**：
把 `components` 去掉 text_encoder，跑一次，看 encoder 时间与峰值显存。

## 七、合规

只读源码与日志；**本轮未占机器**；未加载权重；未改模型数学。
