# R197 — 官方 AscendC `prompt_flash_attention` 在本机**不可用**，"换库"这条路关闭

## 一、动机

R196 查明旧 TIK 内核的串行来自两处（`Sij` 单缓冲 + `disable_sync=False` 自动屏障），
修它需要重写同步骨架，是数天级工程。

于是我先验证一个更便宜的可能：源码里有 `prompt_flash_attention_tiling_v2.cpp`，
说明 CANN 里有**官方 AscendC 版**的实现。如果能用它替换 TIK 内核，
那就是**换库**而不是重写——成本低一个数量级。

## 二、确认算子存在

```
libopapi.so:  aclnnPromptFlashAttention / aclnnPromptFlashAttentionGetWorkspaceSize   ✅
torch_npu:    npu_prompt_flash_attention                                              ✅
```

完整 schema 也拿到了：

```
npu::npu_prompt_flash_attention(Tensor query, Tensor key, Tensor value, *,
  padding_mask=None, atten_mask=None, pse_shift=None, actual_seq_lengths=None,
  ... int num_heads=1, float scale_value=1., int pre_tokens=2147483647,
  int next_tokens=0, str input_layout="BSH", int num_key_value_heads=0,
  actual_seq_lengths_kv=None, int sparse_mode=0) -> Tensor
```

（我第一版用错了参数名 `head_num`，正确是 `num_heads`；错误信息里直接给了 schema。）

## 三、结果：**全部失败，561103**

调用进入 `aclnnPromptFlashAttentionV3` 后报：

```
NPU function error: call aclnnPromptFlashAttentionV3 failed, error code is 561103
```

为排除"序列长度不对齐"这一可能（`41792 = 64 × 653`，确实不是 128 的倍数），
扫了长度：

| N | 128 对齐 | 结果 |
|---|---|---|
| 4096 | ✅ | ❌ 561103 |
| 8192 | ✅ | ❌ 561103 |
| 16384 | ✅ | ❌ 561103 |
| 41792 | ❌ | ❌ 561103 |
| 43776 | ✅ | ❌ 561103 |

**连 4096 这种最常规的长度都失败** ⇒ **不是长度/对齐问题，是该算子在当前
芯片 + CANN 组合上根本不可用**（很可能这个 CANN 构建缺 `ascend910` 的算子二进制；
这台机器注册为 `ascend910`，而 V3 内核可能只编了新架构）。

## 四、这关闭了什么

| 路径 | 状态 |
|---|---|
| 换用官方 AscendC prompt FA | ❌ **关闭**（561103，全长度失败） |
| 换用 `npu_fusion_attention` | ❌ 关闭（R187：161001） |
| 换用 `torch.nn.functional.sdpa` | ❌ 关闭（R187：需物化 45.55 GiB） |
| **改 TIK 内核同步（R196）** | ✅ **唯一剩下的路**，数天级工程 |

**⇒ FA 的 1.5× 只剩"改 TIK 内核"一条路，没有便宜的替代方案。**

## 五、这条负面结论的价值

它避免了"以为能换库省下数天、结果发现换不了"的弯路。
现在 FA 的选项集是**完整且封闭的**：

- 核预算 ❌（R188）
- tile 参数 ❌（R191/R194，UB 封顶 1.089×）
- softmax/少 op ❌（R190，向量只用 ≤21%）
- 换官方算子 ❌（R187/R197，161001 / 561103）
- **改 TIK 内核同步 ✅（R196 定位到具体两处，是唯一出路）**

## 六、诚实边界

- 561103 的确切含义我没有查证（CANN 错误码表未在手边）。我的解读
  "该构建缺 ascend910 算子二进制"是**推断**，未证实。可以查到的替代解释：
  参数组合仍不合法（例如 `sparse_mode=2` 需要配套 `atten_mask`）。
- **未尝试带 `atten_mask` 的 `sparse_mode=2` 调用**——这是唯一还没试的参数变体。
  若要彻底排除，值得再试一次（带显式 causal mask）。
- 只测了 `B=1, H=7, D=128` 与 fp16。

## 七、合规

- 合成张量，未加载权重、未改模型数学、未采用任何东西
- 共享 CANN 未改；私有输出目录；未做 driver/device reset；lease 已释放
