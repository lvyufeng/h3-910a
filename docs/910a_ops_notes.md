# 910A 算子适配笔记（MiniMax H3 / ComfyUI）

记录时间：2026-11-09。机器：`ts-217`（8×910B 无后缀，按 910A 代际处理），CANN 9.2.0，torch_npu 2.10.0。

## 一句话结论

910A 上 **SDPA、FlashAttention、NanToNum、bf16 softmax 全部不可用**。可行路径是：
`--use-split-cross-attention` + 全程 fp16 + 社区补丁集（`ComfyUI-Ascend910A-Adaptation`）。
这条路径能跑通，但很慢（见文末 baseline）。

## 实测证据

### 1. `torch.nn.functional.scaled_dot_product_attention` 整条不可用

探针 `scripts/probe_npu_sdpa.py`，同一报错覆盖了所有组合：

| 场景 | 结果 |
|---|---|
| `enable_gqa=True`，无 mask | FAIL `can not cast format when output is input` |
| 手动扩头后 SDPA | FAIL 同上 |
| float 加性 mask / bool mask | FAIL 同上 |
| bf16 | FAIL 同上 |

报错码 `ERR01007 OPS feature not supported`。

### 2. `torch_npu.npu_fusion_attention` 也不可用

`aclnnFlashAttentionScore failed`（`FlashAttentionKernelNpuOpApi.cpp`）。
所以既没有 SDPA，也没有融合 flash attention——**只能走 matmul/softmax 的 split 实现**。

### 3. eager 数学本身是对的

`scripts/probe_npu_correctness.py` 对照 CPU（fp16，`max|diff|` 容差 2e-2）：

```
OK matmul 0.0078 | bmm 0.0097 | baddbmm 0.0156 | softmax(fp16) 0.00025
OK silu 0.00097 | gelu 0.0014 | conv2d 0.0135 | group_norm 0.00097 | layer_norm 0.00097
OK 分块 attention 0.0004
BAD exp: max|diff|=42914（fp16 在 |x|>11 时溢出，属预期，不是 bug）
```

结论：**算得慢但算得对**，非常适合当 baseline。

### 4. bf16 softmax 编译失败（黑屏/崩溃的元凶之一）

```
SoftmaxV2 compile failed: Unsupported cast type! src_dtype: bfloat16, dst_dtype: float32
```

触发链：DiT 的 `unet_dtype` 在 NPU 上被判成 bf16 → 整条采样图 bf16 → `SoftmaxV2` 编译失败。
**两个 LoRA 权重本身也是 BF16**（`fl2v_turbo_4step` 与 `turbo_v4_step600_ema` 均为 `{'BF16': ...}`），
合并后进一步把计算拖进 bf16。

处置：
1. LoRA 离线转 fp16（`scripts/convert_lora_fp16.py`），工作流改指向 `..._fp16.safetensors`；
2. `sd.py` 里把 `unet_dtype` 在 Ascend 上钉成 fp16（补丁 ⑤）；
3. 启动加 `--fp16-unet --fp16-text-enc`（`--force-fp16` 压不住 int8 量化模型）。

修好后日志出现 `model weight dtype torch.float16, manual cast: None`，`SoftmaxV2` 不再报错。

## 采用的补丁集

来源：`ComfyUI-Ascend910A-Adaptation` 的《MiniMax-H3 昇腾 910A 完整部署使用指南》。
落地脚本：`artifacts/design/patches/apply_910a_comfy_patches.py`（按内容锚定，可 `--revert`）。

| # | 文件 | 改动 | 解决 |
|---|---|---|---|
| ① | `comfy/model_management.py` | `is_oom()` 识别 NPU OOM | attention 重试/清缓存失效 |
| ② | `comfy/ldm/modules/attention.py` | `attention_split` 分块向上取整 | 长序列一次性分配 ~26GB 注意力矩阵 → OOM |
| ③ | `comfy/sd.py` | 量化 UNet 的 manual_cast 尊重 `unet_dtype` | 强转 bf16 |
| ④ | `comfy/sd.py` | Ascend 上 `unet_dtype` 钉 fp16（本仓库补充） | bf16 softmax 编译失败 |
| ⑤ | `scripts/sitecustomize.py` | 关 `allow_internal_format`；`nan_to_num` 等价实现 | conv3d 布局 + NanToNum 缺失 |

`sitecustomize.py` 只在 `H3_910A_RUNTIME_PATCH=1` 时生效，由启动脚本设置。

## 启动参数

```bash
H3_910A_RUNTIME_PATCH=1 bash runtime/h3-instance-control.sh start a -- \
  --use-split-cross-attention --dont-upcast-attention --fp16-unet --fp16-text-enc
```

- `--use-split-cross-attention`：910A 上唯一可用的 attention 实现（flash/sage/ck 都不可用）
- `--dont-upcast-attention`：没有 fp32 加速，upcast 只会更慢
- `--fp16-unet/--fp16-text-enc`：唯一能编译的精度

## 性能 baseline（eager，单卡 0 号卡）

608×352、107 帧、4 步 turbo LoRA、`npu_count=1`：

| 阶段 | 耗时 |
|---|---|
| 4 步采样 | 4 分 26 秒（66.7 秒/步） |
| 整条 prompt（含 VAE 加载/解码/编码） | 16 分 54 秒 |
| 峰值 HBM | 约 21–30 GB（32GB 卡） |

对比参考：同一台机器多卡（`npu_count=4`）本来是设计路径，但会走自研 HCCL
`HcclAllGatherV`，在 CANN 9.2.0 上返回 **HCCL error 5 = `HCCL_E_NOT_SUPPORT`**，目前不可用。
所以现阶段只能用单卡。

## 已知坑

1. **单卡内存很紧**：608×352 曾经 OOM，主要就是补丁②之前的分块 bug；补丁②之后 21GB 左右。
2. **`torch.npu.config.allow_internal_format` 必须关**，否则视频 VAE 的 `reflection_pad3d` 崩。
3. **`nan_to_num` 替换会把 NaN 变 0**：如果输出整片全黑，先怀疑上游算出了 NaN，而不是 VAE 坏了。
4. **`sitecustomize.py` 依赖 PYTHONPATH**：只有启动脚本把 `scripts/` 放进 `PYTHONPATH` 时才生效。
5. ComfyUI 日志里 `hccl_build_version=unknown` 属探针字段问题，不代表 HCCL 缺失。

## 参考链接

- [ComfyUI-Ascend910A-Adaptation](https://gitcode.com/cssyyy_hpccai/ComfyUI-Ascend910A-Adaptation)（补丁来源，含 910a_minimax.md）
- [minimax-h3-int8](https://ai.atomgit.com/xujiashuai/minimax-h3-int8)（Ascend 910 双卡 H3 部署）
- [ascendai/vllm-omni minimax-h3 镜像](https://hub.docker.com/r/ascendai/vllm-omni)（vLLM-Omni 的 H3 昇腾镜像）