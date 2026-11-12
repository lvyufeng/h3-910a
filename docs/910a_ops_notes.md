# 910A 算子适配笔记（MiniMax H3）

记录时间：2026-11-09。机器：`ts-217`（8×910B 无后缀，按 910A 代际处理），CANN 9.2.0，torch_npu 2.10.0。

本文只记录**硬件层面的算子可用性**，这些结论与上层框架无关，是当前 vLLM-Omni
路线和将来任何路线的共同前提。工程实现与性能优化过程见
[vllm_omni_910a.md](vllm_omni_910a.md)。

## 一句话结论

910A 上 **SDPA、FlashAttention、fp16 NanToNum、bf16 softmax 全部不可用**。
注意力只能走 matmul/softmax 的分块实现；精度必须用 fp16
（bf16 的 matmul 只有 fp16 的 1/168 算力）。

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

此外 `npu_fusion_attention`、`npu_prompt_flash_attention` 均因 SoC 版本门禁失败
（`EZ1009 SoC version ascend910 verification failed`），Triton/bishengir 也没有 910A
的 target（`Cannot find option named 'Ascend910B'`，支持列表从 `Ascend910B1` 起）。

### 2. eager 数学本身是对的

`scripts/probe_npu_correctness.py` 对照 CPU（fp16，`max|diff|` 容差 2e-2）：

```
OK matmul 0.0078 | bmm 0.0097 | baddbmm 0.0156 | softmax(fp16) 0.00025
OK silu 0.00097 | gelu 0.0014 | conv2d 0.0135 | group_norm 0.00097 | layer_norm 0.00097
OK 分块 attention 0.0004
BAD exp: max|diff|=42914（fp16 在 |x|>11 时溢出，属预期，不是 bug）
```

结论：**算得慢但算得对**，适合当 baseline / 回退实现。

### 3. bf16 softmax 编译失败

```
SoftmaxV2 compile failed: Unsupported cast type! src_dtype: bfloat16, dst_dtype: float32
```

`SoftmaxV2` 只接受 fp16/fp32，不接受 bf16。这也是后续所有路径都必须绕开 bf16 的原因之一。

### 4. `nan_to_num` 没有 fp16 kernel

```
No supported Ops kernel and engine are found for [NanToNum542], optype [NanToNum]
```

实测矩阵：

| dtype | `torch.nan_to_num` |
|---|---|
| fp16 | **FAIL（无 kernel）** |
| bf16 | OK |
| fp32 | OK |

处置思路是让调用方不再对 fp16 张量调用它，而不是替换算子本身。

### 5. 编译失败会伪装成"算子不支持"

CANN 的 AscendC JIT 编译器不自动加系统 C++ 头文件目录，
`RmsNorm` 之类内置算子会报 `no supported kernel`，真实原因却是
`fatal error: 'cstdint' file not found`。设置

```bash
export CPLUS_INCLUDE_PATH=/usr/include/c++/11:/usr/include/aarch64-linux-gnu/c++/11
```

之后正常。**排查"算子不支持"之前，先确认不是编译环境问题。**

### 6. 并行编译子进程会打爆主机

CANN 默认每个算子起 128 个 fatbin 编译子进程，8 个 worker 同时编译时会
oversubscribe 主机，表现为 `TBE Subprocess ... main process disappeared`。
`export TBE_PARALLEL_COMPILER=4` 可以解决。

## 已知坑

1. **视频 VAE 编码（encode）路径要注意 `torch.npu.config.allow_internal_format`**：
   conv3d 默认输出 NDC1HWC0/ACL_FRACTAL_Z_3D 布局，`reflection_pad3d` 会崩。
   当前只做文生视频（decode 路径），不触发该问题；将来接图生/视频生视频时需要关掉。
2. **`nan_to_num` 替换会把 NaN 变 0**：如果输出整片全黑，先怀疑上游算出了 NaN，
   而不是 VAE 坏了。
3. **残留进程会伪装成 OOM**：进程名显示为 `vLLM-OmniDiff`，`pkill -f vllm-omni`
   抓不到，要先 `npu-smi info -t proc-mem -i N` 找到 pid 再杀，并清
   `/dev/shm/sem.mp-*`。
4. 日志里 `hccl_build_version=unknown` 属探针字段问题，不代表 HCCL 缺失。

## 参考链接

- [minimax-h3-int8](https://ai.atomgit.com/xujiashuai/minimax-h3-int8)（Ascend 910 双卡 H3 部署）
- [ComfyUI-Ascend910A-Adaptation](https://gitcode.com/cssyyy_hpccai/ComfyUI-Ascend910A-Adaptation)（910A 算子补丁的另一条独立路线，本仓库未采用）