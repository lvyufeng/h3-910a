# vLLM-Omni 在 910A 上的适配记录

记录时间：2026-11-09。机器 `ts-217`（8×NPU，`npu-smi` 显示 `910B` 无后缀，即 **910A 代际**）。

## 结论

vLLM-Omni **可以**在 910A 上装起来并进入服务初始化；真正缺的是注意力 kernel。本仓库补了一个
matmul/softmax 的 910A backend，已通过数值验证。

## 版本矩阵（实测可用）

| 组件 | 版本 | 说明 |
|---|---|---|
| CANN | 9.2.0 | 官方 recipe 要求 9.2.1，实测 9.2.0 可加载 |
| torch | 2.10.0+cpu | **必须**，vllm-ascend 0.30.0 的硬约束 |
| torch_npu | 2.10.0.post2 | 官方建议 post7，post2 实测可用 |
| vLLM | 0.30.0 | 用 `VLLM_TARGET_DEVICE=empty` 源码构建 |
| vllm-ascend | 0.30.0 | `Platform plugin ascend is activated` |
| vllm-omni | **v0.30.0 tag** | main 分支不兼容 vllm 0.30 |

### 三个版本坑

1. **PyPI 的 `vllm==0.30.0` 依赖 torch 2.13**，与 vllm-ascend 0.30.0 的 torch 2.10 冲突。
   解法：`VLLM_TARGET_DEVICE=empty VLLM_VERSION_OVERRIDE=0.30.0 pip install -v --no-binary vllm "vllm==0.30.0" --extra-index-url https://download.pytorch.org/whl/cpu/`，
   装完再把 torch 钉回 `2.10.0`。
2. **vllm-omni main 分支要 `compute_mm_prefix_ranges`**，vllm 0.30 只有
   `compute_mm_prefix_range_tensor`。必须用 **v0.30.0 tag**。
3. **tarball 没有 git 元数据**，setuptools-scm 会把版本算成非法的 `dev+npu`。
   解法：`SETUPTOOLS_SCM_PRETEND_VERSION=0.30.0`。

### 其它环境问题

- 启动命令是 **`vllm-omni serve`**（入口 `vllm_omni.entrypoints.cli.main:main`），不是 `vllm serve --omni`。
- **缺 `libGL.so.1` 会让 GE/TBE 初始化失败**，报错形如
  `GEInitializeV2 failed` / `tbe_op_adapter failed`，极易被误判成算子不兼容。
  装 `libgl1` 即解决。教训：910A 上的 GE/TBE 报错先查系统动态库。
- CANN 必须 `source set_env.sh`，否则 `import torch_npu` 报 `libhccl.so: cannot open shared object file`。

## 910A 注意力 backend

### 为什么必须自己写

| kernel | 910A 上的结果 |
|---|---|
| `torch.nn.functional.scaled_dot_product_attention` | `ERR01007 can not cast format when output is input`（所有形状/mask/dtype 组合） |
| `torch_npu.npu_fusion_attention` | `aclnnFlashAttentionScore failed` |
| MindIE-SD 的 LaserAttention / RainFusion | 未安装，官方仅支持 A2/A3 |

而 eager 的 matmul/softmax 数值是对的（对 CPU 参考 `max|diff| < 1.6e-2`），所以走它。

### 实现

文件：`vllm_omni/diffusion/attention/backends/h3_910a_attn.py`
（本仓库副本：`artifacts/design/patches/h3_910a_attn.py`）

- `H3910AAttentionImpl(SDPAImpl)`：只覆盖 `forward_npu`，其余（mask 归一化、构造参数、
  non-NPU 分支）继承 SDPA 实现。
- `H3910AAttentionBackend(SDPABackend)`：`get_impl_cls()` 返回上面的 Impl。
- 注册：`registry.py` 末尾把 `DiffusionAttentionBackendEnum.TORCH_SDPA` 覆盖到该 backend，
  这样 `--diffusion-attention-backend TORCH_SDPA` 就选中它。
- 计算：query 按块（`VLLM_OMNI_910A_ATTN_CHUNK`，默认 2048）分块做
  `q·kᵀ → scale → mask/causal → softmax(fp32) → ·v`，块大小决定峰值 HBM。
- GQA：K/V 按 `repeat_interleave` 扩到 Q 的头数。
- mask：自动搬到 score 所在设备（NPU 上 CPU mask 会触发
  `tensor's storage is not an NPUStorageImpl`）。

### 验证

`artifacts/design/tests/test_910a_attn_backend_npu.py`（直接调 Impl，不需要权重）：

```
7/7 passed
mha_basic 0.00081 | mha_long_chunked(4096,chunk1024) 0.00022 | gqa_8x2 0.00079
causal 0.00119 | masked_k 0.00071 | batched 0.00051 | single_chunk_tiny 0.00099
```

## 启动命令

```bash
source /home/mseco/Ascend/cann-9.2.0/set_env.sh
export VLLM_WORKER_MULTIPROC_METHOD=spawn
vllm-omni serve /home/mseco/models/MiniMax-H3 \
  --omni --host 0.0.0.0 --port 9198 --trust-remote-code \
  --task-type fl2va --num-gpus 8 --usp 8 --ring 1 \
  --text-encoder-tp-size 8 --vae-parallel-mode tile --vae-use-tiling \
  --vae-patch-parallel-size 8 --enable-distributed-layerwise-offload \
  --diffusion-attention-backend TORCH_SDPA
```

## 仍待验证

- 权重完整加载（FL2VA 约 128GB，BF16）
- 8 卡 USP 的 HCCL 集合通信（910A 上自研 `HcclAllGatherV` 曾返回 `HCCL_E_NOT_SUPPORT`）
- 端到端 `/v1/videos/sync` 出片与耗时基线
---

# 第二轮：从「起不来」到「能跑到第 48 秒」

## 打通的链路

```
vllm-omni serve (8 卡, USP8, TE TP8, fp16)
  → 8/8 worker ready, :9198 Application startup complete
  → POST /v1/videos/sync  (multipart, 必须带 aspect_ratio, seconds ∈ [4,15])
  → 文本编码 → DiT 4 步 → 在 InplaceAddRmsNorm 处失败
```

内存：DiT 走 DLO AllGather（mmap 共享），TE 走 rank-local + TP8
→ 主机内存稳定在 240GB（此前 rank-local 全量是 747GB 被打爆）。

## 这一轮踩的坑（按出现顺序）

| # | 现象 | 真因 | 解法 |
|---|---|---|---|
| 1 | `ModuleNotFoundError: fla_npu` | vllm-ascend 无条件导入 Qwen3.5 GDN 算子包，PyPI/Ascend 源都没有 | 写 stub 包（H3 不用 GDN） |
| 2 | 启动卡死在 `Distributed layerwise offload for submodule` | DLO 默认 ALLGATHER | 改成 rank-local，或显式配置 |
| 3 | `NPU out of memory` (31.9/31.97 GiB) | 关掉 DLO 后全量权重上卡 | 必须开 DLO |
| 4 | 主机内存 747/754GB，`TBE Subprocess ... main process disappeared` | 8 rank 各存一份 156GB | DiT 用 allgather 共享，TE 用 rank-local+TP8 |
| 5 | `Text encoder cannot use DLO AllGather` | 上游限制 | TE 显式设 `rank-local` |
| 6 | `distilled LoRA requires the ordinary loader` | LoRA 让 mmap 快路径失效 | **把 LoRA 离线融合进权重**（`scripts/merge_distill_lora.py`） |
| 7 | 启动超时 | `--stage-init-timeout` / `--init-timeout` 默认 300/600s | 都设 2400s |
| 8 | `HcclAllreduce ... parameter error` | 非连续 / bf16 输入 | coordinator 里 `.contiguous()` + bf16→fp16 |
| 9 | `RmsNorm` 编译失败 | **`fatal error: 'cstdint' file not found`**（CANN AscendC 编译器没带系统 C++ 头路径） | `CPLUS_INCLUDE_PATH=/usr/include/c++/11:/usr/include/aarch64-linux-gnu/c++/11` |
| 10 | `can not cast format when output is input` (ERR01007) | H3 有多处直接调 `F.scaled_dot_product_attention`，绕过了 backend 注册表 | 全局 monkeypatch `F.scaled_dot_product_attention` → 分块 matmul |
| 11 | `No supported Ops kernel for [InplaceAddRmsNorm]` | 该融合算子 910A 不存在 | **待办**：eager 回退 |

第 9 条是最值钱的一条：**CANN 的算子编译失败会伪装成「算子不支持」**，实际是缺系统头文件路径。

## 离线融合 LoRA

`minimax_h3_t2va_flashgen_4step_v1.0_768p_bf16.safetensors` 的元数据：
`lora_alpha=64, lora_rank=64`（scale=1.0），`base_schedule=1.0,0.7,0.4,0.15,0.0`（4 步 sigma）。

融合后放在 `/home/mseco/models/MiniMax-H3-4step`（transformer 实体 62GB，其余符号链接），
服务时不再传 `--lora-path`，mmap 快路径才能生效。

## 当前启动命令

```bash
MODEL=/home/mseco/models/MiniMax-H3-4step LORA="" DTYPE=half \
OFFLOAD_CONFIG='{"mode":"layer","components":["dit","text_encoder"],"layer_options":{"dit":{"weight_transfer":"allgather"},"text_encoder":{"weight_transfer":"rank-local"}}}' \
bash runtime/omni-serve-910a.sh --daemon
```

## 剩余待办

1. `InplaceAddRmsNorm` eager 回退（大概率还有其它 910B+ 融合算子：AddRmsNorm / fused_add_rms_norm 等）
2. `aclrtAllocatorGetByStream` 仍偶发 —— 说明还有别的线程/stream 在发集合通信
3. 出片后测速：4s / 10s，与 300 秒目标对比
4. int8 量化当前不可用（`dynamicQuantParams.x not implemented`），需专门适配

---

# 第三轮：算子策略判定 + 补丁清单

## 结论：eager 为主，不写 AscendC

三条证据：

1. **SwiGlu 是 SoC 白名单拦截，不是缺实现**
   ```
   EZ1009: Failed to execute operator SwiGlu_0.
   Reason: 1.SoC version ascend910 verification failed.
           This SoC is not configured through the AddConfig API of the OpDef class.
   ```
   写 AscendC 等于重造一个 CANN 已有、只是没为 ascend910 编译的 kernel。

2. **H3 直接用的 torch_npu 算子只有 2 个**（`npu_swiglu`、`npu_rms_norm`），
   eager 化面积很小，不是无底洞。

3. **eager 替代全部实测可用**：`silu(x1)*x2`、分块 matmul 注意力（7/7 数值验证）、
   native RMSNorm。

什么时候才该写 AscendC：**测速后确认注意力占预算大头**。SwiGLU/RMSNorm 是访存密集型，
eager 代价小；注意力是 O(S²) 计算密集型，那里才值得投入 kernel。

## 补丁清单（按需应用，全部幂等）

| 补丁 | 作用 |
|---|---|
| `h3_910a_attn.py` + `register_910a_attn.py` | 910A 分块 matmul 注意力后端，注册为 TORCH_SDPA |
| `h3_910a_sdpa_shim.py` | 全局替换 `F.scaled_dot_product_attention`（H3 有 3 处直接调用绕过注册表） |
| `patch_910a_norm_fallback.py` | `InplaceAddRmsNorm` 无 kernel → 走 `forward_native` |
| `patch_910a_swiglu.py` | vllm-ascend `AscendSiluAndMul` eager 回退 |
| `patch_910a_omni_silu.py` | vllm_omni `SiluAndMul.forward_npu` eager 回退（**DiT 实际走这条**） |
| `patch_910a_h3_swiglu.py` | H3 自己 `npu_swiglu_from_packed` eager 回退 |
| `patch_910a_allreduce.py` | all_reduce 前 `.contiguous()` + bf16→fp16 |
| `patch_910a_object_broadcast.py` | 用 tensor 广播替换 `broadcast_object_list`（SymInt 崩溃） |
| `patch_910a_nonzero.py` | 布尔索引搬 CPU（NonZero 动态形状不支持） |
| `patch_910a_timesteps_cpu.py` | `torch.unique` + 掩码 scatter 搬 CPU（AICPU EE9999） |
| `patch_910a_index_copy.py` | `index_add_` → `index_copy_`（索引集不相交，等价） |
| `patch_910a_stream_allocator.py` / `patch_910a_stream_warmup_all.py` | stream 分配器绑定（**已弃用**，会破坏文件） |

## 环境层面的坑（比补丁更重要）

| 现象 | 真因 | 解法 |
|---|---|---|
| `GEInitializeV2 failed` / `tbe_op_adapter failed` | 缺 `libGL.so.1` | `apt install libgl1` |
| `RmsNorm ... no supported kernel` | **`fatal error: 'cstdint' file not found`**（CANN AscendC 编译器缺系统 C++ 头路径） | `CPLUS_INCLUDE_PATH=/usr/include/c++/11:/usr/include/aarch64-linux-gnu/c++/11` |
| `hcclCommInitRootInfoConfig error 7 (resource unavailable)` | `/dev/shm` 积压数百个 `sem.mp-*` / `psm_*` 残留 | 清 `/dev/shm`，用 pidfile 杀进程而非 `pkill -9` |
| `NPU out of memory (1.36 GiB already allocated)` | `PYTORCH_NPU_ALLOC_CONF=expandable_segments:True` 假性 OOM | 去掉该变量（改为 opt-in） |
| `ModuleNotFoundError: fla_npu` | vllm-ascend 无条件导入 Qwen3.5 GDN 包 | 写 stub（H3 不用 GDN） |

## 当前状态

- 8 卡服务能起（内存 240GB），请求能跑到 DiT 采样
- 剩余问题：**rank 0 在加载阶段 NPU OOM**（fp16 + DLO allgather + TE 三重压力集中在 device 0）
- 尚未产出第一支视频

---

# 第四轮：从"算子全崩"到"跑到采样循环"

## 本轮打通的算子（全部 eager 化）

| 算子 | 报错 | 根因 | 处理 |
|---|---|---|---|
| `SwiGlu` | `EZ1009 SoC version ascend910 verification failed` | aclnn 二进制没有 ascend910 编译版本 | 3 处调用点全改 `silu(x1)*x2` |
| `InplaceIndexAdd` | aclnn 拒绝 | 断言语义等价，换 `index_copy_`（索引集不相交） | 改了但非真因 |
| `NonZero` | `OPS function error: NonZero, 500002` | 布尔索引是动态形状 | 掩码选择搬 CPU |
| AICPU `EE9999` | `The aicpu execution is abnormal` | `torch.unique` 走 AICPU | unique + scatter 全部搬 CPU |
| **Triton** | `bishengir-compile: Cannot find option named 'Ascend910B'` | **bishengir 最小目标就是 Ascend910B1，根本没有 910A/裸 910B** | `indexed_modulation.py` 4 处守卫改走 eager |

Triton 这条最致命：CANN 9.2 的 bishengir 目标列表里**没有任何 910A 变体**，不是配置问题，是这个工具链就不支持。所以 910A 上任何 `@triton.jit` 内核都没救，只能绕。

## 本轮最大的三个环境坑

1. **`TBE_PARALLEL_COMPILER` 默认 128**
   CANN 每个算子编译默认开 **128 个并行 fatbin 子进程**。8 个 diffusion worker 同时编译 → 上千进程 → `TBE Subprocess raised error, main process disappeared` → 引擎静默死亡。
   解法：`TBE_PARALLEL_COMPILER=4`。

2. **孤儿进程占满 device 0**
   `pkill -f vllm-omni` 抓不到它们——进程名显示为 `vLLM-OmniDiff`。两个孤儿各占 10GB，导致后续每次启动都"假 OOM"。
   正确清理：`npu-smi info -t proc-mem -i 0` 按 pid 杀 + 清 `/dev/shm/sem.mp-*`。
   **这是好几轮"莫名 OOM"的真因。**

3. **`expandable_segments` 锅**
   早期"1.36 GiB already allocated 却 OOM"是孤儿进程，不是它。别再当元凶。

## 架构层面的硬约束（关键结论）

**USP 只切激活值，不切权重。** 所以每张卡都要放完整权重：

- DiT 全量 62GB / 单层 1.24GB
- 单卡 32GB，减去 TE+VAE 常驻约 11GB，只剩约 21GB 给 DiT
- 即 `resident_layers` 最多约 16 层，其余 34 层必须 DLO 流式
- 而 `resident_layers` 要求 `weight_transfer=rank-local`（allgather 会报错）

实测：
| resident_layers | 结果 |
|---|---|
| 46 | 启动 OOM |
| 38 / 30 / 20 / 18 | 采样时 OOM，恒缺 **1.20 GiB** |
| 8 | 进入采样循环（首次出现 `0/4` 进度条），随后 MatMul OOM |

那个恒定的 1.20 GiB = 89878 行 × 7168 × 2 字节，是打包后的 `embeddings` 张量，与分辨率无关。

## 已验证可用的启动配置

```bash
OFFLOAD_CONFIG='{"mode":"layer","components":["dit"],
  "layer_options":{"dit":{"weight_transfer":"rank-local","resident_layers":8}}}' \
MODEL=/home/mseco/models/MiniMax-H3-4step LORA="" QUANT="" DTYPE=half \
TBE_PARALLEL_COMPILER=4 \
bash runtime/omni-serve-910a.sh --daemon
```
8/8 worker 就绪、health 200、主机内存 179GB、无 OOM 启动。

## 下一步方向（按收益排序）

1. **真正切分 DiT 权重**：让 USP/TP 把权重也切开（当前只有激活值切）。这是唯一能同时解决显存和速度的路。
2. 若做不到，改用 **`mode=module` 的粗粒度 offload**，把 DiT 整块放 CPU，用 pinned memory + 异步 prefetch。
3. 速度方面当前连一次 4 步 4 秒视频都没测出来，离 5 分钟目标还差很多轮优化。

---

# 第五轮：性能优化的决定性突破 —— bf16 是元凶

## 核心发现：910A 上 bf16 matmul 只有 fp16 的 1/170

同一个 GEMM（[11250,5376] × [5376,28672]），只换 dtype：

| dtype | 耗时 | 算力 |
|---|---|---|
| bf16 | 3417 ms | **1.01 TFLOPS** |
| fp16 | 20.3 ms | **170.57 TFLOPS** |

**fp16 快 168 倍。** 910A 的 bf16 走的是残废路径（仿真/回退），只有约 1% 算力。
这一条解释了之前所有异常现象：每层 14.7 秒、1% MFU、显存抖动。

### 换算成端到端

| 配置 | 每层 DiT | 每步 | 4 步总计 |
|---|---|---|---|
| bf16 + HSDP | 14.7 s | 735 s | **59 分钟** |
| fp16 + tiled 注意力 | ~1.1 s | 57 s | 7 分 37 秒 |
| **fp16 + 简单分块注意力** | **0.064 s** | **2.2 s** | **1 分 30 秒** |

**39 倍提升**，且 `dit 50/50 stack=2.226s` 证明 DiT 本身已接近合理速度。

## 附带发现：torch 层做 FlashAttention 式分块反而更慢

tiled + online-softmax 版本（每 tile 约 12 个 elementwise op）比简单分块版慢 5 倍。
原因：昇腾每个 op 都是独立 kernel launch，几百个 tile × 12 个 op 全部是启动开销。
**结论：910A 上要减少 op 数量，不是减少访存。**

简单分块版每 tile 只有 3 个 op（QK、softmax、PV），chunk=128。

## 踩到的算子级细节

1. **SoftmaxV2 不支持 bf16**：`E90002 Unsupported cast type! src_dtype: bfloat16`
2. **NanToNum 在 910A 上没有 kernel**：`No supported Ops kernel ... optype [NanToNum]`
   触发条件：fp16 点积溢出成 inf，softmax 走 NanToNum 兜底。
   修法：softmax 前 `clamp(-50000, 50000)`，并避免用 `-inf` 做掩码（改用 finfo.min）。
   验证：`OK finite: True`（溢出输入也能得有限结果）。
3. **`-inf` 掩码本身就会触发**，因为 Ascend 的 softmax 实现内部有 inf 处理路径。

## 端到端时间构成（重要）

### 启动：7-8 分钟，**不是编译**
```
Model loading took 25.0660 GiB and 404.96 seco
Model loading took 25.0660 GiB and 423.30 seco
compile starts: 0        ← JIT 编译次数为 0，kernel 缓存是热的
```
全是读权重：25GB/卡 × 8。

### 单次请求 90 秒，DiT 只占 9 秒
```
4/4 [01:29]                 ← 整个 4 步生成 89 秒
dit 50/50 stack=2.226s      ← 50 层 DiT 只要 2.2 秒/步，4 步共 8.9 秒
```
**剩下约 80 秒在 DiT 循环之外**，主要是文本编码器的 DLO 流式加载
（每次请求从主机内存搬 34GB fp16 的 TE）。

## 一个失败的尝试（记录避免重复）

想关掉 TE 流式让权重全常驻（因为 fp16 后模型减半、显存放得下）。
结果：**主机内存被打爆**（671/754 GB，只剩 6GB），加载从 17s/分片劣化到 105s/分片。
原因：8 个 rank 各自在主机内存里放全量副本再分片。
**结论：TE 的 DLO 流式是保护主机内存的必要手段，不能简单关掉。**

要真正消掉这 80 秒，需要让 TE 也走 HSDP 切分（当前 `_hsdp_shard_conditions` 只覆盖 DiT 的
transformer block），但 vllm-omni 的 HSDP 目前只包装 DiT。

---

# 第六轮：**首次成功出片** + 瓶颈定位

## 里程碑：910A 上跑通了 MiniMax-H3 端到端

```
status = completed
file   = video_gen_8b7590255efc42ec9fa40b24fb6ab755.mp4
640x384, 24fps, 107 帧, 4.46 秒, 1.65 MB
帧间平均亮度 89.9 → 101.0 → 107.1（有真实运动）
```

产物：`artifacts/outputs/h3_910a_4s_640x384.mp4`

## VAE 的 NanToNum：三层伪装，最后一层才是真的

这个问题吃掉了十几轮重启，值得完整记录：

| 层 | 现象 | 真相 |
|---|---|---|
| 1 | 以为是我们的 SDPA shim 产生 | 用 `-inf` 掩码会被 softmax 处理成 inf，但改掉后仍然报错 |
| 2 | 以为是 `torch.compile`（inductor 插 nan_to_num） | 关掉 dynamo/TORCH_COMPILE_DISABLE 后仍然报错 |
| 3 | 以为是 VAE 的 dtype（fp16/bf16/autocast） | 改 autocast、改 linear 权重 dtype 后仍然报错 |
| **真因** | **HuggingFace 每次加载都从模型目录重新生成 `transformers_modules/`，覆盖我对缓存目录的所有修改** | 必须改**模型目录里的源文件** |

正确修改位置：
```
/home/mseco/models/MiniMax-H3-4step/FL2VA/video_vae/flash.py:72
- return out.transpose(1, 2).nan_to_num(0.0)
+ out = out.transpose(1, 2)
+ return torch.where(torch.isfinite(out), out, torch.zeros((), dtype=out.dtype, device=out.device))
```

**教训：改 HF dynamic module 一定要改模型目录的源文件，改 `.cache/huggingface/modules/` 无效。**

## 算子支持矩阵（实测，910A）

| 算子 | fp16 | bf16 | fp32 |
|---|---|---|---|
| matmul | **170 TFLOPS** | 1.01 TFLOPS（残废） | — |
| `nan_to_num` | **✗ 无 kernel** | ✓ | ✓ |
| `F.rms_norm` | ✓ | ✓ | ✓ |
| `SoftmaxV2` | ✓ | ✗ E90002 | ✓ |
| `npu_swiglu` | ✗ EZ1009 SoC 拦截 | ✗ | ✗ |

**DiT 必须 fp16（170x 算力），VAE 解码里的 nan_to_num 必须非 fp16 —— 这是两个互相冲突的约束，
只能靠分模块控制 dtype 来解决。**

## 端到端耗时分解（4 秒 640x384，实测 463.9 秒）

```
07:43:04  请求开始
07:43:04 → 07:45:15   文本编码 + 4 步去噪 = 131 秒（其中去噪 87 秒）
07:45:15 → 07:50:48   VAE 解码        = 333 秒  ← 72% 的总时间！
07:50:48  完成，e2e_wall_time_ms = 463,912
```

### VAE 解码为什么是 333 秒

日志给出答案：
```
WARNING [vae.py:960] MiniMax-H3 VAE decode splits into 6 tile(s) but the tile group
has 8 ranks; decoding rank-locally for this shape instead
```

**6 个 tile 无法均分给 8 个 rank → 退化成每个 rank 独立解码整段视频 → 8 倍冗余工作。**

已把 `--vae-patch-parallel-size` 改为可配（`VAE_PP` 环境变量），下一步用 6 测试。

## 距离目标还差多少

目标：10 秒视频 5 分钟（300 秒）。

当前 4 秒视频 465 秒。若 VAE 并行修复（333 → ~55 秒），4 秒约 186 秒；
10 秒按帧数 2.5 倍外推约 400-450 秒，仍需进一步优化。**但已经从"完全跑不通"变成"能跑通且接近可优化"了。**

---

# 第七轮：可用配置确定 + 真实耗时结构

## 确定的可用配置（fp16 全栈）

```
HSDP=1 DTYPE=half VLLM_OMNI_910A_PERF=1 VLLM_OMNI_DLO_DP_WAVE_TIMEOUT=14400
PYTORCH_NPU_ALLOC_CONF=expandable_segments:True
OFFLOAD_CONFIG='{"mode":"layer","components":["text_encoder"],
                "layer_options":{"text_encoder":{"weight_transfer":"rank-local"}}}'
MODEL=/home/mseco/models/MiniMax-H3-4step LORA="" QUANT=""
```

配套的源文件级修改（**必须改模型目录，不是 HF 缓存**）：
`/home/mseco/models/MiniMax-H3-4step/FL2VA/video_vae/flash.py:72`
用 `torch.where(torch.isfinite(out), out, 0)` 替换 `nan_to_num(0.0)`。

## 实测端到端（640x384，4 步）

| 场景 | 耗时 | 说明 |
|---|---|---|
| 冷启动的第一个请求（4 秒视频） | 313 s | 含 ~50s 首步预热 |
| **热引擎的第二个请求（4 秒视频）** | **93 s** | 稳定态 |
| 热引擎（10 秒视频） | 894 s | 目标尺寸 |

**重要：冷启动首步有约 50 秒的一次性预热**（算子编译/惰性初始化），
之后每步只要 2.5 秒。所以反复压测时必须先跑一个暖机请求，否则数据失真。

## 热请求（4 秒视频，93 秒）的真实分解

```
文本编码 + 去噪    24 s   （其中 DiT 本身仅 ~10 s）
VAE 解码 + MP4 编码 67 s   ← 73%
```

## 10 秒视频（894 秒）的分解

```
去噪进度条 4/4 [06:50] = 410 s
  └─ 其中 DiT forward 仅: 17.6 + 7.5 + 7.5 + 7.5 ≈ 40 s
VAE 解码 + 编码 ≈ 470 s
文本编码 ≈ 14 s
```

## 核心结论：瓶颈不在 DiT，而在两处结构性开销

1. **每步 DiT 之外的开销 ~90 秒/步**（10 秒视频）。阶段计时证明
   `[H3PERF] phase total=7.5s`，但进度条是 102 秒/步 —— 90% 的时间在
   `model(**fk)` 之外（行准备、x0 转换、euler step、`video_rows.clone()`、
   `check_request_cancellation(synchronize=True)`）。
2. **VAE 解码 ~470 秒**，占总时间一半以上。

## 反直觉的失败尝试（记录避免重复）

**把 `vae_tile_size` 从 256 改成 192**，目的是让 tile 数从 6 变成 8，正好匹配 8 卡走并行路径：

- 模拟验证：tile=256 → 6 tiles（与日志实测一致 ✓）；tile=192 → 2×4 = 8 tiles
- 实际效果：**477 秒，比原来的 313 秒更慢 1.5 倍**
- 原因：tile 从 256 缩到 192，重叠 64 占比从 25% 升到 33%，冗余计算增加；
  同时 tile 数翻倍带来更多 gather/blend 开销。**并行的收益被这两项吃掉还有余。**

## 算子/框架层面的经验

- `--vae-patch-parallel-size` 只接受 **1 或完整 DiT 组大小**，中间值直接报错；
  且只有在 `num_tiles >= parallel_size` 时才走并行，否则退化 rank-local。
- 昇腾 `check_request_cancellation(synchronize=True)` 每步强制同步，
  会把异步流水打平，是潜在的性能杀手。

---

# 第八轮：**目标达成** —— 10 秒视频 5 分钟内生成

## 最终实测（暖引擎，连续两次）

| 请求 | 分辨率 | 时长 | 帧数 | 耗时 | 是否达标 |
|---|---|---|---|---|---|
| 红狐雪地 (seed=3) | 640×384 | 10.12 s | 243 | **219 s** | ✅ 3.65 分钟 |
| 宇航员登月 (seed=7) | 640×384 | 10.12 s | 243 | **287 s** | ✅ 4.78 分钟 |

目标：10 秒视频 300 秒（5 分钟）内 → **达成**，两次测量均在目标内。

产物（`artifacts/outputs/`）：
- `h3_910a_10s_640x384_FINAL.mp4`（6.6 MB，243 帧，10.12 秒）

## 关键结论：冷/热差异极大，必须先暖机

| 场景 | 10 秒视频耗时 |
|---|---|
| 冷启动后第一个请求 | 894 s |
| **暖引擎** | **219–287 s** |

差异来源是首次执行的**一次性预热**：
- 首步 DiT：`phase pre=16.1s blocks=29.4s total=45.6s`（后续步仅 7.5s）
- 首次 VAE 解码：也会编译该分辨率对应的 kernel

**这意味着任何性能压测都必须先跑一个暖机请求，否则数据完全失真。**

## 稳定态精确分解（10 秒视频，暖引擎）

每步去噪（插桩 `[H3PERF] dstep`）：
```
dstep 3: prep=1.76s  model=7.46s  update=3.04s   = 12.3 s
dstep 4: prep=1.81s  model=7.47s  update=3.04s   = 12.3 s
4/4 [01:36]   → 去噪总计 96 s
```

DiT forward 内部（`[H3PERF] phase`）：
```
phase pre=0.58s  blocks=6.87s  total=7.45s
```
即 50 层 block 占 6.9 s，预处理仅 0.58 s。

按 4 步 + VAE 解码估算：96 s 去噪 + ~120 s VAE/编码 ≈ 219 s。

## 与 bf16 基线的总对比

| 阶段 | bf16 | fp16（最终） | 加速 |
|---|---|---|---|
| 单层 DiT | 14.7 s | 0.077 s | 191× |
| 单步去噪 | 735 s | 12.3 s（10s 视频） | 60× |
| 4 步去噪（4s 视频） | 59 min | 10 s | 354× |
| 首次成功出片 | 从未成功 | ✅ | — |

## 最终可用配置

```bash
VAE_PP 不设（保持 NUM_GPUS）
vae_tile_size 保持 256（改成 192 反而慢 1.5 倍）
HSDP=1 DTYPE=half
VLLM_OMNI_DLO_DP_WAVE_TIMEOUT=14400
PYTORCH_NPU_ALLOC_CONF=expandable_segments:True
OFFLOAD_CONFIG='{"mode":"layer","components":["text_encoder"],
                "layer_options":{"text_encoder":{"weight_transfer":"rank-local"}}}'
MODEL=/home/mseco/models/MiniMax-H3-4step LORA="" QUANT=""
```

启动后必须先跑一个暖机请求再压测。

---

# 第九轮：画质归因分析（为什么"效果不够好"）

## 先说结论

画质问题**不是数值 bug**。用官方 768p 参考视频（`H3-source/assets/`）做基准，
我们的 640×384 输出在归一化指标上**落在官方区间内**。真正的差距来自三个结构性因素：

| 因素 | 事实 | 影响 |
|---|---|---|
| **分辨率** | 我们 640×384（短边 384），H3 原生短边 **768**（官方参考 1344×768） | 4 倍像素差，最直接的画质差距 |
| **提示词** | 我们 8 个词；官方是 300–500 词的 H3-Context-IR 结构化提示词 | 音频结构、色彩丰富度明显变差 |
| **步数** | 4 步蒸馏 LoRA | 细节与稳定性弱于全步数 |

## 证据 1：色彩没有被打压（修正一个错误结论）

一开始用"平均饱和度"比较，得到"我们掉色 3 倍"的结论——**这是错的**，
被亮度差异误导（我们的雪景 L*=60，参考场景 L*=17–42）。

改用 CIE Lab 色度**分布**（亮度无关）：

| 视频 | C* p50 | C* p99 | C* max |
|---|---|---|---|
| ref fl2va | 17.76 | 46.88 | 61.30 |
| ref r2va | 10.15 | 37.87 | 51.76 |
| ref t2va | 4.14 | 32.25 | 71.75 |
| **我们（简短提示词）** | 3.39 | **40.11** | **52.06** |

我们的 **p99 和 max 都落在官方区间内**，说明没有全局色度钳制。
p50 低是因为雪地大部分画面接近中性白、只有狐狸是彩色——**这正是提示词要求的内容**。

## 证据 2：清晰度也在官方区间内

把官方 768p 视频 lanczos 缩到 640×384（这才是"好的 640×384 视频"长什么样），
再和我们的比：

| 视频 | laplacian var | 平均梯度 |
|---|---|---|
| ref fl2va | 3.17e-03 | 0.0193 |
| ref r2va | 6.01e-03 | 0.0199 |
| ref t2va | 2.24e-02 | 0.0386 |
| **我们** | **3.65e-03** | 0.0232 |

我们的清晰度**正好落在参考区间内**（介于 fl2va 和 r2va 之间）。
之前"模糊 3 倍"的结论是只比了 h3_direct_768p 这一个最锐利的样本，不成立。

> 注意：换成官方风格提示词后，我们的 laplacian 掉到 7.39e-04。但那是因为
> 我写的提示词里明确写了 "soft diffuse light / fades into soft atmospheric haze"，
> 模型忠实地做柔了——**这是提示词跟随正确的表现，不是缺陷**。

## 证据 3：音频是真的差，而且提示词能修好

同一个场景，只改提示词（加入 `overall_soundscape` / `non_diegetic_music` 段）：

| 频段能量 | 简短提示词 | 官方 Context-IR |
|---|---|---|
| 0–200 Hz | 0.5% | **1.6%** |
| 200–2000 Hz | 44.4% | **82.4%** |
| 2000–8000 Hz | 43.3% | 15.5% |
| 8000–20000 Hz | **11.8%** | **0.4%** |

简短提示词下高频嘶声占 11.8%、几乎没有低频——听起来像噪声；
换成官方格式后高频噪声降到 0.4%、中频（人声/环境体感）升到 82.4%。
**H3 的音频是提示词驱动的，不给 soundscape/music 段就只能得到噪声。**

## 为什么不能用 768p（关键工程约束）

既然分辨率是最大差距，我尝试直接跑原生 1344×768，结果两次都失败，原因不同：

**第一次：设备显存 OOM**
```
NPU out of memory. Tried to allocate 1.96 GiB
(NPU 3; 32.00 GiB capacity; 27.19 GiB already allocated)
```
OOM 发生在 `h3_910a_attn.py:88`，即 `torch.softmax(scores.float(), dim=-1)`。
**把整个 score tile 升到 fp32 让注意力峰值显存翻倍。**

修法：910A 的 `SoftmaxV2` 原生支持 fp16（只拒绝 bf16），所以只有 bf16 需要
fp32 往返。改成按 dtype 决定：

```python
upcast_softmax = query.dtype == torch.bfloat16   # 只有 bf16 需要
weights = torch.softmax(scores if not upcast_softmax else scores.float(), dim=-1)
```
同时把 `-inf` 掩码换成 `torch.finfo(dtype).min`（避免触发 NanToNum 回退）。
后端测试 **7/7 通过**（max|diff| ≈ 1e-3）。

**第二次：主机内存耗尽 → HCCL 死锁**
修掉设备 OOM 后，768p 不再 OOM（`oom: 0`），但任务卡死 16 分钟无输出：

```
[rank:5] Process group work ALLGATHER, seq_num 3 dispatch timeout
[rank:6] Process group work ALLGATHER, seq_num 3 dispatch timeout
```
现场：主机内存 **715/754 GB（仅剩 4 GB）**，swap 疯狂换页
（`vmstat` 显示 si=171004 / so=333384），显存 30.1/32 GB，AICore 0%。

**结论：768p 的瓶颈从设备显存转移到了主机内存。** 8 个 rank 各持一份权重
staging（rank-local DLO），768p 的 4 倍激活值把最后的余量吃光，swap 抖动导致
HCCL allgather 超时，进而死锁。

## 可行的下一步（按性价比排序）

1. **换官方 Context-IR 提示词格式** —— 零成本，音频改善可量化（高频噪声 11.8%→0.4%），
   色彩 p50 也上升（3.39→5.38）。
2. **中间分辨率**：1344×768 装不下，但 1024×576（短边 576）或 896×512 值得试，
   像素量是我们现在的 2–2.4 倍，主机内存需求远低于 768p。
3. **提升步数**：4 步是速度档；质量档可以试 8 步（耗时约翻倍）。
4. **要真正上 768p 必须先解决主机内存**：改成共享权重（DLO_MODE=allgather 需要
   HCCL all_gather 可用）或减少 rank 数，把 8×权重副本降下来。

---

# 第十轮：主机内存为什么吃这么多（实测）

## 一句话结论

**文本编码器（Qwen3-VL，63 GB）被 8 个 rank 各复制了一份私有副本 = 504 GB。**
原因是 DLO 的 `weight_transfer: rank-local` 走的是"每个 worker 摊平一份
模型大小的私有分配"，而不是共享页缓存。

## 实测数据

8 个 worker 的 RSS **完全一致**，这是复制而非共享的直接证据：

```
pid        RSS      PSS     Private
2010072   77.3 GB  76.7 GB  76.7 GB
2010071   77.2 GB  76.7 GB  76.7 GB
2010075   76.3 GB  75.8 GB  75.7 GB
2010069   76.2 GB  75.6 GB  75.6 GB
2010074   75.9 GB  75.4 GB  75.4 GB
2010076   74.6 GB  74.0 GB  74.0 GB
2010073   73.7 GB  73.2 GB  73.2 GB
2010070   71.8 GB  71.3 GB  71.2 GB
TOTAL    603.3 GB 599.1 GB 598.9 GB
```

**关键：PSS ≈ RSS ≈ Private（599 / 603 GB 都是私有）。**
如果用 mmap 共享文件，PSS 会远小于 RSS（共享页按比例分摊）。这里三者相等，
说明是**真的 8 份互不共享的私有内存**，不是页缓存重复计数。

稳态 `free -g`：**666–678 GB 已用，剩 38 GB**。

## 每个 worker 74 GB 的构成

| 项目 | 大小 | 说明 |
|---|---|---|
| 文本编码器 Qwen3-VL | **63 GB** | 完整私有副本，**纯浪费 7/8** |
| DiT 权重分片 | ~7.75 GB | 62 GB ÷ 8，每 rank 一份，**合理** |
| 运行时/缓冲 | ~3 GB | Python、torch、激活缓冲 |

8 × 74 GB ≈ 592 GB，与实测 595 GB 吻合。

## 为什么不得不用 rank-local

DLO 默认的权重传输是 AllGather（8 个 rank 共享一份主机权重，按需 all_gather 到设备）。
但 910A 上这条路径直接失败：

```
HcclAllGatherV ... HCCL_E_NOT_SUPPORT
```

（`docs/910a_ops_notes.md` 里有记录。）所以只能退到 `rank-local`，
代价就是注释里写的 "costs ~8x host RAM"。

代码层面，`distributed_layerwise_backend.py` 把两条路径分得很清楚：

```python
# 非 mmap（我们现在走的）
self.cpu_shards: dict[...] = {}      # "flattened into a model-sized private allocation"

# mmap 共享路径
self.cpu_sources: dict[...] = {}     # "immutable views of the checkpoint and are never
                                     #  pinned or flattened into a model-sized private allocation"
```

## 已有的解药：Host Weight Runtime（HWR）

代码里有现成开关，用主机侧共享存储替代私有拷贝：

```
--host-weight-runtime-mode {disabled,preferred,required}   # 默认 disabled
--host-weight-runtime-root <dir>                            # 启用时必填
```

启用后走 `backing_kind == "checkpoint_mmap"`，8 个 rank 只读映射同一份文件，
`_using_rank_local_mmap = True`。理论上 TE 从 504 GB 降到 ~63 GB，**省出约 440 GB**。

注意两个约束（`distributed_layerwise_backend.py`）：

```python
if rank_local_mmap and dp_size != 1:
    raise ValueError("rank_local_mmap requires dp_size=1")
```

我们的配置是 `--usp 8 --ring 1`，dp_size 是否为 1 需要在启用时确认。

**这意味着 768p 很可能可以救回来**：目前卡在主机内存（715/754 GB → swap 抖动 →
HCCL ALLGATHER 超时死锁），而不是设备显存。HWR 若生效，768p 的内存约束会大幅缓解。

## 顺带说明：之前"关掉 TE 流式"为什么更糟

曾经试过不 offload 文本编码器，想让权重全常驻，结果主机内存反而先爆
（671/754 GB，加载从 17s/分片劣化到 105s/分片）。
因为 TE 会整份加载到设备再分片，8 个 rank 各自在主机侧材料化全量副本，
比流式更糟。**TE 流式是必要手段，但 rank-local 的实现方式才是内存问题的根源。**

---

# 第十一轮：主机内存的分配性质确认 + 一个物理上限

## 76 GB 是匿名私有内存，不是可共享的文件映射

之前只测了 RSS/PSS，这轮直接看 smaps 里映射的性质：

```
total private    : 76.4 GB
anonymous private: 76.0 GB
file-backed private:
    0.5 GB  [heap]
    0.0 GB  libtorch_cpu.so
    0.0 GB  libtvm.so ...
```

**76.0 / 76.4 GB 全部是匿名私有映射。** 如果权重是 mmap 自 `.safetensors` 文件，
它会显示为 file-backed，且 8 个进程通过 page cache 天然共享（物理内存只存一份）。
现在不是——**权重被复制进了每个 worker 的私有堆，8 份真实占用 8 × 76 = 608 GB。**

这解释了为什么 `free -g` 的 used 和 RSS 总和几乎相等，也解释了 PSS ≈ RSS。

## Host Weight Runtime 这条路走不通（已实测）

启用 `--host-weight-runtime-mode preferred --host-weight-runtime-root <dir>` 后启动直接报错：

```
ValueError: diffusion_offload_config cannot be combined with Host Weight Runtime
```

HWR 与显式 offload 配置互斥，只能改用 legacy 开关
（`--enable-distributed-layerwise-offload --dlo-no-use-allgather`）。
按 legacy 方式启动后确实起来了，但：

* **HWR 存储目录始终是空的（4.0K）**——HWR 从未真正发布或复用任何东西
* 内存只从 595 GB 降到 522 GB，而且这是配置基线变了（legacy 路径把 DiT 也 offload 了），不是 HWR 的功劳
* 加载完成后卡死 8 分钟无输出，主机内存只剩 5 GB

更关键的是，真正想要的 mmap 共享路径在 `host_weight_plan.py` 里被直接拒绝：

```python
if tensor_parallel_size != 1: return None, "TP requires the ordinary loader"
if use_hsdp:                  return None, "HSDP requires the ordinary loader"
if online_quantization:       return None, "..."
if has_distilled_lora:        return None, "..."
```

我们的配置**同时命中 TP=8、HSDP=1、蒸馏 LoRA 三项**，全部是否决条件。
而且这个 plan 只收集 DiT 张量（`_collect_required_dit_tensors`），
**根本不覆盖文本编码器**——即使可用也解决不了那 63 GB。

## 一个物理上限：768p 10s 不可能在 5 ms 内生成

需求本身要过一遍量级检查。1344×768、243 帧（H3 的 17n+5 网格）：

| 项目 | 数值 | 5 ms 内需要 | 对比 |
|---|---|---|---|
| 输出像素原始数据 | 752 MB | 0.2 TB/s 写入 | 尚可，但只是"写" |
| **权重只需读一遍** | 125 GB | **25 TB/s** | 8 卡 HBM 聚合约 6.4 TB/s，**差 4 倍** |
| DiT 计算（4 步，约 24 万 token） | ~6.0e16 FLOPs | **12 EFLOPS** | 8×910A fp16 约 3 PFLOPS，**差约 4000 倍** |

**光把权重读一遍就装不下 5 ms**，这已经是纯访存、零计算的下限；
计算侧差三到四个数量级。这不是优化问题，是物理约束。

当前实测：768p 最好的情况是**跑不通**（主机内存死锁），
640×384 10 秒热机 219–287 秒。若目标是"768p / 10 秒 / 5 分钟"，还差 3 倍左右；
需要先解决上面的 8×63 GB 私有副本问题。

---

# 第十二轮：768p 跑通了 —— checkpoint mmap 解决主机内存

## 突破：主机内存 595 GB → 29 GB

### 找到真正的开关

上一轮确认了 76 GB/worker 是**匿名私有内存**。这一轮顺着"vllm-omni 有没有支持按分片读"
查下去，结论是：

* `_shard_and_pin()` **确实**支持按分片分配主机内存，但只沿 `dp_size` 维度切：
  ```python
  shard_size = (total_numel + dp_size - 1) // dp_size
  shard = torch.zeros(shard_size, ...)   # 只分配 1/dp_size
  ```
* 我们是 HSDP + 序列并行，`dp_size = 1`，所以"分片"就等于整个模型 → 每 rank 一份全量。
* 真正的共享路径是 `checkpoint_mmap`：把权重保留为 checkpoint 的只读 mmap 视图，
  8 个进程共享同一份 page cache。

### 两个否决条件，逐个拆掉

`build_checkpoint_mmap_plan()` 有四个拒绝条件：

```python
if tensor_parallel_size != 1: ...   # 我们用 usp，tp=1 ✓
if use_hsdp:                  ...   # ← 真正的拦路虎
if online_quantization:       ...   # 无量化 ✓
if has_distilled_lora:        ...   # LORA="" ✓
```

**关掉 HSDP 后 mmap 计划就被生成出来了**，但立刻又撞上第二个条件：

```
DLO direct checkpoint mmap unavailable; using ordinary loader:
dtype mismatch for 'transformer.condition_proj.weight':
checkpoint=BF16, transformed=torch.bfloat16, runtime=torch.float16
```

### 离线转 fp16（关键一步）

mmap 是零拷贝，要求 **文件 dtype == 运行时 dtype**。checkpoint 是 BF16，而我们为了
GEMM 吞吐必须跑 fp16（bf16 在这卡上慢 168 倍），所以不能在线转换——在线转换等于把
权重读进来再转，又变回匿名私有内存。

**做法：离线把 DiT checkpoint 转成 fp16 存一份**
（`/home/mseco/models/MiniMax-H3-fp16`，13 个分片共 62 GB，F32 张量保持 F32，
文本编码器继续符号链接到原 bf16 版本）：

```
fp16 dir dtypes: {'F32': 13, 'F16': 522}
```

### 结果

```
DLO host-weight plan active (AllGather, checkpoint_mmap): skipping ordinary materialization
```

| 指标 | HSDP=1（旧） | HSDP=0 + fp16 mmap（新） |
|---|---|---|
| 每 worker RSS | 74 GB | **2.9 GB** |
| 8 worker 总计 | 595 GB | **29.1 GB** |
| 主机内存 used/avail | 676 / 24 GB | 331 / 237 GB |
| 每卡 HBM | 30 / 32 GB（爆满） | **3.2 / 32 GB** |

**主机内存降 20 倍，HBM 降 10 倍。**

## 768p 现在能跑了

之前 768p 是**根本跑不通**的（主机内存 715/754 GB → swap 抖动 → HCCL ALLGATHER 超时死锁）。
换新配置后：

| 分辨率 | 旧配置 | 新配置 |
|---|---|---|
| 640×384 / 10s | 219–287 s | **167 s**（还更快了） |
| 1344×768 / 10s | ❌ 死锁 | **1554 s（25.9 分钟）** |

产物已验证：`tkhd dims: 1344x768`，21.3 MB，243 帧，无 OOM。

**HSDP=0 不但没变慢，640p 反而从 219–287 s 降到 167 s** —— 说明 mmap 省掉的
主机→设备搬运开销，比 HSDP 带来的权重分片收益更大。

## 距离目标还差多少

目标：768p / 10s / ≤300 s。实测 1554 s，**差 5.2 倍**。

注意像素比只有 4.2 倍（640p 167 s → 768p 若线性应是 ~700 s），实际 1554 s，
多出来的约 2.2 倍来自注意力 O(n²) 和 VAE 解码的非线性成本。

下一步需要拿到阶段分解（`VLLM_OMNI_910A_PERF=1` 已内置但这次没开），
定位是去噪还是 VAE 占主导，再决定优化重点。

---

# 第十三轮：768p 的真实耗时与瓶颈定位

## 热机 vs 冷启动（差 2.9 倍）

同一个 768p / 10s 请求：

| 状态 | 耗时 |
|---|---|
| 冷启动（引擎刚起） | 1554 s（25.9 min） |
| **热机** | **543 s（9 min）** |

冷启动那次多花 1000 s 全是一次性 kernel 编译 + 权重预热。
**后续所有数字都必须用热机值，否则会误导。**

## 每步分解（768p / 10s / 4 步，ATTN_CHUNK=1024）

```
[H3PERF] dstep 4 prep=7.5s model=86.2s update=13.8s     -> 107.5 s/步
[H3PERF] dit 50/50 block=1.960s stack=84.6s
```

4 步去噪 = 430 s，其余（文本编码 + VAE 解码 + 封装）≈ 113 s，合计 543 s。

## 瓶颈是注意力，且它是 O(n²)

用两个分辨率的单 block 耗时反推（设 block = 线性项 L + 注意力项 A）：

| 分辨率 | 单 block | 像素比 |
|---|---|---|
| 640×384 | 0.126 s | 1.0 |
| 1344×768 | 1.960 s | 4.2 |

解方程：
```
L + A            = 0.126          -> A = 0.105 s, L = 0.021 s
4.2L + 17.6A     = 1.960
```

**注意力占 640p 单 block 的 83%，占 768p 单 block 的 94%（1.84 s / 1.96 s）。**

50 blocks × 1.84 s = 92 s/步纯注意力，4 步 = **368 s**，占总时长的 68%。

## 试过但无效：调 chunk size

`ATTN_CHUNK` 从 1024 提到 2048：

| | chunk=1024 | chunk=2048 |
|---|---|---|
| model | 86.2 s | **74.8 s**（快 13%） |
| update | 13.8 s | **22.7 s**（慢 65%） |
| 每步合计 | 107.5 s | 105.3 s |
| 总时长（热机） | **543 s** | 804 s |

model 变快但 update 等量变慢，**净收益 2%，可忽略**。
chunk 只改变 score tile 的大小，不改变总访存量，所以到不了根本瓶颈。

（另注：非去噪阶段耗时波动很大，不同次测量在 113 s ~ 383 s 之间，
所以单次总时长噪声很大，判断优化效果要盯 `dstep` 的分解而不是总时长。）

## 为什么注意力快不起来

910A 上**没有可用的融合注意力算子**：

* `torch.nn.functional.scaled_dot_product_attention` → `ERR01007`
* `npu_fusion_attention` → `EZ1009 SoC version ascend910 verification failed`
* `npu_prompt_flash_attention` → 同样 SoC 校验失败

所以只能用三个独立 matmul（QK → softmax → PV），中间必须把完整的
`[heads, chunk, N]` score 矩阵写进 HBM 再读回来。理论算力占用率约 39%，
剩下的都耗在 score 矩阵的反复读写上。

## 目标可达性评估

目标：768p / 10 s / ≤300 s。

当前最好 543 s，需要 **1.8 倍**加速。

理论上若注意力能跑到 100% GEMM 效率（0.72 s/block 而非 1.84 s），
去噪可降到 ~144 s，加 VAE 113 s ≈ 257 s，**刚好进 300 s**。
但这要求把注意力效率从 39% 提到 100%，在**没有融合 kernel** 的前提下做不到——
这正是 910A 的硬限制（算子被 SoC 版本门禁，不是缺实现）。

可行的方向只剩：
1. 等 CANN 提供 910A 可用的融合注意力（外部依赖）
2. 自己写 AscendC 融合注意力（工作量大，且此前判断优先级低）
3. 换更省注意力的模型/采样策略（改变画质）

---

# 第十四轮：注意力微优化 + AscendC 工具链验证

## 实测到的真实注意力形状

在 `forward_npu` 里加一次性形状日志（`VLLM_OMNI_910A_SHAPE_LOG=1`）后拿到真相：

```
[SHAPE] q=(1, 73408, 7, 128) k=(1, 73408, 7, 128) v=(1, 73408, 7, 128) dtype=torch.float16
```

permute 后是 `(1, 7, 73408, 128)`，即 **H=7, S=73408, D=128**。
（另有 16 次 `(1, 8, 56, 128)` 是极小序列的预热调用。）

**Ultysses 序列并行是把 head 切开、序列保持完整**，所以 8 个 rank 各自对全长序列做
注意力，只是每人 7 个 head。这解释了为什么注意力是 O(n²) 且没有随卡数下降。

## 免费优化：K^T 连续化 + scale 折入 Q

真实形状下测单次注意力调用：

| 实现 | 耗时 | 等效算力 |
|---|---|---|
| 原实现（chunk=1024） | 1.350 s | 14.3 TFLOPS |
| 优化后 | **1.175 s** | 16.4 TFLOPS |

两处改动：

1. **`key.transpose(-1,-2).contiguous()`**：原来 K^T 是非连续的，72 个 chunk 每次
   都要 matmul 去 gather 跨步的行。一次拷贝换掉 72 次跨步访问，实测 matmul 快约 24%。
2. **`query = query * scale`**：scale 原来是乘在 score 矩阵上（S×N），每 chunk 一遍。
   折进 Q（S×D）后这次全量大矩阵遍历直接消失。

在模型里的实际效果：

| 指标 | 优化前 | 优化后 | 倍数 |
|---|---|---|---|
| 单 block | 1.960 s | **1.792 s** | 1.09× |
| model（每步） | 86.2 s | **77.0 s** | 1.12× |
| 每步合计 | 107.5 s | **99.3 s** | 1.08× |
| 4 步去噪 | 430 s | **397 s** | 1.08× |

后端测试仍 7/7 通过，`max|diff| ≈ 1e-3`。

## 试过但没用：npu_attn_softmax_

`torch_npu.npu_attn_softmax_` 在该 SoC 上**可用**（不报错、结果有限），
但速度与 `torch.softmax` 完全相同（0.0204 s vs 0.0206 s）。不是优化点。

## 为什么 4.8% 的效率能救

优化后注意力仍是 14–16 TFLOPS，约为 910A fp16 峰值的 **4.8%**。原因很明确：

单次注意力要处理的 score 矩阵是 `73408 × 7 × 73408` 个 fp16，约 **75 GB**，
而三个独立 matmul 的流程是：

```
matmul 写 scores  ->  softmax 读 scores 写 weights  ->  matmul 读 weights
```

即 **75 GB 数据在 HBM 上往返三次**。softmax 单独就占了单 block 的 38%。

若用 flash-attention 式的分块在线 softmax，score 只存在于片上，这 75 GB×3 的往返
就全部消失。按效率从 4.8% 提到 40% 估算，注意力可从 1.35 s/block 降到约 0.16 s：

```
4 步 × 50 blocks × (1.792 - 1.35 + 0.16) ≈ 120 s 去噪  + 113 s VAE ≈ 233 s
```

**这就是能在 300 s 内完成的唯一路径。**

## AscendC 工具链在 910A 上可用（已验证）

这是写融合算子的前提，本轮把它验证掉了。

平台配置里有 `Ascend910A.ini`，编译器是 CANN 自带的 `ccec`。可用的编译命令：

```bash
source $ASCEND_HOME_PATH/set_env.sh
export CPLUS_INCLUDE_PATH=/usr/include/c++/11:/usr/include/aarch64-linux-gnu/c++/11:/usr/include:/usr/include/aarch64-linux-gnu
$ASCEND_HOME_PATH/aarch64-linux/bin/ccec -xcce -std=c++17 -O2 \
  --cce-aicore-arch=dav-c100 \
  -I$ASCEND_HOME_PATH/aarch64-linux/asc/include \
  -I$ASCEND_HOME_PATH/aarch64-linux/asc/include/basic_api \
  -I$ASCEND_HOME_PATH/aarch64-linux/asc/include/highlevel_api \
  -I$ASCEND_HOME_PATH/aarch64-linux/tikcpp/tikcfw \
  -c kernel.cpp -o kernel.o
```

最小 kernel（`TPipe` + `TQue` + `GlobalTensor` + `DataCopy` + `Muls`）**编译通过，
产出 32,928 字节目标文件**。三个曾经踩到的坑：

1. `cstdint` 找不到 → 必须设 `CPLUS_INCLUDE_PATH`
2. `kernel_tpipe.h` 找不到 → 要显式加 `asc/include/basic_api`
3. **默认语言标准太旧**（`'auto' not allowed in template parameter until C++17`）
   → 必须加 `-std=c++17`

注意：编译通过 ≠ 能运行。内核还需要走算子注册（aclnn）才能在 torch 里调用，
这是下一步的工作。

---

# 第十五轮：精确阶段分解 + 910A 的真实 matmul 峰值

## 阶段分解（768p / 10s，实测）

在 pipeline 的 `forward()` 里插了三个计时点（`[H3PERFP]`），终于不用再靠
"总时长减去去噪"去猜了：

| 阶段 | 耗时 | 占比 |
|---|---|---|
| **encode**（Qwen3-VL 文本编码） | **40.2 s** | 6% |
| **denoise**（4 步 DiT） | **465.1 s** | 70% |
| **decode**（VAE 解码 243 帧） | **155.4 s** | 23% |
| 其他 | ~7 s | 1% |
| **总计** | **668 s** | 100% |

此前"非去噪阶段在 113~383 s 之间波动"的谜团解开了：**decode 本身就不稳定**
（多次测到 154.5 / 155.4 / 160.4 / 162.4 s，另一些场景下会更高），
而 encode 是稳定的 40 s。

去噪内部分解（每步）：

```
[H3PERF] dstep 4 prep=8.0s model=76.6s update=14.8s   -> 99.4 s/步
4 步 = 397.6 s，但 diffuse() 总共耗 465 s -> DiT 之外还有 ~67 s
```

## 910A 的真实 matmul 峰值只有 ~77 TFLOPS

之前一直按"910A fp16 ≈ 300 TFLOPS"估算，实测差 4 倍：

| 形状 | 实测 |
|---|---|
| 4096³ matmul | **74.8 TFLOPS** |
| 8192³ matmul | **77.4 TFLOPS** |
| H=7, M=1024, K=128, N=73408（真实注意力形状） | **32.0 TFLOPS** |

真实注意力形状只有 32 TFLOPS，因为 K=128 太短，cube 单元摊不开。

这直接改变了融合 kernel 的收益上限：

```
注意力 FLOPs = 1.93e13
按 32 TFLOPS  -> 理论下限 0.60 s   （当前 1.35 s，即 2.25 倍差距）
按 77 TFLOPS  -> 理论下限 0.25 s
```

不是之前估的 13 倍，而是 **2.25 倍**。

## 访存带宽标定

顺带标定了有效带宽（fp16，1.05 GB 张量）：

| 操作 | 有效带宽 |
|---|---|
| `mul_` 就地 | **999.3 GB/s** |
| `add_` 就地 | 999.3 GB/s |
| `amax(dim=-1)` | 1093.5 GB/s |
| `copy_` | 691.2 GB/s |
| `torch.softmax` fp16 | **228.4 GB/s** |
| `npu_attn_softmax_` | 205.6 GB/s |

softmax 只有 228 GB/s 看起来很低，但它是多趟操作（fp16 读 → fp32 中间量 →
计算 → fp16 写，约 6.3 GB 实际流量），折算下来约 685 GB/s，即**已达峰值 68%**，
不是实现问题。

`npu_attn_softmax_` 比 torch 的还慢一点，没有使用价值。

## 两个失败的优化尝试

**1. 文本编码器改用 fp16 —— 无效**
`encoder.py` 里 TE 硬编码 `torch.bfloat16`，本以为 40 s 的 encode 是 bf16 GEMM 造成的。
实测改成 fp16 后 **encode=38.0 s**（bf16 是 40.2 s），几乎没变。
说明 40 s 主要是 **DLO 把 67.7 GB 的 TE 权重逐层从主机搬到设备**（≈1.7 GB/s
的有效搬运速率），不是计算。

**2. 让 TE 常驻设备 —— 破坏 mmap**
想通过显式传 `OFFLOAD_CONFIG`（只列 `dit`、不列 `text_encoder`）让 TE 常驻，
结果是灾难：

* `mmap: 0` —— **显式传 diffusion_offload_config 会让 checkpoint_mmap 路径失效**
* 主机内存从 331 GB 涨到 **703 GB（只剩 8 GB）**

印证了之前的结论：显式 offload 配置与 mmap 路径互斥，不能这样组合。

## 关于 300 s 目标的物理下限

把各阶段压到理论最好情况估算：

| 项目 | 当前 | 理论最好 |
|---|---|---|
| 注意力（200 次 × 1.35 s） | 270 s | 120 s（32 TFLOPS 满效率） |
| 非注意力 DiT | 36 s | 36 s |
| prep + update | 91 s | ~10 s（这两个只处理几十 MB） |
| diffuse 额外开销 | 67 s | ? |
| encode | 40 s | ~3 s（若权重能常驻） |
| decode | 155 s | 155 s（计算量固定） |
| **合计** | **668 s** | **~330–400 s** |

即：**即使注意力跑满硬件算力、encode 完全消除，总量仍约 330–400 s，
距离 300 s 还有一截**，而 decode 的 155 s 和 prep/update 的异常开销是关键未知。
