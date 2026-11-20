# R124：原始 H3 同输入数值门与 910A 完整出片验证

本轮完成 **5 个同输入代表位置的完整 prediction 比较**，以及 **910A 自身 encoder + 原始 VAE 的完整 49 次前向生成**。这不是 encoder 零误差、端到端逐像素一致、通用画质/音频评分或性能目标达标的声明。

## 本轮证据

- 数值门：`||NPU-GPU||₂ / ||GPU||₂ ≤ 1%`，目标 `≤ 0.5%`。这是双方约定的工程验收值，不是行业通用阈值。5 个位置都达到目标；视频比较全部 `37296×96` target 元素，音频全部 `414×32` 元素。详见[数值摘要](<artifacts/diagnostics/round124/all_sigma_gates_summary124.json>)与[原始逐位置记录](<artifacts/diagnostics/round124/sigma_checks/sigma_checks_rank0.json>)。
- GPU 参考来自先前成功的原始 vLLM-Omni 管线：live encoder、tiled keyframe VAE、50 主层、49 次真实前向。5 个快照的 GPU conditioning **只用于同输入诊断**，不会替代 NPU 正式生成。详见[GPU 轨迹完成回执](<artifacts/diagnostics/round124/GPU_pipeline_reference/GPU_pipeline_trajectory_result124c.json>)。
- NPU 正式生成使用自身 FP16 encoder、原始 50 主层、原始 FP32 VAE 参数及正常 FP16 视频解码 autocast；8 rank 每个完成 49 次前向。每次检查 155 个 Half matrix module，输出有限，没有观察到 Half 饱和边界。详见[本地综合核验](<artifacts/diagnostics/round124/verified_full_generation124.json>)和[实际生成回执](<artifacts/diagnostics/round124/NPU_generated/generation_result124.json>)。
- 输出为 1344×768、124 帧、24fps，实际时长约 5.167s；32kHz 立体声原生音轨。见[910A 视频](<artifacts/diagnostics/round124/NPU_generated/h3_910a_vllm_omni_fp16.mp4>)与[未经增益处理的原生音轨](<artifacts/diagnostics/round124/NPU_generated/h3_910a_vllm_omni_audio.wav>)。
- 124 帧均检测到红球，首尾位置约 `(320,595) → (1024,595)`，有垂直弹跳；[接触表](<artifacts/diagnostics/round124/NPU_generated/contact_sheet.png>)和两个完整分辨率抽样帧未见明显文字/logo/黑帧。统计不能证明所有伪影不存在。见[全帧质检](<artifacts/diagnostics/round124/NPU_generated/all_frame_quality_inspection124.json>)。
- 音轨有限且非空，但非常轻：peak −75.13 dBFS、RMS −98.78 dBFS。未增加增益，不声称音频感知质量达标。
- encoder 独立诊断仍有约 6.13% 全张量差异（visual 6.53%，text 0.057%）。这是 encoder 诊断，不是同输入 DiT 门，也不是已对齐的声明。见[encoder 对照](<artifacts/diagnostics/round124/own_NPU_encoder_vs_successful_GPU124.json>)。

## 精度与来源约束

没有本地下载或中转模型权重；GPU 使用此前在 4090 上已核验的缓存，NPU 使用自己已核验的原始权重。FP16 大矩阵、FP32 residual/norm/宽输出 GEMM、原始 FP32 patch/time/final 参数与精度保护沿用 R123。没有 BF16 回退、student、LoRA、隐藏 GPU encoder 缓存或人为 clipping。

居民 NPU 元数据额外写入了 4-interval `base_schedule`，官方 GPU pinned revision 元数据没有该字段。隔离视图采用核验 git blob `ed4dfb751d8417e5ab8cf58a6915214bda6ba676` 的官方元数据，并以 symlink 指向原有 NPU component 目录；原文件与权重不改动。见[元数据视图回执](<artifacts/diagnostics/round124/original_metadata_view124f.json>)。

CANN 的认证 manager 与 helper bootstrap 已隔离；正确 key 能重建 proxy、错误 key 会拒绝。native NanToNum 在原始 910A 上不可用，浮点谓词试验也没有通过，因此兼容实现使用 IEEE 整数位判断。24 个 FP16/FP32 小算子用例通过；本次实际 VAE 解码的 compatibility call count 为 **0**，不能把未调用的适配器当作加速结果。见[IPC 检查](<artifacts/diagnostics/round124/private_IPC_bootstrap_preflight124e.json>)、[NanToNum 小算子检查](<artifacts/diagnostics/round124/Half_nan_to_num_equivalent_probe124k.json>)和[执行源 manifest](<artifacts/diagnostics/round124/executed_source_manifest124g.json>)。

## 复核与运行

本地证据核验不加载 Torch、不跑模型：

```bash
python3 artifacts/diagnostics/round124/verify_round124_evidence.py
python3 -m py_compile artifacts/diagnostics/round124/*.py
```

证据核验脚本不覆盖完成回执；重复运行会核对当前导出的结果与原回执语义完全相同，有差异则报错。实际运行入口为[成功的 NPU 驱动](<artifacts/diagnostics/round124/generate_full_NPU124g.py>)、[worker 精度/审计接入](<artifacts/diagnostics/round124/alignment_hooks124.py>)、[选择性 process bootstrap](<artifacts/diagnostics/round124/npu_process_bootstrap124e.py>)。

重新跑模型前必须通过 fleet arbiter 领取 owned NPU、准备一个新的私有 IPC key（32 byte、0600）和新的输出 label/work 目录。实验脚本故意拒绝覆盖，不能直接复制旧启动命令重跑。关键运行设置：

```bash
VLLM_OMNI_910A_TE_DTYPE=fp16
VLLM_OMNI_910A_TEXT_HF_MATH=1
H3_NPU_ALIGNMENT124_BOOTSTRAP=0
H3_NPU_ALIGNMENT124_SELECTIVE_BOOTSTRAP=1
H3_NPU_ALIGNMENT124_IPC_KEY_FILE=<new-0600-key-under-/tmp/h3_align12401>
CANN_KNOWLEDGE_BANK_PROCESS_NUM=2
ASCEND_WORK_PATH=<new-private-work-directory>
VLLM_OMNI_DLO_DP_WAVE_TIMEOUT=7200
H3_NPU_VAE_PRESERVE_FP32_PARAMS124=1
H3_NPU_SAVE_FINAL_LATENTS124=1
H3_NPU_NAN_COMPAT124=1
H3_NPU_CAPTURE_NATIVE_MEDIA124=1
H3_NPU_RAW_OUTPUT_DIR124=<new-output-directory>
```

早期失败保留：简化 GPU 回放第7次前向非有限值，被排除出参考；旧 GPU audit 缺少环境变量；CANN manager 认证；居民 NPU 4 步元数据；默认 600s RPC watchdog 在第5次前向后停止正在推进的长请求。均没有被重新解释为成功数值结果，也没有放宽 1% 门槛。

## 尚未达成的目标

本轮是带 offload、输出检查、启动/搬运的正确性运行，不是 A2/A3/GPU 速度基准。观察值：encoder stage 65.11s、warm DiT median 38.33s/step、native VAE decode 138.55s，generation wall 2316.96s（不含 engine init 232.21s）。**encoder <2s、DiT <15s/step、VAE <20s 均尚未达成，A2/A3 本轮没有实测。**

GPU/NPU 工作进程均已退出，临时 HTTP 服务与 key 已清理，两个 owned 租约均已释放。见[NPU 空闲核验](<artifacts/diagnostics/round124/NPU_post_generation_resource_usage124.txt>)和[GPU 清理回执](<artifacts/diagnostics/round124/GPU_reference_cleanup124.json>)。长期性能目标保持未完成；本轮仅完成数值门、真实生成和有边界的样例检查。
