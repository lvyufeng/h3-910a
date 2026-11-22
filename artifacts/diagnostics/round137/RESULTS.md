# R137 — 原数学固定输入 encoder / 固定 latent 音频 decoder 重复性定位

用户授权：仅做原数学不变（default False）、固定输入 encoder TP8 与固定 native audio latent
decoder 的重复性定位；无候选、无新容差、无 49 步生成、无 DiT/VideoVAE 计算。

## 结论（实测）

**原组件在 910A 上本身就不可逐位重复**，这不是候选或诊断引入的：

| 组件 | 重复次数 | 结果 |
|---|---|---|
| 原 encoder（真实 TP8，FP16，8 rank） | 8 次/rank，固定输入 | 全部互不相同；96.1–96.4% 输出词不同（10.0–10.8M / 10,460,160），relRMS **0.88–2.32%**，maxabs **320–976**（FP16） |
| 原 audio decoder（FP32，每 rank 完整副本） | 8 次/rank，固定 latent | 全部互不相同；99.99% 词不同（331,173–331,196 / 331,200），maxabs **4.96e-5…6.11e-5**，relRMS **82.7–84.3%**（参考输出近静音，RMS≈1.15e-5） |

**首次分叉边界（8 个 rank 完全一致）**

- encoder：`encoder.text.layers.0` 的 **post** 输出（事件序号 62）。
  input_changed=False、parameter_changed=False、output_changed=True。
  即：输入词完全相同、参数/CPU master 完全相同，**第一个 text layer 的输出就已经不同**，
  之后逐层放大到整段 hidden。
- audio decoder：`audio.decoder.resblocks.9.convs1.0.parametrizations.weight.0` 的 **post**
  输出（事件序号 270），这是原生 `parametrizations.weight_norm` 由同值 weight_g/weight_v
  **当次计算出来的**规范化权重；input/parameter 均未变，derived weight 变了。
  resblocks.0–8 及之前全部边界两次完全相同。

**输入门（完整词域，零差异）**

- encoder 实际原生 `_encode` 参数：[2043] int64 ids 0/2043 异；pixel_values [8064,1536] fp16
  0/12,386,304 异；image_grid_thw [2,3] int64 0/6 异；videos 为 None。冻结后 NPU clone 与原输入
  逐词一致，可用 format/shape/stride 一致。
- audio 固定 latent：R136 `r01_original_false` 的 `final_latents` 整文件 SHA
  `bf9daec8…f0f5d02e0`，audio_latent [2,32,207] fp32，原 CPU 与 NPU 冻结副本
  0/13,248 词差异。

**参数/状态门**：两组件 before/after 参数、buffer、layerwise hook 的 immutable CPU master
（含指针）与 stager 状态完全相同；原方法 `_encode` descriptor 精确还原；全程
default False、warn False、HF32 off、parametrize cache 关闭，final runtime 与 initial 相同。

## 时间（组件级，含同步+D2H+哈希审计，不与 R136 stage 口径混用）

- engine 初始化 251.69 s。
- encoder：native 中位 3.047 s（8 次 2.93–3.12 s）；trace 6.18 s；移除 observer 后 3.07 s。
- audio decoder：native 首次 59.28 s（含 staging 冷启），其后中位 0.360 s；trace 1.87 s；
  移除 observer 后 0.359 s。峰值 PyTorch allocated：encoder 5.251 GiB，audio 2.152 GiB。

## 对 R136 归因的影响（不采用任何容差）

R136 记录的 originalFalse→restoredFalse encoder 差 3.6968%、candidate→original 2.5306%，
与本轮原组件自身重复波动（0.88–2.32%，跨进程历史值到 3.7%）同量级。因此 R136 中这些
encoder 级差异**不能**作为候选/恢复的因果证据；audio 侧 6e-5 级差异同理落在原噪声内。
本轮不据此接受或否决任何候选，也不改变任何数值策略。

## 失败尝试（均保留，未 replay）

- 137a：`set_forward_context(od_config=…)` 参数名错误 → 准备阶段 TypeError，0 次固定重放；
  private atomic fatal 清理正常，Issue 已记录。
- 137b：同环境同 kwargs 下 rank7 worker 在 layerwise 原生 `_to_cpu` pinned Half copy 中
  原生 SIGSEGV（exit -11），orchestrator 收 EOF；host 722 GiB free、8 卡 health OK、
  无残留进程，判为瞬态原生异常；crash trace 存 `attempt137b_failure/`。
- 137c：同 pinned 数学重试成功（上表数据即此轮）。

## 边界与限制

- trace 是旁路同步观测（D2H+完整 SHA），nonclean；trace 一致/不一致不等于 kernel 物理确定性，
  也不能证明 CANN 内部 padding/算子因果；audio 侧 boundary 选择到 resblock/up/pre/post conv
  与各 weight_norm 参数化，未覆盖每个 AMP 内部 c1/c2 Conv/activation。
- 本轮没有任何候选启用、没有新确定性策略、没有 155 checks/49 步声明；无权重或激活归档传到客户端。
- private 进程/key 已清理，ts-217 lease 已释放。