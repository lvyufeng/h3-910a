## R137 结果：原数学固定输入组件本身就不可逐位重复

**状态：Awaiting approval（需要用户决定下一步数值策略，本轮不采用任何容差）**

### 实测（原 default False、原 FP16 encoder TP8、原 FP32 audio VAE，无候选/无 49 步）

| 组件 | 8 次固定输入重复 | 差异规模 |
|---|---|---|
| encoder TP8 | 全部互不相同 | 10.0–10.8M / 10,460,160 词不同（≈96%），relRMS 0.88–2.32%，maxabs 320–976 |
| audio decoder FP32 | 全部互不相同 | 331,173–331,196 / 331,200 词不同，maxabs 4.96e-5…6.11e-5，relRMS 82.7–84.3%（输出近静音） |

**首次分叉（8 rank 一致）**

- encoder → `encoder.text.layers.0` post（序号 62）：输入不变、参数不变、输出变化。
- audio → `audio.decoder.resblocks.9.convs1.0.parametrizations.weight.0` post（序号 270）：
  输入与 weight_g/weight_v 不变，但原生 `parametrizations.weight_norm` **当次物化出的权重**变了；
  resblocks.0–8 及之前全部边界两次完全一致。

**输入门（完整词域零差异）**：encoder ids 0/2043、pixel_values 0/12,386,304（fp16）、grids 0/6；
audio latent 0/13,248（fp32，源文件 SHA `bf9daec8…f0f5d02e0`）。
参数/buffer/immutable CPU master/hook 状态 before-after 完全一致，原 `_encode` 精确还原，
final runtime == initial runtime（default False、warn False、HF32 off、parametrize cache 关闭）。

### 对 R136 归因的直接影响

R136 的 originalFalse→restoredFalse encoder 差 3.6968%、candidate→original 2.5306%，与本轮原组件
自身跨次波动（0.88–2.32%）**同量级**；audio 6e-5 级差异同样落在原噪声内。因此这些 encoder 级差异
不能作为候选或恢复的因果证据。本轮既不接受也不否决候选，未修改任何数值策略。

### 时间（组件级，含同步+D2H+哈希审计，不与 R136 stage 口径混用）

engine init 251.69 s；encoder native 中位 3.047 s（trace 6.18 s / 移除 observer 3.07 s）；
audio native 首次 59.28 s（staging 冷启）其后中位 0.360 s（trace 1.87 s / 移除后 0.359 s）。
峰值 allocated：encoder 5.251 GiB，audio 2.152 GiB。

### 失败尝试（保留、未 replay）

- 137a：`set_forward_context(od_config=…)` 参数名错，准备阶段 TypeError，0 次重放，清理正常。
- 137b：同环境同 kwargs 下 rank7 worker 在 layerwise 原生 `_to_cpu` pinned Half copy 中 SIGSEGV(-11)；
  host 722 GiB free、8 卡 OK、无残留进程，判瞬态原生异常，crash trace 已存。
- 137c：同 pinned 数学重试成功，上表为其数据。

### 需要用户决定

原数学重复性已不成立，继续做 candidate/restored 的差异归因没有统计基础。可选方向：
1. 保持现状：把"原组件不可逐位重复"作为已确认事实记录，不再做数值归因（不新增策略）。
2. 允许**诊断性**定位非确定性来源（例如临时关闭已接受的 910A FA 路径，或临时启用确定性算法开关）——
   这些都会偏离已接受参考路径，仅用于诊断、不采纳、不用于性能口径。
3. 其他你指定的边界。

未获明确授权前不会运行上述任一诊断，也不会采用任何 tolerance。