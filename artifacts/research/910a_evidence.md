# MiniMax H3 在 910A 代际 8×32 GiB 平台上的公开证据审计

**检索与访问截止：2026-11-09（Asia/Shanghai）。** 本报告审查两个独立 4 卡实例部署 H3 的公开可检查证据。目标假设由本项目给出：8 张 32 GiB HBM 的 910A 代际设备，分为两个 4 卡 HCCS 组，并按 2D-mesh 组织；CANN 目标为 9.2.0；主精度为 FP16；INT8 仅用于已分别验证的量化权重或算子；不假设 BF16、FP8、NVFP4 或 SLA 在目标平台可用；每个实例优先使用一个 4 卡组，即 `0-3` 和 `4-7`。无后缀 `910B` 在本报告中按 910A 代际处理；出现 `910B3` 时保留原称，且不把两种标识互换。

## 范围与证据等级

审查范围包括 MindIE-SD issue、模型与实现仓库、官方模型规格、平台公告和可检索的公开传播材料。结论按“单条断言”分级，不按发布者整体继承：

- **A，可检查原始材料：** 有命令、配置、源码、原始日志、可下载基准回执或官方模型规格可直接核对。A 级只表示材料可核查，不保证其硬件属于目标平台。
- **B，第一方未附完整运行轨迹：** 厂商、平台或项目对自身工作的公告、README 性能表或部署自述。可作为适配线索，不能替代可复现回执。
- **C，二手或残缺材料：** 搜索结果摘要、已删除原文、AI 生成页面或无法取得附件内容的转述。只记录“该说法存在”，不把说法升级为运行事实。

“未披露”表示来源没有提供该字段；“未找到”表示在本报告列出的检索范围内没有发现，不表示发布方从未做过、互联网上不存在或该能力已被否定。

## 来源与状态台账

访问日期均为 **2026-11-09**。HTTP 状态为本次直接请求或带普通浏览器 User-Agent 的请求结果；反爬策略可能使同一页面在不同网络环境下返回不同状态。

| 编号 | 来源 | URL | 状态 | 在报告中的用途 |
|---|---|---|---|---|
| S1 | GitCode MindIE-SD issue `#339` | https://gitcode.com/Ascend/MindIE-SD/issues/339 | HTTP 200 | 目标代际 FP16 失败日志、环境、请求和评论时间线 |
| S2 | GitCode issue `#339` API | https://api.gitcode.com/api/v5/repos/Ascend/MindIE-SD/issues/339 | HTTP 200 | 标题、`closed`、标签、日期与正文元数据 |
| S3 | GitCode issue `#339` 评论 API | https://api.gitcode.com/api/v5/repos/Ascend/MindIE-SD/issues/339/comments | HTTP 200 | 5 条评论、报告者目标值和关闭意见 |
| S4 | GitCode milestones API | https://api.gitcode.com/api/v5/repos/Ascend/MindIE-SD/milestones | HTTP 200，响应为 `[]` | 确认 issue 未关联公开 milestone |
| S5 | GitHub 同号 issue | https://github.com/Ascend/MindIE-SD/issues/339 | **HTTP 404** | 证明 GitHub 路径本次不可访问 |
| S6 | GitHub 同号 issue API | https://api.github.com/repos/Ascend/MindIE-SD/issues/339 | **HTTP 404** | 复核 GitHub 仓库 API 同样没有该号 issue |
| S7 | 成都智算中心公告 | https://www.cdzs.cn/news/208 | HTTP 200（浏览器 UA） | 910A 适配公告，B 级 |
| S8 | 成都智算中心首页 | https://www.cdzs.cn/ | HTTP 200（浏览器 UA） | 平台通用算力背景，不含 H3 运行参数 |
| S9 | 搜狗结果页 | https://www.sogou.com/web?query=%E4%BA%91%E5%8D%97%E6%95%B0%E6%B4%BE%20MiniMax%20H3%20%E6%96%87%E7%94%9F%E8%A7%86%E9%A2%91 | HTTP 200（可能转入验证页） | 云南数派标题、摘要和索引时间 |
| S10 | 云南数派微信原文跳转 | http://mp.weixin.qq.com/s?src=11&timestamp=1794157543&ver=7076&signature=*UrE7w73K8wnJc0VLpO*vT6aYiZhopEmukV9G9HyyPEUa3Za3kohpr3ex2tPLlgrotDfZEUQfS2VCyt0UExMj6LUD58eNXGJjPOSzGbTPkAxRLGorp-ykKGwNGWR7NVo&new=1 | HTTP 200，正文显示“该内容已被发布者删除” | 确认原文已不可作为正文证据 |
| S11 | MiniMax-H3 官方仓库 | https://github.com/MiniMax-AI/MiniMax-H3 | HTTP 200 | 开源模块、checkpoint 和通用部署示例 |
| S12 | 官方 README 原文 | https://raw.githubusercontent.com/MiniMax-AI/MiniMax-H3/main/README.md | HTTP 200 | 模型架构、默认输出规格和初始开源边界 |
| S13 | Hugging Face 官方模型页 | https://huggingface.co/MiniMaxAI/MiniMax-H3 | HTTP 200 | checkpoint 与模型卡交叉核对 |
| S14 | Ascend-H3 仓库 | https://github.com/Huage001/Ascend-H3 | HTTP 200；提交 `c1572731d188174cd22155acd30b0ded403608be` | 910B3/CANN 9.2.0/INT8 的相邻代际回执，非目标证据 |
| S15 | Ascend-H3 INT8 回执 | https://github.com/Huage001/Ascend-H3/blob/c1572731d188174cd22155acd30b0ded403608be/benchmarks/ascend-int8-e2e.json | HTTP 200 | `40/40` E2E 记录及四卡数值 |
| S16 | Ascend-H3 SP 回执 | https://github.com/Huage001/Ascend-H3/blob/c1572731d188174cd22155acd30b0ded403608be/benchmarks/ascend-sp-e2e.json | HTTP 200 | SP4 记录与独立来源哈希 |
| S17 | minimax-h3-ascend | https://github.com/zhenhanchen1-cyber/minimax-h3-ascend | HTTP 200；提交 `14363607c8801458b27ce9f87f1c6d3dc7528247` | 裸 `910B`、2×64 GiB、INT8/ComfyUI 自述，B 级，缺少原始运行回执 |
| S18 | GitCode 镜像 | https://gitcode.com/nxxwjbl/minimax-h3-ascend-uncensored | HTTP 200 | `minimax-h3-ascend` 的镜像入口与交叉入口 |

`#339` 的 GitHub 与 GitCode 状态需要分开表述：同号 GitHub issue 和 GitHub API 都是 **404**，而同号 GitCode 页面、issue API 和评论 API 都是 **200**。[S1][S2][S3][S5][S6] GitCode API 返回的 `state` 为 `closed`，标签为 `usage`、`resolved`，`milestone` 为空，评论数为 5。本次直接请求 GitCode milestones API 得到空数组 `[]`；公开元数据没有给出 milestone 名称。[S4]

四份 issue 附件在本次访问中都返回 **HTTP 401**：两个 CSV 和两个 MP4。响应正文为 `{"error_code":401,"error_code_name":"UNAUTHORIZED","error_message":"You are not signed in."}`。因此，原始 profiler 表格与两段视频文件的内容没有被纳入性能计算。

## 官方模型规格

MiniMax 官方 README 把 H3 描述为通用全模态生成系统。最高 2K、最长 15 秒、24 FPS、32 kHz 立体声是模型输出能力描述，不是 910A 实测。与部署直接相关的事实是：

- H3-Base 发布的 checkpoint 精度为 **BF16**；H3-Omni-Transformer 是 **33B dense** 单流 Transformer。[S11][S12]
- 约 **13B 参数**位于 AdaLN 相关分支；官方称这些分支的输出可预计算和缓存，因此仅推理部署不必加载这些参数。[S12]
- 默认输出短边 **768 像素**，输出帧率 **24 FPS**，输出时长 **4–15 秒**，音频为 **32 kHz stereo**。[S12]
- 初始开源版本只提供 **full attention**；原生 sparse attention 的训练和推理能力在模型层面存在，但实现没有随初始版本发布。[S12]
- 2K 输出依赖 **H3-Regenerate-2K**，该模块尚未开源。[S12]
- 官方 SGLang 示例使用 `--num-gpus 4 --ulysses-degree 4`，这是通用框架示例，没有登记 910A、CANN、HBM 或成功运行日志。[S12]

这些规格可用于判断接口和模型边界，不能用来证明任何昇腾平台的实际内存占用或时延。Ascend-H3 的 INT8 路径把 Q/K/V 以 BF16 送入 attention；这不构成 BF16 在 910A 上可用的证据。

## 公开性能对照表

下表合并公开条目，但保留平台、精度和计时常口径。没有证据的格子写“未披露”，不从 FPS 与时长反推帧数，也不把采样、VAE 和 E2E 混成同一列。

| 来源 | 等级 | 平台 / 卡数 / 每卡 HBM | 分辨率 / 帧数 / 时长 | 步数 / 精度 / 并行 | 采样耗时 | VAE 耗时 | E2E 耗时 | 并发与口径 |
|---|---|---|---|---|---|---|---|---|
| S1–S3 MindIE-SD `#339` 失败请求 | A（原始失败日志） | `910PremiumA`，8 卡，约 32 GiB（`32741–32768 MiB`） | 1280×720 / 未披露 / 5 秒 | 4 步 / FP16 / USP8、ring1、文本 TP8、VAE tile 与 patch parallel 8 | 未披露 | 未披露 | 未披露 | 单实例单请求；在视频 VAE 中 OOM |
| S1–S3 `#339` 失败请求 profiler 行 | A（失败跨度） | `910PremiumA`，8 卡，约 32 GiB（`32741–32768 MiB`） | 1280×720 / 未披露 / 5 秒 | 4 步 / FP16 / USP8、ring1、文本 TP8、VAE tile 与 patch parallel 8 | 未披露 | 失败请求 `decode` 跨度 91.720247 秒 | 失败请求 `forward` 跨度 628.084641 秒（不是成功 E2E） | 两项都不是成功计时 |
| S7 成都智算中心公告 | B（第一方公告） | 910A 集群（Ascend 910 Premium A）/ 未披露 / 未披露 | 未披露 / 未披露 / 未披露 | 未披露 / 未披露 / 未披露 | 未披露 | 未披露 | 未披露 | 未披露；公告未给运行回执 |
| S9–S10 云南数派搜索摘要 | C（原文已删除） | 910A / 未披露 / 未披露 | 未披露 / 未披露 / 成片约 5.18 秒 | 未披露 / 未披露 / 未披露 | 未披露 | 未披露 | 未披露；摘要称“平均用时约 133s”，测量范围未定义 | 未披露 |
| S17 `minimax-h3-ascend` 0.4MP、4 步 | B（第一方自述；无原始 trace） | 裸 `910B`，2 卡，每卡 64 GiB | 864×480 / 124 / 5 秒 | 4 步 / INT8 ConvRot / 2 卡 ComfyUI，机制未披露 | 17.8 秒 | 未披露 | 冷启 71 秒；热机 20–25 秒 | 未披露 |
| S17 `minimax-h3-ascend` 0.4MP、8 步 | B（第一方自述；无原始 trace） | 裸 `910B`，2 卡，每卡 64 GiB | 864×480 / 124 / 5 秒 | 8 步 / INT8 ConvRot / 2 卡 ComfyUI，机制未披露 | 36.4 秒 | 未披露 | 89.7 秒 | 未披露 |
| S17 `minimax-h3-ascend` 0.6MP | B（第一方自述；无原始 trace） | 裸 `910B`，2 卡，每卡 64 GiB | 1024×576 / 192 / 8 秒 | 4 步 / INT8 ConvRot / 2 卡 ComfyUI，机制未披露 | 约 40 秒 | 未披露 | 约 60–90 秒 | 未披露 |
| S17 `minimax-h3-ascend` 1.0MP、长片 | B（第一方自述；无原始 trace） | 裸 `910B`，2 卡，每卡 64 GiB | 1152×864 / 294 / 12 秒 | 8 步 / INT8 ConvRot / 2 卡 ComfyUI，机制未披露 | 约 71 秒/步，合计约 570 秒 | 未披露 | 约 11 分钟 | 未披露 |
| S14–S15 Ascend-H3，124 帧 Full TP4 | A（可检查回执；非目标平台） | 自述仅 `910B3`，4 卡，HBM 未披露 | 1344×768 / 124 / 未披露 | 实际 8 次 DiT forward / INT8，Q/K/V 以 BF16 进入 attention / TP4 | 未披露 | 未披露 | 64.78 秒 | 未披露；非 910A |
| S14–S16 Ascend-H3，124 帧 Full SP4 | A（可检查回执；非目标平台） | 自述仅 `910B3`，4 卡，HBM 未披露 | 1344×768 / 124 / 未披露 | 实际 8 次 DiT forward / INT8，Q/K/V 以 BF16 进入 attention / SP4 | 未披露 | 未披露 | 52.20 秒 | 未披露；非 910A |
| S14–S15 Ascend-H3，124 帧 Veda TP4 | A（可检查回执；非目标平台） | 自述仅 `910B3`，4 卡，HBM 未披露 | 1344×768 / 124 / 未披露 | 实际 8 次 DiT forward / INT8 Veda 90%，Q/K/V 以 BF16 进入 attention / TP4 | 未披露 | 未披露 | 43.38 秒 | 未披露；非 910A |
| S14–S16 Ascend-H3，124 帧 Veda SP4 | A（可检查回执；非目标平台） | 自述仅 `910B3`，4 卡，HBM 未披露 | 1344×768 / 124 / 未披露 | 实际 8 次 DiT forward / INT8 Veda 90%，Q/K/V 以 BF16 进入 attention / SP4 | 未披露 | 未披露 | 29.59 秒 | 未披露；非 910A |
| S14–S15 Ascend-H3，345 帧 Full TP4 | A（可检查回执；非目标平台） | 自述仅 `910B3`，4 卡，HBM 未披露 | 1344×768 / 345 / 未披露 | 实际 8 次 DiT forward / INT8，Q/K/V 以 BF16 进入 attention / TP4 | 未披露 | 未披露 | 328.63 秒 | 未披露；非 910A |
| S14–S16 Ascend-H3，345 帧 Full SP4 | A（可检查回执；非目标平台） | 自述仅 `910B3`，4 卡，HBM 未披露 | 1344×768 / 345 / 未披露 | 实际 8 次 DiT forward / INT8，Q/K/V 以 BF16 进入 attention / SP4 | 未披露 | 未披露 | 299.26 秒 | 未披露；非 910A |
| S14–S15 Ascend-H3，345 帧 Veda TP4 | A（可检查回执；非目标平台） | 自述仅 `910B3`，4 卡，HBM 未披露 | 1344×768 / 345 / 未披露 | 实际 8 次 DiT forward / INT8 Veda 90%，Q/K/V 以 BF16 进入 attention / TP4 | 未披露 | 未披露 | 130.11 秒 | 未披露；非 910A |
| S14–S16 Ascend-H3，345 帧 Veda SP4 | A（可检查回执；非目标平台） | 自述仅 `910B3`，4 卡，HBM 未披露 | 1344×768 / 345 / 未披露 | 实际 8 次 DiT forward / INT8 Veda 90%，Q/K/V 以 BF16 进入 attention / SP4 | 未披露 | 未披露 | 98.70 秒 | 未披露；非 910A |

Ascend-H3 的 E2E 口径为：从新 prompt 提交开始，到完整 MP4 接收并写盘结束；包含文本编码、全部去噪更新、音频与视频解码、MP4 编码和响应传输，加载、已完成的 LoRA merge、启动时 AdaLN 准备与 warmup 不计入。每个方法及基线执行 8 次实际 DiT forward，seed 为 7。来源没有提供采样子项或 VAE 子项耗时，故两者均记“未披露”。[S14][S15][S16]

## 已证实事实

**1. 公开的 910A 代际直接运行证据只有失败请求，未见目标规格的成功 E2E 日志。** `#339` 报告的机器有 8 张 `910PremiumA`，`npu-smi` 显示每卡总量 `32741` 或 `32768 MiB`，错误日志显示 `31.97 GiB total capacity`。系统为 EulerOS 2.0 SP8，CANN 9.1.0，TorchNPU 2.10.0，vLLM 0.26.0，`npu-smi` 25.5.2；驱动版本未单独披露。[S1][S2]

**2. `#339` 的请求是 FP16 720p，但不是 CANN 9.2.0 目标验证。** 请求参数为 `width=1280`、`height=720`、`fps=24`、`num_inference_steps=4`、`flow_shift=12`、`audio_flow_shift=3.0`、`seed=2026`、`duration=5`。启动使用 `--dtype float16 --num-gpus 8 --usp 8 --ring 1 --text-encoder-tp-size 8 --enable-distributed-layerwise-offload --vae-parallel-mode tile --vae-use-tiling --vae-patch-parallel-size 8`。帧数没有被请求脚本或正文披露，不能把 24 FPS × 5 秒写成已观测帧数。[S1][S3]

**3. 环境存在一处 CANN 表述差异。** 正文版本表写 CANN 9.1.0，但启动脚本设置 `CANN_OPP=/mnt/models/cann910-full/cann-9.0.1/opp`，并把该目录挂到容器内的 `cann-9.0.1/opp`。这既不能确认为 CANN 9.2.0 成功，也不能仅凭目录名把整机版本改写成 9.0.1；可核实结论是版本声明与 OPP 路径不一致。[S1][S3]

**4. 失败发生在视频 VAE 解码路径。** 原始栈为 `decode_latent()` → `decode_base()` → `decode_temporal()` → `_decode_temporal_streaming()`，最后在 `clip_dec = clip_dec.to(z.device)` 触发 OOM。错误显示 NPU 0 总容量 31.97 GiB、已分配 28.01 GiB、空闲 271.64 MiB，尝试再分配 290.00 MiB。[S1]

**5. `91.720247` 秒和 `628.084641` 秒是失败请求中的 profiler 跨度，不是成功计时。** 原始行分别为 `MiniMaxH3Pipeline.decode took 91.720247s` 和 `MiniMaxH3Pipeline.forward took 628.084641s`，随后同一请求报 OOM。可用结论是它执行到了 VAE 和整条 forward 的失败位置；成功的采样、VAE 和 E2E 时延均未披露。缺失的 CSV 也不能补成成功结果。[S1][S3]

**6. `#339` 的 `resolved` 标签和关闭状态与最后一条维护意见存在矛盾。** issue 元数据为 `closed`，标签包含 `resolved`，5 条评论，创建于 `2026-09-01T09:51:02+08:00`，更新于 `2026-09-28T21:57:47+08:00`；公开 milestone 为空。[S2][S3][S4] 最后一条评论是：

> “该issue涉及的机器不属于维护范围也没有有效prof分析，暂时关闭”

同日的维护意见还说：

> “目前我们只适配过64G以上显存的机器，32G机器的OOM问题正在解决中”

因此，关闭状态不能被解释为维护者已在 32 GiB 机器上完成验证。[S3]

**7. 报告者的“OOM 已解决”与“100 秒内”没有公开成功证据。** 2026-09-04 的评论写道：

> “以上的OOM的问题已经解决，对标其他平台910A生成的视频，我们当前生成的视频性能和精度还有差异。”

随后在“期望的性能和精度”下提出：

> “1. 5s 768的t2v视频，可以在100s之内完成生成；”

这是目标表达。该评论没有修复命令、成功启动参数、版本变化、成功 trace 或成功 E2E；两个视频附件本次均无法下载。报告者同日另一条说法提到“468分辨率”可以推理到 10 秒，但“468”不是可核对的分辨率规格，本报告不把它解释成具体宽高或成功基准。[S3]

**8. 成都是官方适配公告，不是可复现成功日志。** 页面标题为“成都智算中心基于910A成功适配MiniMax H3”，发布时间为 `2026-09-04 10:36:19`。原文称：

> “成都智算中心联合生态伙伴基于昇腾910A国产算力平台，完成该模型的适配验证工作。”

> “此次适配，由技术团队逐层自研、逐层镜像整条推理链路，完成专项技术攻坚，在昇腾910A集群（Ascend 910 Premium A），打通了MiniMax H3 全链路推理与调优流程。”

页面没有披露每实例卡数、每卡 HBM、CANN/驱动、精度、分辨率、用于测量的帧数与时长、步数、并行策略、采样/VAE/E2E、并发或实际 launch 状态。[S7] 公告还有一处真实日期矛盾：发布时间是 9 月 4 日，正文却写“该模型预计将于 8 月13日24:00前正式上线星海智算平台”。本报告不据此认定已上线。[S7]

**9. 云南数派目前只能引用搜索摘要，原文已经删除。** 搜狗结果标题为“模型适配进展｜从昇腾910A上的MiniMax H3文生视频，到多参创作”，摘要可见“成片约5.18秒”“生成时间：9月21日”“平均用时约133s”。结果页索引时间戳 `1793675910` 对应 **2026-11-03 11:18:30 +08:00**，但这是索引时间，不是已核实发布时间。[S9] 微信跳转页面 HTTP 200，正文显示“该内容已被发布者删除”。[S10] “5.18 秒”是成片时长；“133s”没有说明包含采样、VAE 还是 E2E，也没有给出单次/多次、卡数、HBM、版本、精度、分辨率、帧数、步数、并行策略或并发。[S9][S10]

**10. 官方 H3 的 BF16 checkpoint 和 sparse attention 状态是可核对的模型事实。** 官方 checkpoint 为 BF16；初始开源版本只提供 full attention，sparse 实现待发布；2K 依赖尚未开源的 H3-Regenerate-2K；官方四卡 SGLang 示例没有 910A 运行记录。[S11][S12][S13] 这些事实不能被转写成“910A 支持 BF16、sparse 或 2K”。

**11. Ascend-H3 的四卡回执可检查，但硬件明确是 `910B3`，不属于目标 910A 代际。** 仓库自述硬件为 `Ascend 910B3 only`，每卡 HBM 和驱动未披露。环境为 Python 3.12.14、Torch 2.10.0+cpu、Torch-NPU 2.10.0.post2、CANN 9.2.0。[S14] 验收范围为：

> “**Validated scope:** 40/40 INT8 E2E cells at 1344×768, 124/345 frames and 1/2/4/8 NPUs (37 fresh measurements and 3 previously validated layer-Offload measurements).”

此外，`30/30 fresh SP cells` 覆盖 2/4/8 NPU；选中的 SP 记录保持 DiT resident，文本编码器 Offload 单独标记。[S14][S16] 回执可以复核该平台的 INT8 E2E 与并行结果，不能借用为 910A FP16 结果。

**12. Ascend-H3 的 INT8 不是“全算子 INT8”。** 仓库写明：

> “INT8 projections feed BF16 Q/K/V into attention. Attention accumulators, Veda predictions and VDN state keep their prescribed precision. This is INT8 model inference, not a claim that every attention operation is INT8.”

并写明：

> “The native recipe retains unsupported layers in BF16.”

因此其 attention 输入仍为 BF16，未支持层仍为 BF16；该路线不能证明目标机器的 BF16 或全链路 INT8 可用。[S14]

**13. `minimax-h3-ascend` 的环境是裸 `910B`、2×64 GiB，不是目标 4×32 GiB。** 仓库固定在提交 `14363607c8801458b27ce9f87f1c6d3dc7528247`，其 README 的环境段原文为：

> “- 硬件: 2x Ascend 910B (64GB HBM each, 合计 128GB), 229GB RAM, 40 core, 196GB disk”
>
> “- 驱动: 25.5.5 (V100R001C23SPC009B220)”
>
> “- CANN: 9.0.0 …”
>
> “- torch: 2.7.1+cpu / torch_npu: 2.7.1.post4”

按本题约定，裸 `910B` 归入 910A 代际；报告不把它改写成 `910B3`、`910C` 或其他型号。2 张 64 GiB 卡合计 128 GiB 仍是 2 卡拓扑，不等于 4×32 GiB，也不是双 4 卡实例。该仓库的 CANN 9.0.0 与驱动 25.5.5 不构成 CANN 9.2.0 目标验证。[S17]

**14. `minimax-h3-ascend` 使用 INT8 ConvRot 和 ComfyUI，性能自述缺少原始回执。** README 给出 864×480、124 帧、4 步的采样 17.8 秒、冷启 E2E 71 秒、热机 E2E 20–25 秒；864×480、124 帧、8 步的采样 36.4 秒、E2E 89.7 秒；1024×576、192 帧、4 步的采样约 40 秒、E2E 约 60–90 秒；1152×864、294 帧、8 步的采样约 570 秒、E2E 约 11 分钟。[S17] 另有单卡 32 GiB、480p/5 秒、稳态约 27–30 GiB、约 2–5 分钟的自述，以及“8 卡以上可多实例并行增加吞吐”的建议。[S17] 这些内容都没有逐次原始 trace、每卡 HBM 曲线、完整服务器响应或并发日志；4×64 GiB/910C 的描述也超出目标平台。它们只能支持路线线索，不能作为目标验收。

**15. 本次检索没有找到目标配置的成功 E2E 日志。** 没有找到同时满足“910A 代际、FP16、单个 4×32 GiB 实例或两个独立 4×32 GiB 实例、H3 完整 E2E、可检查 log/trace/receipt”的公开材料。尤其没有找到成功的 **FP16 4×32 GiB** 或 **dual 4×32 GiB** H3 E2E 日志。这里的 NOT FOUND 仅指本次列出的来源范围。[S1]–[S18]

## 合理推断

以下是由已核对材料推导出的部署判断，不是新增公开事实。

**双 4 卡都要先跨过“模型能否加载”和“VAE 能否完成”两道门。** 目标每个四卡实例约有 128 GiB 总 HBM，低于 `#339` 的 8×32 GiB 单实例约 256 GiB 总容量，且该请求已在 VAE 的 290 MiB 小分配上失败；4 卡实例需要重新配置文本 TP、USP/SP、VAE 分块和 offload。公开的 Ascend-H3 四卡回执说明规范性四卡路径在 `910B3` 上可检查，但该平台的 INT8/Q/K/V BF16 路线、HBM 和驱动都没有迁移到 910A 的证据。[S1][S14][S16]

**SP 和 TP 的降内存作用不能按卡数线性折算。** Ascend-H3 写明 SP 在 FFN 与残差上分配 token 行，但仍保留复制的 DiT master weights；TP 才切分相应权重。`#339` 已经同时启用 `--usp 8`、文本 TP8、layerwise offload 与 VAE tiling 仍 OOM，因此“4 卡序列并行”不能推出“每卡内存必然足够”。[S1][S14]

**VAE 是需要独立门禁的阶段。** 目标代际 A 级日志的确定失败点是视频 VAE 的 temporal streaming；同时，现有第三方时延自述没有可靠拆分 VAE 与采样。若两路实例都在同一阶段达到内存上限，仅错开启动不一定解决问题；需要先验证单实例 VAE 是否能稳定完成，再检查双路并发。[S1][S17]

**每实例绑定一个四卡 HCCS 组是合理的拓扑起点。** 题目给定两个 4 卡 HCCS 组组成 2D-mesh。将 A 放在 `0-3`、B 放在 `4-7` 可使实例内部通信留在组内，避免把跨组路径放进关键通信链。这是依据目标拓扑的工程推断；公开材料没有该拓扑的双实例 H3 结果，因此仍需实际 HCCL 与生成验收。

**CANN 9.2.0 上的相邻代际结果只能证明版本与代码路径可工作于其自身平台。** Ascend-H3 的环境登记为 CANN 9.2.0，硬件为 `910B3`；目标机器仍需验证 CANN 9.2.0、Torch-NPU、算子包、驱动和模型补丁的组合。版本号相同不等于算子设备兼容。

**INT8 只适合作为 FP16 基线之后的逐项优化。** Ascend-H3 的 INT8 依赖 BF16 Q/K/V 和保留 BF16 的未支持层，`minimax-h3-ascend` 的自述也没有给出逐算子数值与内存回执。目标若要使用 INT8 权重或算子，必须分别检查加载、数值一致性、峰值 HBM 和端到端产物；不能把第三方的 INT8 秒数直接换算成目标 FP16 上限或容量。

**两个 4 卡实例的吞吐不能由单路热机时间相加得到。** 双实例会叠加主机 CPU、内存、模型加载/换入、VAE 工作区、存储和网络压力；公开材料未给出目标拓扑的并发采样、两路 VAE 重叠、失败隔离或产物写盘记录。因此只能先建立单路稳定基线，再测同时提交双路时的有效吞吐。

## 未证实宣传

- **未证实：成都公告足以证明 910A 上“稳定落地”或“全链路调优”已完成可复现交付。** 它属于第一方 B 级公告，缺少每实例卡数、HBM、软件版本、精度、分辨率、帧数、步数、并行、分阶段时延、并发和原始回执；发布日与“8 月13日24:00前上线”的表述还发生日期倒置。[S7]

- **未证实：“高性价比、全栈国产化 AI 视频生产方案”可作为目标 SLA 或 TCO 结论。** 公告没有单条视频成本、吞吐、资源占用或并发数据。成都首页的“近千颗昇腾 910 处理器、300P FLOPS”是集群通用介绍，不是 H3 单实例或双实例性能。[S7][S8]

- **未证实：云南数派“平均用时约 133s”可以代表 910A 的目标性能。** 目前只有搜索摘要，原文已删除；摘要没有定义该平均值是采样、VAE 还是 E2E，也没有说明卡数、HBM、版本、精度、分辨率、帧数、步数、并行和并发。“5.18 秒”只能作为成片时长引用。[S9][S10]

- **未证实：`minimax-h3-ascend` 的冷/热机秒数、单卡 32 GiB“稳态 27–30 GiB”和“超过 8 卡可多实例”已经满足可复现标准。** 仓库自述给出了工作流和环境快照，但没有原始 trace、逐卡 HBM 曲线、完整版本锁、服务器响应、重复次数和并发日志；2×64 GiB 与单卡 32 GiB 的叙述都不是双 4×32 GiB 实测。[S17][S18]

- **未证实：`#339` 的“OOM 已解决”和 32 GiB 可用性。** 评论没有修复命令、成功版本、成功 trace 或 E2E；“100 秒内”是期望指标。维护者同日称 32 GiB 的 OOM 仍在解决，最后又以“不属于维护范围也没有有效prof分析”关闭。`resolved` 标签与关闭意见存在冲突，不能单独当作成功凭据。[S2][S3]

- **未证实：Ascend-H3 的 910B3 INT8 数字可迁移为 910A FP16 数字。** 该仓库硬件明确是 `910B3`，每卡 HBM 与驱动未披露，INT8 路径仍有 BF16 Q/K/V 和 BF16 未支持层；回执没有采样/VAE 子项。即使四卡 E2E 数字可检查，也缺少目标平台的 FP16 复现。[S14][S15]

- **未证实：BF16、FP8、NVFP4、SLA 或 sparse 可作为目标默认加速开关。** 官方 checkpoint 是 BF16，但目标主精度为 FP16；官方初始版本只有 full attention；Ascend-H3 的 INT8 依赖 BF16 attention 输入；`minimax-h3-ascend` 把 FP8/NVFP4 标为昇腾不适用或 N 卡路线。公开材料没有这些能力在目标 910A/CANN 9.2.0 上可用的日志。[S11][S12][S14][S17]

## 对双路部署的直接含义

**当前可证明的不是“双路可跑”，而是目标代际最接近的一次 FP16 请求在 8×32 GiB 上于 VAE OOM。** 该请求不能直接代表 4×32 GiB 单实例，也不能代表两个 4 卡实例同时执行；它证明的是现有低容量路径还没有跨越完整 E2E 的成功门槛。[S1]

**双 4 卡应被视为待验证的部署形态，而不是已知容量结果。** `0-3` 和 `4-7` 的隔离能减少实例内部的跨组通信，但每个实例都必须独立加载或换入所需模型、文本编码器和 VAE，并独立管理采样与解码峰值。现有公开材料没有该拓扑的并发样本。

**不能把 `#339` 的 8 卡配置机械除以二。** 其 8 卡实例使用 USP8、文本 TP8、VAE patch parallel 8、distributed layerwise offload 和 tiled VAE；改为每实例 4 卡后，这些并行度、tensor 切分、VAE 分块规则和通信组都要重新配置。公开的 Ascend-H3 SP4/TP4 数字来自 `910B3` INT8，不能补上这个缺口。[S1][S14][S16]

**720p 是尚未通过的目标门槛。** 目标代际的 A 级日志在 1280×720、5 秒、4 步请求上失败；B 级第一方自述虽有较低分辨率成功数据，但平台是裸 `910B` 2×64 GiB、INT8/ComfyUI，口径不同。先证明低分辨率 FP16 单实例，再逐级提高 token、帧数和分辨率，才可讨论 720p 与并发。[S1][S17]

**双实例验收必须测“总量和互相影响”。** 需要同时记录两份请求的开始/结束、每 rank HBM 峰值、文本编码、DiT 更新、视频 VAE、音频 VAE、mux/写盘、两路时间重叠、HCCL/端口隔离与失败隔离。两次顺序运行的热机时间不能代替并发吞吐。

**CANN 9.2.0 与 FP16 必须由目标机器自行验证。** `#339` 是 CANN 9.1.0 声明并引用 9.0.1 OPP 路径；成都未披露版本；云南未披露精度；Ascend-H3 是非目标 `910B3` 的 CANN 9.2.0 INT8 路径；`minimax-h3-ascend` 是 CANN 9.0.0 的裸 `910B` INT8 自述。现有材料没有目标版本的 FP16 成功回执。

**量化路线应逐权重、逐算子准入。** INT8 只在权重和算子经过目标栈上的数值、内存与产物验证后使用。BF16、FP8、NVFP4、SLA、sparse attention 和 VSA/Veda/Spark 不能因第三方仓库出现相关字样就进入目标默认配置；无目标日志时，它们应保持不可用假设。

## 对 4 卡单实例和双 4 卡实例最有价值的 5 条实施建议

1. **先在 `0-3` 建立一个 FP16 单实例基线，并固定完整栈与权重的可审计身份。** 目标使用 CANN 9.2.0，记录驱动、Torch-NPU、框架/补丁提交、容器镜像、模型与 LoRA 的 SHA256、启动参数和完整日志。开启 tiled VAE 与必要的 offload 后，从最低内存可行的分辨率、帧数和 4 步请求开始，逐 rank 记录 HBM 峰值与成功 MP4。第二个实例在单实例通过前不启动。

2. **建立目标兼容性门禁，顺序覆盖 CANN 9.2.0 算子冒烟、FP16 模型加载、FP16 forward、视频 VAE 解码和音频/MP4 完成。** 每一关都要保存失败点、版本和输入参数；只有在上述阶段全部通过后，才允许验证经过明确标注的 INT8 量化权重或算子，并单独比对数值、精度、HBM 与产物。不得把 BF16、FP8、NVFP4 或 SLA 当作默认回退。

3. **单实例稳定后，再在 `4-7` 启动独立第二实例。** 两进程分别绑定 `0-3` 与 `4-7`，使用不同端口、rendezvous/master 端口、HCCL socket 范围、缓存、临时目录、输出目录和日志目录；每个实例的通信组限定在自身四卡组内，先做 HCCL/AllReduce 冒烟，再提交同一 H3 请求。独立设备集合用于避免显存和通信组重叠，不能把 HTTP 健康或 HCCL 探针通过当成模型推理通过。

4. **单独隔离并测量 VAE 瓶颈。** 在同一单实例基线下分别记录视频 VAE 的 temporal tiling 分片、拼接与峰值 HBM，再测试每个实例在本组内独立执行解码的路径；比较串行化解码与实例本地并行解码，避免共享工作区、同一设备集合或跨实例复用中间 tensor。任何 VAE 优化都要以完整音视频 MP4 成功和目视/数值质量检查为终点。

5. **为两个并发 MP4 定义可复现的验收基准。** 固定 prompt、分辨率、帧数、步数、seed、精度与 offload；记录文本编码、全部 DiT 更新、视频 VAE、音频 VAE、mux/写盘的逐阶段耗时，以及每 rank HBM 峰值、两路重叠时间、失败隔离和连续多轮稳定性。报告单路与双路的 P50/P95、成功率、产物规格和 SHA256；不以一次冷启或热机时间作为 SLA。

## 局限与证据缺口

- 目标平台最接近的 A 级证据是失败请求，成功的 **FP16 4×32 GiB** 与 **dual 4×32 GiB** E2E 日志均 **NOT FOUND**。这不表示相关能力不存在，只表示本次来源范围没有可检查的成功记录。
- `#339` 的 GitHub 同号 issue/API 为 404，GitCode 同号页面与 API 为 200；附件 CSV/MP4 为 401。报告不能复核缺失附件，也无法把报告者的“已解决”升级为成功运行。
- 成都是 B 级官方公告，不是 launch 日志或性能报告；页面日期与“8 月13日24:00前上线”的表述相互矛盾，未提供可用于复现的实验参数。
- 云南数派只剩 C 级搜索摘要，原文显示已被发布者删除；“133s”没有测量范围，不能用于 E2E、吞吐或 SLA。
- Ascend-H3 的 `40/40` 与 `30/30` 回执均属明确自述的 **910B3 only** 平台，且是 INT8、BF16 Q/K/V、未支持层 BF16 路线；没有目标 910A、FP16、4×32 GiB 或双实例证据。
- `minimax-h3-ascend` 是裸 `910B`、2×64 GiB、CANN 9.0.0、驱动 25.5.5 的 INT8/ComfyUI 自述；没有原始逐次 trace、每卡 HBM 曲线、完整服务器返回或并发日志。报告按题目约定把裸 `910B` 归于 910A 代际，但不把它改称为 `910B3`。
- 公开材料普遍缺少两路实例同时运行时的逐阶段耗时、每 rank 内存曲线、并发下的失败隔离、连续重复次数和产物校验。报告中的任何“100 秒”“133 秒”“20–25 秒”都不是目标 FP16 双路 SLA。

报告结论限于上述检索范围和访问状态。**NOT FOUND** 指本次审查没有找到，不是对公开互联网或未公开工程记录的绝对否定。

## 来源与访问日期

- **[S1]** MindIE-SD issue `#339`，GitCode 页面；状态 HTTP 200。访问日期：2026-11-09。https://gitcode.com/Ascend/MindIE-SD/issues/339
- **[S2]** GitCode issue API；状态 HTTP 200，`state=closed`，标签 `usage`/`resolved`，milestone 为空。访问日期：2026-11-09。https://api.gitcode.com/api/v5/repos/Ascend/MindIE-SD/issues/339
- **[S3]** GitCode comments API；状态 HTTP 200，5 条评论。访问日期：2026-11-09。https://api.gitcode.com/api/v5/repos/Ascend/MindIE-SD/issues/339/comments
- **[S4]** GitCode milestones API；状态 HTTP 200，响应 `[]`。访问日期：2026-11-09。https://api.gitcode.com/api/v5/repos/Ascend/MindIE-SD/milestones
- **[S5]** GitHub MindIE-SD issue `#339`；状态 **HTTP 404**。访问日期：2026-11-09。https://github.com/Ascend/MindIE-SD/issues/339
- **[S6]** GitHub issue API；状态 **HTTP 404**。访问日期：2026-11-09。https://api.github.com/repos/Ascend/MindIE-SD/issues/339
- **[S7]** 成都智算中心，《成都智算中心基于910A成功适配MiniMax H3》，发布时间 2026-09-04；状态 HTTP 200（浏览器 UA）。访问日期：2026-11-09。https://www.cdzs.cn/news/208
- **[S8]** 成都智算中心首页；状态 HTTP 200（浏览器 UA），含“近千颗昇腾 910 处理器、300P FLOPS”通用描述。访问日期：2026-11-09。https://www.cdzs.cn/
- **[S9]** 搜狗结果页；状态 HTTP 200，可能转入验证页，云南数派结果只剩摘要。访问日期：2026-11-09。https://www.sogou.com/web?query=%E4%BA%91%E5%8D%97%E6%95%B0%E6%B4%BE%20MiniMax%20H3%20%E6%96%87%E7%94%9F%E8%A7%86%E9%A2%91
- **[S10]** 云南数派微信原文跳转；状态 HTTP 200，正文“该内容已被发布者删除”。访问日期：2026-11-09。http://mp.weixin.qq.com/s?src=11&timestamp=1794157543&ver=7076&signature=*UrE7w73K8wnJc0VLpO*vT6aYiZhopEmukV9G9HyyPEUa3Za3kohpr3ex2tPLlgrotDfZEUQfS2VCyt0UExMj6LUD58eNXGJjPOSzGbTPkAxRLGorp-ykKGwNGWR7NVo&new=1
- **[S11]** MiniMax-AI/MiniMax-H3 官方仓库；状态 HTTP 200。访问日期：2026-11-09。https://github.com/MiniMax-AI/MiniMax-H3
- **[S12]** 官方 README；状态 HTTP 200，BF16 checkpoint、33B dense、full attention 等规格出处。访问日期：2026-11-09。https://raw.githubusercontent.com/MiniMax-AI/MiniMax-H3/main/README.md
- **[S13]** MiniMaxAI/MiniMax-H3 Hugging Face 模型页；状态 HTTP 200。访问日期：2026-11-09。https://huggingface.co/MiniMaxAI/MiniMax-H3
- **[S14]** Huage001/Ascend-H3，提交 `c1572731d188174cd22155acd30b0ded403608be`；仓库状态 HTTP 200，自述 `Ascend 910B3 only`，CANN 9.2.0。访问日期：2026-11-09。https://github.com/Huage001/Ascend-H3
- **[S15]** Ascend-H3 INT8 E2E 回执；状态 HTTP 200。访问日期：2026-11-09。https://github.com/Huage001/Ascend-H3/blob/c1572731d188174cd22155acd30b0ded403608be/benchmarks/ascend-int8-e2e.json
- **[S16]** Ascend-H3 SP E2E 回执；状态 HTTP 200。访问日期：2026-11-09。https://github.com/Huage001/Ascend-H3/blob/c1572731d188174cd22155acd30b0ded403608be/benchmarks/ascend-sp-e2e.json
- **[S17]** zhenhanchen1-cyber/minimax-h3-ascend，提交 `14363607c8801458b27ce9f87f1c6d3dc7528247`；状态 HTTP 200，裸 `910B`、2×64 GiB、CANN 9.0.0、驱动 25.5.5、INT8/ComfyUI 自述。访问日期：2026-11-09。https://github.com/zhenhanchen1-cyber/minimax-h3-ascend
- **[S18]** minimax-h3-ascend GitCode 镜像；状态 HTTP 200，作为交叉入口。访问日期：2026-11-09。https://gitcode.com/nxxwjbl/minimax-h3-ascend-uncensored
