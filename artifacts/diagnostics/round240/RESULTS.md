# R240 — 撤回 R236/R237 的“3× 差值 / wrapper 2.78 s”：内联插事件改变了执行

## 一、方法

在同一次运行里同时做三件事（不改数学、无流切换、forward 内零同步）：

1. 在 `qkv_proj` / `mlp.fc1` 模块上挂 pre/post 钩子（模块 span）；
2. patch `UnquantizedLinearMethod.apply`（GEMM 本体 span）；
3. 尝试记录 attention 的真实布局（`q/k/v` shape、`packed_total`、`max_seqlen`、`cu_seqlens`）。

**步时不变性**：17.9237 s vs 基准 17.928 s = **−0.004 s**（最好的一次，几乎零扰动），8 rank 完成。

## 二、结果：与 R236/R237 直接冲突

| 层 | R240 模块 span | R240 apply span | R235（钩子，100 样本） | R237/R236（内联事件） |
|---|---|---|---|---|
| `qkv` | **17.074** | **17.063** | 17.067 | **41.561** |
| `fc1` | **23.184–23.489** | **23.181–23.479** | 23.407 | **54.430** |

- R240 的模块 span 与 apply **几乎相等**（差 0.01 ms）⇒ wrapper 没有额外设备工作。
- R240/R235 用**两种独立方式**（钩子、apply 包装）都得到 qkv≈17.07、fc1≈23.4，
  且两者步时都在基准附近（R240 −0.004 s、R235 +0.062 s）。
- R237 的 41.56 / 54.43 是用**替换 forward 并内联插事件**得到的，步时 +0.061/+0.097 s。

## 三、结论：撤回两条结论

1. **撤回“in-model GEMM 是独立 mm 的 3.11×”**（R237）：该结论基于内联事件测得的 41.56/54.43，
   而两种不替换 forward 的方法都得到 17.07/23.4，与 R146 独立基准（12.47/19.04）同量级。
2. **撤回“wrapper 额外 2.78 s/step”**（R239）：R240 显示模块 span ≈ apply span，
   wrapper 差额约 0.01 ms，不存在 24.5/31.1 ms 的 wrapper 开销。
3. **方法学教训**：在 `forward` 里替换语句并插入事件，会改变执行（很可能打乱了
   offloader 的 prefetch/wait 与编译区域），使被替换区域的测量系统性偏大；
   **凡是替换 forward 得到的绝对时间都不能用于归因**，只能用不替换 forward 的钩子/patch。

## 四、仍未解决的问题（回到 R235 的缺口）

R235（钩子，100 样本，步时 +0.062 s）仍有：`attn` 父区间比子项多 25.0 ms/块、
`mlp` 父区间比 fc1+fc2 多 34.8 ms/块，合计 3.457 s/step。
R240 证明这些子项（qkv 17.07、fc1 23.4）本身没有被低估，所以缺口**确实存在于
父区间内、子模块之外**。

结合 R228（权重等待在慢 H2D rank 上无差异）与 R230（审计同步只值 0.036 s），
最可能的位置是**块边界处 offloader 的 `wait_event(下一块权重)`**：
- 它发生在 block 的 post_forward（`clear_block_storage`），位于 `mlp` 父区间**之后**；
- 但 `attn`/`mlp` 子项之间的间隔（如 `q_norm` 前后、`out_proj` 后）也可能承载流等待；
- 需要一次**专门测块边界**的探针：在 block pre/post 钩子里记录 `wait_event` 前后的事件，
  并把每次等待的时长累加，与 3.457 s 对照。

## 五、布局记录未成功

本轮尝试在 `_run_packed_attention` 里记录 `packed_total/max_seqlen/cu_seqlens` 形状，
8 rank 的 `layout_samples` 均为空 ⇒ patch 未被调用（该路径可能被编译捕获）。
`M` 口径问题仍未解决，**继续暂停引用任何 TFLOPS/MFU 数字**。

## 六、合规

- 未改盘上源码；未安装候选；FA trim 仍 OFF；未下载/回传张量。
- 主动撤回两条自己刚发布的结论，并把被撤回的原因写清，避免后续基于错误前提做候选。
- 步时 −0.004 s，数据质量优于前几轮。

## 七、下一步（R241）

1. 用**不替换 forward** 的方法测块边界等待：在 block pre/post 钩子与
   `LayerwiseOffloadHook.pre_forward/offload_layer` 上记录事件，累加每次 `wait_event` 时长，
   与 3.457 s 对照；
2. 用 `torch.npu.Event` 在 copy stream 与 compute stream 之间做跨流配对，量化暴露等待；
3. 布局记录改用 hook 在 `AttentionMetadata` 构造处抓取（不再 patch 方法体）。