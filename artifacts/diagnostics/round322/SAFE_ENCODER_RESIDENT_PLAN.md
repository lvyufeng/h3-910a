# R322 — encoder 常驻的**分步安全验证**方案（吸取 R320 事故教训）

## 一、事故回顾（R320）

`H3_SKIP_ENCODER_OFFLOAD=1` 在 **8 rank 一次性**验证：host RAM 用到 750/754 GB（仅 1 GB 可用），
约 10 分钟后 host 失联，靠 OOM killer 清进程自行恢复。**这是不可接受的实验方式。**

## 二、新方案：三道门，逐级放行

### 门 1：单 rank 容量与耗时（不起 Omni，纯离线）

- 只用 **1 个进程 / 1 张卡**，从已验证的 on-host 权重加载 text encoder 分片（TP8 的 1/8 ≈ **7.9 GB**）；
- 直接 `encode_prompt` 等价前向（固定种子、固定输入；
- 记录：host RSS 峰值、NPU HBM 峰值、单次前向耗时；
- **放行条件**：host RSS 峰值 < 80 GB、HBM < 28 GB、前向 < 20 s（单卡 1/8 分片）。

### 门 2：单 rank 常驻 + 局部驻留验证

- 在门 1 基础上保持常驻，连续跑 3 次前向；
- 记录：RSS 是否稳定（无泄漏）、第 2/3 次耗时（应为稳态）、HBM 是否回收；
- **放行条件**：RSS 波动 < 2 GB、稳态耗时 < 10 s。

### 门 3：完整 8 rank 常驻（受控）

- 启动前先记录 `free -g`，并在 launcher 里加**硬门**：若可用 < 300 GB 直接 abort；
- 启动后每 10 s 采样一次 `free -g` 与各 rank RSS，写入日志；一旦可用 < 80 GB 立即 abort 并清理；
- **放行条件**：encoder < 2 s 且全程可用内存 > 80 GB。

## 三、为什么必须这样

- 63 GB encoder ÷ 8 rank = 每 rank 7.9 GB，看似宽裕，但实际还要叠加：
  权重原始 FP32 副本、HF safetensors 映射、page cache、DiT 32.3 GB、各 rank 的激活缓冲；
- R320 就是把这些**叠加项**一次性引入，没人知道是哪一项压垮了内存；
- 分步验证可以把「哪一项吃内存」测出来，而不是靠猜。

## 四、当前状态

- ts-217 已恢复（576/754 GB，OOM killer 已清理我方进程）；
- stage323（安全配置 + 探针）正在跑 49 步，用于同时拿到 **encoder 62.95 s** 与 **video/audio VAE** 的精确值；
- 交付不变：**16.34 s/step（−8.9%）**，`H3_PERF_COMBO=1` 默认 OFF。
