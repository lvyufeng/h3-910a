# R330 — 全线状态汇总与需要用户决定的事项

## 一、四个目标：实测值 + 判定

| 目标 | 实测（本轮会话新证据） | 判定 | 依据 |
|---|---|---|---|
| **DiT ≤10 s/step**（11-23 上调） | **16.34 s/step**（中位，−8.9% vs 17.928 s 基线） | **当前硬件不可达** | 交付已在 49 步×2 验证冻结；FA 内核 TIK 结构下限 **126.19 ms**（R291 配置饱和）；本机 910A 的 AscendC cube 不可用（R313，六种配置全失败） |
| DiT ≤15 s/step | 同上 | 同上 | 需要 FA ≤105 ms（−17%）+ 非 FA −0.3 s，均无可行路径 |
| **encoder <2 s** | warm **3.196 s** / cold 62.95 s | **差 1.6×，配置不可解** | TP8 固定（合同）、offload 已是 warm 最优；需代码级优化 |
| **video VAE <20 s** | warm **20.672 s**（tiling=False）/ 20.746–20.998 s（tiling=True）；cold 80.2 s | **差 3.4%，配置不可解** | 配置空间探尽：mode 仅 `tile`、patch_parallel_size 仅 1/8、fast_path 已 lossless（R327/R328/R329） |
| audio VAE（记录项） | warm **0.20 s** | 达标 | 与既有收据 0.210 s 吻合 |
| A2/A3 对比 | 未开始 | 缺机器 | 只有 8×910A |

## 二、本轮会话新增的关键实测（全部可复现、含同步）

- `encode_prompt`：cold **62.15 / 62.95 s** → warm **3.196 s**（两趟同 engine，R326）；
- `VideoVAE.decode_latent`：cold **80.12–80.85 s** → warm **20.672–20.998 s**（三种配置实测）；
- `AudioVAE.decode_latent`：cold **56.0–64.7 s** → warm **0.199–0.266 s**；
- 三组数字与既有收据（cold combined 142.143 s、warm video 20.565 s、warm audio 0.210 s）**完全自洽**；
- 生成整轮：engine_init 216–407 s、pass1 wall 1176–1206 s、pass2 wall 890–913 s、denoise 稳态 17.0–17.9 s/it。

## 三、本轮修复的环境级问题（供后续复用）

1. **TBE 缓存损坏** → `Orchestrator initialization failed`（空消息）：把 `~/atc_data` 移开即恢复（备份 `~/atc_data.bak_h3_1918`）；
2. **探针被官方 sitecustomize 遮蔽**：PYTHONPATH 把私有目录放最前 + 私有 sitecustomize **chain-load** 官方那份；
3. **orchestrator 异常被吞**：装 `threading.excepthook` 才能看到真实栈；
4. **spawn 重入陷阱**：脚本必须 `if __name__ == '__main__'` 守卫，否则 8 个 spawn 子进程重复 `mkdir`；
5. **启动方式**：单进程 `python generate_full_NPU126.py --output-label …`，不要用 `torch_npu_run`（8 进程 mkdir 竞态）；
6. **排障顺序**：先看 worker 行 `(DiffusionWorker_…)`，顶层 `RuntimeError` 没有信息量。

## 四、需要用户决定的三件事

1. **是否批准代码级优化**（唯一能达标的路径）：
   - video VAE：warm 再快 3.4%（tile 内部尺寸/重叠、通信与计算重叠）；
   - encoder：warm 再快 1.6×（视觉塔/文本塔前向优化、更细的重叠）。
   两者都**没有产出保证**，且会改动 vllm_omni 代码（需你明确授权）。
2. **A2/A3 对比**：需要对应硬件，当前只有 8×910A。
3. **DiT 10 s 目标**：当前硬件（910A + CANN 9.2.0）下不可达；如果你有 910B/310P 机器，可重启 AscendC 路线。

## 五、交付状态（不变）

**16.34 s/step（−8.9%）**，49 步 ×2 验证，video relRMS 全在原生自重复带内（9.84e-04–1.077e-03），
`H3_PERF_COMBO=1` 默认 OFF（可一键回退），远端 `master` 含全部记录与 receipts。
