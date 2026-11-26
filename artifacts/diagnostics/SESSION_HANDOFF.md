# 会话交接（2026-11-26，R332–R359）

## 一、交付现状（未变更，仍为唯一可部署状态）

| 项 | 值 |
|---|---|
| DiT 步时 | **16.34 s/step（−8.9% vs 17.928 s）** |
| 验证 | 49 步 × 2 路径，全部视频 relRMS 落在原生自重复带 9.84e-04–1.077e-03，无 NaN/Inf |
| 开关 | `H3_PERF_COMBO=1`，**默认 OFF**，一键回退 |
| 位置 | `artifacts/deploy/h3_perf_switches.py` + `README.md` |
| 结论 | DiT ≤10/15 s 在本机不可达（有结构性证据）；10 s 未达成，故不宣布总体完成 |

## 二、encoder / VAE 现状

| 目标 | 实测 | 说明 |
|---|---|---|
| encoder <2 s | **2.805 s**（常驻后；基线 3.196 s） | H2D 仅 0.39 s，其余 2.8 s 是计算（text_model 25.6 s + vision 23.3 s 于首轮 60 s 中）⇒ 需 1.4× 计算优化 |
| video VAE <20 s | **20.672 s**（warm 最优） | 差 3.4%；改 tile/overlap 会有微小数值变化，需用户批准 |
| audio VAE | 0.20 s | ✅ 达标 |

## 三、AscendC 路线：证据完整关闭（本会话核心产出）

1. **cube 可用**：框架 `Matmul`（A=I）第 0 行逐元素精确；
2. 但**反汇编证据**：手写 `Mmad` 版 kernel **0** 条 MAD/CBUF；高层 `Matmul` 版 **0** 条 MAD（仅标量 LD./ST./ADD.）；
3. **对照**：CANN 自带 910A `Conv2D` 含 **144** 条 `MAD.` ⇒ 工具链能发射，CANN 自家算子用 TBE/TIK 构建；
4. 编译器拒绝 `KERNEL_TYPE_AIC_ONLY` / `KERNEL_TYPE_MIX_VECTOR_CORE`（910A 是 v200 代）；
5. 结论：**AscendC 在本机对 910A 不生成 cube 指令**，无法写 FA 内核。

## 四、下一步（唯一路径：TIK）

1. 用 `python/site-packages/te/tik/api/cube/matmul.py` 的 API 写 16×16 matmul；
2. **先反汇编确认 `MAD.` 存在**（本会话最重要的方法论教训）；
3. 数值判据：A=I ⇒ `bad=0`；
4. 通过后移植 64×512×128 FA，接回 `VLLM_OMNI_910A_FLASH_ATTN` 开关；
5. 数值门禁：155 条 Half 检查 + relRMS 在原生带内；任何数值变化先报用户。

## 五、工程与证据位置（远端 ts-217，`/tmp/h3_r221/`）

- `faA/proj`：msopgen 工程（`ai_core-Ascend910A`），当前为高层 Matmul 版；
- `test_id32.cpp` / `test_m16.cpp` / `test_mmad16.cpp` / `test_k16.cpp`：判据 harness；
- `dis.txt` / `mm_dis.txt` / `cann_mm.txt`：反汇编证据；
- `resident_patch.py`：encoder 常驻补丁（opt-in `H3_ENCODER_RESIDENT=3`）；
- `warmvae33x.py` / `warm340.log`：常驻与两趟测量脚本与日志。

## 六、启动配方（勿改）

单进程 `/home/mseco/vllm910a/bin/python -u <script.py>`，`cwd=/tmp/h3_budget149`，
`source /home/mseco/Ascend/cann-9.2.0/set_env.sh`，R126 环境变量全套，
`VLLM_OMNI_DLO_DP_WAVE_TIMEOUT=7200`，`ASCEND_WORK_PATH=<新目录>`。

## 七、未完成项（并行目标）

- **A2/A3 对比（#13）**：需 910B/310P 硬件，未开始；
- encoder <2 s：需计算优化（非本会话范围）；
- video VAE <20 s：需用户批准数值变化。
