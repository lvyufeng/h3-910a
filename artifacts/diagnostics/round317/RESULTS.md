# R317 — 延迟 import 钩子已做成；encoder 常驻仍未跑通（Orchestrator 初始化受阻）

## 一、本轮成果：零侵入延迟探针

`sitecustomize.py`（放 PYTHONPATH 上即被所有 Python 进程自动加载）：

- 只在 `vllm_omni...pipeline_minimax_h3` **被 import 的那一刻** 通过 `sys.meta_path`
  finder+loader 包装 `exec_module`，随后 patch `encode_prompt` / `decode_latent`；
- 与官方脚本**零冲突**：不修改 `generate_full_NPU126.py`，不需要在 `main()` 之前 import pipeline
  （这正是上一轮 `SetPrecisionMode 500001` 的根因）；
- 日志确认钩子生效：`[STAGE] lazy hook armed for vllm_omni...pipeline_minimax_h3`。

## 二、本轮失败：Orchestrator 初始化失败

- 带 encoder-offload skip → `RuntimeError: Orchestrator initialization failed: `（内部异常为空）；
- **去掉 skip 只装探针 → 同样失败** ⇒ 与 monkeypatch 无关，是**启动方式**问题：
  我们直接 `torch_npu_run ... generate_full_NPU126.py`，而 R126 成功时的启动是**外层 launcher**
  （`subprocess.Popen(['/bin/bash','-c', ...])`，`cwd=/tmp/h3_budget149`，且 env 里带
  `H3_BUDGET149_LABEL`、`OMP_NUM_THREADS=4`、`OPENBLAS_NUM_THREADS=1`）。
- 证据：`/tmp/h3_budget149/budget149a_process.json` 完整记录了当年命令与环境变量。

## 三、下轮执行配方（已确定）

1. 新建 launcher：照 `launch_budget149.py` 的 `Popen` 结构，把 entry 换成
   `/tmp/h3_budget149/generate_full_NPU126.py --output-label NPU_full_generation126_stage<NNN>`；
2. env 里加 `PYTHONPATH=/tmp/h3_r221:$PYTHONPATH`（launcher 会再追加官方两路径）
   + `H3_STAGE_PROBE=1`（+ 可选 `H3_SKIP_ENCODER_OFFLOAD=1`）；
3. `sitecustomize.py` 自动生效，`atexit` 写 `H3_STAGE_OUT`；
4. 判据：encoder 阶段 < 2 s；若显存/数值不符则回退并记录。

## 四、状态

- 交付不变：**16.34 s/step（−8.9%）**，49 步 ×2 验证，relRMS 在原生带内，`H3_PERF_COMBO=1` 默认 OFF；
- encoder 基线 ~109 s（既有收据）/ 目标 <2 s；video VAE 20.565 s / 目标 <20 s；
- DiT 15 s：R313 路线级结论（本机 AscendC cube 不可用、TIK 126.19 ms 下限）⇒ 当前条件不可达；
- 进程清理：上一轮残留 `gen_stage292` 进程已全部 kill（`left=0`）；
- 机器 ts-217 直连正常；远端 master 含本轮记录与探针源码。
