# R355 — TIK 的完整 cube 实现在主机上（`te/tik/api/cube/matmul.py`）；这是 910A 的官方正路

## 一、本轮发现

主机上存在 **TIK 的完整 Python 实现**：

```
python/site-packages/te/tik/api/cube/matmul.py     ← 完整 matmul 实现（含 inst.mmad 调用）
python/site-packages/te/tik/api/cube/cube_common.py
python/site-packages/te/tik/tik_lib/tik_api_constants.py  ← 含 load_cbuf_to_ca 等 builtin 名称
python/site-packages/te/tik/tik_lib/tik_mmad_convert_api/
```

`matmul.py` 里可以看到 `inst.mmad(...)` 的实际调用序列；`inst` 是 TIK 的指令编码器，
最终会生成对 `__builtin_cce_*` 的调用。

⇒ **TIK 才是 910A 上官方支持的 cube 编程接口**（CANN 自带的 910A 算子就全是 TBE/TIK 写的），
而 AscendC 的 C++ 高级 API（`LoadData`/`Mmad`）在 v200 上语义未被正确接线（R353/R354 的一系列现象）。

## 二、三条可选路线（按可行性）

| 路线 | 说明 | 代价 |
|---|---|---|
| **A. 直接用 TIK API 写 FA**（推荐） | 用 `te.tik.api` 的 `load`/`mmad`/`vec_*`，经 TBE 编译成 910A 算子 | 中：需跑通 TBE op 构建流程 |
| B. 用 builtin 直调 | 在 AscendC kernel 里叫 `load_cbuf_to_ca(...)`，参数顺序抄 TIK 源码 | 中低：需精确抄参数 |
| C. 继续 AscendC 高层 API | 已被证实在 v200 上不工作（多轮） | 不建议 |

## 三、结论与建议

- 至此可以给出**有依据的定性**：
  **910A 上用 AscendC C++ 高层 API 手写 cube kernel 不可行**（不是硬件不行，是这层封装在 v200 上没接线）；
  **可用的是 TIK（TBE DSL）**，它在这台机器上有完整实现与官方算子作参考。
- 这与"cube 可用"不矛盾：框架 `Matmul`（编译期走 TBE）能算出精确第 0 行。
- **建议**：在新会话启动"TIK 重写 FA"立项——先用 TIK 写 16×16 matmul 验证（判据 A=I 时 `bad=0`），
  通了再按 64×512×128 的形状移植，最后接回 `VLLM_OMNI_910A_FLASH_ATTN` 的部署开关。

## 四、状态

- 交付不变：**16.34 s/step（−8.9%）**，`H3_PERF_COMBO=1` 默认 OFF；
- 目标：DiT ≤10/15 s 仍**未达成**；本季探索的结论是"路线需从 AscendC 改为 TIK"；
- encoder 2.805 s（常驻后，<2 s 未达）；video VAE 20.672 s（<20 s 未达）；
- 所有工程与判据留在 `/tmp/h3_r221/`；远端 master 含 R332→R355 完整记录。
