# R357 — 对照证据：CANN 自带 910A Conv2D 内核含 **144 条 `MAD.`**，工具链确实能发射 cube

## 一、本轮对照实验

| 二进制 | 反汇编行数 | `MAD`/`MMAD` 命中 |
|---|---|---|
| **CANN 910A `Conv2D`**（`ops_legacy/conv2d/Conv2D_5HD_FP16_...o`） | 436,379 | **144（`MAD.`）** |
| CANN 910A `SquaredDifference`（纯 vector 算子，对照） | 309,220 | 0 |
| **我们 msopgen 生成的 kernel**（手写 Mmad 版） | 686 | **0** |

## 二、由此可确定

1. **910A 工具链能正常发射 cube（`MAD.`）指令** —— 硬件与编译器都没问题；
2. CANN 的 910A 算子是用 **TBE/TIK** 构建的（`ops_legacy`），产出里 MAD 正常；
3. **我们的 AscendC msopgen 工程里，手写 `Mmad`/`LoadData` 没有生成任何指令**
   —— 这不是硬件问题，而是**这条 AscendC 基础原语路径在该目标上不发射指令**。

## 三、与"框架 Matmul 部分可用"的关系

框架 `Matmul`（R338）在同一台机器上**能算出精确的第 0 行** ⇒ 那一版二进制里必然含 MAD。
所以更精确的结论是：

- **AscendC 高层 `Matmul` 能发射 cube**（但我们的用法只算出 1 行）；
- **AscendC 底层 `Mmad`/`LoadData` 不发射**（整段被丢弃）；
- **TIK** 是 CANN 自家 910A 算子的构建方式，肯定能发射。

## 四、下一步（两条具体路线）

### 路线 1（推荐）：走 TIK
用 `python/site-packages/te/tik/api/cube/matmul.py` 的 API 写 16×16 matmul，
构建后**先反汇编确认出现 `MAD.`**，再跑数值（判据 A=I ⇒ bad=0）。
理由：CANN 的 910A 算子就是这么构建的，唯一被证明能产出 MAD 的路径。

### 路线 2：继续 AscendC 但只用高层 `Matmul`
既然它能发射 MAD，就把它当作唯一 cube 原语：把 FA 写成
`Matmul(Q,K^T)` → vector softmax → `Matmul(P,V)`，把 M 方向拆成能被正确迭代的粒度并逐个校验。
关键未解点是"为什么同一份数据只算出第 0 行"，需要在能发射 MAD 的前提下重新排查
（此前 R341 的 M=16 结论也是在"可能未发射 MAD"的二进制上得到的，需要重做）。

## 五、状态

- 交付不变：**16.34 s/step（−8.9%）**，`H3_PERF_COMBO=1` 默认 OFF；
- 反汇编证据：远端 `/tmp/h3_r221/dis.txt`、`cann_mm.txt`；
- 下一步建议：先做路线 1 的 16×16 TIK 验证（含反汇编检查），预计 2–3 轮内可判定 TIK 是否能承载 FA。
