# R353 — 关键结论：910A 的正确 cube 编程接口是 **TBE/TIK**，不是 AscendC 的 C++ 高级 API

## 一、本轮实验链（逐步收窄）

| 实验 | 结果 | 含义 |
|---|---|---|
| `KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIC_ONLY)` | 编译报错：**not support in current core version** | 910A 非 v220 代 |
| `KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_MIX_VECTOR_CORE)` | 同样 **not support** | mix 模式也不可用 |
| 去掉 task type（默认 = `CodeMode.AIC`）+ 标量循环替换 `Cast` | Mmad 路径仍**全 0** | 不是 vector 指令问题 |
| **把 L0A 内容读回 GM**（DIAG2） | **全 0** | ⇒ `LoadData`（L1→L0A）**没有把数据装进去** |

结合此前已确认的事实：
- `GM→L1→UB` 精确正确（R344）；
- 框架 `Matmul` 能算出精确第 0 行（R338）；
- `Mmad` 是编译器内建（头文件里无定义），无 v200 专用重载暴露给我们。

## 二、推断（有依据）

CANN 安装目录里 **`opp/built-in/op_impl/ai_core/tbe/impl/` 下的 910A 算子全是用 TBE/TIK DSL 写的**
（`ops_nn/`、`ops_math/` 等），而不是 AscendC C++。同时 `ops_cv/ascendc/*` 里 **`AddConfig("ascend910")` 数量为 0**（R336 实测）。

⇒ 对 910A 而言：
- **TBE/TIK**（`tik.load`、`tik.mmad`、`tbe.dsl`）是官方支持且自带 910A 实现的 cube 编程路径；
- **AscendC 的 C++ 高级 API**（`LoadData`/`Mmad`/`Matmul`）在这台机器上对 910A 的**数据搬入路径是静默无效的**
  （不报错、不生效），只有 `Matmul` 的框架封装能工作（因为它内部走了别的路径）。

## 三、这意味着什么

1. 我此前"AscendC 在 910A 上不可用"的**粗结论是错的**（R336 已撤回）——**框架 Matmul 确实能算**；
2. 但更准确的说法是：**AscendC 高层 API 的裸 cube 原语在 910A 上不生效**，
   所以"用 AscendC 手写 FA kernel"这条路在这台机器上走不通——**应该改用 TIK**；
3. TIK 是 Python/C++ DSL，编译产出与 CANN 内置算子同构，910A 上**有成熟实现**。

## 四、下一步（方向明确）

1. 在 `impl/ops_nn` 或 `impl/ops_math` 里找**已编译到 ascend910 的 TIK 算子**作为模板；
2. 用 TIK 写 16×16 matmul 验证（`tik.load` + `tik.mmad`），判据 A=I 时 `bad=0`；
3. 通了之后再移植 FA（`tik` 支持 `set_fmatrix`、`mmad`、`vec_add` 等，FA 需要的原语齐全）。

## 五、状态

- 交付不变：**16.34 s/step（−8.9%）**，`H3_PERF_COMBO=1` 默认 OFF；
- AscendC：cube 可用（框架级）；**裸 AscendC cube 原语在 910A 上静默无效**（本轮的 L0A 回读证据）；
- 建议路线：**TIK 内核重写**（这是我此前完全没试过的、且与 CANN 自身 910A 实现一致的路径）；
- 工程与所有判据 harness 保留在 `/tmp/h3_r221/`。
