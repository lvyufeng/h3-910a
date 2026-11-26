# R368 — AscendC cube 运行时调试：cube 能跑、第 0 行数值正确，M 维不前进（待修）

## 一、已完成的正向验证

| 检查项 | 结果 |
|---|---|
| 内核是否被启动 | ✅ 入口写哨兵 `o[0..2]=7/8/9`，harness 读到 7.0000 |
| `GetValue/SetValue` 的 offset 语义 | ✅ `td_vals` 在 idx 0/1/512/513 分别读到 11/22/33/44（完全正确） |
| Matmul 是否写 workspace | ✅ 预置哨兵 `tGm[0]=3` 被覆盖为真实结果 |
| Matmul 数学（第 0 行） | ✅ 随机数据下 `got=-3.121` vs `ref=-3.122`（误差 ~1e-3） |
| 运行时 tiling | ✅ M=64 N=512 Ka=128 usedCoreNum=1 singleCoreM=64 singleCoreN=512 singleCoreK=128 baseM=64 baseN=256 baseK=64 |

## 二、核心异常（已最小化复现）

确定性数据：`q[i][d]=i+1`、`k[d][j]= (d==0 ? j+1 : 0)` ⇒ 期望 `o[i][j]=(i+1)*(j+1)`。

| 位置 | got | ref |
|---|---|---|
| 0 (i=0,j=0) | 1.000 | 1.000 ✅ |
| 1 (i=0,j=1) | 2.000 | 2.000 ✅ |
| 512 (i=1,j=0) | **1.000** | 2.000 ❌ |
| 1024 (i=2,j=0) | **1.000** | 3.000 ❌ |
| 1536 (i=3,j=0) | **1.000** | 4.000 ❌ |

⇒ 每一行都用 **q 的第 0 行** 参与计算（A 的 M 维步进为 0），列方向正确。

## 三、已排除

- 不是启动问题（哨兵可写）；
- 不是读回 API 问题（offset 语义已单独验证）；
- 不是 tiling 字段错（dump 全部合理）；
- 不是 B 的转置约定（改成非转置 K×N 后第 0 行精确正确）；
- 不是 UB 容量（改为标量读回后现象不变）。

## 四、下一步排查方向

1. **ND→NZ 转换**：A 以 `CubeFormat::ND` 传入时依赖运行期 ND2NZ；怀疑转换只处理了第一块（16 行）并复制。可改测 M=16/32，或预先在 host 侧把 A 转成 NZ 再用 `CubeFormat::NZ`；
2. **`SetOrgShape`/`SetShape` 参数顺序**（M,N,K）在 v200 tiling 中的实际语义；
3. 改用低层 `Mmad`（TIK 生成代码的等价 intrinsic 序列）绕开 Matmul 封装——TIK 版本已在设备端验证可生成 `MMAD`，可作对照；
4. 检查 `Matmul` 的 `isTransA` 参数是否需要显式 `false`。

## 五、状态

- 交付不变：**16.34 s/step（−8.9%）**，`H3_PERF_COMBO=1` 默认 OFF；
- cube 代码生成已彻底打通（R366/R367），当前唯一障碍是 Matmul 运行时 M 维行为；
- 远端工程：`/tmp/h3_r221/faA/proj`；harness：`test_mmrows.cpp`（确定性）、`test_mmfull.cpp`、`test_dump10.cpp`。
