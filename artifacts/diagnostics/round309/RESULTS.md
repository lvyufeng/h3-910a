# R309 — tiling 正确，但 C 内容 = 第 0 行复制 16 次（A 行 stride 或 fixpipe M 步长问题）

## 一、诊断链（本轮三次实验）

1. **最近邻映射不可用**：32768 个参考值里总能找到 1e-4 内的巧合匹配，`MAPCHECK` 结论无意义；
2. **整行匹配（512 元素 L2）**给出决定性结果：

```
ROWMAP out_row -> best_ref_row (rms):
[0->0 rms=0.001] [1->0 rms=0.001] [2->0 rms=0.001] ... [15->0 rms=0.001]
```

⇒ **输出的每一行都等于参考矩阵的第 0 行**（`q[0]·k[j]`），rms 仅 0.001（不是巧合）；
3. **M=16 单 tile 复现同样现象**（`RESULT M=16 N=512 K=128 maxdiff=21.55 relRMS=1.348`），
   排除多块/分块问题。

## 二、tiling 字段 dump（host 侧真实值）

```
[TILINGDUMP] usedCore=1 M=16 N=512 Ka=128 Kb=128
             scM=16 scN=512 scK=128 baseM=16 baseN=256 baseK=64 stepM=1 stepN=2 isBias=0
```

⇒ **tiling 完全正常**：M/N/Ka/Kb、singleCore、base、step 都对。

## 三、结论与剩余两种可能

现象：C 的 16 行内容完全相同，且都等于 `q[0]·k[j]`。tiling 正常 ⇒ 只能是：

| # | 可能 | 判据 |
|---|---|---|
| **A** | kernel 侧 **A 的行 stride 被当作 0**（L1/L0A 加载时 `m` 方向步长丢失） | 换 A/B 角色做对照实验：若「常数行」变成「常数列」，则问题跟随第一个操作数 |
| **B** | fixpipe **C 写出的 M 步长 = 0**（框架 C 布局参数未生效） | 若 A/B 对调后现象不变（仍常数行），则问题在 C 写出 |

## 四、下一步（R310，一次判别）

**对照实验**：交换 A/B。

```cpp
mm.SetTensorA(kGm);          // k (512×128)
mm.SetTensorB(qGm, true);    // q (16×128) 转置
mm.IterateAll(tGm);          // 输出 512×16
```

- 若输出变成「每列相同」（即 C[i][j] = k[i]·q[0]）⇒ 问题跟随**第一个操作数**（A 加载 stride）；
- 若仍是「每行相同」⇒ 问题在 **C 写出**（fixpipe M 步长）；
- 两种结果对应两种完全不同的修法，一次实验即可定。

## 五、状态

- 交付不变：**16.34 s/step（−8.9%）**，49 步 ×2 验证，relRMS 在原生带内，`H3_PERF_COMBO=1` 默认 OFF；
- AscendC：S0 ✅、S0.5 ✅、S1a-0 ✅；**S1a-1 已能跑、tiling 正确、C 内容异常已精确刻画**；
- 机器：ts-217 在线（直连正常；fleet bridge 探测误报 OFFLINE，已记录）；
- 远端：本轮提交。
