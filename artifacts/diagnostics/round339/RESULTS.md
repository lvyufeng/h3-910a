# R339 — A 走位问题：tiling 形状参数调整无效；下一步换 `Iterate()`/`SetTail` 路径

## 一、本轮尝试（全部无变化）

| 尝试 | 结果 |
|---|---|
| host `SetBType(..., true→false)` | C 仍为「每行 = A[0]·B」 |
| host 4 参 `SetOrgShape(32,32,32,32)` + `SetSingleShape(32,32,32)` | **无变化** |
| 加 `TILINGDUMP` printf | 未能出现在运行输出里（op host 的 stdout 未透出），需要改用文件写出或 tiling 断言 |

实测始终为：

```
ID32 maxdiff=96 relRMS=7.813e-01 bad=960/1024
row0 out = 1 2 3 4 5 6    ← 正确
row1 out = 1 2 3 4 5 6    ← 应为 33 34 35 36 37 38
```

## 二、当前最强假设

`Matmul` 在 910A 走的是 `MatmulImpl` 分支（`matmul_intf.h` 中 `__NPU_ARCH__ != 2201/3510` 的 else 分支），
而 `IterateAll` 的完整实现路径主要面向 `MatmulClient`（2201/3510）。
**`MatmulImpl` 在 910A 上可能只正确写出第一个 M 块（baseM 行），后续行不迭代。**

## 三、下一步（有序，逐条验证）

1. **改用显式迭代**：`mm.SetTail(m, n, k)` + `mm.Iterate()` 按 M 块循环（每块 16 行），
   自己累加 `offsetM`，把 C 写到对应位置 —— 绕开 `IterateAll` 的 M 走位；
2. 或 `mm.IterateAll(cGm, /*enAtomic*/0, /*enSequentialWrite*/true)`；
3. tiling dump 改为**写文件**（`/tmp/h3_r221/td.txt`）以确认真实字段；
4. 判据：M=32 时 `bad=0`。

## 四、状态

- 交付不变：**16.34 s/step（−8.9%）**，`H3_PERF_COMBO=1` 默认 OFF；
- **cube 已验证可算**（R338，第 0 行精确），缺陷是 M 迭代/走位；
- 工程与 harness 就绪：`/tmp/h3_r221/faA/proj` + `test_id32.cpp`（A=I 判据）；
- 远端 master 含本轮记录。
