# R182 — **卡了 7 轮的 `cstdint` 打通了**；卡点转移到内核 API 类型支持

## 主突破：根因是 PYTHONPATH 被 CANN 包装脚本抢占

`$CANN/bin/asc_opc` 是个 bash 包装脚本，它这样设 PYTHONPATH：

```bash
PYTHON_PATH="${PKG_PATH}/python/site-packages"
export PYTHONPATH="${PYTHON_PATH}:${PYTHONPATH}"     # CANN 自己排最前
```

**CANN 自己的 site-packages 永远在外部 PYTHONPATH 之前**，所以我 R179 起的
`PYTHONPATH=$OVERLAY:...` 全部被遮蔽——补丁从未被导入。这解释了 R175–R181 七轮"补丁生效但报错不变"。

### 解法：私有 `asc_opc` 包装

```bash
#!/bin/bash
export PYTHONPATH="$OVL:$R/python/site-packages:${PYTHONPATH:-}"
exec ${HI_PYTHON:-python3} "$R/python/site-packages/asc_opc_tool/opc.py" "$@"
```

放在 `PATH` 最前的 `/tmp/h3_bin182/asc_opc`，绕开 CANN 包装直接调 `opc.py`。

**验证（关键证据）**：

```
asc_opc resolves to: /tmp/h3_bin182/asc_opc
imported from: /tmp/h3_overlay182/asc_op_compile_base/__init__.py
patch present: True
```

### 补丁锚点（也修正了 R179 的错误）

R179 的脚本用 `compile_options = ` 正则找锚点——**在那个文件里不存在**，断言失败。
真正的位置是 `ascendc_compile_v200.py` 的 `gen_compile_cmd_v200()`：

```python
    for option in compile_option_tuple.compile_options:
        compile_cmd += [option]
    ...
    return compile_cmd
```

在 `return compile_cmd` 前追加三个 `-isystem` 即可。`_v200` 1 处 + `_v220` 4 处，共 5 个锚点。

## 效果：`cstdint` 从 10 处 → **0 处**

```
=== cstdint: 0
CompileStage.LINKRELOCATE      # 编译已推进到链接阶段
```

这是 R172 以来第一次越过头文件问题。

## 新的卡点：内核 API 的类型支持比声明窄

| API | 头文件声明支持 | 实际链接 |
|---|---|---|
| `Ands` | `SupportType<PrimType, uint16_t, int16_t, int64_t, uint64_t>` | **half / int16 / uint16 全部 `undefined symbol`** |
| `ReduceMax` | `static_assert(SupportType<PrimType, half, float>())` | half / float 可用 |

`Ands` 的 Level-2（`count` 版）对 16 位类型**只有声明没有实现**，很可能只实现了 32 位。

已试过并失败：`Ands<half>`、`Ands<int16_t>`、`Ands<uint16_t>` —— 都是同一个 undefined symbol 模式。

## 下一轮的候选路线

### A. 32 位 `Ands`（需加宽，代价是访存翻倍）
把 half 视图为 int32（每 32 位含两个 fp16），**不能用 Ands 做逐元素掩码**（会同时改两个值）。
除非用 `Ands` 掩 0x7fff7fff 得到"两个幅值并列"，然后……无法用 ReduceMax 求单值 max。
→ **此路不通**，除非自己做配对比较。

### B. 绕开 `Ands`：用 `Abs` 得到幅值（half 运算，确定支持）
`Abs` 把 0x8000|m 变成 m，正好等价于掩符号位，且是 half 运算。
问题仍在：Abs 后 NaN 位模式（0x7c01..0x7fff）仍是 NaN，`ReduceMax` 对 NaN 不可靠（R165）。

### C. `Compare` + `ReduceMax` on half（推荐）
- 饱和：`CompareScalar(mask, x, (half)65504.0, EQ)` → 掩码，Cast 到 half，`ReduceMax` 得 0/1
- NaN：`Compare(mask2, x, x, EQ)` 的**取反**即 NaN（NaN != NaN）
- Inf：`CompareScalar(mask3, x, (half)INFINITY, EQ)`
全部只用 half 的 Compare/ReduceMax，绕开整数 AND。
代价：3 次扫描（但可以做在一个 tile 循环里，共享同一次 CopyIn）。

### D. 退回 CPU 侧判定（不建议）
把 155 个张量的 max 位模式用 `ReduceMax(Abs(x))` 粗筛，仅当接近阈值时回退精确路径。
但要先确认 `ReduceMax` 在含 NaN 输入上的真实行为（R165 未测 ReduceMax，只测了 `max()`）。

## 已修正的内核文件

`artifacts/diagnostics/round182/src/h3_range_stats.cpp`（当前版本用 uint16 Ands + Cast to float + float ReduceMax，
**能编译到链接阶段，但 `Ands` 未定义**）。下一轮按 C 方案改写即可。

## 状态

- 目标已 resume 并处于 active
- 本轮 lease 已持有将释放；无遗留进程
- 未改共享 CANN（overlay 在 `/tmp`，wrapper 在 `/tmp`）
