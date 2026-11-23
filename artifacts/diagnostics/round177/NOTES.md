# R177 — 三处认知更正；构建注入口仍被构建自身覆盖

## 本轮更正了三件事（都是前几轮的错判）

### 1. 我一直在补错的文件

构建目录里有**两个** dynamic python：

| 文件 | 是否被内核编译使用 |
|---|---|
| `build_out/op_kernel/ascendc_kernels/binary/dynamic/h3_range_stats.py` | **是（这个才是真的）** |
| `build_out/op_kernel/H3RangeStats_ascend910b/dynamic/H3RangeStats.py` | 否（R175/R176 补的就是它） |

R176 "补丁到位但无效"的真正原因是补错了对象。

### 2. R175 说 `custom_compile_options.ini` 没有读取方 —— **错**

`opdesc_parser.py:396` 明确读它（R175 的 grep 方式没匹配到）。

### 3. 它的格式现在确定了（读 `opdesc_parser` 385–425）

每行 3 个逗号分隔字段：`op_type , soc_ver , option;option;option`
- `op_type` 写 `ALL` → 所有算子；`soc_ver` 留空 → 所有 SOC；选项用**分号**

## 但两条路都失败

- 补**真实** dynamic py → 仍 10 处 `cstdint`
- 写 `custom_compile_options.ini` → **构建把它清空重写**（构建后 `head -1` 为空）

⇒ 该 ini 是**构建产物**，不能当预置输入。

## 建议：换路线

**A. 私有 PYTHONPATH overlay**：复制 `compiler/tikcpp/ascendc_kernel_cmake/` 到私有目录，
在 `get_op_desc()` 返回前给每个 opdesc 塞三个 `-isystem`，`PYTHONPATH=$OVERLAY:$PYTHONPATH bash build.sh`。

**B. 兜底**：用 R174 已验证可用的手工 bisheng 编 `.o`，或先用直调/TBE 路径拿到**单次耗时**这个数字。

## 为什么那个数字重要

阶段 A 的全部价值是"18.5 ms/次 → ?"。到 1–2 ms 则每步省 2.5–2.9 s；只到 10 ms 则省 1.3 s。
**拿到实测前不应再按 2.5 s 规划。**

## 合规

共享 CANN 未改；未下载权重；未做 driver/device reset；lease 已释放。
