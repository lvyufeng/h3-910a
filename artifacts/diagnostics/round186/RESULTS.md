# R186 — 契约 B：真实模型路径逐张量判定比对，**310/310 一致**

## 一、做法

给审计加了一个 `verify` 模式（`H3_FAST_AUDIT=verify`）：**每层同时算原生判定和快速判定**，
逐条比对并落盘。这只是验证，不是采用；默认路径未变。

8 卡真实模型运行（`--steps 10`，warmup 1 + 1 clean repeat = 2 次前向）。

## 二、结果

```
rank0: comparisons=310 mismatches=0 agree=True
TOTAL comparisons=310 mismatches=0 nonfinite_hits=0 saturated_hits=0
VERDICT: ALL AGREE
```

| 项 | 值 |
|---|---|
| 比对次数 | **310** |
| 不匹配 | **0** |
| 非有限命中 | **0** |
| 饱和命中 | **0** |
| warm step 中位（verify 模式，双路都算） | 18.248 s |

（310 = 155 条检查 × 2 次前向。verify 模式的 18.248 s **慢于** native 的 17.987 s 属预期，
因为它每层跑了两条路径。）

## 三、**必须说清的局限（这是本轮最重要的诚实部分）**

`nonfinite_hits=0`、`saturated_hits=0` —— **真实张量里没有任何一个触发非有限或饱和判定**。

所以这 310 次比对证明的是：**在真实权重/激活上，两条路径对"干净张量"的判定完全一致**。
它**没有**证明在真正出现 NaN/Inf/饱和时两者一致——那些区分性用例来自
**R183 的 15 个对抗用例（15/15，含 `nan_and_saturation`、`inf_and_saturation`、`many_bad`）**。

两类证据合起来才完整：

| 证据 | 覆盖 |
|---|---|
| R186 310 次真实张量比对 | 真实分布下判定一致（全部干净） |
| R183 15/15 对抗用例 | 边界与病态输入下判定一致（含 NaN/Inf/饱和/混合） |

## 四、只拿到 rank0 的回执

`atexit` 里用 `dist.get_rank()` 取 rank，进程退出时分布式组可能已销毁，
异常被 `except` 吞掉后回落到 rank 0，于是 8 个 rank 都写到同一个 `rank0.json`（后写覆盖）。
`310` 与"单 rank 310 次"吻合，说明**不是 8 rank 合计**。

不影响结论（判定等价是逐 rank 独立成立的），但下一轮若要全 rank 证据，
应改为在 `close()` 里（分布式组尚未销毁时）取 rank，或用 `RANK` 环境变量。

## 五、本轮修的一个错

启动脚本里 `export PYTHONPATH="$D/audit:$D:..."` **覆盖**了 `set_env.sh` 加入的 CANN 路径，
运行直接死在 `No module named 'tbe'`。改为追加 `${PYTHONPATH:-}` 后正常。

这是本轮第二个"环境变量处理不当"的教训（R185 是 `HOME` 被 env 字典整体替换掉），
两者症状都表现为与真实原因无关的 ACL/GEMM 初始化错误。

## 六、合规

- 复用已核主机权重，未下载
- 共享 CANN 未改；`bisheng` wrapper、overlay、`.so` 全在 `/tmp`
- 未做 driver/device reset；未改模型数学；默认路径未变
- lease 已释放
