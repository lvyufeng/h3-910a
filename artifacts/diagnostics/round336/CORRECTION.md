# R336 — 更正：不能断言「AscendC 在本机不可用」；证据只支持「我的 matmul 用法没跑通」

## 一、更正结论

R313 我写下"Ascend910A 的 AscendC cube 能力不足"——**这个结论下得太早，应更正**。
六种失败配置（高层 Matmul 4 种 + 手工 Mmad 2 种）只能证明**我的用法/布局有问题**，
不能证明硬件或 CANN 不支持。

## 二、本次核查得到的**中性证据**

| 检查 | 结果 | 含义 |
|---|---|---|
| `asc/include/utils/tiling/platform/platform_ascendc.h:61` | `ASCEND910 = 0, // Ascend910A, Ascend910B` | AscendC **把 910A 和 910B 视为同一 SoC 版本**，说明框架层面是支持的 |
| 我们的 msopgen 工程 | 以 `ai_core-Ascend910A` 编译、打包、安装、**成功执行** identity kernel | 加载链路（aclnn→op→kernel）在 910A 上**完全可用** |
| CANN 内置 `ops_cv/ascendc/*` 中带 `AddConfig("ascend910")` 的算子数 | **0** | CANN **没有为 910A 出货 AscendC 版内置算子**（这只说明官方实现不面向 910A，不说明 API 不能用） |

⇒ 正确表述：**AscendC 在 910A 上"能编译、能装载、能执行"；cube（Matmul/Mmad）失败是用法问题，未定性**。

## 三、为什么之前的失败更像"用法错"

1. 高层 `Matmul` 六种配置全部产出"C 每一行相同"——典型是 **C 的 stride/布局参数不对**（fixpipe 写同一个地址），
   而不是算不对；tiling dump 显示 M/N/Ka/Kb/baseM/step 全部正常；
2. 手工 `Mmad` 输出**全 0**——典型是 **L0A/L0B 没加载进去**（`LoadData2dParams` 的
   `repeatTimes/srcStride` 单位用错），Mmad 空操作 → 0，而不是硬件不支持。

## 四、决定性验证方案（分两步，避免再盲试）

### 步骤 1（半天内可完成）：验证搬运链路
- L1 填已知数据 → `LoadData` 到 L0A → **不经 Mmad**，用 `DataCopy` 把 L0A 内容读回 UB → 存 GM → 与输入比对；
- 若不一致 ⇒ 定位 `LoadData2dParams`（repeat=?, srcStride 单位=16 元素块?）与 L1 的 NZ 摆放；
- 若一致 ⇒ 搬运正确，问题只在 Mmad 参数或 C 布局。

### 步骤 2：16×16 最小 matmul，C 用**两种读法**各测一遍
- 读法 A：`DataCopyCO12DstParams(nSize=?, mSize=?)`（nz2ndEn true/false）；
- 读法 B：把 L0C 视为 16×16 NZ 分形，按 fractal 索引手工展开；
- 只要有一种读法与 CPU 参考一致 ⇒ **cube 可用**，随后迁移到 64×512×128。

### 步骤 3（若两步都过）：把这条路线接回 FA 内核重写（S1a-1 → S1 → S2）

## 五、对目标的意义

如果 cube 可用，则 **DiT ≤15 s（乃至 10 s）的路重新打开**——此前"硬件不可达"的结论应撤回，
改为"待验证的 AscendC cube 用法问题"。

## 六、状态

- 交付不变：**16.34 s/step（−8.9%）**，`H3_PERF_COMBO=1` 默认 OFF；
- 用户已指出结论有误，本条为更正记录；
- 下一步：按上面步骤 1 立即验证搬运链路（预计 1–2 轮）。
