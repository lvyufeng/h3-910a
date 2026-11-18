# Round86 — CPU 第24层重置：输入误差被晚段放大，SiLU 并非大精度缺口

保持[CPU FP16 基准](<../round78/final_CPU_reference_verification78.json>)及原始参数不变，
将其保存的 raw layer24 状态位级校验传入 NPU layer25，只运行25..49。
没有重新注入 DeepStack、应用 final norm、修改条件或重算金标准，也不把这条精度隔离
路径的耗时当作完整 encode 时间。

## 真实晚段结果

[全8 rank 汇总](<summary86.json>)与[rank0 原始回执](<results/late01/ranks/tp8_rank0.json>)：
全部25层执行，3个案例重复位级一致，原生路径恢复，原参数与 CPU 输入未变。

| 由精确 CPU layer24 出发，与原始 CPU raw layer49 比较 | 相对 RMS |
|---|---:|
| 原生 attention + 原生 row | **0.5213%** |
| 精确 attention + 原生 row | 0.7214% |
| 精确 attention + FP32 row | 0.5450% |

对照 R85 从初始 CPU 条件跑完整50层的2.4757%/1.8193%/2.7148%，输入重置显著降低晚段
最终误差，但仍不等于整体精度通过。原生 row 的0.5213%与全图2.4757%差异表明：前段
微小输入差异会被晚段放大；本实验没有证明某个单一晚层或算子就是根因。

## 实际第25层叶算子

对同一组真实 Half 输入/权重，在 CPU Half 上运行对应叶算子；不使用 Float 权重教师，
不把局部 down_proj 当作完整 HCCL 行并行输出。rank0 的相对误差：

| 算子 | 相对 RMS |
|---|---:|
| HF 两舍入点 RMSNorm：Float归约→Half归一化→Half gamma | 6.425e-6 |
| 本地 gate / up Half Linear | 1.306e-5 / 1.345e-5 |
| 实际原生 Half gate 的 SiLU | **4.423e-7** |
| SiLU × up（同一原生值上的 CPU Half 对照） | 3.132e-7 |
| 本地 down Half Linear | 1.228e-5 |

**将 SiLU 激活扩展至 Float 再转 Half，输出与原生 SiLU 位级完全相同**。因此这组实际晚层
数据不支持“Half SiLU 大精度缺口”假设；没有为它增加无效 FP32 激活路径。
[原始叶报告](<leaf/late_leaf86.py>)保留冻结/重复门禁及 CPU Half 参考资格。

## 源码和验收边界

[晚段驱动](<late_rebase86.py>)只构建和加载原始层25..49，使用 ModuleDict 原始权重名称；
FP32 row 适配器明确配置50个行权重（25层×2），没有误用完整50层的100权重假设。
所有参考仍为原始 CPU FP16，数值容差未另行选择，完整精度通过字段保持 false。

另从已安装 HF 原始源码核验了[两舍入点 RMSNorm与 Half RoPE](<modeling_qwen3_vl_original86.py#L353-L437>)，
确认没有把 RoPE 或 gamma 运算“修复”成与基准不同的舍入顺序。最初 inspect 导入因缺少
libhccl 环境失败；改为只读复制原始源码检查，失败 stdout 保留，不冒充成功读取结果。

[最终核验](<final_verification86.json>)确认23生产文件、13安装API文件、CPU基准与运行库
未变，相关worker为零，owned ts217租约已释放。所有执行来源由
[只读源码 SHA 核验](<executed_sources86.json>)记录。32项源码/实际证据测试通过。

完整 CPU FP16 图精度、编码<2s、真实 encoder/DiT共驻及官方同任务比较仍未完成。
[下一步](<NEXT.md>)继续分离晚段输入敏感性和单层真实误差，保留原生 row 默认对照，
不根据某一叶算子更精确就叠加全图，也不放宽容差。
