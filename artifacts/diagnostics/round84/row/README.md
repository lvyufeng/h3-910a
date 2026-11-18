# Round84 —私有 FP32 行并行求和适配器（仅源码，未在 NPU 执行）

本目录只包含 **task-24** 的产物。它把 Round83 **已实跑验证**的
Half×Half→FP32 输出 kernel 接进 Round79 候选编码器的私有
`MiniMaxH3Qwen3VLRowParallelLinear`，让 TP 部分和以 **FP32** 进入
`group.all_reduce`，最后只做 **一次** Half 收窄。

> **非声明**：这里的源码没有在本任务中编译、没有 launch、没有跑模型、没有
> SSH/远程编译/装包，也没有改动生产文件或提交 git。它不是精度结论、不是加速
> 结论，也不是整编码器验收。NPU 上的数值与耗时由 root driver 在租赁主机上测。

## 它修的是什么

Round79 候选的 Row 路径：

```python
output_parallel = self.quant_method.apply(self, input_parallel)  # GEMM 结果已经是 Half
output_parallel = output_parallel.float()                        # 再提回 FP32，太晚
output_parallel = self.group.all_reduce(output_parallel)         # FP32 集合通信
output_parallel = output_parallel.to(self.output_dtype)          # 第二次 Half 收窄
```

单卡参考路径把整个 `K` 累加在一个 FP32 累加器里、最后只收窄一次。原生 TP 路径
因此让 **每个 rank 的部分和先被 Half 收窄一次** 再求和，FP32 集合通信无法恢复
已经丢掉的低位。

Round83 已经实际执行过的 kernel 恰好提供“Half×Half→FP32 累加器→**FP32 GM 输出**”
（M=2043；K=1024/3456；N=5120），重复输出位级一致、对 FP32 累加诊断的相对 RMS
1.08e-6。本适配器只是把这个已验证原语接到编码器上。

## 文件

| 文件 | 说明 |
| --- | --- |
| [row_fp32_adapter84.py](row_fp32_adapter84.py) | 适配器本体（`SHA256 eb75f43c…`，1120 行） |
| [test_row_adapter84.py](test_row_adapter84.py) | 纯 stdlib 测试，38 项全过（`SHA256 ff0a4972…`） |
| [manifest84.json](manifest84.json) | 来源固定、内存契约、不变量、root 验证计划 |

## API（root driver 调用）

```python
from row_fp32_adapter84 import install_row   # 别名 install 同样可用

adapter = install_row(candidate, runtime=Runtime5GM(rank),
                      kernel_dir=ROOT / "kernels", sequence_length=2043)
adapter.set_enabled(False)          # 先保持禁用；禁用时调用原 forward
adapter.prime(encoder.text_model)   # 参数 load+move 之后：缓存 100 个权重 + 准备 2 个 kernel
adapter.controls(sample_input=x, k_local=1024)   # 可选的 pre-warm 打包证明
adapter.set_enabled(True)           # 打开 FP32 部分和路径
snap = adapter.snapshot()           # 计数器/实际 dtype/字节/显存（纯数据）
adapter.release_scratch()           # 释放激活缓冲，保留权重缓存
adapter.uninstall()                 # 恢复原始 forward，清空全部
```

`install_row` 用闭包里的外层 `adapter` 调用 `adapter._forward(self, input_)`，
不依赖 `self.forward` 的绑定方法身份；只替换 `MiniMaxH3Qwen3VLRowParallelLinear.forward`。

## 打开后的数值顺序

1. 原始 Half 输入；`input_is_parallel=False` 时按原生 `.split(...)[tp_rank].contiguous()` 精确复刻；
2. Half 输入打包为 `(K1, Mp, 16)`，M 的 5 行 padding 保持 **精确 0**；
3. 已验证 kernel：Half×Half→FP32 累加器→FP32 GM（单个 L0C 跨全部 K tile，无 Half 部分和）；
4. FP32 输出还原为 `(M,N)` 或 `(1,M,N)`（与原生空间形状一致）；
5. 用层自己的 `group.all_reduce` 对 **FP32** 部分和求和，前后 dtype 都断言并计数；
6. 可选 FP32 bias（H3 的 row 层是 `bias=False`，该分支不激活但单独计数）；
7. 恰好一次 `.to(layer.output_dtype)`。

## 内存契约

- 每 rank 缓存 **Half** 打包权重 `50*(5120*1024+5120*3456)*2 = 2,293,760,000 B`（≈2.14 GiB），
  无 FP32 权重影子；
- 激活 scratch 合计 `102,236,192 B`（≈97.5 MiB）：两个几何的 A 缓冲、各一个 40 MB 的 FP32 输出、
  4 个 16 元素 aux。

## 只在下列条件成立时才启用（否则 fail closed）

层类型必须是固定的 Row 类本身；`candidate._is_quantized_linear(layer)` 为 False；
权重/输入/`output_dtype` 都是 Half；设备是 NPU 且与 `runtime.device` 一致；
权重形状是已验证的 `(5120,1024)` 或 `(5120,3456)`；`tp_size == group.world_size == 8`；
激活是 `(M,K)` 或 `(1,M,K)` 且 `M == sequence_length`；缓存的权重签名（逻辑
`data_ptr` + `storage_offset` + shape/stride/dtype/device/`_version`）未变。
跨 stream 复用 scratch 默认拒绝，除非显式 `allow_stream_switch` 或 `bind_stream()`。

## 并发与共享 kernel 目录

8 个 rank 共用一个只写一次的 kernel 目录。Round83 builder 自身的 `.compile.lock`
只覆盖编译，它写 `spec83.json` 是就地非原子写。适配器因此：

- 用独立的 `.row84_build.lock` 做跨进程 `flock`，把构建/缓存命中串行化；
- 构建后校验 meta（`dst_dtype=float32`、`block_dim=30`、`m_pad=2048`、K tile 数、
  无 Half 部分和、无 bias 操作数、obj SHA）；
- 若 `spec83.json` 仍解析失败，则用 `os.replace` 原子重写。

root 在 8 个 rank 之前串行预构建两个 kernel，rank 侧走缓存命中路径即可。

## 测试覆盖（38 项，纯 stdlib，本机可跑）

```bash
python3 artifacts/diagnostics/round84/row/test_row_adapter84.py
```

覆盖：候选文件/forward/类源码摘要与固定值一致；改动 forward 后校验失败；R83
kernel 与 runtime 文件摘要一致；launcher 库摘要出现在 runtime 源码中；原始
forward 确实在集合通信前做了 Half 收窄（即被移除的缺陷）；几何与激活形状的
接受/拒绝；输出形状不会塌成 `(N,)`；`m_pad=2048`；权重签名区分同 storage 的
不同视图与 `_version` 变化；install 只改 forward 且默认禁用；禁用时走原 forward；
未 prime 不能启用；uninstall 恢复原函数并能容忍第三方替换；模块顶层不导入
torch/vllm；无 bfloat16/量化 API/subprocess/socket/clamp；FP32 在集合通信前后
都有断言；复用 R83 的 `pack_a/pack_w/unpack_c` 而非重写；kernel 目录有锁与
原子修复；跨 stream 默认拒绝。

## 未建立的部分（不要在报告里当结论）

- 适配器尚未在 NPU 主机上被导入过；
- 两个 kernel 不是这里构建的（由 root 预构建）；
- 没有任何数值：FP32 部分和相对原生“先收窄部分和”的增益未在此测量；
- 没有任何耗时：打包/还原/集合通信开销未测量；
- 无整编码器精度或速度结论；
- FP32 bias 分支已实现但从未被执行（H3 row 层 `bias=False`）。

## root 侧建议验证顺序

1. 8 rank 之前串行预构建 `(K=1024,N=5120)` 与 `(K=3456,N=5120)`；
2. `install_row` → `set_enabled(False)` → load/move 后 `prime`，核对 100 层、
   2,293,760,000 B、2 个 kernel；
3. `controls(sample_input=<真实 Half 激活>, k_local=…)`，要求打包与已验证 helper
   位级一致且 padding 行为 0；
4. 单层 A/B：disabled / enabled / repeat；再扩展到 50 层 CPU 条件与 native 条件；
5. 关闭适配器后，完整 native 路径必须恢复其先前的输出哈希；
6. 打开适配器 warm `encode_ids`，与“本次 enabled 运行自己的 native 参考”比较；
7. 记录 `snapshot()` 计数器、缓存字节、显存高水位与耗时；
8. worker 全部退出后释放租赁。