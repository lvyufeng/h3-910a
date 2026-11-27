# 910A 机器运行 CANN 9.2 的修复记录（ts-87）

## 现象
`torch_npu` 能 import、`torch.npu.is_available()` 为 True、8 张卡可见，但第一个真实算子就失败：
`AclSetCompileopt(ACL_PRECISION_MODE)` → `GEInitializeV2 failed`；`plog` 里真正的根因链是
`CANNKB: ModuleNotFoundError`（先是 `decorator`，随后 `scipy` / `attr`）→ `CannKbInit failed`
→ `TeFusion init failed` → `GEInitializeV2 failed`。

## 根因
CANN 的 CANNKB/TBE 代码在**宿主 Python 解释器内**执行（不是独立 3.7 子进程；plog 的
`(pid,python)` 就是调用方 PID）。出厂机器上 `$ASCEND_HOME_PATH/python/site-packages`
只有 tbe/auto_tune 代码，缺 TBE 运行时需要的第三方包，于是 GE 初始化失败。
`vllm-ascend` 的 sdist 构建同样因 `ascend910a` 未列入 SoC 表而失败，属于另一处（可选）问题。

## 修复（ts-87 已验证）
向 `$ASCEND_HOME_PATH/python/site-packages` 安装 **cp312** 版本的 TBE 依赖（与运行时解释器同版本，
避免 PYTHONPATH 污染；实测 cp37 版本会因 ABI 不匹配失败）：

```bash
uv pip install --python <runtime py3.12> \
  --index-url https://pypi.tuna.tsinghua.edu.cn/simple \
  --target $ASCEND_HOME_PATH/python/site-packages \
  numpy sympy scipy decorator cffi pyyaml psutil attrs absl-py
```

其中 `attrs` 必须提供 `attr` 兼容模块（attrs>=25 自带）。完成后实测：

| 项目 | 结果 |
|---|---|
| 单卡 FP16 matmul 2048/5376/8192 | 82.3 / 90.9 / 100.0 TFLOPS |
| 8 卡 HCCL all_reduce（32 MB fp16） | 2.63 ms，算法带宽 22.35 GB/s |
| 8 卡 all_reduce 正确性 | sum = 7168 = 256×(0+…+7) ✓ |

## 结论
ts-87/ts-88 这类 **真 Ascend910A** 机器在补齐上述依赖后可以运行 torch_npu + HCCL，
不需要 vllm-ascend（0.30 的 SoC 表不含 ascend910a；本次路线为自带 TP 的独立 runner）。
