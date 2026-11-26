# R352 — 决定性证据：910A 是 **v200 代核心**（`KERNEL_TYPE_AIC_ONLY` 不被支持）；task-type 假设**排除**

## 一、本轮拿到的硬证据

按构建脚本的判定逻辑（`find_kernel_type_by_source`：正则抓
`__enable_feature_for_compile_default = <token>` 并映射 `CodeMode`），
我在 kernel 里用 `static constexpr int KERNEL_TYPE_AIC_ONLY = 1;` 保证 token 合法，
然后写 `KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIC_ONLY);`。

构建输出（决定性）：

```
[ERROR] [op_compilation.py:346] Exception occured,
        current kernel type: KERNEL_TYPE_AIC_ONLY is not support in current core version
```

## 二、由此得到的确定事实

1. **构建脚本确实在读源码里的 task-type token**（我的写法被正确识别）；
2. **910A 不支持 `KERNEL_TYPE_AIC_ONLY`** ⇒ 它不属于 v220（910B）代，而是 **v200 代（m200 构建模式）**；
3. v200 代支持的 token（来自 `constants.py` 的 `STR_TO_KERNEL_TYPE_V200`）只有：
   - `KERNEL_TYPE_AICORE` → `CodeMode.AIC`
   - `KERNEL_TYPE_VECTORCORE` → `CodeMode.AIC`
   - `KERNEL_TYPE_MIX_VECTOR_CORE` → `CodeMode.MIX_VECTOR_CORE`
4. 而 `find_kernel_type_by_source` 在**没有 token 时**对非 v220/c310 构建模式**直接返回 `CodeMode.AIC`**。

⇒ **我们的 kernel 本来就是 AIC 模式**，`Mmad` 不该被"编译掉"。
**"task type 是 AIV-only"这个假设被排除**（我上一轮的猜测错了）。

## 三、这条证据的真正价值

910A 是 **v200 代核心**，这意味着：CANN 文档与样例绝大多数面向 v220（910B），
其 `LoadData`/`Mmad`/`DataCopyCO12Dst` 的**参数语义可能是 v220 版本**，
在 v200 上并不适用——这与我们"参数怎么改结果都不变/全 0"的现象高度一致。

## 四、下一步（方向已换，更具体）

1. 到 `tikcpp/tikcfw/` 下找 **v200 专属**的 intrinsic 定义（例如
   `kernel_operator_data_copy_intf.h` 里 `#if __CCE_AICORE__ == 100/200` 分支），
   确认 910A 上 `LoadData2dParams` 的字段语义、`MmadParams` 的字段集合；
2. 用 v200 分支要求的参数重写 Mmad 调用（可能还需要 `set_fmatrix` 类的配置）；
3. 判据不变：A=I 时 `bad=0`。

## 五、状态

- 交付不变：**16.34 s/step（−8.9%）**，`H3_PERF_COMBO=1` 默认 OFF；
- AscendC：cube 可用（框架行 0 精确）；**已确认 910A=v200 代核心**；
  task-type 假设排除；剩余问题聚焦到 **v200 与 v220 的 intrinsic 参数差异**；
- 工程：`/tmp/h3_r221/faA/proj`（kernel 里现在带着 `KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIC_ONLY)`
  与 constexpr，需在下一步按 v200 改写）。
