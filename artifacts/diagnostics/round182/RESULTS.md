# R182 — 自定义算子**首次跑通**：2.07 / 1.58 / 0.09 ms；判定正确性**未成立**

## 一、卡了 8 轮的构建问题，本轮两个解决方案

### 根因：CANN 的 `asc_opc` 抢占 PYTHONPATH

`$CANN/bin/asc_opc` 里 `export PYTHONPATH="${CANN_SITE_PACKAGES}:${PYTHONPATH}"`，
CANN 自己的包永远在最前，R179–R181 的外部 overlay 全部被遮蔽。

### 方案 1（已验证有效）：私有 `asc_opc` 包装

绕开 CANN 包装，用 `PYTHONPATH=$OVL:$CANN_SITE:$PYTHONPATH` 直接调 `opc.py`。
验证：`imported from: /tmp/h3_overlay182/...`、`patch present: True`。

### 方案 2（**最稳，推荐**）：包装 `bisheng` 本身

CANN 用 `shutil.which("bisheng")` 解析编译器（`ascendc_common_utility.py:267`），
所以在 `PATH` 最前放一个 wrapper 即可拦到**所有**设备编译，与哪个 python 模块拼命令无关：

```bash
exec $REAL "$@" -isystem $H/include -isystem $H/include/c++/14.3.0 \
    -isystem $H/include/c++/14.3.0/aarch64-target-linux-gnu
```

**这个方法同时覆盖 `ascend910` 和 `ascend910b` 两条编译路径**——方案 1 的 5 个锚点只覆盖了后者。
（实测：soc 改成 `ascend910` 后方案 1 失效、`cstdint` 复现；方案 2 立刻 0 错误。）

## 二、soc 必须是 `ascend910`，不是 `ascend910b`

R176 我把 `ASCEND_COMPUTE_UNIT` 和 `AddConfig` 都改成了 `ascend910b`，方向**错了**。

运行时错误说明了原因：

```
SoC version ascend910 verification failed.
This SoC is not configured through the AddConfig API of the OpDef class.
```

设备注册名是 **`ascend910`**（任务原始目标写的也是 8×910A）。已改回。

## 三、op_host 骨架必须改（否则 aclnn 参数校验失败）

msopgen 生成的骨架是**逐元素算子**模板：

```cpp
*y_shape = *x1_shape;                              // 输出形状 = 输入形状
context->SetOutputDataType(0, inputDataType);      // 输出类型 = 输入类型
tiling->size = data_sz;  context->SetBlockDim(8);  // coreNum 从未赋值！
```

对一个归约算子这是错的。已改为：
- `InferShape`：输出 = `[H3_CORE_NUM]`（每核一个槽）
- `InferDataType`：输出 = `DT_INT32`
- `TilingFunc`：`tiling->coreNum = H3_CORE_NUM`（骨架漏了）

改完后 `GetWorkspaceSize` 通过。

## 四、**首次实测计时（本阶段最重要的数字）**

| 形状 | 内核 | R178 torch 代理 | 倍数 |
|---|---|---|---|
| `mlp_fc1` (10448×14336, 285.7 MiB) | **2.076 ms** | 3.905 ms | **1.88×** |
| `qkv_proj` (10448×10752, 214.3 MiB) | **1.582 ms** | 2.947 ms | **1.86×** |
| `adaln` (3×96768, 0.6 MiB) | **0.087 ms** | 0.146 ms | 1.68× |

换算每步（50 / 50 / 55 次）：

```
50×2.076 + 50×1.582 + 55×0.087 = 103.8 + 79.1 + 4.8 = 187.7 ms
```

| 口径 | 每步 |
|---|---|
| 当前审计（R162） | **2870 ms** |
| 本轮内核（8 核） | **188 ms** |
| **隐含节省** | **≈2.68 s** |

**比 R178 代理预期的 2.52 s 更多**，因为真实内核比 torch 归约还快（8 核已优于单队列归约）。

## 五、但**判定结果是错的**——不可用

15 个对抗用例只有 6 个"通过"，且**这种通过与正确性无关**：

```
all_finite_normal   maxMag=0x3473b907 ... MISMATCH
zeros               maxMag=0x3de2b9b8 ... MISMATCH
```

问题特征：
1. `maxMag` 出现 `0x3d443d5e`、`0x3a68c102` 这类**超过 16 位**的值——我读的是 int32，
   但内核写的 `blockMax` 上界只有 `0x7fff`，所以读回的**不是内核写的值**。
2. 这些值在不同用例间**重复出现**，不随输入变化——典型的是读到了固定的残留/未初始化内存。
3. `0x3d443d5e` 作为 **float32** 约为 0.048，`0x3a68c102` 约 0.00089——像是**浮点张量的残留数据**。

已尝试 `pipe_barrier(PIPE_ALL)` 同步 V/S 流水，**无效**（可能不是同步问题）。

### 未验证的怀疑方向（下一轮按序排查）
1. `ReduceMax` Level-2 的 `dst` 是否要求特定大小/对齐，`dst.GetValue(0)` 是否取错位置；
2. `TQue<VECCALC>` 的缓冲区是否被 `ReduceMax` 的 `sharedTmpBuffer` 写坏；
3. 每核写 `yGm_[coreIdx]` + `DataCopy(yGm_, oc, 8)` 的写出是否真的落到 `y[coreIdx]`
   （`SetGlobalBuffer` 用 `y + coreIdx` 偏移，但 `DataCopy` 长度 8 可能越界覆盖）；
4. 输出的 `int` 视图与 `TQue<VECOUT>` 的 32B 对齐要求。

**第 3 条最可疑**：`yGm_.SetGlobalBuffer((__gm__ int*)y + coreIdx, 1)` 后
`DataCopy(yGm_, oc, 8)` 想写 8 个 int，但 buffer 只声明了 1 个元素，且每核都会写 8 个——
这会造成 8 核互相覆盖写成混乱值。

## 六、状态判定

| 项 | 状态 |
|---|---|
| 编译 | ✅ 通（`bisheng` wrapper） |
| 安装 | ✅ 通（`SUCCESS`，vendor 目录就绪） |
| 运行 | ✅ 通（aclnn 调用成功） |
| **计时** | ✅ **2.076 / 1.582 / 0.087 ms，值 2.68 s/步** |
| **判定正确性** | ❌ **失败，不可采用** |
| 阶段 A 是否成立 | ⏳ 计时支持，但要等判定修好 |

**最重要的结论**：性能前提**成立且好于预期**（2.68 s vs 预期 2.52 s），
所以值得继续投入修正确性——这不再是"值不值得做"的问题，而是"能不能修对"的问题。

## 七、合规

- 共享 CANN 未改：overlay 与 wrapper 都在 `/tmp`
- vendor 目录安装了自定义算子包（CANN 官方支持的扩展位置，非共享 SDK 修改）
- 未下载权重；未做 driver/device reset；lease 将释放
