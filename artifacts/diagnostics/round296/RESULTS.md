# R296 — S1a-1 第一版：cube 链路写出来并编译到只剩一个 API 错误

## 一、本轮做了什么

按 S1a-1 目标写了 `S = q(64×128) @ k(512×128)ᵀ` 的 AscendC 内核（源码已入库
`round296/deployed/fa_kernel_s1a1_v1.cpp`），流程：

1. `TQue<A1/B1>` 申请 L1，`Nd2NzParams` + `DataCopy` 把 q/k 从 GM 搬到 L1（ND→NZ）；
2. `TQue<A2/B2>` + `LoadData2dParams` + `LoadData` 搬 L1→L0A/L0B（k 用 `ifTranspose=true`，实现 Kᵀ）；
3. `MmadParams` + `Mmad` 计算到 L0C（FP32 累加，`cmatrixInitVal=true`）；
4. 尝试 L0C→UB 的 `DataCopy`，再 `Cast` 到 half，最后 `DataCopy` 回 GM。

## 二、编译结果：只剩 1 个错误

```
flash_attention_custom.cpp:79:5: error: no matching function for call to 'DataCopy'
```

即 **L0C→UB 的搬运 API 用错**。检索 AscendC 头确认：

- `ascendc/basic_api/interface/kernel_operator_data_copy_intf.h:57` 只有
  `DataCopy(const LocalTensor<T>& dst, const GlobalTensor<T>& src, const DataCopyParams&)`；
- L0C（CO1）到 UB 的搬运不走 `DataCopy`，需要 **`Fixpipe`** 系列接口
  （`basic_api` 里尚未定位到其声明，下一步要先找 `Fixpipe` 头与 `FixpipeParams*` 结构）。

## 三、其余部分没有报错

q/k 的 ND→NZ、L1→L0A/L0B 的 `LoadData`、`Mmad` 参数、`Cast`、GM 回写都**通过了编译**，
说明 S1a-1 的结构是对的，只差 L0C 出口这一步。

## 四、下一步（很具体）

1. 定位 `Fixpipe` 的声明与参数结构（`FixpipeParamsV220` / `FixpipeParams`），
   用 `Fixpipe(ubFp32, cL0, params)` 把 L0C 搬到 UB；
2. 重新 build；通过后安装、用 `test_fa294.cpp` 扩展（k 扩到 512 行、输出 64×512）与 CPU 参考对比；
3. 数值通过后进入完整 FA（softmax 向量段 + PV + 循环）。

## 五、状态

- 交付不变：16.34 s/step（−8.9%），49 步 ×2 验证，数值在原生带内；
- 机器 ts-217（910A）租约有效（至 20:52 UTC）；
- AscendC 阶段：S0 ✅、S0.5 ✅、S1a-0 ✅、**S1a-1 编译推进到只剩 L0C 出口**。
