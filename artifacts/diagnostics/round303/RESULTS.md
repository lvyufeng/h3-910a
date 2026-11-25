# R303 — 查清 `GetSysWorkSpacePtr` 语义；AICore 异常另有原因

## 一、源码证据

`asc/include/basic_api/kernel_operator_swap_mem_intf.h:41-58`：

```cpp
__aicore__ inline __gm__ uint8_t* __gm__ GetSysWorkSpacePtr()
{
    if constexpr (__NPU_ARCH__ == 2201 || __NPU_ARCH__ == 3510) {
        return __get_kfc_workspace_addr();
    } else {
        return g_sysWorkspaceReserved;
    }
    ...
    // framework launch
#if (WORKSPACE_PARAM_OFFSET != 0xffffffff)
    return ((GM_ADDR*)get_para_base())[WORKSPACE_PARAM_OFFSET];
#else
    return g_sysWorkspaceReserved;
#endif
}
```

要点：

- 系统工作区指针由**框架在 launch 时注入**（`g_sysWorkspaceReserved` 或从 param base 取）；
- 文档只保证「系统工作区从这里开始」，**没有保证它与 kernel 第 5 个参数 `workspace` 的相对位置**；
- 我们之前假设「用户缓冲 = `workspace + sysWs`」可能不成立（若框架把系统工作区放在
  `workspace` 之外的独立分配，则我们的用户缓冲就在 `workspace` 里，且 `workspace` 大小只有
  `sysWs + 512KB` 也够；但若框架认为 `workspace` 全部属于用户，则 sys 区在别处，也可能是越界）。

## 二、当前可用的两个事实

1. `ACL_FAIL 507015 = ACL_ERROR_RT_AICORE_EXCEPTION`（kernel 崩溃，非 host 参数错）；
2. 把手工缓冲从 192 KB 降到 48 KB **不能**消除该异常 ⇒ 不是 UB 超限。

## 三、下一步（R304，最小化定位）

不再猜语义，用「二分 kernel」定位崩溃点：

| 步骤 | kernel 内容 | 预期 |
|---|---|---|
| K1 | 只有 `REGIST_MATMUL_OBJ` + `SetTensorA/B` + `IterateAll`（写 workspace 起始） | 若崩溃→Matmul/tiling 问题 |
| K2 | K1 + 我们自己的 offset（`workspace+sysWs`） | 若崩溃→工作区布局问题 |
| K3 | 只有 `DataCopy(workspace→UB→o)`（不调 Matmul） | 用于确认 workspace 可读写 |

先跑 K1，并把 `sysWs` 打点（用 `printf`/`Assert` 不可用，可先把 C 写到 `workspace` 起始并令
`currentWorkspace[0] = 512KB`，看是否仍崩）。

## 四、状态

- 交付不变：**16.34 s/step（−8.9%）**，49 步 ×2 验证，relRMS 在原生带内，`H3_PERF_COMBO=1` 默认 OFF；
- AscendC：S0 ✅、S0.5 ✅、S1a-0 ✅；S1a-1 编译 ✅、运行期 AICore 异常未解；
- 机器 ts-217（910A）租约至 20:52 UTC。
