# R243 — 三次尝试定位 offloader 等待：找到了钩子，但测量未成功且末轮步时 +0.531 s（作废）

## 一、三次尝试与结果

| 尝试 | 做法 | 结果 |
|---|---|---|
| a | 在 `block._forward_hooks` 里找 `LayerwiseOffloadHook` | `blocks_hooked=0`（找错位置） |
| b | 改进为检查绑定方法的 `__self__` | 仍 `0`（仍找错位置） |
| c | 按源码改到 `module._hook_registry` | **`blocks_hooked=50/50`**，但 `delta=null`，且**步时 18.4594 s（+0.531 s）** |

- 尝试 a/b 的步时：18.0233 s（+0.095）、18.0507 s（+0.123），虽在 ±0.15 内但已偏离常规。
- **尝试 c 步时 18.4594 s 远超阈值 ⇒ 该轮计时作废**，不引用任何时间结论。

## 二、源码事实（本轮新增，值钱的部分）

1. offloader 的钩子**不在** `_forward_hooks`，而是注册进 RT-DiT 风格的
   `HookRegistry`：`registry.register_hook(LayerwiseOffloadHook._HOOK_NAME, hook)`，
   挂在 `module._hook_registry` 上。这是前两次失败的原因。
2. 每块的等待点只有一个：`post_forward → offload_layer →
   clear_block_storage(..., self._prefetch_done)` 内的 `compute.wait_event(下一块拷贝完成)`，
   随后 `self._prefetch_done = None`。
3. `pre_forward` 只有在本块未 materialize 时才同步回退（`non_blocking=False`），稳态不走此分支。

## 三、为什么仍测不到（诚实记录）

- 我用 `register_forward_hook(prepend=True)` 想让自己的钩子**先于** offloader 的 `post_forward` 执行，
  以便在 `_prefetch_done` 被清空前读到它；实测 `delta` 全为 `null` ⇒ 这个“抢跑”没有生效
  （Registry 的钩子调度顺序与 `_forward_hooks` 的 prepend 无关）。
- 同时，“在 block 上注册前置钩子”本身**不是零扰动**：块 span 从 R235 的 347.0 ms
  升到 **357.9 ms（+10.9 ms/块 ≈ +0.55 s/步）**，正好等于本轮步时增量。
  ⇒ 该方法学路线被证伪，不能用于测量等待。

## 四、方法学结论（第三条规则）

**不要为了测量而向 block 模块注册钩子**：block 是 offloader 的调度边界，
在它上面加前置/后置钩子会改变拷贝与计算的交错，+11 ms/块的代价与要测的量同量级。

## 五、下一步（R244，唯一还没试过的零扰动路线）

**包装 `clear_block_storage` 函数本身**（`vllm_omni/diffusion/offloader/tensor_utils.py`）：

```python
real = tensor_utils.clear_block_storage
def wrapped(params, bufs, ready_event):
    if ready_event is not None:
        now = Event(); now.record()                     # compute 流
        delta = ready_event.elapsed_time(now)           # 正=拷贝已就绪，负=仍在飞
        record(max(0.0, -delta))
    return real(params, bufs, ready_event)
```

- 它在**每块的最后一步**被调用，是唯一真正承载等待的位置；
- 包装它只在 Python 层加几十微秒，不向任何 forward 插入事件、不改钩子注册；
- 若 `stall_total_ms` 累加仍接近 3.457 s ⇒ offloader 等待就是缺口；
  若接近 0 ⇒ 缺口不在权重流，需要转向“父区间内未归属的设备工作”其它解释。

## 六、合规

- 未改盘上源码；未安装候选；FA trim 仍 OFF；未下载/回传张量。
- 三次尝试全部留档；末轮 +0.531 s 的计时数据主动作废；主机已无残留进程（三次运行均已退出）。