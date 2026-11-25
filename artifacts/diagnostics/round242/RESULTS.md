# R242 — 只读源码确认 offloader 的真实等待结构（解释 R241 为何采不到样本）

## 一、源码事实（`vllm_omni/diffusion/offloader/layerwise_backend.py`）

```python
def prefetch_layer(self, non_blocking: bool = True) -> None:      # line 155
    self.copy_stream.wait_stream(current_omni_platform.current_stream())
    evt = Event()
    with stream(self.copy_stream):
        for dtype, cpu_weight in self.dtype_cpu_flattened_weights.items():
            gpu_weight = torch.empty(...)
            gpu_weight.copy_(cpu_weight, non_blocking=non_blocking)
        evt.record(self.copy_stream)          # line 178
    ... set_tensor_storage(...)  # 只重绑定 storage，不拷贝
    self._prefetch_done = evt                 # line 199

def offload_layer(self) -> None:              # line 202
    clear_block_storage(self.block_parameters, self.block_buffers, self._prefetch_done)
    self._prefetch_done = None                # line 207  ← 事件对象在此被丢弃

def pre_forward(self, module, *args, **kwargs):                   # line 228
    if not self.is_materialized and self._prev_hook is not None:
        self._prev_hook.prefetch_layer(non_blocking=False)        # 同步回退（阻塞 H2D）
    self.prefetch_layer(non_blocking=True)                        # 预取“下一块”
    return args, kwargs

def post_forward(self, module, output):                            # line 239
    self.offload_layer()          # 内部 current_stream().wait_event(下一块的 _prefetch_done)
    return output
```

### 由此得到的三条事实

1. **R241 采不到样本的原因**：`_prefetch_done` 在 **`post_forward → offload_layer` 里被置 None**（line 207）。
   等到块 N 的 `pre_forward` 去读 `self._prev_hook._prefetch_done` 时，它已经是 `None`。
   我的属性名假设其实是对的，是**读取时机**错了。
2. **稳态下每块没有显式 host 等待**：`pre_forward` 只在“当前块未 materialize（例如块级缓存跳过）”时
   才走同步回退（`non_blocking=False`）；R227 观测到 `synchronous_fallback.count = 1`（每 rank 一次，
   应是首块），与此一致。
3. **每块末尾有一次“计算等拷贝”**：`offload_layer` 里的 `clear_block_storage(..., _prefetch_done)`
   会让 compute 流 `wait_event(下一块的拷贝完成)`。由于该拷贝在**本块 `pre_forward` 时**就发起、
   本块要跑约 347 ms，而 646 MB 在 21.3 GB/s 下约 30 ms，稳态应早已完成 ⇒ **理论上不产生暴露等待**。

## 二、对 3.457 s 缺口的判断（本轮不下结论，只排除）

- offloader 的显式等待路径在稳态**应当无代价**，这与 R228（把慢 H2D 的 rank 1/5 与快 rank 对比，
  `rest` 只差 0.0042 s）方向一致；
- 因此 3.457 s 缺口**不能**简单归给 offloader 等待，需要下一步用**正确时机**的测量来定性：
  在 `post_forward` 的 `offload_layer` 调用**之前**读取 `self._prefetch_done`（此时尚未置 None），
  用 `elapsed_time(block_end_event)` 得到“拷贝完成 → 块结束”的间隔；
  若该值为负即说明拷贝尚未完成、块末尾存在暴露等待。

## 三、R243 设计（零扰动）

不改任何 forward，只**在 block 级钩子**里做两件事（钩子已证明不改变步时，R240 = −0.004 s）：

1. block pre-hook：记录 compute 流事件 `E_enter`；
2. block post-hook：记录 `E_exit`，并在**该钩子里**读取 `hook._prefetch_done`（此时 `post_forward`
   的 offload 尚未执行，事件仍有效），计算 `elapsed_time(E_exit)`：
   - 正值 = 下一块权重在本块结束前已就绪（无暴露等待）；
   - 负值 = 尚未就绪，`max(0, -值)` 即暴露等待时长；50 块累加后与 3.457 s 对照。

## 四、合规

- 本轮纯只读（grep/sed 源码），未在主机上跑任何计算、未加载模型、未改任何文件。
- 未安装候选；FA trim 仍 OFF；未下载/回传张量。
- 主机状态：无残留进程，`/tmp/h3_budget149` 占用约 15 MB/目录。