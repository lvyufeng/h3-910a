"""R126 low-interference profiling with preallocated NPU events.

Hooks record stream events; there is no device/stream synchronization in a module
hook, no activation copy and no reduction. CPU dispatch time and device-span time
are separate. Parent and child spans overlap and must NEVER be summed together.
"""
from __future__ import annotations
from collections import defaultdict
import time
import torch


class WarmEventProfile126:
    def __init__(self, model, slots_per_region=160):
        self.model = model
        self.slots = slots_per_region
        self.pairs = {}
        self.counters = defaultdict(int)
        self.host_start = {}
        self.host_ms = defaultdict(float)
        self.handles = []
        self.parents = []
        targets = []
        for index, block in enumerate(model.blocks):
            for name, module in [('block',block),('attention',block.attn),('MLP',block.mlp),
                                 ('AdaLN',block.adaln_proj),('norm1',block.norm1),('norm2',block.norm2)]:
                targets.append(('main.'+name,module))
            targets.extend([('attention.QKV',block.attn.qkv_proj),('attention.out',block.attn.out_proj),
                            ('MLP.fc1',block.mlp.fc1),('MLP.fc2',block.mlp.fc2)])
        for name in ['token_refiner','time_embedder','final_layer','video_patch_proj','audio_patch_proj','condition_proj']:
            module=getattr(model,name,None)
            if module is not None:targets.append((name,module))
        for label in {label for label,_ in targets}:
            self.pairs[label]=[(torch.npu.Event(enable_timing=True),torch.npu.Event(enable_timing=True))
                               for _ in range(slots_per_region)]
        for label,module in targets:
            self.handles.extend([module.register_forward_pre_hook(self._pre(label)),
                                 module.register_forward_hook(self._post(label))])

    def _pre(self,label):
        def pre(module,inputs):
            index=self.counters[label]
            if index>=self.slots:raise RuntimeError('profiling slot overflow: '+label)
            self.pairs[label][index][0].record()
            self.host_start[(label,id(module))]=(index,time.perf_counter())
        return pre

    def _post(self,label):
        def post(module,inputs,result):
            index,start=self.host_start.pop((label,id(module)))
            self.host_ms[label]+=(time.perf_counter()-start)*1000
            self.pairs[label][index][1].record()
            self.counters[label]+=1
        return post

    def reset(self):
        self.counters.clear();self.host_start.clear();self.host_ms.clear()

    def summarize(self):
        assert not self.host_start
        rows={}
        for label,count in self.counters.items():
            milliseconds=[a.elapsed_time(b) for a,b in self.pairs[label][:count]]
            rows[label]={'calls':count,'device_span_total_ms':sum(milliseconds),
                         'device_span_mean_ms':sum(milliseconds)/count,
                         'host_dispatch_total_ms':self.host_ms[label]}
        return {'regions':rows,'hooks_synchronize_device':False,
                'parent_child_spans_overlap_do_not_sum':True,
                'device_span_includes_waits_on_current_stream':True}

    def close(self):
        for h in self.handles:h.remove()
        self.handles.clear()
