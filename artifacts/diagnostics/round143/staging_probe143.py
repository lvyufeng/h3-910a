"""R143 staging side channel: when does each block's weights actually arrive?

Measurement only. Records torch events on the native copy stream and on the compute
stream and computes the alignment after the step, so no synchronize is inserted into
the model forward and no value, plan, stream or buffer is changed.
"""
from __future__ import annotations
from collections import defaultdict

_STATE={'installed':False,'prefetch':{},'blocks':{},'order':[]}


def install(torch, blocks):
    """Wrap the native prefetch and stamp every block with compute-stream events."""
    from vllm_omni.diffusion.offloader.layerwise_backend import LayerwiseOffloadHook
    index_by_block={id(b):i for i,b in enumerate(blocks)}
    original=LayerwiseOffloadHook.prefetch_layer
    if not getattr(original,'_staging143',False):
        def prefetch(self,*args,**kwargs):
            index=index_by_block.get(id(getattr(self,'next_block',None)))
            start=torch.npu.Event(enable_timing=True);end=torch.npu.Event(enable_timing=True)
            start.record(self.copy_stream)
            result=original(self,*args,**kwargs)
            end.record(self.copy_stream)
            if index is not None:
                _STATE['prefetch'].setdefault(index,[]).append((start,end))
                done=getattr(self,'_prefetch_done',None)
                if done is not None:_STATE['prefetch'][index][-1]=(start,end,done)
            return result
        prefetch._staging143=True
        LayerwiseOffloadHook.prefetch_layer=prefetch
    for i,block in enumerate(blocks):
        def pre(module,inputs,_i=i):
            event=torch.npu.Event(enable_timing=True);event.record()
            _STATE['blocks'].setdefault(_i,{})['start']=event
            _STATE['order'].append(_i)
        def post(module,inputs,output,_i=i):
            event=torch.npu.Event(enable_timing=True);event.record()
            _STATE['blocks'].setdefault(_i,{})['end']=event
            return None
        block.register_forward_pre_hook(pre)
        block.register_forward_hook(post)
    _STATE['installed']=True
    return True


def collect(torch):
    """Align copy-stream and compute-stream events once, after the step."""
    if not _STATE['installed']:
        return {'installed':False}
    torch.npu.synchronize()
    rows=[]
    for i in sorted(_STATE['blocks']):
        block=_STATE['blocks'][i]
        entry={'block':i}
        spans=_STATE['prefetch'].get(i) or []
        if spans:
            start,end,*rest=spans[-1]
            try:
                entry['prefetch_span_ms']=round(start.elapsed_time(end),3)
            except BaseException:
                entry['prefetch_span_ms']=None
            if 'start' in block:
                try:
                    # positive: copy finished after the block started -> the block waits
                    entry['prefetch_end_after_block_start_ms']=round(end.elapsed_time(block['start']),3)
                except BaseException:
                    entry['prefetch_end_after_block_start_ms']=None
        if 'start' in block and 'end' in block:
            try:entry['block_ms']=round(block['start'].elapsed_time(block['end']),3)
            except BaseException:entry['block_ms']=None
        rows.append(entry)
    stalls=[r['prefetch_end_after_block_start_ms'] for r in rows
            if r.get('prefetch_end_after_block_start_ms') is not None]
    spans=[r['prefetch_span_ms'] for r in rows if r.get('prefetch_span_ms') is not None]
    result={'installed':True,'blocks_observed':len(rows),'block_order':_STATE['order'][:60],
            'per_block':rows,
            'prefetch_blocks_with_events':len(spans),
            'prefetch_span_ms_sum':round(sum(spans),1) if spans else None,
            'prefetch_span_ms_max':round(max(spans),1) if spans else None,
            'blocks_whose_weights_arrived_late':len([s for s in stalls if s>0]),
            'late_total_ms':round(sum(s for s in stalls if s>0),1) if stalls else None,
            'late_max_ms':round(max(stalls),1) if stalls else None,
            'note':'positive late value = copy-stream completion happened after the block started consuming weights',
            'measurement_only_no_math_change':True,'no_synchronize_inside_forward':True}
    _STATE['prefetch'].clear();_STATE['blocks'].clear();_STATE['order']=[]
    return result
