"""Default-off complete DiT block fixtures for same-input standalone replay."""
import hashlib
import json
import os
from pathlib import Path
import re
import torch

_DONE=set()


def begin(module,x,kwargs):
    path=os.environ.get('VLLM_OMNI_910A_BLOCK_AUDIT_DIR')
    if not path or not torch.distributed.is_initialized():return None
    rank=torch.distributed.get_rank()
    match=re.search(r'blocks\.(\d+)\.',module.adaln_proj._cache_name)
    if match is None:return None
    layer=int(match.group(1))
    if layer not in (0,24,49) or (rank,layer) in _DONE:return None
    root=Path(path)/('rank%d'%rank);root.mkdir(parents=True,exist_ok=True)
    target=root/('block%02d.pt'%layer)
    if target.exists():return None
    _DONE.add((rank,layer))
    payload={'x':x.detach().cpu().contiguous().clone(),'kwargs':{},'layer':layer,'rank':rank}
    for key,value in kwargs.items():
        if isinstance(value,torch.Tensor):value=value.detach().cpu().contiguous().clone()
        if key=='video_layout':
            # Dense FA replay ignores VSA layout; the replay must still match
            # original block output before treating this as equivalent.
            payload['video_layout_present']=value is not None
            value=None
        payload['kwargs'][key]=value
    tensors=[payload['x']]+[v for v in payload['kwargs'].values() if isinstance(v,torch.Tensor)]
    assert sum(t.numel()*t.element_size() for t in tensors)+x.numel()*x.element_size()<512*1024*1024
    payload['_target']=target
    return payload


def finish(payload,out):
    if payload is None:return
    target=payload.pop('_target')
    payload['output']=out.detach().cpu().contiguous().clone()
    with target.open('xb') as f:torch.save(payload,f)
    tensors=[payload['x'],payload['output']]+[v for v in payload['kwargs'].values() if isinstance(v,torch.Tensor)]
    meta={'layer':payload['layer'],'rank':payload['rank'],'shape':list(payload['x'].shape),
          'kwargs':{k:list(v.shape) if isinstance(v,torch.Tensor) else v for k,v in payload['kwargs'].items()},
          'tensor_bytes':sum(t.numel()*t.element_size() for t in tensors),
          'x_sha256':hashlib.sha256(memoryview(payload['x'].view(torch.uint8).numpy())).hexdigest(),
          'output_sha256':hashlib.sha256(memoryview(payload['output'].view(torch.uint8).numpy())).hexdigest(),
          'scope':'Actual first-step dense DiT block input/output, complete local rows on each rank, instrumentation-only'}
    target.with_suffix('.json').write_text(json.dumps(meta,indent=2)+'\n')
