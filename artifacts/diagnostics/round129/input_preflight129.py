#!/usr/bin/env python3
"""Read-only on-host R128 native-input metadata/positions, no model inference."""
from __future__ import annotations
import hashlib
import json
from pathlib import Path
import os

ROOT=Path('/tmp/h3_repeat12901')
OLD=Path('/tmp/h3_overlap12801')


def sha(path):
    digest=hashlib.sha256()
    with path.open('rb') as file:
        for chunk in iter(lambda:file.read(4*1024**2),b''):digest.update(chunk)
    return digest.hexdigest()


def main():
    assert ROOT.is_dir() and not (ROOT/'input_preflight129.json').exists()
    import torch
    torch.set_num_threads(4)
    result={'scope':__doc__,'completed':False,'model_inference':False,'local_model_or_activation_transfer':False,'inputs':[]}
    first=None
    for rank in range(8):
        path=OLD/f'native_worker_input_step25_rank{rank}.pt'
        expected=json.loads((OLD/f'formal_replay_rank{rank}128.json').read_text())['input_sha256']
        actual=sha(path);assert actual==expected
        data=torch.load(path,map_location='cpu',weights_only=True)
        def meta(value):
            if isinstance(value,torch.Tensor):return {'type':'tensor','shape':list(value.shape),'dtype':str(value.dtype),'stride':list(value.stride()),'storage_offset':value.storage_offset(),'bytes':value.numel()*value.element_size()}
            if isinstance(value,dict):return {str(k):meta(v) for k,v in value.items()}
            if isinstance(value,(list,tuple)):return [meta(v) for v in value]
            assert value is None or isinstance(value,(str,float,int,bool));return value
        positions={}
        for name in ('img_pos_info','audio_pos_info','text_pos_info','img_pos_for_infer_output_info'):
            values=data[name]['position_ids'].view(-1).long();unique,counts=torch.unique(values,return_counts=True)
            positions[name]={'rows':values.numel(),'unique_rows':unique.numel(),'repeated_rows':values.numel()-unique.numel(),
                             'min':int(values.min()),'max':int(values.max()),'max_multiplicity':int(counts.max())}
        sets={name:set(data[name]['position_ids'].view(-1).tolist()) for name in ('img_pos_info','audio_pos_info','text_pos_info')}
        overlap={a+':'+b:len(sets[a]&sets[b]) for a,b in [('img_pos_info','audio_pos_info'),('img_pos_info','text_pos_info'),('audio_pos_info','text_pos_info')]}
        result['inputs'].append({'rank':rank,'file_sha256':actual,'file_bytes':path.stat().st_size,'positions':positions,'cross_modality_position_overlaps':overlap,'structure':meta(data) if rank==0 else None})
        del data
    result['completed']=True
    (ROOT/'input_preflight129.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2),flush=True)


if __name__=='__main__':main()
