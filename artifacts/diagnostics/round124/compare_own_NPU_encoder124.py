#!/usr/bin/env python3
"""Compare full live NPU encoder output with the actual successful GPU activation.

Activation statistics only on CPU; this is not CPU H3 inference or a performance
baseline. Both tensors already exist on the owned NPU host, so no extra large
activation transfer or any model-weight transfer is required. No BF16 roundtrip.
"""
import hashlib
import json
from pathlib import Path

ROOT=Path('/tmp/h3_align12401')


def main():
    import torch
    torch.set_num_threads(4)
    out=ROOT/'own_NPU_encoder_vs_successful_GPU124.json';assert not out.exists()
    source=ROOT/'own_NPU_encoder_probe124b/own_NPU_encoder_hidden124.pt'
    receipt=json.loads((ROOT/'own_NPU_encoder_probe124b/encoder_rank0.json').read_text())
    assert receipt['completed'] and hashlib.sha256(source.read_bytes()).hexdigest()==receipt['capture_sha256']
    n=torch.load(source,map_location='cpu',weights_only=True)
    g=torch.load('/tmp/h3_align12301/same_initial_sampling123.pt',map_location='cpu',weights_only=True)
    assert n['hidden_states'].shape==g['text_embeddings'].shape==(2043,5120)
    assert torch.equal(n['token_tags'],g['text_tags'])
    def metric(a,b):
        assert a.dtype==b.dtype==torch.float16
        x=a.double();y=b.double();d=x-y
        return {'relative_RMS':float(d.square().sum().sqrt()/y.square().sum().sqrt().clamp_min(1e-30)),
                'RMSE':float(d.square().mean().sqrt()),'max_abs_error':float(d.abs().max()),
                'GPU_max_abs':float(y.abs().max()),'NPU_max_abs':float(x.abs().max()),
                'finite':bool(torch.isfinite(x).all() and torch.isfinite(y).all()),'rows':int(a.shape[0])}
    r={'scope':__doc__,'completed':True,'GPU_actual_successful_encoder_reference':True,'NPU_encoder_ran_live':True,
       'GPU_encoder_cache_used_to_compute_NPU_output':False,'CPU_model_inference':False,
       'full_tensor_shape':[2043,5120],'token_tags_identical':True,'all_rows':metric(n['hidden_states'],g['text_embeddings']),'by_token_tag':{},
       'full_video_quality_acceptance_pending':True}
    for tag in torch.unique(n['token_tags']):
        mask=n['token_tags']==tag;r['by_token_tag'][str(int(tag))]=metric(n['hidden_states'][mask],g['text_embeddings'][mask])
    out.write_text(json.dumps(r,indent=2)+'\n');print(json.dumps(r,indent=2),flush=True)


if __name__=='__main__':main()
