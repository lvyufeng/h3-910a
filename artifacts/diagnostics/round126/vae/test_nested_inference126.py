#!/usr/bin/env python3
"""NPU regression for cache prepare nested inside generation inference_mode.

Synthetic tiny NPU Linear modules only: no checkpoint weight access and no CPU
forward/model inference. Checks private-copy version counters and exact native
outputs while every original FP32 parameter stays untouched.
"""
import hashlib
import json
from pathlib import Path


def main():
    import torch
    import torch_npu
    from autocast_operand_cache126 import AutocastOperandCache126
    torch.npu.set_device(0);torch.npu.matmul.allow_hf32=False;torch.npu.conv.allow_hf32=False
    torch.manual_seed(126)
    decoder=torch.nn.Module()
    decoder.transformer_blocks=torch.nn.ModuleList([
        torch.nn.ModuleDict({name:torch.nn.Linear(16,16,bias=True,dtype=torch.float32,device='npu:0')
                             for name in ('to_qkv','to_out','w1','w2')}) for _ in range(36)])
    assert not any(torch.is_inference(p) for p in decoder.parameters())
    with torch.inference_mode():
        cache=AutocastOperandCache126(decoder,budget_bytes=1024**2)
        cache.prepare()
        assert torch.is_inference_mode_enabled()
        assert all(not torch.is_inference(row['half_weight']) and row['half_weight']._version>=0 for row in cache.rows)
        cache.begin_proof()
        with torch.autocast('npu',dtype=torch.float16):
            x=torch.full((17,16),0.1,dtype=torch.float32,device='npu:0')
            for row in cache.rows:row['module'](x)
        torch.npu.synchronize();proof=cache.end_proof()
        assert proof['leaf_comparisons']==144 and proof['all_same_input_leaf_words_equal'] and proof['all_leaf_outputs_finite']
        assert cache.snapshot()['original_parameters_unchanged']
        result={'scope':__doc__,'completed':True,'CPU_model_inference':False,'checkpoint_weights_accessed':False,
                'outer_inference_mode_stayed_enabled':True,'private_copies_track_version':True,
                'same_input_Linear_comparisons':proof['leaf_comparisons'],'all_words_equal':True,
                'all_original_FP32_parameters_unchanged':True,
                'cache_source_sha256':hashlib.sha256(Path('/tmp/h3_perf12601/vae/autocast_operand_cache126.py').read_bytes()).hexdigest()}
        cache.close()
    path=Path('/tmp/h3_perf12601/cache_nested_inference_unit126.json');assert not path.exists()
    path.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2),flush=True)

if __name__=='__main__':main()
