#!/usr/bin/env python3
"""Actual 910A exhaustive FP16 safety-predicate proof, no model inference."""
import hashlib
import json
from pathlib import Path


def main():
    import torch
    import torch_npu
    from equivalent_safety126 import half_safety_counts
    from npu_half_nan_compat124 import ieee_nonfinite_masks
    torch.npu.set_device(0)
    signed=torch.arange(-32768,32768,dtype=torch.int32).to(torch.int16)
    values=signed.view(torch.float16).to('npu:0')
    native=half_safety_counts(values,False);candidate=half_safety_counts(values,True)
    old_nan,old_posinf,old_neginf=ieee_nonfinite_masks(values)
    words=values.view(torch.int16).to(torch.int32)&0x7fff
    new_bad=words>=0x7c00;new_boundary=words==0x7bff
    expected_bad=((signed.to(torch.int32)&0x7fff)>=0x7c00)
    expected_boundary=((signed.to(torch.int32)&0x7fff)==0x7bff)
    rows={'native_counts':[int(t) for t in native],'candidate_counts':[int(t) for t in candidate],
          'bad_per_word_matches_original':bool(torch.equal(new_bad.cpu(),(old_nan|old_posinf|old_neginf).cpu())),
          'bad_per_word_matches_independent_integer_spec':bool(torch.equal(new_bad.cpu(),expected_bad)),
          'boundary_per_word_matches_independent_integer_spec':bool(torch.equal(new_boundary.cpu(),expected_boundary))}
    assert rows['native_counts']==rows['candidate_counts']==[2048,2]
    assert all(rows[k] for k in rows if k.endswith('spec') or k.endswith('original'))
    result={'scope':__doc__,'completed':True,'model_weights_loaded':False,'CPU_model_inference':False,
            'every_65536_Half_bit_pattern_tested_on_NPU':True,'results':rows,'verification_passed':True,
            'source_sha256':hashlib.sha256(Path('/tmp/h3_perf12601/equivalent_safety126.py').read_bytes()).hexdigest()}
    out=Path('/tmp/h3_perf12601/equivalent_safety_unit126.json');assert not out.exists()
    out.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2),flush=True)

if __name__=='__main__':main()
