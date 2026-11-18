#!/usr/bin/env python3
"""Serial cache/object verification before eight-rank model priming."""
from pathlib import Path
import json
import hashlib
ROOT=Path('/tmp/h3_round91_full01')


def main():
    import batch_gemm85
    import vision_bias87
    import erf_gelu_fixed90
    expected={'batch_gemm85.py':'546cd211ddb3938c3d8dac9d5207a79646c1458a3887df7e8c8562426069854a',
              'vision_bias87.py':'f8655b5fa35aecec0753d3e657f16fd2e235f402a57e441eb8959f4c057f0379',
              'erf_gelu_fixed90.py':'5a5e85aaad5697a7a890ecb0b5b724bc3acd93d819978ab55d5eb68038cd0552'}
    for name,sha in expected.items():assert hashlib.sha256((ROOT/name).read_bytes()).hexdigest()==sha
    text=[batch_gemm85.build_kernel(batch=8,m=m,k=k,n=n,out_dir=ROOT/'attention_kernels')for m in(512,507)for k,n in((128,2048),(2048,128))]
    vision=[vision_bias87.build_kernel(m=m,k=k,n=n,out_dir=ROOT/'bias_kernels')for m,k,n in((8064,1152,3456),(8064,1152,1152),
            (8064,1152,4352),(8064,4352,1152),(2016,4608,4608),(2016,4608,5120),(8064,1536,1152))]
    erf=erf_gelu_fixed90.build_kernel(numel=9289728,out_dir=ROOT/'erf_kernels')
    for meta in text+vision+[erf]:
        assert hashlib.sha256(Path(meta['obj_path']).read_bytes()).hexdigest()==meta['obj_sha256']
        assert meta['block_dim']==30
    receipt={'scope':__doc__,'serial_setup_before_any8rank_model_load':True,'text_kernels':text,'vision_kernels':vision,
             'erf_kernel':erf,'all_original_source_and_object_SHA_verified':True,'no_FP16_model_input_executed':True}
    p=ROOT/'prebuild91.json';assert not p.exists()
    with p.open('x')as f:f.write(json.dumps(receipt,indent=2)+'\n')
    print('Verified4text+7vision+1erfkernelobjectsseriallybeforeTP8modelpriming')


if __name__=='__main__':main()
