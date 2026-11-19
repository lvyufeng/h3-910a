#!/usr/bin/env python3
"""Serial 13 existing kernel identity verification before original Half TP8 priming."""
from pathlib import Path
import json
import hashlib
ROOT=Path('/tmp/h3_round102_full01')


def main():
    import batch_gemm85
    import vision_bias87
    import erf_gelu_fixed90
    import flash_vision101
    text=[batch_gemm85.build_kernel(batch=8,m=m,k=k,n=n,out_dir=ROOT/'attention_kernels')for m in(512,507)for k,n in((128,2048),(2048,128))]
    vision=[vision_bias87.build_kernel(m=m,k=k,n=n,out_dir=ROOT/'bias_kernels')for m,k,n in((8064,1152,3456),(8064,1152,1152),(8064,1152,4352),(8064,4352,1152),(2016,4608,4608),(2016,4608,5120),(8064,1536,1152))]
    erf=erf_gelu_fixed90.build_kernel(numel=9289728,out_dir=ROOT/'erf_kernels')
    flash=flash_vision101.build_kernel(batch=16,m=4032,n=4032,d=72,out_dir=ROOT/'vision_attention_kernels')
    for meta in text+vision+[erf,flash]:
        assert hashlib.sha256(Path(meta['obj_path']).read_bytes()).hexdigest()==meta['obj_sha256']
        assert meta['block_dim']==30
    receipt={'scope':__doc__,'all13objects_SHA_verified_before_8rank_model_priming':True,'text_kernels':text,'vision_kernels':vision,'erf_kernel':erf,'flash_kernel':flash,'no_FP16_model_inputs_executed':True}
    with(ROOT/'prebuild102.json').open('x')as f:f.write(json.dumps(receipt,indent=2)+'\n')
    print('Verified thirteen existing binaries before full TP8 model loading',flush=True)


if __name__=='__main__':main()
