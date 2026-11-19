#!/usr/bin/env python3
"""Serial18 existing kernels verified before originalHalf textTP8+VisionTP8 priming."""
from pathlib import Path
import json
import hashlib
ROOT=Path('/tmp/h3_round108_full01')


def main():
    import batch_gemm85
    import vision_bias87
    import erf_gelu_fixed90
    import flash_vision101
    text=[batch_gemm85.build_kernel(batch=8,m=m,k=k,n=n,out_dir=ROOT/'attention_kernels')for m in(512,507)for k,n in((128,2048),(2048,128))]
    vision=[vision_bias87.build_kernel(m=m,k=k,n=n,out_dir=ROOT/'bias_kernels')for m,k,n in((8064,1152,3456),(8064,1152,1152),(8064,1152,4352),(8064,4352,1152),(2016,4608,4608),(2016,4608,5120),(8064,1536,1152))]
    erf=erf_gelu_fixed90.build_kernel(numel=9289728,out_dir=ROOT/'erf_kernels')
    flash=flash_vision101.build_kernel(batch=16,m=4032,n=4032,d=72,out_dir=ROOT/'vision_attention_kernels')
    head2=flash_vision101.build_kernel(batch=2,m=4032,n=4032,d=72,out_dir=ROOT/'head2_kernels')
    import blocked_fp32_gemm83
    tp=[]
    for mode,k,n in(('column',1152,512),('row',256,1152),('column',1152,640),('row',640,1152)):
        kernel=vision_bias87 if mode=='column'else blocked_fp32_gemm83
        tp.append(kernel.build_kernel(m=8064,k=k,n=n,out_dir=ROOT/'tp_backend_kernels'/mode))
    for meta in text+vision+[erf,flash,head2]+tp:
        assert hashlib.sha256(Path(meta['obj_path']).read_bytes()).hexdigest()==meta['obj_sha256']
        assert meta['block_dim']==30
    receipt={'scope':__doc__,'all18objects_SHA_verified_before_8rank_model_priming':True,'local_head2_kernel':head2,'TP_GEMM_kernels':tp,'text_kernels':text,'vision_kernels':vision,'erf_kernel':erf,'flash_kernel':flash,'no_FP16_model_inputs_executed':True}
    with(ROOT/'prebuild108.json').open('x')as f:f.write(json.dumps(receipt,indent=2)+'\n')
    print('Verified eighteen existing binaries before full TP8 model loading',flush=True)


if __name__=='__main__':main()
