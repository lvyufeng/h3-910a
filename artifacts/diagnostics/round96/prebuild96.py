#!/usr/bin/env python3
"""Serial original twelve-object verification before actual correctedvision A/B."""
from pathlib import Path
import json
ROOT=Path('/tmp/h3_round96_vision01')


def main():
    import batch_gemm85
    import vision_bias87
    import erf_gelu_fixed90
    records=[batch_gemm85.build_kernel(batch=16,m=m,k=k,n=n,out_dir=ROOT/'attention_kernels')
             for m in(512,448)for k,n in((128,4096),(4096,128))]
    records += [vision_bias87.build_kernel(m=m,k=k,n=n,out_dir=ROOT/'bias_kernels')
                for m,k,n in((8064,1152,3456),(8064,1152,1152),(8064,1152,4352),(8064,4352,1152),
                            (2016,4608,4608),(2016,4608,5120),(8064,1536,1152))]
    records.append(erf_gelu_fixed90.build_kernel(numel=9289728,out_dir=ROOT/'erf_kernels'))
    import hashlib
    for meta in records:
        assert hashlib.sha256(Path(meta['obj_path']).read_bytes()).hexdigest()==meta['obj_sha256']
        assert meta['block_dim']==30
    with(ROOT/'prebuild96.json').open('x')as f:f.write(json.dumps({'scope':__doc__,'records':records,'all12objSHAsverified':True},indent=2)+'\n')
    print('Verified4originalvisionattention+7currentvisionbias+1erfkernelobjectsserially')


if __name__=='__main__':main()
