#!/usr/bin/env python3
"""Serial small/full Vision TP8 GEMM prebuild before any HCCL worker/model priming."""
from pathlib import Path
import json
import os
ROOT=Path(os.environ.get('R107_ROOT','/tmp/h3_round107_gemm_tp01'))


def main():
    import vision_bias87 as col
    import blocked_fp32_gemm83 as row
    r={'scope':__doc__,'completed':False,'kernels':[]}
    try:
        for m in(128,8064):
            for mode,k,n in(('column',1152,512),('row',256,1152),('column',1152,640),('row',640,1152)):
                kernel=col if mode=='column'else row
                meta=kernel.build_kernel(m=m,k=k,n=n,out_dir=ROOT/'tp_backend_kernels'/mode)
                assert meta['dst_dtype']==('float16'if mode=='column'else'float32')
                assert meta['GM_argument_count']==5 and 0<meta['block_dim']<=30
                r['kernels'].append({'mode':mode,'m':m,'k':k,'n':n,'meta':meta})
        r['completed']=True
    except BaseException as error:r['failure']={'type':type(error).__name__,'message':str(error)};raise
    finally:
        with(ROOT/'prebuild107.json').open('x')as f:f.write(json.dumps(r,indent=2)+'\n')
        print(json.dumps({'completed':r['completed'],'kernel_count':len(r['kernels']),'failure':r.get('failure')}),flush=True)


if __name__=='__main__':main()
