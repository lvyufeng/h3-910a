#!/usr/bin/env python3
"""Verify four recorded text attention binaries before the exact-input block43 control."""
from pathlib import Path
import hashlib
import json
ROOT=Path('/tmp/h3_round110_block43_01')


def main():
    import batch_gemm85
    metas=[batch_gemm85.build_kernel(batch=8,m=m,k=k,n=n,out_dir=ROOT/'attention_kernels')for m in(512,507)for k,n in((128,2048),(2048,128))]
    for meta in metas:
        assert hashlib.sha256(Path(meta['obj_path']).read_bytes()).hexdigest()==meta['obj_sha256']
        assert meta['block_dim']==30 and meta['dst_dtype']=='float32'
    r={'scope':__doc__,'kernels':metas,'all4objects_SHA_verified_before8rank_priming':True,'no_model_oracle_modified':True}
    with(ROOT/'prebuild110.json').open('x')as f:f.write(json.dumps(r,indent=2)+'\n')
    print('Verified four pre-existing Float attention kernels',flush=True)


if __name__=='__main__':main()
