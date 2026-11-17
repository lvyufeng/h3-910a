#!/usr/bin/env python3
"""Direct TIK FP16 operands -> FP32 L0C -> GM;no generic DSL scheduler.

Microgeometry only. Operands use the same (K1,M,16)/(K1,N,16) layouts
as the executed flash-attention matmul helper. Output is (N1,M,16),
decoded only after measurement. Bias,if used,is the exact original
Half value promoted to FP32 in L1 before the single final Half cast.
"""
import hashlib
import json
import os
from pathlib import Path
import tempfile
import gemm_fp16_fp32out82 as contracts

MICRO_M,MICRO_K,MICRO_N=128,512,128
VARIANTS=contracts.VARIANTS
capability_report=contracts.capability_report


def build_kernel(variant='fp32_out',m=128,k=512,n=128,out_dir='kernels'):
    if (m,k,n)!=(128,512,128)or variant not in VARIANTS:
        raise ValueError('TIKmicro geometry fixed to128x512x128')
    from tbe import tik
    contracts.set_compile_soc()
    spec=VARIANTS[variant];dtype=spec['dst_dtype'];use_bias=spec['use_bias']
    name='tik_gemm82_legacy5_'+variant+'_m128_k512_n128'
    t=tik.Tik(disable_debug=True)
    ag=t.Tensor('float16',(k//16,m,16),name='a_gm',scope=tik.scope_gm)
    bg=t.Tensor('float16',(k//16,n,16),name='b_gm',scope=tik.scope_gm)
    cg=t.Tensor(dtype,(n//16,m,16),name='c_gm',scope=tik.scope_gm)
    biasg=t.Tensor('float32',(n,),name='bias_gm',scope=tik.scope_gm)if use_bias else None
    aux=[t.Tensor('float16',(16,),name='aux_gm%d'%i,scope=tik.scope_gm)for i in range(1 if use_bias else 2)]
    with t.for_range(0,1,block_num=1)as core:
        al=t.Tensor('float16',(k//16,m,16),name='a_l1',scope=tik.scope_cbuf)
        bl=t.Tensor('float16',(k//16,n,16),name='b_l1',scope=tik.scope_cbuf)
        c0=t.Tensor('float32',(n//16,m,16),name='c_l0c',scope=tik.scope_cc)
        cu=t.Tensor(dtype,(n//16,m,16),name='c_ub',scope=tik.scope_ubuf)
        t.data_move(al,ag,0,1,m*k*2//32,0,0)
        t.data_move(bl,bg,0,1,n*k*2//32,0,0)
        biasl=None
        if use_bias:
            biasl=t.Tensor('float32',(n,),name='bias_l1',scope=tik.scope_cbuf)
            t.data_move(biasl,biasg,0,1,n*4//32,0,0)
        t.matmul(c0,al,bl,m,k,n,init_l1out=True,bias=biasl)
        # The source burst is FP32 L0C bytes. When cu is Half,this is the
        # one final conversion;the unbiased variant retains allFP32 bits.
        t.tensor_mov(cu,c0,'m',1,m*n*4//1024,0,0)
        t.data_move(cg,cu,0,1,m*n*(4 if dtype=='float32'else 2)//32,0,0)
    root=Path(out_dir).resolve();root.mkdir(parents=True,exist_ok=True)
    work=Path(tempfile.mkdtemp(prefix='tik_gemm82_build_',dir=root)).resolve()
    assert work.parent==root and work.name.startswith('tik_gemm82_build_')
    previous=os.getcwd()
    try:
        os.chdir(work)
        t.BuildCCE(kernel_name=name,inputs=[ag,bg]+([biasg]if use_bias else[])+aux,outputs=[cg])
    finally:os.chdir(previous)
    obj=work/'kernel_meta'/(name+'.o');meta=work/'kernel_meta'/(name+'.json')
    if not obj.is_file()or not meta.is_file():raise RuntimeError('NoTIKkernelartifacts')
    data=json.loads(meta.read_text());assert data['blockDim']==1
    return {'variant':variant,'dst_dtype':dtype,'uses_fp32_bias':use_bias,
            'geometry':{'m':m,'k':k,'n':n},'compile_soc':'Ascend910B',
            'kernel_name':name,'kernel_symbol':data['kernelName'],'block_dim':1,
            'obj_path':str(obj),'obj_sha256':hashlib.sha256(obj.read_bytes()).hexdigest(),
            'json_path':str(meta),'core_type':data.get('coreType'),
            'input_layout':'K1_M_16','weight_layout':'K1_N_16','output_layout':'N1_M_16',
            'bias_dtype':'float32'if use_bias else None,'GM_argument_count':5,'unused_aux_inputs':len(aux),'built':True,'executed':False,
            'contract_source_sha256':hashlib.sha256(Path(contracts.__file__).read_bytes()).hexdigest()}
