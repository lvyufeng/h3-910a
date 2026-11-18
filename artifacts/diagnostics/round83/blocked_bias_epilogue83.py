#!/usr/bin/env python3
"""Blocked Half GEMM with portable FP32-UB bias and one final Half cast.

Derived from the verified R83 blocked FP32 accumulation and executed R83 bias
micro-kernel. This source is not an executed result. The original Half bias
value is exactly promoted in the host;weights and inputs remain Half. Padded
M rows return the bias,not zero,so the host must crop them before model use.
"""
import hashlib
import json
import os
from pathlib import Path
import tempfile
import blocked_fp32_gemm83 as blocked


BASE_SHA256='78896adadaf47801dd92d69bdfce394d2628f7885be643dd568647c8b4c8c2ee'


def build_kernel(m,k,n,out_dir='kernels',variant='fp32_bias_fp16_out'):
    if variant!='fp32_bias_fp16_out':raise ValueError('Only the one-Half-cast bias output is supported')
    if hashlib.sha256(Path(blocked.__file__).read_bytes()).hexdigest()!=BASE_SHA256:
        raise RuntimeError('Reviewed blocked GEMM source changed')
    plan=blocked.plan_tiling(m,k,n);capability=blocked.capability_report()
    from tbe import tik
    mp=plan['padded']['m_pad'];mb=nb=kb=128;k1=k//16;kb1=8;n1=n//16
    grid=plan['grid'];cores=grid['cores'];steps=grid['steps'];nt=grid['n_tiles'];tasks=grid['tasks']
    name=f'blocked_bias83_m{m}_k{k}_n{n}_c{cores}'
    t=tik.Tik(disable_debug=True)
    ag=t.Tensor('float16',(k1,mp,16),name='a_gm',scope=tik.scope_gm)
    bg=t.Tensor('float16',(k1,n,16),name='b_gm',scope=tik.scope_gm)
    biasg=t.Tensor('float32',(n,),name='bias_gm',scope=tik.scope_gm)
    aux=t.Tensor('float16',(16,),name='aux_gm',scope=tik.scope_gm)
    cg=t.Tensor('float16',(n1,mp,16),name='c_gm',scope=tik.scope_gm)
    with t.for_range(0,cores,block_num=cores)as core:
        al=t.Tensor('float16',(kb1,mb,16),name='a_l1',scope=tik.scope_cbuf)
        bl=t.Tensor('float16',(kb1,nb,16),name='b_l1',scope=tik.scope_cbuf)
        c0=t.Tensor('float32',(nb//16,mb,16),name='c_l0c',scope=tik.scope_cc)
        c32=t.Tensor('float32',(nb//16,mb,16),name='c32_ub',scope=tik.scope_ubuf)
        c16=t.Tensor('float16',(nb//16,mb,16),name='c16_ub',scope=tik.scope_ubuf)
        biasub=t.Tensor('float32',(nb,),name='bias_ub',scope=tik.scope_ubuf)
        with t.for_range(0,steps)as step:
            task=core+step*cores
            with t.if_scope(task<tasks):
                base_m=(task//nt)*mb;base_n=(task%nt)*nb
                for kt in range(k//kb):
                    with t.new_stmt_scope(disable_sync=False):
                        t.data_move(al,ag[(kt*kb1*mp+base_m)*16],0,kb1,mb,mp-mb,0)
                        t.data_move(bl,bg[(kt*kb1*n+base_n)*16],0,kb1,nb,n-nb,0)
                        t.matmul(c0,al,bl,mb,kb,nb,init_l1out=(kt==0),bias=None)
                with t.new_stmt_scope(disable_sync=False):
                    t.tensor_mov(c32,c0,'m',1,mb*nb*4//1024,0,0)
                    t.data_move(biasub,biasg[base_n],0,1,nb*4//32,0,0)
                    for group in range(nb//16):
                        t.vec_add(16,c32[group,:,:],c32[group,:,:],biasub[group*16],mb,2,2,0)
                    t.h_cast(c16,c32,'none')
                    # Half output row16 is32B,not the64B FP32 output unit.
                    t.data_move(cg[((base_n//16)*mp+base_m)*16],c16,0,nb//16,mb,0,mp-mb)
    root=Path(out_dir).resolve();root.mkdir(parents=True,exist_ok=True)
    work=Path(tempfile.mkdtemp(prefix='blocked_bias83_build_',dir=root)).resolve()
    if work.parent!=root:raise RuntimeError('Unexpected build workspace')
    previous=os.getcwd()
    try:
        os.chdir(work);t.BuildCCE(kernel_name=name,inputs=[ag,bg,biasg,aux],outputs=[cg])
    finally:os.chdir(previous)
    obj=work/'kernel_meta'/(name+'.o');meta=work/'kernel_meta'/(name+'.json')
    if not obj.is_file()or not meta.is_file():raise RuntimeError('No blocked bias kernel artefacts')
    info=json.loads(meta.read_text());assert int(info['blockDim'])==cores
    return {'kernel_name':name,'kernel_symbol':info['kernelName'],'block_dim':cores,
            'obj_path':str(obj),'obj_sha256':hashlib.sha256(obj.read_bytes()).hexdigest(),'json_path':str(meta),
            'dst_dtype':'float16','uses_fp32_bias':True,'single_final_half_cast':True,
            'input_layout':'K1_Mp_16','weight_layout':'K1_N_16','output_layout':'N1_Mp_16',
            'logical_geometry':{'M':m,'K':k,'N':n},'padded_M':mp,'grid':grid,
            'FP32_L0C_preserved_across_K_tiles':True,'matmul_bias_operand_used':False,
            'padded_output_rows_are_bias_not_zero':True,'GM_argument_count':5,
            'base_source_sha256':BASE_SHA256,'capability':capability,'built':True,'executed':False,
            'full_encoder_precision_accepted':False,'whole_goal_complete':False}
