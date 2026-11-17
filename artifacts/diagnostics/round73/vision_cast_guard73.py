"""Exact BF16 strided vision attention input cast with native contiguous BNSD output.

Read full16x72 BF16 token rows (aligned), integer-decode FP32 in UB, then write
72-float rows per head. This fuses cast/physical transpose without FP arithmetic,
head padding, QK/softmax/PV changes or copying input through contiguous().
"""
import ctypes
import fcntl
import hashlib
import json
import os
from pathlib import Path
import shutil
import sys
import tempfile
import torch
import torch_npu
sys.path.insert(0,'/tmp/h3_round58_vec01')
from native_bf16_vector58 import unpack_bf16
from fused_vector58 import library,core_budget

ROOT=Path('/tmp/h3_round73_cast02');KERNELS=ROOT/'kernels'
_HASH=hashlib.sha256(Path(__file__).read_bytes()+Path('/tmp/h3_round58_vec01/native_bf16_vector58.py').read_bytes()).hexdigest()[:12]
_CACHE={};_HANDLES={}


def build(tokens,pitch,cores):
    from tbe import tik
    import tbe.common.platform as platform
    platform.set_current_compile_soc_info('Ascend910A')
    width,padded=1152,1280
    name=f'visioncast73_n{tokens}_p{pitch}_c{cores}_{_HASH}'
    obj=KERNELS/name/(name+'.o')
    if obj.is_file():
        assert json.loads(obj.with_suffix('.json').read_text())['blockDim']==cores
        return str(obj),name+'__kernel0',cores
    t=tik.Tik(disable_debug=True)
    length=(tokens-1)*pitch+width
    xgm=t.Tensor('uint16',(length,),name='xgm',scope=tik.scope_gm)
    agm=t.Tensor('uint16',(length,),name='agm',scope=tik.scope_gm)
    bgm=t.Tensor('uint16',(length,),name='bgm',scope=tik.scope_gm)
    cgm=t.Tensor('uint16',(length,),name='cgm',scope=tik.scope_gm)
    ygm=t.Tensor('float32',(16,tokens,72),name='ygm',scope=tik.scope_gm)
    with t.for_range(0,cores,block_num=cores)as core:
        words=t.Tensor('uint16',(padded,),name='words',scope=tik.scope_ubuf)
        transposed=t.Tensor('uint16',(padded,),name='transposed',scope=tik.scope_ubuf)
        zeros=t.Tensor('uint16',(padded,),name='zeros',scope=tik.scope_ubuf)
        values=t.Tensor('float32',(padded,),name='values',scope=tik.scope_ubuf)
        t.vector_dup(128,zeros,0,padded//128,1,8)
        with t.for_range(0,(tokens+cores-1)//cores)as step:
            row=core+step*cores
            with t.if_scope(row<tokens):
                t.vector_dup(128,words,0,padded//128,1,8)
                t.data_move(words,xgm[row*pitch],0,1,width//16,0,0)
                unpack_bf16(t,words,values,transposed,zeros,padded)
                for head in range(16):
                    t.data_move(ygm[head,row,0],values[head*72],0,1,9,0,0)
    work=Path(tempfile.mkdtemp(prefix='visioncast73_build_',dir=ROOT)).resolve(strict=True)
    if work.parent!=ROOT.resolve()or not work.name.startswith('visioncast73_build_'):raise RuntimeError('Unexpected cast build dir')
    old=os.getcwd()
    try:
        os.chdir(work);t.BuildCCE(kernel_name=name,inputs=[xgm,agm,bgm,cgm],outputs=[ygm])
        src=work/'kernel_meta'/(name+'.o');assert src.is_file()
        obj.parent.mkdir(exist_ok=False)
        shutil.copyfile(src.with_suffix('.json'),obj.with_suffix('.json'));shutil.copyfile(src,obj)
    finally:
        os.chdir(old)
        if work.resolve(strict=True)!=work or work.parent!=ROOT.resolve()or not work.name.startswith('visioncast73_build_'):raise RuntimeError('Unexpected cast cleanup target')
        shutil.rmtree(work)
    assert json.loads(obj.with_suffix('.json').read_text())['blockDim']==cores
    return str(obj),name+'__kernel0',cores


def cast(value):
    if value.ndim!=4 or value.shape[0]!=1 or value.shape[1]!=16 or value.shape[-1]!=72 or not 0<value.shape[2]<=16384:raise ValueError('Measured vision cast requires[1,16,N,72]')
    n=value.shape[2];pitch=value.stride(2)
    if (value.dtype!=torch.bfloat16 or value.device.type not in('npu','privateuseone')or value.device.index!=torch.npu.current_device()
        or pitch not in(1152,3456)or value.stride(1)!=72 or value.stride(3)!=1
        or value.data_ptr()%32 or torch_npu.get_npu_format(value)not in(0,2)
        or value.storage_offset()*2+((n-1)*pitch+1152)*2>value.untyped_storage().nbytes()):
        raise ValueError('Vision cast requires current-device alignedND strided token rows with16adjacent72wideheads')
    if torch.is_grad_enabled()and value.requires_grad:raise ValueError('Inference-only strided vision cast')
    cores=min(n,core_budget());key=(n,pitch,cores)
    if key not in _CACHE:
        KERNELS.mkdir(exist_ok=True)
        with(KERNELS/'.compile.lock').open('a')as lock:
            fcntl.flock(lock,fcntl.LOCK_EX);_CACHE[key]=build(n,pitch,cores)
    obj,symbol,cores=_CACHE[key]
    if obj not in _HANDLES:
        h=library().fa910a_register(obj.encode(),symbol.encode())
        if not h:raise RuntimeError('Vision cast registration failed')
        _HANDLES[obj]=h
    output=(torch.empty_strided(value.shape,value.stride(),dtype=torch.float32,device=value.device) if n==1
            else torch.empty((1,16,n,72),dtype=torch.float32,device=value.device))
    if not output.is_contiguous()or output.data_ptr()%32 or torch_npu.get_npu_format(output)not in(0,2):raise RuntimeError('Invalid native contiguousFP32 output')
    p=value.data_ptr()
    rc=library().fa910a_launch(_HANDLES[obj],symbol.encode(),cores,p,p,p,p,output.data_ptr(),ctypes.c_void_p(torch.npu.current_stream().npu_stream))
    if rc:raise RuntimeError('Vision cast launch failed:%d'%rc)
    return output
