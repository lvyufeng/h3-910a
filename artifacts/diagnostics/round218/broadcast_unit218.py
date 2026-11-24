#!/usr/bin/env python3
"""Small TIK unit for exact FP16 broadcast alias safety, before any FA execution."""
import ctypes
import hashlib
import json
import os
from pathlib import Path
import numpy as np
import torch
import torch_npu
from tbe import tik
import tbe.common.platform as plat
from flash_attention.ops.flash_attention.tik_ops_utils import TikOpsUtils

root=Path('/tmp/h3_broadcast218')
root.mkdir(mode=0o700,exist_ok=True)
label=os.environ['H3_BROADCAST_LABEL']
out=root/label;out.mkdir(mode=0o700,exist_ok=False)
torch.npu.set_device(0)
plat.set_current_compile_soc_info('Ascend910A')
M=128
instance=tik.Tik(disable_debug=True)
gm_in=instance.Tensor('float16',(M,),name='input_gm',scope=tik.scope_gm)
gm_out=instance.Tensor('float16',(M,16),name='output_gm',scope=tik.scope_gm)
# Match the known five-GM-pointer native launcher ABI; three arguments are unused.
dummy1=instance.Tensor('float16',(16,),name='unused1',scope=tik.scope_gm)
dummy2=instance.Tensor('float16',(16,),name='unused2',scope=tik.scope_gm)
dummy3=instance.Tensor('float16',(16,),name='unused3',scope=tik.scope_gm)
with instance.for_range(0,1,block_num=1):
    vec=instance.Tensor('float16',(M,),name='vec',scope=tik.scope_ubuf)
    instance.data_move(vec,gm_in,0,1,M//16,0,0)
    rows=TikOpsUtils(instance).broadcast_vec_from_M_to_MN0(vec)
    instance.data_move(gm_out,rows,0,1,M,0,0)
old=os.getcwd();os.chdir(out)
key='broadcast218_'+label
try:instance.BuildCCE(kernel_name=key,inputs=[gm_in,dummy1,dummy2,dummy3],outputs=[gm_out])
finally:os.chdir(old)
obj=out/'kernel_meta'/(key+'.o')
meta=json.loads(obj.with_suffix('.json').read_text())
lib=ctypes.CDLL('/tmp/libfa910a.so')
lib.fa910a_init.argtypes=[ctypes.c_int];lib.fa910a_init.restype=ctypes.c_int
lib.fa910a_register.argtypes=[ctypes.c_char_p,ctypes.c_char_p];lib.fa910a_register.restype=ctypes.c_void_p
lib.fa910a_launch.argtypes=[ctypes.c_void_p,ctypes.c_char_p,ctypes.c_uint32]+[ctypes.c_void_p]*6
lib.fa910a_launch.restype=ctypes.c_int
assert lib.fa910a_init(0)==0
handle=lib.fa910a_register(str(obj).encode(),(key+'__kernel0').encode())
assert handle
# Check all bit-copy results on representative finite values, signs, subnormals
# and NaN payloads. No arithmetic should alter these words.
words=np.array([0,0x8000,0x0001,0x03ff,0x0400,0x3c00,0x7bff,0xfbff,
                0x7c00,0xfc00,0x7e00,0x7d01,0x3c01,0xbc01,0x3555,0xb555]*8,dtype=np.uint16)
source=torch.from_numpy(words.view(np.float16).copy()).to('npu:0')
output=torch.empty((M,16),dtype=torch.float16,device='npu:0')
dummies=[torch.empty((16,),dtype=torch.float16,device='npu:0') for _ in range(3)]
stream=ctypes.c_void_p(torch.npu.current_stream().npu_stream)
torch.npu.synchronize()
for _ in range(3):
    rc=lib.fa910a_launch(handle,(key+'__kernel0').encode(),meta['blockDim'],
        source.data_ptr(),dummies[0].data_ptr(),dummies[1].data_ptr(),dummies[2].data_ptr(),output.data_ptr(),stream)
    assert rc==0,rc
    torch.npu.synchronize()
actual=output.cpu().contiguous().view(torch.int16).numpy().view(np.uint16)
expected=np.repeat(words[:,None],16,axis=1)
different=int((actual!=expected).sum())
result={'compiled':True,'label':label,'object_sha256':hashlib.sha256(obj.read_bytes()).hexdigest(),
        'blockDim':meta['blockDim'],'actual_launch_count':3,'total_words':int(actual.size),
        'different_words_vs_expected_broadcast':different,'pass':different==0,
        'target':'FP16 bit-copy broadcast, including signed zero/subnormal/NaN payloads'}
(out/'result.json').write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps(result))
raise SystemExit(0 if result['pass'] else 2)
