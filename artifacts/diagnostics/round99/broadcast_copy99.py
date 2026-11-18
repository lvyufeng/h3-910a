#!/usr/bin/env python3
"""Native original-910 bit-copy Half-row transpose broadcast to exact Float, operator probe only."""
from pathlib import Path
import json
import os
import hashlib
ROOT=Path('/tmp/h3_round99_broadcast01')


def main():
    import torch
    import torch_npu
    from tbe import tik
    import tbe.common.platform as platform
    from runtime5gm83 import Runtime5GM
    import best_text87 as previous
    torch.set_num_threads(2);torch.npu.set_device(0);platform.set_current_compile_soc_info('Ascend910')
    r={'scope':__doc__,'completed':False,'source_SHA256':previous.file_sha(Path(__file__)),'no_model_precision_or_speed_claim':True}
    try:
        assert platform.intrinsic_check_support('Intrinsic_vor','uint16')
        assert platform.intrinsic_check_support('Intrinsic_vtranspose','float16')
        t=tik.Tik(disable_debug=True)
        src=t.Tensor('float32',(64,),name='src',scope=tik.scope_gm)
        aux=[t.Tensor('float32',(16,),name='aux%d'%i,scope=tik.scope_gm)for i in range(3)]
        out=t.Tensor('float32',(4,64,16),name='out',scope=tik.scope_gm)
        with t.for_range(0,1,block_num=1)as core:
            rowf=t.Tensor('float32',(64,),name='rowf',scope=tik.scope_ubuf)
            rowh=t.Tensor('float16',(64,),name='rowh',scope=tik.scope_ubuf)
            scratch=t.Tensor('float16',(256,),name='copy_matrix',scope=tik.scope_ubuf)
            broad=t.Tensor('float16',(1024,),name='broadcast_half',scope=tik.scope_ubuf)
            broadf=t.Tensor('float32',(1024,),name='broadcast_float',scope=tik.scope_ubuf)
            t.data_move(rowf,src,0,1,8,0,0);t.h_cast(rowh,rowf,'none')
            words=rowh.reinterpret_cast_to('uint16');matrix=scratch.reinterpret_cast_to('uint16')
            for block in range(4):
                t.vec_or(16,matrix,words[block*16],words[block*16],16,1,0,0)
                t.vec_trans(broad[block*256],scratch,1,0,0)
            t.h_cast(broadf,broad,'none')
            for group in range(4):t.data_move(out[group*1024],broadf,0,1,128,0,0)
        name='half_row_transpose_copy99';work=ROOT/'compile';work.mkdir();before=os.getcwd()
        try:os.chdir(work);t.BuildCCE(kernel_name=name,inputs=[src]+aux,outputs=[out])
        finally:os.chdir(before)
        obj=work/'kernel_meta'/f'{name}.o';raw=json.loads((work/'kernel_meta'/f'{name}.json').read_text())
        meta={'obj_path':str(obj),'obj_sha256':hashlib.sha256(obj.read_bytes()).hexdigest(),'kernel_symbol':raw['kernelName'],'block_dim':int(raw['blockDim'])};r['meta']=meta
        row=torch.arange(-32,32,dtype=torch.float16);row[32]=torch.tensor(-0.0,dtype=torch.float16)
        row=row.float();source=row.to('npu:0');dst=torch.empty((4,64,16),dtype=torch.float32,device='npu:0')
        dummy=torch.empty((16,),dtype=torch.float32,device='npu:0');runtime=Runtime5GM(0)
        runtime.launch(meta,[source,dummy,dummy,dummy,dst])
        gold=row.reshape(1,64,1).expand(4,64,16).contiguous()
        r['exact_Float_copy']=previous.compare(dst,gold);assert r['exact_Float_copy']['raw_bits_equal']
        runtime.launch(meta,[source,dummy,dummy,dummy,dst]);assert previous.compare(dst,gold)['raw_bits_equal']
        r['completed']=True
    except BaseException as e:r['failure']={'type':type(e).__name__,'message':str(e)};raise
    finally:
        with(ROOT/'broadcast_copy99.json').open('x')as f:f.write(json.dumps(r,indent=2)+'\n')
        print(json.dumps({'completed':r['completed'],'exact_copy':r.get('exact_Float_copy'),'failure':r.get('failure')},indent=2),flush=True)


if __name__=='__main__':main()
