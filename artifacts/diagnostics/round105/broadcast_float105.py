#!/usr/bin/env python3
"""Original910 exact Float bits broadcast; no Half arithmetic rounding or model benchmark."""
from pathlib import Path
import hashlib
import json
import os
ROOT=Path('/tmp/h3_round105_broadcast01')


def main():
    import torch
    import torch_npu
    from tbe import tik
    import tbe.common.platform as platform
    from runtime5gm83 import Runtime5GM
    import best_text87 as previous
    from float_row_bits105 import broadcast_float_row_bits
    torch.set_num_threads(2);torch.npu.set_device(0);platform.set_current_compile_soc_info('Ascend910')
    r={'scope':__doc__,'completed':False,'source_SHA256':previous.file_sha(Path(__file__)),
       'helper_SHA256':previous.file_sha(Path(__file__).with_name('float_row_bits105.py')),
       'no_model_precision_or_speed_claim':True,'cases':{}}
    try:
        assert platform.intrinsic_check_support('Intrinsic_vor','uint16')
        assert platform.intrinsic_check_support('Intrinsic_vtranspose','float16')
        t=tik.Tik(disable_debug=True)
        src=t.Tensor('float32',(64,),name='src',scope=tik.scope_gm)
        aux=[t.Tensor('float32',(16,),name='aux%d'%i,scope=tik.scope_gm)for i in range(3)]
        out=t.Tensor('float32',(5,64,16),name='out',scope=tik.scope_gm)
        with t.for_range(0,1,block_num=1)as core:
            row=t.Tensor('float32',(64,),name='row',scope=tik.scope_ubuf)
            scratch=t.Tensor('float16',(256,),name='word_matrix',scope=tik.scope_ubuf)
            trans=t.Tensor('float16',(256,),name='word_planes',scope=tik.scope_ubuf)
            broad=t.Tensor('float32',(1024,),name='broadcast_Float_bits',scope=tik.scope_ubuf)
            dst=t.Tensor('float32',(5120,),name='out_UB',scope=tik.scope_ubuf)
            t.data_move(row,src,0,1,8,0,0)
            broadcast_float_row_bits(t,dst,row,5,64,scratch,trans,broad)
            t.data_move(out,dst,0,1,640,0,0)
        name='broadcast_exact_Float_words105';work=ROOT/'compile';work.mkdir();before=os.getcwd()
        try:os.chdir(work);t.BuildCCE(kernel_name=name,inputs=[src]+aux,outputs=[out])
        finally:os.chdir(before)
        obj=work/'kernel_meta'/f'{name}.o';raw=json.loads((work/'kernel_meta'/f'{name}.json').read_text())
        meta={'obj_path':str(obj),'obj_sha256':hashlib.sha256(obj.read_bytes()).hexdigest(),'kernel_symbol':raw['kernelName'],'block_dim':int(raw['blockDim'])};r['meta']=meta
        runtime=Runtime5GM(0);dummy=torch.empty((16,),dtype=torch.float32,device='npu:0')
        torch.manual_seed(105)
        rows={'finite_general':torch.randn((64,),dtype=torch.float32),'softmax_rescale':torch.rand((64,),dtype=torch.float32),
              'softmax_denominator':1+4032*torch.rand((64,),dtype=torch.float32)}
        bits=[0,0x80000000,1,0x80000001,0x007fffff,0x807fffff,0x00800000,0x80800000,
              0x3f800001,0xbf800001,0x3f7fffff,0x40000001,0x7f7fffff,0xff7fffff,0x7f800000,0xff800000,
              0x7fc01234,0xffc04321,0x7f801234,0xff801234,0x12345678,0x9abcdef0,0xabcdef01,0x7f001234]
        bits=(bits*3)[:64];signed=[v if v<2**31 else v-2**32 for v in bits]
        rows['arbitrary_Float_word_patterns']=torch.tensor(signed,dtype=torch.int32).view(torch.float32)
        for name,row in rows.items():
            source=row.to('npu:0');gold=source.cpu().reshape(1,64,1).expand(5,64,16).contiguous()
            # This is bitwise memory replication, not an arithmetic metric. NaN
            # payload and subnormal tests must not be discarded by isfinite().
            source_bits=source.cpu().view(torch.int32).clone()
            assert torch.equal(source_bits,row.view(torch.int32))
            output=torch.empty((5,64,16),dtype=torch.float32,device='npu:0')
            runtime.launch(meta,[source,dummy,dummy,dummy,output])
            actual=output.cpu().contiguous();assert actual.shape==gold.shape
            exact=torch.equal(actual.view(torch.int32),gold.view(torch.int32));assert exact
            first=actual.view(torch.int32).clone()
            runtime.launch(meta,[source,dummy,dummy,dummy,output])
            repeat=torch.equal(output.cpu().contiguous().view(torch.int32),first);assert repeat
            assert torch.equal(source.cpu().view(torch.int32),source_bits)
            r['cases'][name]={'exact_all_Float_bits':True,'repeated_all_bits_equal':True,'input_bits_unchanged':True,
                              'SHA256_output_bytes':hashlib.sha256(actual.view(torch.uint8).numpy().tobytes()).hexdigest(),
                              'includes_non_Half_representable_Float':True,'shape':list(actual.shape)}
        r['runtime']=runtime.snapshot();r['completed']=True
    except BaseException as error:r['failure']={'type':type(error).__name__,'message':str(error)};raise
    finally:
        with(ROOT/'broadcast_float105.json').open('x')as f:f.write(json.dumps(r,indent=2)+'\n')
        print(json.dumps({'completed':r['completed'],'cases':r['cases'],'failure':r.get('failure')},indent=2),flush=True)


if __name__=='__main__':main()
