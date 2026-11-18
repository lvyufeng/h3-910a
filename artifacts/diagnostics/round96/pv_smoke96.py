#!/usr/bin/env python3
"""Cheap two-L0C fused PV compile/launch smoke against unchanged two-kernel arithmetic."""
from pathlib import Path
import json
ROOT=Path('/tmp/h3_round96_fused01')


def main():
    import torch
    import torch_npu
    import batch_gemm85 as old
    import pv_high_low96 as fused
    import full_row_fp32_fixed84 as previous
    from runtime5gm83 import Runtime5GM
    torch.set_num_threads(2);torch.npu.set_device(0);runtime=Runtime5GM(0)
    r={'scope':__doc__,'completed':False,'source_SHA256':previous.file_sha(Path(__file__)),
       'synthetic_smoke_only_not_model_timing':True,'no_Float_Linear_teacher_or_BF_goldcasts':True,
       'whole_goal_complete':False,'full_encoder_precision_accepted':False,'numerical_tolerance':None}
    try:
        torch.manual_seed(96)
        p=torch.softmax(torch.randn(2,128,256,dtype=torch.float32),dim=-1)
        hi=p.to(torch.float16).to('npu:0');lo=((p-p.to(torch.float16).float())*1024).to(torch.float16).to('npu:0')
        v=torch.randn(2,128,256,dtype=torch.float16).to('npu:0')
        ah=old.pack_a(hi);al=old.pack_a(lo);bv=old.pack_w(v)
        oldmeta=old.build_kernel(batch=2,m=128,k=256,n=128,out_dir=ROOT/'smoke_old')
        newmeta=fused.build_kernel(batch=2,m=128,k=256,n=128,out_dir=ROOT/'smoke_fused')
        r['oldmeta']=oldmeta;r['newmeta']=newmeta
        auxh=torch.empty((16,),dtype=torch.float16,device='npu:0');auxf=torch.empty((16,),dtype=torch.float32,device='npu:0')
        ch=torch.empty((2,8,128,16),dtype=torch.float32,device='npu:0');cl=torch.empty_like(ch)
        co=torch.empty((2,8,128,16),dtype=torch.float16,device='npu:0')
        inputs=[previous.tensor_info(x)['sha256']for x in(hi,lo,v)]
        with torch.inference_mode():
            runtime.launch(oldmeta,[ah,bv,auxh,auxh,ch]);runtime.launch(oldmeta,[al,bv,auxh,auxh,cl])
            baseline=(old.unpack_c(ch,128)+old.unpack_c(cl,128)*(1/1024)).to(torch.float16)
            runtime.launch(newmeta,[ah,al,bv,auxf,co]);out=fused.unpack_out_half(co,128)
            repeat=torch.empty_like(co);runtime.launch(newmeta,[ah,al,bv,auxf,repeat])
        r['fused_vs_old_two_products']=previous.compare(out,baseline)
        r['fused_repeat']=previous.compare(fused.unpack_out_half(repeat,128),out)
        assert r['fused_vs_old_two_products']['raw_bits_equal']and r['fused_repeat']['raw_bits_equal']
        assert inputs==[previous.tensor_info(x)['sha256']for x in(hi,lo,v)]
        r['all_inputs_unchanged']=True;r['completed']=True
    except BaseException as error:r['failure']={'type':type(error).__name__,'message':str(error)};raise
    finally:
        with(ROOT/'pv_smoke96.json').open('x')as f:f.write(json.dumps(r,indent=2)+'\n')
        print(json.dumps({'completed':r['completed'],'fused_vs_old':r.get('fused_vs_old_two_products'),'failure':r.get('failure')},indent=2),flush=True)


if __name__=='__main__':main()
