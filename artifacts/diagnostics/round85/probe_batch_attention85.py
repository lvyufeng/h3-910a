#!/usr/bin/env python3
"""Test batched accurate FP32 QK/PV on frozen real decoder0 Half operands.

Actual CPU Half MATH SDPA is the teacher. Float score/PV intermediates and the
high/scaled-low P representation are leaf diagnostics,not another model oracle.
"""
from pathlib import Path
import hashlib
import json
import statistics
import time

ROOT=Path('/tmp/h3_round85_batch01')
CAPTURE=Path('/tmp/h3_round85_capture01/QKV_rank0.pt')


def main():
    import torch
    import torch_npu
    from runtime5gm83 import Runtime5GM
    import batch_gemm85 as kernel
    from full_row_fp32_fixed84 import compare,tensor_info,file_sha
    torch.set_num_threads(2);torch.npu.set_device(0)
    target=ROOT/'attention_result85.json';assert not target.exists()
    original=torch.load(CAPTURE,map_location='cpu',weights_only=True)
    q=original['q'];k=original['k'];v=original['v']
    assert q.dtype==k.dtype==v.dtype==torch.float16
    groups=q.shape[1]//k.shape[1];kc=k.repeat_interleave(groups,1);vc=v.repeat_interleave(groups,1)
    assert q.shape==(1,8,2043,128)
    runtime=Runtime5GM(0)
    # Hardware smoke proves distinct heads and two K blocks before actual SDPA.
    torch.manual_seed(8501)
    sa=torch.randn((2,128,256),dtype=torch.float16)*.125
    sb=torch.randn((2,128,256),dtype=torch.float16)*.125
    sm=kernel.build_kernel(batch=2,m=128,k=256,n=128,out_dir=ROOT/'kernels')
    spa=kernel.pack_a(sa.to('npu:0'));spb=kernel.pack_w(sb.to('npu:0'))
    sc=torch.empty(sm['gm_shapes']['c'],dtype=torch.float32,device='npu:0')
    sax=[torch.zeros(16,dtype=torch.float16,device='npu:0')for _ in range(2)]
    runtime.launch(sm,[spa,spb,*sax,sc],synchronize=True)
    smoke_first=kernel.unpack_c(sc,m=128).cpu()
    runtime.launch(sm,[spa,spb,*sax,sc],synchronize=True)
    smoke_repeat=compare(kernel.unpack_c(sc,m=128),smoke_first)
    assert smoke_repeat['raw_bits_equal']
    smoke=compare(smoke_first,torch.matmul(sa.float(),sb.float().transpose(-2,-1)))
    assert smoke['relative_rms']<1e-3 # predeclared wrong-layout sanity,not model tolerance
    metas={}
    for m in(512,507):
        for kk,nn in((128,2048),(2048,128)):
            metas[(m,kk,nn)]=kernel.build_kernel(batch=8,m=m,k=kk,n=nn,out_dir=ROOT/'kernels')
    qn=q.squeeze(0).to('npu:0');kn=kc.squeeze(0).to('npu:0');vn=vc.squeeze(0).to('npu:0')
    before={name:tensor_info(tensor)['sha256']for name,tensor in(('q',qn),('k',kn),('v',vn))}
    def gemm(a,b):
        h,m,kk=a.shape;nn=b.shape[-2];meta=metas[(m,kk,nn)]
        pa=kernel.pack_a(a);pb=kernel.pack_w(b)
        c=torch.empty((h,nn//16,((m+127)//128)*128,16),dtype=torch.float32,device=a.device)
        aux=[torch.zeros(16,dtype=torch.float16,device=a.device)for _ in range(2)]
        runtime.launch(meta,[pa,pb,*aux,c],synchronize=True)
        return kernel.unpack_c(c,m=m)
    diagnostics=[]
    def accurate_attention(record=False):
        # K/V are activation-derived packed copies,not model-weight shadows.
        kp=torch.zeros((8,2048,128),dtype=torch.float16,device=qn.device);kp[:,:2043]=kn
        vp=torch.zeros((8,2048,128),dtype=torch.float16,device=qn.device);vp[:,:2043]=vn
        result=torch.empty_like(qn)
        for start in range(0,2043,512):
            end=min(start+512,2043);m=end-start
            scores=gemm(qn[:,start:end].contiguous(),kp)[:,:,:2043]*(128**-0.5)
            mask=torch.arange(2043,device=qn.device)[None,:]>torch.arange(start,end,device=qn.device)[:,None]
            p=torch.softmax(scores.masked_fill(mask,float('-inf')),-1)
            ph=p.to(torch.float16)
            pl=((p-ph.float())*1024.0).to(torch.float16)
            ap=torch.zeros((8,m,2048),dtype=torch.float16,device=qn.device);ap[:,:,:2043]=ph
            al=torch.zeros_like(ap);al[:,:,:2043]=pl
            high=gemm(ap,vp.transpose(-2,-1).contiguous())
            low=gemm(al,vp.transpose(-2,-1).contiguous())
            pv=high+low*(1.0/1024.0)
            result[:,start:end]=pv.to(torch.float16)
            if record:
                cpu_scores=torch.matmul(q.float().squeeze(0)[:,start:end],kc.float().squeeze(0).transpose(-2,-1))*(128**-0.5)
                cpu_mask=mask.cpu();cpu_p=torch.softmax(cpu_scores.masked_fill(cpu_mask,float('-inf')),-1)
                cpu_pv=torch.matmul(cpu_p,vc.float().squeeze(0))
                diagnostics.append({'query_start':start,'query_end':end,
                    'score_vs_CPU_Float':compare(scores,cpu_scores),
                    'probability_vs_CPU_Float':compare(p,cpu_p),
                    'P_reconstruction':compare(ph.float()+pl.float()/1024,p),
                    'PV_vs_CPU_Float':compare(pv,cpu_pv),
                    'half_output_vs_actual_CPU_Half_MATH':compare(result[:,start:end],original['cpu_half_math_out'].squeeze(0)[:,start:end])})
        return result.unsqueeze(0)
    with torch.inference_mode():first=accurate_attention(record=True);second=accurate_attention()
    repeat=compare(second,first);assert repeat['raw_bits_equal']
    samples=[];warm_hashes=[]
    for _ in range(3):
        torch.npu.synchronize();begin=time.perf_counter()
        with torch.inference_mode():out=accurate_attention()
        torch.npu.synchronize();samples.append(time.perf_counter()-begin);warm_hashes.append(tensor_info(out)['sha256'])
    assert len(set(warm_hashes))==1
    after={name:tensor_info(tensor)['sha256']for name,tensor in(('q',qn),('k',kn),('v',vn))};assert before==after
    r={'scope':__doc__,'completed':True,'kernel_launch_executed':True,
       'source_sha256':file_sha(Path(__file__)),'kernel_source_sha256':file_sha(Path(kernel.__file__)),
       'frozen_capture_sha256':file_sha(CAPTURE),'actual_CPU_Half_MATH_reference_sha256':tensor_info(original['cpu_half_math_out'])['sha256'],
       'batched_distinct_head_twoK_smoke':smoke,'smoke_repeat_bits':smoke_repeat,
       'smoke_wrong_layout_guard_is_not_model_precision_tolerance':True,
       'custom_vs_CPU_Half_MATH':compare(first,original['cpu_half_math_out']),
       'saved_native_vs_CPU_Half_MATH':compare(original['native_out'],original['cpu_half_math_out']),
       'custom_repeat_bits':repeat,'original_operands_unchanged':before==after,
       'no_FP32_model_weight_shadows':True,'FP32_score_and_PV_before_single_Half_attention_output':True,
       'query_chunks':diagnostics,'resident_leaf_SDPA_samples_s':samples,
       'resident_leaf_SDPA_median_s':statistics.median(samples),
       'timing_includes_activation_padding_packing_score_softmax_split_two_PV_and_Halfcast':True,
       'not_whole_encoder_timing':True,'full_encoder_precision_accepted':False,'whole_goal_complete':False,
       'builds':list(metas.values())}
    with target.open('x')as f:f.write(json.dumps(r,indent=2)+'\n')
    print(json.dumps({k:r[k]for k in('custom_vs_CPU_Half_MATH','saved_native_vs_CPU_Half_MATH','resident_leaf_SDPA_median_s')},indent=2))


if __name__=='__main__':main()
