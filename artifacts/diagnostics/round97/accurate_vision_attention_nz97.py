#!/usr/bin/env python3
"""Accurate vision attention with direct Float-NZ softmax/dual-Half-NZ probability packing.

Prepared source only. The root must validate both new softmax kernels and the
fused PV source against actual frozen QKV/CPU Half before any whole-model claim.
No global torch or production edits, no cached image-conditioning or Float weights.
"""
from pathlib import Path
import hashlib
from accurate_vision_attention_pv96 import (
    AccurateVisionAttentionPV96,check_activation_spec,query_chunks,
    NUM_HEADS,ORIGINAL_HEAD_DIM,PADDED_HEAD_DIM,ORIGINAL_KEY_LEN,PADDED_KEY_LEN)


class AccurateVisionAttentionNZ97(AccurateVisionAttentionPV96):
    def __init__(self,candidate,runtime,kernel_dir):
        self._softmax_metas={}
        super().__init__(candidate,runtime,kernel_dir)
        self._stats['softmax_pack_launches']=0
        self._stats['fused_pv_direct_NZ_launches']=0

    def prepare(self):
        import softmax_pack97
        receipt=super().prepare()
        for m in(512,448):
            meta=softmax_pack97.build_kernel(batch=16,m=m,n=4096,valid_keys=4032,out_dir=self._kernel_dir/'softmax')
            assert meta['block_dim']==30
            assert hashlib.sha256(Path(meta['obj_path']).read_bytes()).hexdigest()==meta['obj_sha256']
            self._softmax_metas[m]=meta
        receipt['softmax_pack_metas']=self._softmax_metas
        receipt['no_Torch_score_unpack_or_softmax_or_probability_pack']=True
        receipt['launch_plan']={'query_chunks':8,'qk_launches':8,'softmax_pack_launches':8,'fused_pv_launches':8,'kernel_launches':24,'Torch_probability_pack_calls':0}
        return receipt

    def set_enabled(self,value):
        if value and len(self._softmax_metas)!=2:
            raise RuntimeError('direct-NZ attention requires both real softmax/pack kernels')
        return super().set_enabled(value)

    def forward(self,q,k,v):
        if not self._enabled:
            self._stats['calls_disabled']+=1
            return self._original(q,k,v)
        import torch
        import batch_gemm85 as batch
        import pv_high_low96 as pv
        check_activation_spec(q.shape,str(q.dtype),q.device.type,q.device.index,self._runtime.device)
        assert q.shape==k.shape==v.shape and q.dtype==k.dtype==v.dtype==torch.float16
        assert q.device==k.device==v.device
        self._stats['calls_enabled']+=1
        qp=torch.zeros((NUM_HEADS,ORIGINAL_KEY_LEN,PADDED_HEAD_DIM),dtype=q.dtype,device=q.device);qp[...,:ORIGINAL_HEAD_DIM]=q[0]
        kp=torch.zeros((NUM_HEADS,PADDED_KEY_LEN,PADDED_HEAD_DIM),dtype=q.dtype,device=q.device);kp[:,:ORIGINAL_KEY_LEN,:ORIGINAL_HEAD_DIM]=k[0]
        vp=torch.zeros_like(kp);vp[:,:ORIGINAL_KEY_LEN,:ORIGINAL_HEAD_DIM]=v[0]
        packed_k=batch.pack_w(kp);packed_v=batch.pack_w(vp.transpose(1,2).contiguous())
        self._stats['pack_w_calls']+=2
        auxh=torch.empty((16,),dtype=torch.float16,device=q.device)
        auxf=torch.empty((16,),dtype=torch.float32,device=q.device)
        result=torch.empty((NUM_HEADS,ORIGINAL_KEY_LEN,ORIGINAL_HEAD_DIM),dtype=q.dtype,device=q.device)
        for start,end,m in query_chunks():
            packed_q=batch.pack_a(qp[:,start:end,:]);m_pad=512
            scores=torch.empty((16,256,m_pad,16),dtype=torch.float32,device=q.device)
            self._runtime.launch(self._qk_metas[(m,128,4096)],[packed_q,packed_k,auxh,auxh,scores])
            hi=torch.empty((16,256,m_pad,16),dtype=torch.float16,device=q.device);lo=torch.empty_like(hi)
            self._runtime.launch(self._softmax_metas[m],[scores,auxf,auxf,hi,lo])
            out=torch.empty((16,8,m_pad,16),dtype=torch.float16,device=q.device)
            self._runtime.launch(self._pv_metas[(m,4096,128)],[hi,lo,packed_v,auxf,out])
            result[:,start:end,:]=pv.unpack_out_half(out,m)[:,:,:ORIGINAL_HEAD_DIM]
            self._stats['kernel_launches']+=3;self._stats['qk_launches']+=1
            self._stats['softmax_pack_launches']+=1;self._stats['fused_pv_launches']+=1
            self._stats['fused_pv_direct_NZ_launches']+=1;self._stats['query_chunks_processed']+=1
            self._stats['float_before_half_chunks']+=1;self._stats['half_output_chunks']+=1
        return result.unsqueeze(0)

    def snapshot(self):
        r=super().snapshot()
        r.update(round=97,direct_NZ_probability_packing=True,
                 launch_plan={'query_chunks':8,'qk_launches':8,'softmax_pack_launches':8,'fused_pv_launches':8,'kernel_launches':24,'Torch_probability_pack_calls':0},
                 bytes={'score_FloatNZ_per_chunk':16*256*512*16*4,'dual_HalfNZ_probability_per_chunk':2*16*256*512*16*2,'HalfNZ_output_per_chunk':16*8*512*16*2,'Torch_probability_unpack_or_pack_bytes':0},
                 no_Torch_score_unpack_softmax_probability_decomp_pack=True,
                 softmax_kernels=len(self._softmax_metas),
                 Float_reduction_order_may_change_from_R88=True,
                 full_encoder_precision_accepted=False,whole_goal_complete=False,numerical_tolerance=None,
                 source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
        return r

    def uninstall(self):
        if self._installed:
            current=getattr(self._candidate,'_vision_sdpa_bf16_compatible',None)
            if getattr(current,'__self__',None)is not self or getattr(current,'__func__',None)is not type(self).forward:
                raise RuntimeError('refusing to clobber another helper patch')
        r=super().uninstall();self._softmax_metas.clear();r['softmax_kernels']=0;return r
