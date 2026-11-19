#!/usr/bin/env python3
"""Original-Half Vision-TP matrix backend: fused biased Half columns, Float GM rows.

Reuses executed R87/R83 Cube kernels; original weights stay Half, only bias widens.
Logical shard channels are NOT kernel padding. No activation cache across calls.
"""
from pathlib import Path
import copy
import hashlib


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def ceil128(value):return((value+127)//128)*128


def geometry(m,k,n):
    if any(not isinstance(v,int)or isinstance(v,bool)or v<=0 for v in(m,k,n)):raise ValueError('positive integer matrix shapes required')
    return{'m':m,'k':k,'n':n,'mp':ceil128(m),'kp':ceil128(k),'np':ceil128(n)}


class VisionGemmBackend107:
    def __init__(self,runtime,kernel_dir):
        self.runtime=runtime;self.kernel_dir=Path(kernel_dir);self.entries=[];self.metas={}
        self.counts={'column_launches':0,'row_Float_launches':0,'fresh_input_packs':0,'Half_output_columns':0,'Float_output_rows':0}
    def prime(self,name,weight,bias,mode,m=8064):
        import torch
        import vision_bias87 as col
        import blocked_fp32_gemm83 as row
        assert mode in('column','row')and weight.ndim==2 and weight.dtype==torch.float16
        assert weight.device.type in('npu','privateuseone')and weight.device.index==self.runtime.device
        assert mode!='row'or bias is None,'row bias must be added ONCE after Float HCCL'
        n,k=map(int,weight.shape);g=geometry(m,k,n);key=(mode,m,g['kp'],g['np'])
        if key not in self.metas:
            module=col if mode=='column'else row
            meta=module.build_kernel(m=m,k=g['kp'],n=g['np'],out_dir=self.kernel_dir/mode)
            assert meta['dst_dtype']==('float16'if mode=='column'else'float32')
            assert meta['GM_argument_count']==5 and 0<meta['block_dim']<=30
            assert sha(meta['obj_path'])==meta['obj_sha256']
            self.metas[key]=meta
        padded=torch.zeros((g['np'],g['kp']),dtype=torch.float16,device=weight.device);padded[:n,:k]=weight.detach()
        packed=padded.reshape(g['np'],g['kp']//16,16).permute(1,0,2).contiguous()
        assert torch.equal(packed.permute(1,0,2).reshape(g['np'],g['kp'])[:n,:k],weight.detach())
        assert int(torch.count_nonzero(padded[n:]).item())==0 and int(torch.count_nonzero(padded[:n,k:]).item())==0
        pb=None
        if mode=='column':
            pb=torch.zeros((g['np'],),dtype=torch.float32,device=weight.device)
            if bias is not None:
                assert bias.dtype==torch.float16 and tuple(bias.shape)==(n,)and bias.device==weight.device
                pb[:n]=bias.detach().float();assert torch.equal(pb[:n].half(),bias.detach())
            assert int(torch.count_nonzero(pb[n:]).item())==0
        entry={'name':str(name),'mode':mode,**g,'weight':packed,'bias':pb,'meta':self.metas[key],
               'weight_roundtrip_exact':True,'zero_only_K_N_padding':True}
        self.entries.append(entry)
        if not hasattr(self,'aux'):self.aux=torch.empty((16,),dtype=torch.float16,device=weight.device)
        return entry
    def pack_input(self,entry,input_):
        import torch
        assert input_.dtype==torch.float16 and tuple(input_.shape)==(entry['m'],entry['k'])
        assert input_.device.index==self.runtime.device and input_.device.type in('npu','privateuseone')
        # Fresh A every call: no saved pixels, hidden states or conditioning.
        flat=torch.zeros((entry['mp'],entry['kp']),dtype=torch.float16,device=input_.device)
        flat[:entry['m'],:entry['k']]=input_
        self.counts['fresh_input_packs']+=1
        return flat.reshape(entry['mp'],entry['kp']//16,16).permute(1,0,2).contiguous()
    def run(self,entry,input_):
        import torch
        a=self.pack_input(entry,input_);mode=entry['mode'];dtype=torch.float16 if mode=='column'else torch.float32
        c=torch.empty((entry['np']//16,entry['mp'],16),dtype=dtype,device=input_.device)
        arguments=[a,entry['weight'],entry['bias'],self.aux,c]if mode=='column'else[a,entry['weight'],self.aux,self.aux,c]
        self.runtime.launch(entry['meta'],arguments,synchronize=True)
        nd=c.permute(1,0,2).reshape(entry['mp'],entry['np'])[:entry['m'],:entry['n']].contiguous()
        assert nd.dtype==dtype and tuple(nd.shape)==(entry['m'],entry['n'])
        if mode=='column':self.counts['column_launches']+=1;self.counts['Half_output_columns']+=1
        else:self.counts['row_Float_launches']+=1;self.counts['Float_output_rows']+=1
        return nd
    def controls(self,entry,input_):
        import torch
        a=self.pack_input(entry,input_);flat=a.permute(1,0,2).reshape(entry['mp'],entry['kp'])
        assert torch.equal(flat[:entry['m'],:entry['k']],input_)
        assert int(torch.count_nonzero(flat[entry['m']:]).item())==0
        assert int(torch.count_nonzero(flat[:entry['m'],entry['k']:]).item())==0
        return{'input_prefix_bit_equal':True,'zero_only_M_K_padding':True,'geometry':{k:entry[k]for k in('m','k','n','mp','kp','np')},'mode':entry['mode']}
    def snapshot(self):
        return{'source_SHA256':sha(__file__),'entries':[{k:e[k]for k in('name','mode','m','k','n','mp','kp','np','weight_roundtrip_exact','zero_only_K_N_padding')}for e in self.entries],
               'kernels':copy.deepcopy(list(self.metas.values())),'counts':dict(self.counts),
               'packed_Half_weight_bytes':sum(e['weight'].numel()*2 for e in self.entries),
               'Float_bias_bytes':sum(e['bias'].numel()*4 for e in self.entries if e['bias']is not None),
               'Float_weight_shadows':0,'row_outputs_Float_before_collective':True,'column_one_Half_after_Float_bias':True,
               'no_activations_cached_across_calls':True,'full_encoder_precision_accepted':False,'whole_goal_complete':False,'numerical_tolerance':None}
