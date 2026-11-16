#!/usr/bin/env python3
"""Teacher-forced actual DiT block replay with USP8/TP1 and bounded AdaLN A/B.

Complete single-block timing excludes all model offload/stage work. The native
baseline MUST match actual captured block output before claiming integration
identity. Non-bit-exact norm fusion is an explicit exploratory comparison.
"""
import gc
import hashlib
import json
import os
from pathlib import Path
import statistics
import sys
import time
import torch
import torch_npu
from torch import nn
from safetensors import safe_open

ROOT=Path('/tmp/h3_round62_blocks01')
MODEL=Path('/home/mseco/models/MiniMax-H3-native-fp16-audit53/FL2VA/transformer')
rank,size=int(os.environ['RANK']),int(os.environ['WORLD_SIZE']);assert size==8
OUT=ROOT/('block_replay02_rank%d.json'%rank)
if OUT.exists():raise RuntimeError('Refuse DiT block replay receipts overwrite')
torch.npu.set_device(rank);torch.set_num_threads(2)
for key in ('VLLM_OMNI_910A_ADALN_AUDIT_DIR','VLLM_OMNI_910A_BLOCK_AUDIT_DIR'):
    os.environ.pop(key,None)
os.environ.update(VLLM_OMNI_910A_FP16_SEQ='1',VLLM_OMNI_910A_FLASH_ATTN='1',VLLM_OMNI_910A_EAGER_MODULATION='1',
                  VLLM_OMNI_910A_NO_FUSED_NORM='1',VLLM_OMNI_910A_NO_FUSED_ACT='1')
from vllm.config import VllmConfig, set_current_vllm_config
from vllm_omni.diffusion.distributed.parallel_state import init_distributed_environment, initialize_model_parallel, destroy_model_parallel
from vllm_omni.diffusion.config import set_current_diffusion_config
from vllm_omni.diffusion.data import OmniDiffusionConfig, DiffusionParallelConfig, AttentionConfig
from vllm_omni.diffusion.forward_context import set_forward_context
from vllm_omni.diffusion.models.minimax_h3 import minimax_h3_transformer as dit
sys.path.insert(0,'/tmp/h3_round61_adaln01')
from adaln_vector61 import launch, PreparedIndices

r={'rank':rank,'layers':[], 'scope':'Actual first-step complete resident DiT blocks, USP8/TP1. Not DLO, full step, whole generation, BF16 equivalence or official A2/A3 benchmark.'}
originals={name:getattr(dit,name) for name in ('indexed_gate','rms_norm_indexed_scale_shift','indexed_gate_rms_norm_scale_shift')}
MODE='baseline';PREPARED=None


def norm_exact(x,weight,eps):
    f=x.float();var=f.pow(2).mean(-1,keepdim=True)
    return (weight.float()*(f*torch.rsqrt(var+eps))).to(x.dtype)


def gate_fn(x,gate,other,indices):
    if MODE=='baseline':return originals['indexed_gate'](x,gate,other,indices)
    return launch('gate',x,PREPARED,gate=gate,branch=other)


def rms_fn(x,weight,shift,scale,indices,eps):
    if MODE in ('baseline','point_gate'):
        return originals['rms_norm_indexed_scale_shift'](x,weight,shift,scale,indices,eps)
    if MODE=='lossless_split':
        h=norm_exact(x,weight,eps)
        return launch('mod',h,PREPARED,shift=shift,scale=scale)
    return launch('rms_mod',x,PREPARED,weight=weight,shift=shift.contiguous(),scale=scale.contiguous(),eps=eps)


def gate_rms_fn(x,gate,branch,weight,shift,scale,indices,eps):
    if MODE in ('baseline','point_gate'):
        return originals['indexed_gate_rms_norm_scale_shift'](x,gate,branch,weight,shift,scale,indices,eps)
    if MODE=='lossless_split':
        updated=launch('gate',x,PREPARED,gate=gate,branch=branch)
        h=norm_exact(updated,weight,eps)
        return updated,launch('mod',h,PREPARED,shift=shift,scale=scale)
    return launch('gate_rms_mod',x,PREPARED,gate=gate.contiguous(),branch=branch,weight=weight,shift=shift.contiguous(),scale=scale.contiguous(),eps=eps)


for name,fn in [('indexed_gate',gate_fn),('rms_norm_indexed_scale_shift',rms_fn),('indexed_gate_rms_norm_scale_shift',gate_rms_fn)]:setattr(dit,name,fn)


def metric(a,b):
    x,y=a.detach().cpu().contiguous(),b.detach().cpu().contiguous();e=x.float().double()-y.float().double()
    return {'raw_bits_equal':torch.equal(x.view(torch.uint8),y.view(torch.uint8)),'finite':bool(torch.isfinite(x.float()).all() and torch.isfinite(y.float()).all()),
            'elements':x.numel(),'word_mismatches':int(x.view(torch.int16).ne(y.view(torch.int16)).sum()),
            'max_abs_error':float(e.abs().max()),'relative_rms':float(e.square().mean().sqrt()/y.float().double().square().mean().sqrt().clamp_min(1e-30))}


def run(block,x,kwargs,config,vcfg):
    with set_forward_context(vllm_config=vcfg,omni_diffusion_config=config,denoise_step_idx=0):
        return block(x,**kwargs)


def measured(block,x,kwargs,config,vcfg):
    torch.distributed.barrier();torch.npu.synchronize();start=time.perf_counter()
    first=run(block,x,kwargs,config,vcfg);torch.npu.synchronize();cold=time.perf_counter()-start
    groups=[];outputs=[]
    for _ in range(3):
        torch.distributed.barrier();torch.npu.synchronize();start=time.perf_counter()
        out=run(block,x,kwargs,config,vcfg);torch.npu.synchronize();groups.append((time.perf_counter()-start)*1000)
        outputs.append(out.detach().cpu().contiguous())
    return outputs[-1],{'cold_call_s':cold,'warm_groups_ms':groups,'median_warm_ms':statistics.median(groups),
                        'first_second':metric(first,outputs[0]),'second_third':metric(outputs[0],outputs[1]),'third_fourth':metric(outputs[1],outputs[2])}


try:
    init_distributed_environment(world_size=size,rank=rank,local_rank=rank,backend='hccl')
    initialize_model_parallel(sequence_parallel_size=size,ulysses_degree=size,tensor_parallel_size=1,backend='hccl')
    config=OmniDiffusionConfig(model=str(MODEL.parent.parent),model_class_name='MiniMaxH3Pipeline',task_type='fl2va',dtype=torch.float16,num_gpus=8,
                              parallel_config=DiffusionParallelConfig(ulysses_degree=8,ring_degree=1,tensor_parallel_size=1),
                              diffusion_attention_config=AttentionConfig(default='TORCH_SDPA'))
    vcfg=VllmConfig()
    arch=dit.MiniMaxH3DiTArchConfig.from_mapping(json.loads((MODEL/'config.json').read_text()))
    index=json.loads((MODEL/'model.safetensors.index.json').read_text())['weight_map']
    with torch.inference_mode(),set_current_vllm_config(vcfg),set_current_diffusion_config(config),set_forward_context(vllm_config=vcfg,omni_diffusion_config=config,denoise_step_idx=0):
        for layer in (0,24,49):
            fixture=torch.load(ROOT/'capture'/('rank%d'%rank)/('block%02d.pt'%layer),weights_only=True)
            assert fixture['layer']==layer and fixture['rank']==rank
            block=dit.MiniMaxH3DiTBlock(arch,None,prefix='blocks.%d'%layer,adaln_cache=None)
            assert block.attn.attention.parallel_strategy.name=='ulysses', 'Replay must reshard before the full-key mask'
            owner=dit.MiniMaxH3DiTModel.__new__(dit.MiniMaxH3DiTModel);nn.Module.__init__(owner)
            owner.arch=arch;owner.adaln_cache=None;owner._diffusers_weights=False
            owner.blocks=nn.ModuleDict({str(layer):block})
            prefix='blocks.%d.'%layer
            def weights():
                for shard in sorted(set(index[k] for k in index if k.startswith(prefix))):
                    with safe_open(str(MODEL/shard),framework='pt',device='cpu') as f:
                        for key in sorted(k for k in index if k.startswith(prefix) and index[k]==shard):yield key,f.get_tensor(key)
            t=time.perf_counter();loaded=owner.load_weights(weights())
            assert loaded==set(dict(owner.named_parameters())) and len(loaded)==10
            block.to(torch.device('npu',rank)).eval();torch.npu.synchronize()
            assert all(p.dtype==torch.float16 for p in block.parameters())
            x=fixture['x'].to('npu').contiguous()
            kwargs={k:v.to('npu') if isinstance(v,torch.Tensor) else v for k,v in fixture['kwargs'].items()}
            PREPARED=PreparedIndices(kwargs['combined_indices'],kwargs['t_emb'].shape[0]*3)
            row={'layer':layer,'load_and_copy_s':time.perf_counter()-t,'loaded_parameter_tensors':len(loaded),'cases':{},
                 'shape':list(x.shape),'attention_backend':block.attn.attention.attn_backend.get_name(),'parallel_strategy':block.attn.attention.parallel_strategy.name}
            baseline=None
            for mode in ('baseline','point_gate','lossless_split','norm_fused_exploratory'):
                MODE=mode
                output,receipt=measured(block,x,kwargs,config,vcfg)
                receipt['vs_capture']=metric(output,fixture['output'])
                if baseline is None:baseline=output
                receipt['vs_baseline']=metric(output,baseline)
                row['cases'][mode]=receipt;print(json.dumps({'rank':rank,'layer':layer,'mode':mode,**receipt}),flush=True)
            r['layers'].append(row);OUT.write_text(json.dumps(r,indent=2)+'\n')
            del owner,block,fixture,x,kwargs,baseline,output;gc.collect();torch.npu.empty_cache()
    r['completed']=True
except BaseException as exc:
    r['failure']={'type':type(exc).__name__,'message':str(exc)};raise
finally:
    OUT.write_text(json.dumps(r,indent=2)+'\n')
    for name,original in originals.items():setattr(dit,name,original)
    if torch.distributed.is_initialized():destroy_model_parallel();torch.distributed.destroy_process_group()
