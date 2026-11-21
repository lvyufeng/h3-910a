#!/usr/bin/env python3
"""Isolate the original time embedder with true TP2/USP4 and original FP32 guards.

Small component diagnostics only: not model generation, sampler or new arithmetic.
All checkpoint weights and actual timesteps stay on the owned host.
"""
from __future__ import annotations
import hashlib
import json
import math
import os
from pathlib import Path

ROOT=Path('/tmp/h3_repeat12902')
CHECKPOINT=Path('/home/mseco/models/MiniMax-H3/FL2VA/transformer')


def main():
    import torch
    import torch_npu
    from safetensors import safe_open
    from vllm_omni.diffusion.distributed.parallel_state import init_distributed_environment,initialize_model_parallel,destroy_distributed_env
    from vllm_omni.diffusion.data import OmniDiffusionConfig,DiffusionParallelConfig,TransformerConfig
    from vllm_omni.diffusion.config import set_current_diffusion_config
    from vllm_omni.diffusion.forward_context import set_forward_context
    from h3_replay123.transformer import MiniMaxH3TimeEmbedder,MiniMaxH3DiTArchConfig
    from fp32_time_residual_guard123 import install_time_projection_guard123
    from fingerprints129 import fingerprint,normalized_fingerprint,digest
    from vllm.distributed.parallel_state import get_tp_group,get_tensor_model_parallel_world_size
    rank=int(os.environ['RANK']);torch.npu.set_device(rank);torch.set_num_threads(4)
    torch.npu.matmul.allow_hf32=False;torch.npu.conv.allow_hf32=False
    path=ROOT/f'time_component_rank{rank}129.json';assert not path.exists()
    init_distributed_environment(world_size=8,rank=rank,local_rank=rank,distributed_init_method='env://',backend='hccl')
    initialize_model_parallel(tensor_parallel_size=2,sequence_parallel_size=4,ulysses_degree=4,ring_degree=1,backend='hccl')
    pc=DiffusionParallelConfig(tensor_parallel_size=2,sequence_parallel_size=4,ulysses_degree=4,ring_degree=1)
    config=json.loads((CHECKPOINT/'config.json').read_text());arch=MiniMaxH3DiTArchConfig.from_mapping(config)
    od=OmniDiffusionConfig(model=str(CHECKPOINT.parent),model_class_name='MiniMaxH3Pipeline',dtype=torch.float16,
        tf_model_config=TransformerConfig.from_dict(config),parallel_config=pc,num_gpus=8)
    result={'scope':__doc__,'rank':rank,'completed':False,'TP_actual':get_tensor_model_parallel_world_size(),
        'TP_group':list(get_tp_group().ranks),'source_timesteps':'R128 ownlive NPU input, no teacher features',
        'original_arithmetic_changed':False,'passes':[]}
    try:
        assert result['TP_actual']==2
        with set_current_diffusion_config(od),set_forward_context(omni_diffusion_config=od),torch.no_grad():
            class Container(torch.nn.Module):pass
            holder=Container();holder.time_embedder=MiniMaxH3TimeEmbedder(arch,prefix='time_embedder')
            parameters=dict(holder.named_parameters());index=json.loads((CHECKPOINT/'model.safetensors.index.json').read_text())['weight_map']
            for filename in sorted({index[name] for name in parameters}):
                with safe_open(str(CHECKPOINT/filename),framework='pt',device='cpu') as file:
                    for name,parameter in parameters.items():
                        if index[name]!=filename:continue
                        loader=parameter.weight_loader;loader(parameter,file.get_tensor(name))
            assert all(p.dtype==torch.float32 for p in holder.parameters())
            install_time_projection_guard123(holder);holder.to(f'npu:{rank}').eval()
            data=torch.load(f'/tmp/h3_overlap12801/native_worker_input_step25_rank{rank}.pt',map_location='cpu',weights_only=True)
            t=data['unique_timesteps'].to(f'npu:{rank}');del data
            result['actual_timestep_input']=fingerprint(t,torch)
            result['parameters_before']={name:fingerprint(p,torch) for name,p in holder.named_parameters()}
            records=[];reference={};local_old=[]
            # Original local quant_method.apply runs once; capturing before RowParallel reduction distinguishes boundaries.
            for name,layer in holder.named_modules():
                if name not in ('time_embedder.proj_in','time_embedder.proj_out'):continue
                old=layer.quant_method;old_apply=old.apply
                def apply(actual,x,bias=None,name=name,old_apply=old_apply):
                    output=old_apply(actual,x,bias)
                    records.append({'region':name+':local_projection','input':fingerprint(x,torch),
                        'output':fingerprint(output,torch),'bias':fingerprint(bias,torch)})
                    return output
                old.apply=apply;local_old.append((old,old_apply))
                def post(module,args,output,name=name):
                    records.append({'region':name+':post_TP','input':fingerprint(args,torch),'output':fingerprint(output,torch)})
                layer.register_forward_hook(post)
            for repeat in range(4):
                records.clear()
                half=holder.time_embedder.frequency_embedding_size//2
                freqs=torch.exp(-math.log(10000.0)*torch.arange(half,dtype=torch.float32,device=t.device)/half)
                argument=t.float()[:,None]*freqs[None]
                spectrum=torch.cat([torch.cos(argument),torch.sin(argument)],dim=-1)
                records.append({'region':'frequency_construction','input':fingerprint(t,torch),'output':fingerprint(spectrum,torch)})
                output=holder.time_embedder(t);records.append({'region':'time_embedder_output','output':fingerprint(output,torch)})
                cpu=output.detach().cpu().contiguous()
                if repeat==0:reference={r['region']:r for r in records};first=cpu.clone()
                compared=[]
                for record in records:
                    old=reference[record['region']]
                    compared.append({'region':record['region'],'all_record_logical_SHA_equal':digest(normalized_fingerprint(record))==digest(normalized_fingerprint(old))})
                delta=cpu.double()-first.double();norm=float(first.double().square().sum().sqrt().clamp_min(1e-30))
                result['passes'].append({'repeat':repeat,'all_regions':list(records),'compare_to_first':compared,
                    'output_different_FP32_words':int((cpu.view(torch.int32)!=first.view(torch.int32)).sum()),
                    'max_abs_difference':float(delta.abs().max()),'relative_RMS_error':float(delta.square().sum().sqrt())/norm})
            result['parameters_after']={name:fingerprint(p,torch) for name,p in holder.named_parameters()}
            result['FP32_parameters_unchanged']=normalized_fingerprint(result['parameters_before'])==normalized_fingerprint(result['parameters_after'])
            result['completed']=True;path.write_text(json.dumps(result,indent=2)+'\n')
            print(json.dumps({'rank':rank,'time_component_complete':True,'first_changes':[next((r['region'] for r in p['compare_to_first'] if not r['all_record_logical_SHA_equal']),None) for p in result['passes']]}),flush=True)
    except BaseException as exc:result['failure']={'type':type(exc).__name__,'message':str(exc)};path.write_text(json.dumps(result,indent=2)+'\n');raise
    finally:destroy_distributed_env()


if __name__=='__main__':main()
