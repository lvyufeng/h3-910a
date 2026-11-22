#!/usr/bin/env python3
"""Isolated deterministic-control probe of original Half->FP32 GEMM operands.

No model generation/copy candidate; a stable but different result is not adopted.
Standard torch deterministic flag changes only this private test process and is
restored before exit. Original FP32time weights and residual guard are unchanged.
"""
from __future__ import annotations
import hashlib
import json
import os
from pathlib import Path
import time

ROOT=Path('/tmp/h3_time13001')
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
    from fingerprints129 import fingerprint,normalized_fingerprint
    from vllm.distributed.parallel_state import get_tp_group,get_tensor_model_parallel_world_size
    rank=int(os.environ['RANK']);torch.npu.set_device(rank);torch.set_num_threads(4)
    torch.npu.matmul.allow_hf32=False;torch.npu.conv.allow_hf32=False
    path=ROOT/f'deterministic_rank{rank}130.json';assert not path.exists()
    initial_flag=torch.are_deterministic_algorithms_enabled();initial_warn=torch.is_deterministic_algorithms_warn_only_enabled()
    init_distributed_environment(world_size=8,rank=rank,local_rank=rank,distributed_init_method='env://',backend='hccl')
    initialize_model_parallel(tensor_parallel_size=2,sequence_parallel_size=4,ulysses_degree=4,ring_degree=1,backend='hccl')
    pc=DiffusionParallelConfig(tensor_parallel_size=2,sequence_parallel_size=4,ulysses_degree=4,ring_degree=1)
    config=json.loads((CHECKPOINT/'config.json').read_text());arch=MiniMaxH3DiTArchConfig.from_mapping(config)
    od=OmniDiffusionConfig(model=str(CHECKPOINT.parent),model_class_name='MiniMaxH3Pipeline',dtype=torch.float16,
        tf_model_config=TransformerConfig.from_dict(config),parallel_config=pc,num_gpus=8)
    report={'scope':__doc__,'rank':rank,'completed':False,'TP_actual':get_tensor_model_parallel_world_size(),'TP_group':list(get_tp_group().ranks),
        'initial_flag':initial_flag,'initial_warn_only':initial_warn,'cases':[],'candidate_policy_adopted':False,
        'source_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    def compare(a,b):
        assert a.shape==b.shape and a.dtype==b.dtype==torch.float32
        delta=a.double()-b.double();den=float(b.double().square().sum().sqrt().clamp_min(1e-30))
        count=int((a.reshape(-1).view(torch.int32)!=b.reshape(-1).view(torch.int32)).sum())
        return {'shape':list(a.shape),'all_words_equal':count==0,'different_FP32_words':count,
            'max_abs_difference':float(delta.abs().max()),'relative_RMS_error':float(delta.square().sum().sqrt())/den}
    try:
        with set_current_diffusion_config(od),set_forward_context(omni_diffusion_config=od),torch.no_grad():
            class Container(torch.nn.Module):pass
            holder=Container();holder.time_embedder=MiniMaxH3TimeEmbedder(arch,prefix='time_embedder')
            parameters=dict(holder.named_parameters());index=json.loads((CHECKPOINT/'model.safetensors.index.json').read_text())['weight_map']
            for filename in sorted({index[name] for name in parameters}):
                with safe_open(str(CHECKPOINT/filename),framework='pt',device='cpu') as file:
                    for name,parameter in parameters.items():
                        if index[name]==filename:parameter.weight_loader(parameter,file.get_tensor(name))
            install_time_projection_guard123(holder);holder.to(f'npu:{rank}').eval()
            data=torch.load(f'/tmp/h3_overlap12801/native_worker_input_step25_rank{rank}.pt',map_location='cpu',weights_only=True)
            timestep=data['unique_timesteps'].to(f'npu:{rank}');del data
            target=holder.time_embedder.proj_out;method=target.quant_method;original_apply=method.apply;fixed={}
            def capture(layer,x,bias=None):
                fixed['x']=x.detach().clone();fixed['bias']=None if bias is None else bias.detach().clone()
                return original_apply(layer,x,bias)
            method.apply=capture;holder.time_embedder(timestep);torch.npu.synchronize();method.apply=original_apply
            x=fixed['x'];bias=fixed['bias'];xh=x.half();xl=((x-xh.float())*4096.).half()
            wh=method._weight_high_transposed;wl=method._weight_low_transposed
            pairs=[('mm0',xh,wh),('mm1',xh,wl),('mm2',xl,wh),('mm3',xl,wl)]
            report['input_before']=fingerprint(x,torch);report['parameters_before']={name:fingerprint(p,torch) for name,p in holder.named_parameters()}
            baseline={}
            for name,flag in [('native',initial_flag),('deterministic',True),('restored',initial_flag)]:
                torch.npu.synchronize();torch.use_deterministic_algorithms(flag,warn_only=False)
                case={'mode':name,'flag_actually_enabled':torch.are_deterministic_algorithms_enabled(),'operations':[],'errors':[]}
                for op,a,b in pairs+[('full_original_guard',None,None)]:
                    reference=None;rows=[];times=[]
                    before_a=fingerprint(a,torch) if a is not None else None;before_b=fingerprint(b,torch) if b is not None else None
                    for repeat in range(10):
                        try:
                            torch.npu.synchronize();started=time.perf_counter()
                            output=torch.mm(a,b,out_dtype=torch.float32) if a is not None else original_apply(target,x,bias)
                            cpu=output.detach().cpu().contiguous().clone();torch.npu.synchronize();times.append(time.perf_counter()-started)
                        except BaseException as exc:
                            case['errors'].append({'op':op,'type':type(exc).__name__,'message':str(exc)});break
                        if reference is None:reference=cpu.clone()
                        if op not in baseline:baseline[op]=cpu.clone()
                        rows.append({'repeat':repeat,'self_repeat':compare(cpu,reference),'versus_native_first':compare(cpu,baseline[op])})
                    unchanged=(before_a==fingerprint(a,torch) and before_b==fingerprint(b,torch)) if a is not None else True
                    assert unchanged,'The isolated operand values/state changed'
                    case['operations'].append({'op':op,'actual_repeats':len(rows),'all_same_input_self_outputs_bit_identical':bool(rows) and all(r['self_repeat']['all_words_equal'] for r in rows),
                        'results':rows,'times_include_explicit_D2H_sync_s':times,'operands_unchanged':unchanged})
                report['cases'].append(case)
                path.write_text(json.dumps(report,indent=2)+'\n')
            torch.use_deterministic_algorithms(initial_flag,warn_only=initial_warn)
            report['restored_original_deterministic_flags']=torch.are_deterministic_algorithms_enabled()==initial_flag and torch.is_deterministic_algorithms_warn_only_enabled()==initial_warn
            report['parameters_after']={name:fingerprint(p,torch) for name,p in holder.named_parameters()}
            assert normalized_fingerprint(report['parameters_before'])==normalized_fingerprint(report['parameters_after'])
            report['input_after']=fingerprint(x,torch);assert report['input_before']==report['input_after']
            report['completed']=True;path.write_text(json.dumps(report,indent=2)+'\n')
            print(json.dumps({'rank':rank,'completed':True,'cases':[{'mode':r['mode'],'stable_ops':[op['op'] for op in r['operations'] if op['all_same_input_self_outputs_bit_identical']],'errors':r['errors']} for r in report['cases']]}),flush=True)
    except BaseException as exc:report['failure']={'type':type(exc).__name__,'message':str(exc)};path.write_text(json.dumps(report,indent=2)+'\n');raise
    finally:
        torch.use_deterministic_algorithms(initial_flag,warn_only=initial_warn)
        destroy_distributed_env()


if __name__=='__main__':main()
