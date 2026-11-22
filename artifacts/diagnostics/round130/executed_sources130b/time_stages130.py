#!/usr/bin/env python3
"""Eight-rank original time-component decomposition: native/observed/restored, fixed ops.

No model50/49-step generation, no copy optimization, no numeric-policy promotion.
Only original FP32time parameters and the original residual guard are loaded.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import math
import os
from pathlib import Path

ROOT=Path('/tmp/h3_time13001')
CHECKPOINT=Path('/home/mseco/models/MiniMax-H3/FL2VA/transformer')


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--label',default='stages130a');args=parser.parse_args()
    assert args.label in ('stages130a','stages130b')
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
    from projection_observer130 import ProjectionObserver130
    from vllm.distributed.parallel_state import get_tp_group,get_tensor_model_parallel_world_size
    rank=int(os.environ['RANK']);torch.npu.set_device(rank);torch.set_num_threads(4)
    torch.npu.matmul.allow_hf32=False;torch.npu.conv.allow_hf32=False
    out=ROOT/args.label;assert out.is_dir();path=out/f'stages_rank{rank}130.json';assert not path.exists()
    init_distributed_environment(world_size=8,rank=rank,local_rank=rank,distributed_init_method='env://',backend='hccl')
    initialize_model_parallel(tensor_parallel_size=2,sequence_parallel_size=4,ulysses_degree=4,ring_degree=1,backend='hccl')
    pc=DiffusionParallelConfig(tensor_parallel_size=2,sequence_parallel_size=4,ulysses_degree=4,ring_degree=1)
    config=json.loads((CHECKPOINT/'config.json').read_text());arch=MiniMaxH3DiTArchConfig.from_mapping(config)
    od=OmniDiffusionConfig(model=str(CHECKPOINT.parent),model_class_name='MiniMaxH3Pipeline',dtype=torch.float16,
        tf_model_config=TransformerConfig.from_dict(config),parallel_config=pc,num_gpus=8)
    report={'scope':__doc__,'label':args.label,'rank':rank,'completed':False,'TP_actual':get_tensor_model_parallel_world_size(),
        'TP_group':list(get_tp_group().ranks),'source_sha256':{p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in ROOT.glob('*.py')},
        'no_model_generation_or_candidate_change':True,'native':[],'observed':[],'restored':[],'fixed_operand_repeats':[]}
    def publish():path.write_text(json.dumps(report,indent=2)+'\n')
    def compare(actual,baseline):
        actual=actual.detach().cpu().contiguous();baseline=baseline.detach().cpu().contiguous()
        assert actual.shape==baseline.shape and actual.dtype==baseline.dtype==torch.float32
        different=int((actual.view(torch.int32)!=baseline.view(torch.int32)).sum())
        delta=actual.double()-baseline.double();den=float(baseline.double().square().sum().sqrt().clamp_min(1e-30))
        return {'shape':list(actual.shape),'different_FP32_words':different,'all_words_equal':different==0,
            'max_abs_difference':float(delta.abs().max()),'relative_RMS_error':float(delta.square().sum().sqrt())/den}
    try:
        assert report['TP_actual']==2
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
            report['input']=fingerprint(timestep,torch)
            report['parameters_before']={name:fingerprint(p,torch) for name,p in holder.named_parameters()}
            original=[];local_inputs=[];stage_ref=None;retained=None
            original_mm=torch.mm;original_add=torch.Tensor.add_
            target=holder.time_embedder.proj_out;old_method=target.quant_method;old_apply=old_method.apply
            def capture_local(layer,x,bias=None):
                # Copy on consumer stream, before rowparallel's later inplace reduction changes output.
                saved_input=x.detach().clone();saved_bias=None if bias is None else bias.detach().clone()
                output=old_apply(layer,x,bias)
                original.append(output.detach().cpu().contiguous().clone());local_inputs.append((saved_input,saved_bias))
                return output
            old_method.apply=capture_local
            final_ref=None
            for repeat in range(4):
                output=holder.time_embedder(timestep);torch.npu.synchronize()
                if final_ref is None:final_ref=output.detach().cpu().contiguous().clone();local_ref=original[-1].clone()
                report['native'].append({'repeat':repeat,'postTP':compare(output,final_ref),'local_beforeTP':compare(original[-1],local_ref)})
            fixed_input,fixed_bias=local_inputs[0];report['fixed_proj_out_input']=fingerprint(fixed_input,torch)
            old_method.apply=old_apply
            for repeat in range(4):
                observer=ProjectionObserver130(torch,retain_operands=repeat==0)
                with observer.observe():local_output=old_apply(target,fixed_input,fixed_bias)
                assert torch.mm is original_mm and torch.Tensor.add_ is original_add
                expected_ops=[('mm',0),('mm',1),('add_',0),('mm',2),('add_',1),('mm',3),('add_',2)]
                if fixed_bias is not None:expected_ops.append(('add_',3))
                expected_alpha=[2.0**-12,2.0**-12,2.0**-24]+([1.0] if fixed_bias is not None else [])
                report['actual_local_bias_present']=fixed_bias is not None
                report['last_observed_sequence']=[(r['op'],r['index']) for r in observer.rows]
                assert report['last_observed_sequence']==expected_ops, 'Unexpected original op interleaving; do not call incomplete trace a proof'
                assert [r['kwargs'].get('alpha',1.0) for r in observer.rows if r['op']=='add_']==expected_alpha
                if stage_ref is None:stage_ref=observer;observed_ref=local_output.detach().cpu().contiguous().clone();retained=observer.saved_operands
                report['observed'].append({'repeat':repeat,'stages':observer.report(stage_ref),'root_local_output':compare(local_output,observed_ref),
                    'original_vs_observed_output':compare(local_output,local_ref),'capture_sync_is_diagnostic_not_lossless_fix':True})
            for index,(op,a,b,kwargs) in enumerate(retained):
                baseline=None;rows=[]
                before_a=fingerprint(a,torch);before_b=fingerprint(b,torch)
                for repeat in range(5):
                    if op=='mm':output=original_mm(a,b,**kwargs)
                    else:output=original_add(a.clone(),b,**kwargs)
                    cpu=output.detach().cpu().contiguous().clone()
                    if baseline is None:baseline=cpu.clone()
                    rows.append(compare(cpu,baseline))
                assert normalized_fingerprint(before_a)==normalized_fingerprint(fingerprint(a,torch))
                assert normalized_fingerprint(before_b)==normalized_fingerprint(fingerprint(b,torch))
                report['fixed_operand_repeats'].append({'op':op,'sequence_index':index,'shape_a':list(a.shape),'shape_b':list(b.shape),'kwargs':{k:str(v) for k,v in kwargs.items()},
                    'same_inputs_before_after':True,'repeats':rows})
            # Original no-stage-hook control after restoring all process functions.
            old_method.apply=capture_local
            for repeat in range(4):
                output=holder.time_embedder(timestep);torch.npu.synchronize()
                report['restored'].append({'repeat':repeat,'postTP':compare(output,final_ref),'local_beforeTP':compare(original[-1],local_ref)})
            old_method.apply=old_apply
            report['parameters_after']={name:fingerprint(p,torch) for name,p in holder.named_parameters()}
            assert normalized_fingerprint(report['parameters_before'])==normalized_fingerprint(report['parameters_after'])
            report['original_parameter_words_unchanged']=True;report['global_functions_restored']=torch.mm is original_mm and torch.Tensor.add_ is original_add
            report['completed']=True;publish()
            print(json.dumps({'rank':rank,'completed':True,'native_diff':[r['local_beforeTP']['different_FP32_words'] for r in report['native']],
                'first_stage_changes':[next((f"{s['op']}:{s['index']}" for s in row['stages'] if not s['output']['all_raw_words_equal']),None) for row in report['observed']],
                'fixed_op_repeat_different':[any(not r['all_words_equal'] for r in row['repeats']) for row in report['fixed_operand_repeats']]}),flush=True)
    except BaseException as exc:report['failure']={'type':type(exc).__name__,'message':str(exc)};publish();raise
    finally:destroy_distributed_env()


if __name__=='__main__':main()
