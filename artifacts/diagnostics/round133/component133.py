#!/usr/bin/env python3
"""Original final-AdaLN small component: same t_emb, Half input/weight/bias, real TP2.

No new arithmetic, no fullmodel/sampler, no copy candidate or tolerance adoption.
The input must match R132 actual formal-worker final-AdaLN input fingerprint.
"""
import hashlib
import json
import os
from pathlib import Path
import time

ROOT=Path('/tmp/h3_adaln13301')
CHECKPOINT=Path('/home/mseco/models/MiniMax-H3/FL2VA/transformer')


def main():
    import numpy as np
    import torch
    import torch_npu
    from safetensors import safe_open
    from vllm_omni.diffusion.distributed.parallel_state import init_distributed_environment,initialize_model_parallel,destroy_distributed_env,get_sp_group,get_sequence_parallel_world_size,get_ulysses_parallel_world_size
    from vllm_omni.diffusion.data import OmniDiffusionConfig,DiffusionParallelConfig,TransformerConfig
    from vllm.config import set_current_vllm_config
    from vllm_omni.diffusion.vllm_config import create_diffusion_vllm_config
    from cohort128 import Cohort128
    from vllm_omni.diffusion.config import set_current_diffusion_config
    from vllm_omni.diffusion.forward_context import set_forward_context
    from vllm.distributed.parallel_state import get_tp_group,get_tensor_model_parallel_world_size
    from h3_replay123.transformer import MiniMaxH3TimeEmbedder,MiniMaxH3AdalnProj,MiniMaxH3DiTArchConfig
    from fp32_time_residual_guard123 import install_time_projection_guard123
    from fingerprints129 import fingerprint,normalized_fingerprint
    from callable_info133 import callable_info
    rank=int(os.environ['RANK']);torch.npu.set_device(rank);torch.set_num_threads(4)
    torch.npu.matmul.allow_hf32=False;torch.npu.conv.allow_hf32=False
    initial=torch.are_deterministic_algorithms_enabled();warn=torch.is_deterministic_algorithms_warn_only_enabled();assert not initial
    path=ROOT/f'component_rank{rank}133.json';assert not path.exists()
    init_distributed_environment(world_size=8,rank=rank,local_rank=rank,distributed_init_method='env://',backend='hccl')
    initialize_model_parallel(tensor_parallel_size=2,sequence_parallel_size=4,ulysses_degree=4,ring_degree=1,backend='hccl')
    config=json.loads((CHECKPOINT/'config.json').read_text());arch=MiniMaxH3DiTArchConfig.from_mapping(config)
    pc=DiffusionParallelConfig(tensor_parallel_size=2,sequence_parallel_size=4,ulysses_degree=4,ring_degree=1)
    od=OmniDiffusionConfig(model=str(CHECKPOINT.parent),model_class_name='MiniMaxH3Pipeline',dtype=torch.float16,
        tf_model_config=TransformerConfig.from_dict(config),parallel_config=pc,num_gpus=8)
    report={'scope':__doc__,'rank':rank,'completed':False,'TP_actual':get_tensor_model_parallel_world_size(),'TP_group':list(get_tp_group().ranks),
        'SP_actual':get_sequence_parallel_world_size(),'SP_group':list(get_sp_group().ranks),'Ulysses_actual':get_ulysses_parallel_world_size(),
        'module_repeats':[],'local_repeats':[],'restored_module_repeats':[],
        'source_sha256':{p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in ROOT.glob('*.py')},'new_arithmetic_or_flag_baseline_adopted':False}
    def publish():path.write_text(json.dumps(report,indent=2)+'\n')
    cohort=Cohort128(torch,prefix='R133:component',timeout_s=120)
    native_vllm=cohort.run('build_native_worker_config',lambda:create_diffusion_vllm_config(torch.device(f'npu:{rank}'),od))
    report['native_config_factory']=callable_info(create_diffusion_vllm_config)
    report['requested_linear_backend']=str(native_vllm.kernel_config.linear_backend)
    report['VLLM_BATCH_INVARIANT_environment']=os.environ.get('VLLM_BATCH_INVARIANT','unset')
    report['context_scope']='Native worker config factory/default backend, not an independently recorded R132 final-AdaLN runtime config'
    def compare(actual,reference):
        cpu=actual.detach().cpu().contiguous();ref=reference.detach().cpu().contiguous()
        assert cpu.shape==ref.shape and cpu.dtype==ref.dtype
        if cpu.dtype==torch.float16:
            changed=int(np.count_nonzero(cpu.numpy().view(np.uint16)!=ref.numpy().view(np.uint16)))
            x=cpu.float().double();y=ref.float().double();delta=x-y
        else:
            assert cpu.dtype==torch.float32;changed=int(np.count_nonzero(cpu.numpy().view(np.uint32)!=ref.numpy().view(np.uint32)))
            x=cpu.double();y=ref.double();delta=x-y
        den=float(y.square().sum().sqrt().clamp_min(1e-30))
        coords=[]
        if changed:
            indices=np.argwhere(cpu.numpy().view(np.uint16 if cpu.dtype==torch.float16 else np.uint32)!=ref.numpy().view(np.uint16 if cpu.dtype==torch.float16 else np.uint32))
            coords=indices[:16].tolist()
        return {'shape':list(cpu.shape),'dtype':str(cpu.dtype),'total_words':cpu.numel(),'different_words':changed,
            'all_words_equal':changed==0,'max_abs_difference':float(delta.abs().max()),'relative_RMS_ratio':float(delta.square().sum().sqrt())/den,
            'first_changed_coordinates':coords,'complete_logical_words_compared':True}
    method=None;wrapper_installed=False;had_override=False;override=None
    try:
        with set_current_vllm_config(native_vllm),set_current_diffusion_config(od),set_forward_context(vllm_config=native_vllm,omni_diffusion_config=od),torch.no_grad():
            def construct():
                class Container(torch.nn.Module):pass
                holder=Container();holder.time_embedder=MiniMaxH3TimeEmbedder(arch,prefix='time_embedder')
                holder.final_layer=Container();holder.final_layer.adaln_proj=MiniMaxH3AdalnProj(arch,arch.final_adaln_out_features,None,
                    expand_ratio=2,modality_num=1,prefix='final_layer.adaln_proj')
                params=dict(holder.named_parameters());index=json.loads((CHECKPOINT/'model.safetensors.index.json').read_text())['weight_map']
                for name in params:assert name in index
                for filename in sorted({index[name] for name in params}):
                    with safe_open(str(CHECKPOINT/filename),framework='pt',device='cpu') as file:
                        for name,param in params.items():
                            if index[name]==filename:param.weight_loader(param,file.get_tensor(name))
                install_time_projection_guard123(holder);holder.to(f'npu:{rank}').eval()
                return holder,params
            holder,params=cohort.run('construct_original_small_components_all8',construct)
            final=holder.final_layer.adaln_proj;layer=final.linear;method=layer.quant_method;old=method.apply
            def verify_and_prepare():
                assert layer.weight.dtype==layer.bias.dtype==torch.float16
                assert layer.weight.shape==(5376,2688) and layer.bias.shape==(5376,)
                report['installed_dispatch']={'apply':callable_info(old),'gemm_impl':callable_info(getattr(method,'_gemm_impl',old)),
                    'scope':'Installed original Python callable, not an identified CANN kernel'}
                before={name:fingerprint(param,torch) for name,param in holder.named_parameters()};report['parameters_before']=before
                original=json.loads(Path(f'/tmp/h3_audio13201/audio_tail_rank{rank}132.json').read_text());assert original['completed']
                observed=original['actual_local_dispatch']['audio_out']['gemm_impl']
                assert report['installed_dispatch']['gemm_impl']==observed,'Standalone dispatch callable differs from the installed formal default'
                report['same_python_default_dispatch_as_R132_audio_head']=True
                for name,param in params.items():
                    if name.startswith('final_layer.adaln_proj'):
                        ref=original['model_before_state']['materialized_parameters_and_buffers'][name]
                        assert normalized_fingerprint(fingerprint(param,torch))==normalized_fingerprint(ref),('Original loaded final-AdaLN parameter differs',name)
                fixture=Path(f'/tmp/h3_overlap12801/native_worker_input_step25_rank{rank}.pt')
                expected=json.loads(Path(f'/tmp/h3_overlap12801/formal_replay_rank{rank}128.json').read_text())['input_sha256']
                with fixture.open('rb') as file:assert hashlib.file_digest(file,'sha256').hexdigest()==expected
                report['source_input_SHA_verified']=expected
                data=torch.load(fixture,map_location='cpu',weights_only=True)
                timestep=data['unique_timesteps'].to(f'npu:{rank}');del data
                torch.use_deterministic_algorithms(True,warn_only=False)
                assert torch.are_deterministic_algorithms_enabled()
                return before,original,timestep
            before,original,timestep=cohort.run('verify_source_params_dispatch_and_flag_before_TP',verify_and_prepare)
            t_emb=cohort.run('original_time_forward_all8',lambda:holder.time_embedder(timestep));torch.npu.synchronize()
            def verify_time():
                reference_trace=next(r for r in original['tail_traces'][0] if r['module']=='final_layer.adaln_proj' and r['boundary']=='input')
                ref=reference_trace['leaves']['args'][0]['full'];actual=normalized_fingerprint(fingerprint(t_emb,torch))
                assert actual==ref,('Reconstructed time embedding differs from actual R132 input',actual,ref)
                report['actual_formal_t_emb_SHA_matched']=True;report['t_emb']=fingerprint(t_emb,torch)
            cohort.run('verify_formal_t_emb_all8_before_Module',verify_time)
            local_holder={};fixed={};base_output=None;base_local=None;base_input=None
            had_override='apply' in method.__dict__;override=method.__dict__.get('apply')
            def capture(actual_layer,x,bias=None):
                assert actual_layer is layer and x.shape==(3,2688) and x.dtype==torch.float16 and bias.dtype==torch.float16
                cpu_input=x.detach().cpu().contiguous().clone()
                output=old(actual_layer,x,bias)
                local_holder['output']=output.detach().cpu().contiguous().clone()
                local_holder['input']=cpu_input
                if not fixed:fixed.update({'x':x.detach().clone(),'bias':bias.detach().clone()})
                return output
            method.apply=capture;wrapper_installed=True
            for repeat in range(32):
                output=cohort.run(f'original_Module_with_TP_gather_{repeat}',lambda:final(t_emb));torch.npu.synchronize();cpu=torch.cat([v.detach().cpu().contiguous() for v in output],dim=-1)
                if base_output is None:base_output=cpu.clone();base_local=local_holder['output'].clone();base_input=local_holder['input'].clone()
                report['module_repeats'].append({'repeat':repeat,'input':compare(local_holder['input'],base_input),
                    'local_before_TP_gather':compare(local_holder['output'],base_local),'post_gather_output':compare(cpu,base_output),
                    'shift':compare(cpu[:,:5376],base_output[:,:5376]),'scale':compare(cpu[:,5376:],base_output[:,5376:])})
            if had_override:method.apply=override
            else:delattr(method,'apply')
            wrapper_installed=False
            def verify_frozen():
                report['original_vs_frozen_input']=compare(fixed['x'],base_input);assert report['original_vs_frozen_input']['all_words_equal']
                report['actual_local_x_weight_bias_before']={'x':fingerprint(fixed['x'],torch),'w':fingerprint(layer.weight,torch),'bias':fingerprint(fixed['bias'],torch)}
            cohort.run('verify_frozen_original_input_before_local_replay',verify_frozen)
            first=None
            for repeat in range(128):
                output=cohort.run(f'original_local_apply_no_TP_{repeat}',lambda:old(layer,fixed['x'],fixed['bias']));cpu=output.detach().cpu().contiguous().clone()
                if first is None:first=cpu.clone()
                report['local_repeats'].append({'repeat':repeat,'output':compare(cpu,first),'versus_first_original_module_local_output':compare(cpu,base_local)})
            for repeat in range(4):
                result=cohort.run(f'restored_Module_original_TP_{repeat}',lambda:final(t_emb));cpu=torch.cat([v.detach().cpu().contiguous() for v in result],dim=-1)
                report['restored_module_repeats'].append({'repeat':repeat,'output':compare(cpu,base_output)})
            report['actual_local_x_weight_bias_after']={'x':fingerprint(fixed['x'],torch),'w':fingerprint(layer.weight,torch),'bias':fingerprint(fixed['bias'],torch)}
            assert report['actual_local_x_weight_bias_before']==report['actual_local_x_weight_bias_after']
            after={name:fingerprint(param,torch) for name,param in holder.named_parameters()};assert normalized_fingerprint(before)==normalized_fingerprint(after)
            report['original_parameters_unchanged']=True
            torch.use_deterministic_algorithms(initial,warn_only=warn);report['private_flags_restored_false']=not torch.are_deterministic_algorithms_enabled()
            cohort.run('all8_original_component_done',publish)
            report['completed']=True;publish()
            print(json.dumps({'rank':rank,'completed':True,'module_local_changes':sum(not r['local_before_TP_gather']['all_words_equal'] for r in report['module_repeats']),
                'module_postgather_changes':sum(not r['post_gather_output']['all_words_equal'] for r in report['module_repeats']),
                'fixed_local_changes':sum(not r['output']['all_words_equal'] for r in report['local_repeats'])}),flush=True)
    except BaseException as exc:
        report['failure']={'type':type(exc).__name__,'message':str(exc)};publish()
        from cohort128 import publish_fatal128
        publish_fatal128({'ok':False,'rank':rank,'phase':'component_outer_failure','pid':os.getpid(),**report['failure']})
        raise
    finally:
        if wrapper_installed and method is not None:
            if had_override:method.apply=override
            else:delattr(method,'apply')
        torch.use_deterministic_algorithms(initial,warn_only=warn)
        destroy_distributed_env()


if __name__=='__main__':main()
