#!/usr/bin/env python3
"""Actual original-Half NPU conditioning control; not a full encoder/DiT benchmark."""
from pathlib import Path
import argparse
import json
import time


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',type=Path,required=True)
    parser.add_argument('--cpu-root',type=Path)
    parser.add_argument('--construct-only',action='store_true')
    args=parser.parse_args();root=args.root.resolve(strict=True)
    from evidence_io111 import EvidenceWriter
    import construct_original111 as helpers
    import torch
    import torch_npu
    from vllm_omni.diffusion.distributed.parallel_state import init_distributed_environment,initialize_model_parallel,destroy_model_parallel
    filename='original_construct111.json'if args.construct_only else'original_NPU_conditioning111.json'
    writer=EvidenceWriter(root/filename)
    r={'scope':__doc__,'completed':False,'TP1_replicated_preSP_refiner':True,'not_full_encoder_or_co_residency':True,
       'full_encoder_precision_accepted':False,'whole_goal_complete':False,'numerical_tolerance':None}
    writer.write(r);handles=[];model=None;distributed=False
    try:
        r['original_sources']=helpers.verify_sources(root/'expected_original_sources111.json')
        r['runtime']={'torch':torch.__version__,'torch_npu':torch_npu.__version__}
        torch.npu.set_device(0);torch.set_num_threads(2)
        init_distributed_environment(world_size=1,rank=0,local_rank=0,backend='hccl')
        initialize_model_parallel(sequence_parallel_size=1,ulysses_degree=1,tensor_parallel_size=1,backend='hccl');distributed=True
        model,geometry=helpers.construct_original(root/'expected_half_parameters111.json',torch.device('npu',0));r['constructed_original_model']=geometry
        inputs=json.loads((root/'expected_payload_inputs111.json').read_text())['inputs'];before={};payload={}
        for name,row in inputs.items():
            assert helpers.file_sha(row['path'])==row['file_SHA256']
            x=torch.load(row['path'],map_location='cpu',weights_only=True)
            if isinstance(x,dict):
                assert row['tensor_key']in ('pre_final_norm','selected50')and row['tensor_key']in x
                if row['tensor_key']=='selected50':assert x['case']==name
                x=x[row['tensor_key']]
            info=helpers.tensor_info(x)
            assert info['shape']==[1,2043,5120]and info['dtype']=='torch.float16'and info['finite']and info['sha256']==row['tensor_SHA256']
            before[name]=info;payload[name]=x.squeeze(0).contiguous()
        r['actual_inputs_verified']=before
        if args.construct_only:
            r['no_model_forward_executed']=True;r['completed']=True;writer.write(r);return
        assert args.cpu_root is not None,'Require completed original CPU stage reference before NPU timing.'
        cpu_root=args.cpu_root.resolve(strict=True)
        # The root fills the filename schema only after reviewing the fresh CPU implementation.
        cpu_index=json.loads((cpu_root/'cpu_stage_index111.json').read_text())
        assert cpu_index['all_original_CPU_baseline_repeats_exact']
        cpu={}
        for name,entry in cpu_index['cases'].items():
            p=Path(entry['path']);assert p.parent==cpu_root and helpers.file_sha(p)==entry['file_SHA256']
            tensors=torch.load(p,map_location='cpu',weights_only=True)
            assert set(tensors)=={'projected','refiner0','refiner1','finalnorm'}
            for key,v in tensors.items():
                info=helpers.tensor_info(v);assert info['dtype']=='torch.float16'and info['shape']==[2043,5376]and info['finite']
                assert info['sha256']==entry['stages'][key]['sha256']
            cpu[name]=tensors
        assert set(cpu)==set(inputs)
        captured={}
        def hook(name):
            def save(module,unused,output):
                value=output[0]if isinstance(output,tuple)else output
                captured[name]=value.detach().clone()
            return save
        handles.append(model.condition_proj.register_forward_hook(hook('projected')))
        for i,block in enumerate(model.token_refiner.blocks):handles.append(block.register_forward_hook(hook('refiner%d'%i)))
        handles.append(model.token_refiner.final_norm.register_forward_hook(hook('finalnorm')))
        cu=torch.tensor([0,2043,2043],dtype=torch.int32,device='npu')
        r['cases']={}
        with torch.inference_mode():
            for name,rows in payload.items():
                x=rows.to('npu');captured.clear();first=model(x,cu);torch.npu.synchronize()
                first_stages={k:v.cpu().contiguous()for k,v in captured.items()}
                assert set(first_stages)=={'projected','refiner0','refiner1','finalnorm'}
                assert helpers.compare(first.cpu(),first_stages['finalnorm'])['raw_bits_equal']
                record={'same_input_CPU':{k:helpers.compare(v,cpu[name][k])for k,v in first_stages.items()},
                        'vs_CPU_original_input':{k:helpers.compare(v,cpu['CPU_original'][k])for k,v in first_stages.items()},
                        'input_vs_CPU_original':helpers.compare(rows,payload['CPU_original']),
                        'not_full_encoder_time':True,'actual_pre_existing_DiT_finalnorm_not_fake_encoder_norm':True}
                captured.clear();repeat=model(x,cu);torch.npu.synchronize()
                record['repeat_bits_equal']=all(helpers.compare(captured[k].cpu(),v)['raw_bits_equal']for k,v in first_stages.items())
                assert record['repeat_bits_equal']
                # Remove all capture hooks before conditioning-only clocks; reinstall for the next case.
                for h in handles:h.remove()
                handles.clear();captured.clear()
                warm=model(x,cu);torch.npu.synchronize();assert helpers.compare(warm.cpu(),first_stages['finalnorm'])['raw_bits_equal']
                clocks=[]
                for _ in range(5):
                    torch.npu.synchronize();start=time.perf_counter();out=model(x,cu);out_cpu=out.cpu();torch.npu.synchronize();clocks.append(time.perf_counter()-start)
                    assert helpers.compare(out_cpu,first_stages['finalnorm'])['raw_bits_equal']
                record['conditioning_only_wall_s_including_final_D2H']=clocks
                dest=root/('NPU_original_stages_%s111.pt'%name);assert not dest.exists();torch.save(first_stages,dest)
                record['untimed_stage_capture']={'path':str(dest),'file_SHA256':helpers.file_sha(dest),'stages':{k:helpers.tensor_info(v)for k,v in first_stages.items()},'never_stock_encoder_input':True}
                r['cases'][name]=record;writer.write(r)
                handles.append(model.condition_proj.register_forward_hook(hook('projected')))
                for i,block in enumerate(model.token_refiner.blocks):handles.append(block.register_forward_hook(hook('refiner%d'%i)))
                handles.append(model.token_refiner.final_norm.register_forward_hook(hook('finalnorm')))
                del x,first,repeat,warm,out,out_cpu,first_stages
        r['parameters_after']=helpers.parameter_info(model)
        r['original_parameters_unchanged']=r['parameters_after']==geometry['parameters'];assert r['original_parameters_unchanged']
        r['original_sources_after']=helpers.verify_sources(root/'expected_original_sources111.json')
        for name,row in inputs.items():assert helpers.file_sha(row['path'])==row['file_SHA256']
        r['original_input_files_unchanged']=True;r['completed']=True;writer.write(r)
    except BaseException as error:
        r['failure']={'type':type(error).__name__,'message':str(error)};writer.write(r);raise
    finally:
        for h in handles:h.remove()
        if model is not None:del model
        if distributed:destroy_model_parallel()
        if torch.distributed.is_initialized():torch.distributed.destroy_process_group()


if __name__=='__main__':main()
