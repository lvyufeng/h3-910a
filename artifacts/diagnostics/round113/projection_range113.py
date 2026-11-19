#!/usr/bin/env python3
"""Diagnose genuine Half condition-projection overflow, never redefine the oracle.

Power-of-two controls preserve the original Half weights and keep F.linear operands
and its output Half. Float rescaling is diagnostic only, not a replacement model,
Float-weight teacher, hidden conditioning cache, clamp or tolerance waiver.
"""
from pathlib import Path
import argparse
import hashlib
import json


def file_sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def bits_sha(x):
    import torch
    return hashlib.sha256(x.detach().cpu().contiguous().view(torch.uint8).numpy().tobytes()).hexdigest()


def stats(x):
    import torch
    y=x.detach().cpu().contiguous();finite=torch.isfinite(y)
    return {'shape':list(y.shape),'dtype':str(y.dtype),'sha256':bits_sha(y),'finite':bool(finite.all()),
            'NaN_count':int(torch.isnan(y).sum()),'positive_Inf_count':int(torch.isposinf(y).sum()),
            'negative_Inf_count':int(torch.isneginf(y).sum()),'nonfinite_row_count':int((~finite).any(-1).sum()),
            'max_abs_finite':float(y[finite].abs().max())if bool(finite.any())else None}


def main():
    args=argparse.ArgumentParser(description=__doc__)
    args.add_argument('--root',type=Path,required=True);args.add_argument('--device',choices=['cpu','npu'],required=True)
    flags=args.parse_args();root=flags.root.resolve(strict=True)
    import torch
    import torch.nn.functional as F
    from safetensors import safe_open
    if flags.device=='npu':
        import torch_npu
        torch.npu.set_device(0)
    torch.set_num_threads(16 if flags.device=='cpu'else 2)
    filename=root/('projection_range_%s113.json'%flags.device);assert not filename.exists()
    expected=json.loads((root/'expected_half_parameters111.json').read_text())
    inputs=json.loads((root/'expected_payload_inputs111.json').read_text())['inputs']
    checkpoint=Path('/home/mseco/models/MiniMax-H3-native-fp16-audit53/FL2VA/transformer')
    assert file_sha(checkpoint/'config.json')==expected['checkpoint_config_SHA256']
    assert file_sha(checkpoint/'model.safetensors.index.json')==expected['checkpoint_index_SHA256']
    params={}
    with safe_open(str(checkpoint/'model-00001-of-00013.safetensors'),framework='pt',device='cpu')as f:
        for name in('condition_proj.weight','condition_proj.bias'):
            value=f.get_tensor(name);assert value.dtype==torch.float16
            assert bits_sha(value)==expected['parameters'][name]['reference_sha256']
            params[name]=value.to(flags.device)
    w=params['condition_proj.weight'];b=params['condition_proj.bias'];parameter_before={name:bits_sha(p)for name,p in params.items()}
    original_module=None
    if flags.device=='npu':
        from vllm_omni.diffusion.distributed.parallel_state import init_distributed_environment,initialize_model_parallel
        from vllm.model_executor.layers.linear import ColumnParallelLinear
        init_distributed_environment(world_size=1,rank=0,local_rank=0,backend='hccl')
        initialize_model_parallel(sequence_parallel_size=1,ulysses_degree=1,tensor_parallel_size=1,backend='hccl')
        with torch.device('cpu'):
            original_module=ColumnParallelLinear(5120,5376,bias=True,gather_output=True,params_dtype=torch.float16,quant_config=None,prefix='condition_proj')
        original_module.to('npu')
        with torch.no_grad():
            original_module.weight.copy_(w);original_module.bias.copy_(b)
        assert bits_sha(original_module.weight)==parameter_before['condition_proj.weight']
        assert bits_sha(original_module.bias)==parameter_before['condition_proj.bias']
        original_module.eval()
    result={'scope':__doc__,'completed':False,'device':flags.device,'original_Half_parameter_SHAs':parameter_before,
            'original_half_only_Flinear':True,'no_Float_Linear_weights':True,'no_BF16_gold_or_clamp':True,
            'no_full_encoder_or_conditioning_acceptance':True,'full_encoder_precision_accepted':False,
            'whole_goal_complete':False,'numerical_tolerance':None,'cases':{}}
    with torch.inference_mode():
        for name,pin in inputs.items():
            assert file_sha(pin['path'])==pin['file_SHA256']
            payload=torch.load(pin['path'],map_location='cpu',weights_only=True)
            x=payload[pin['tensor_key']];assert bits_sha(x)==pin['tensor_SHA256']
            assert x.shape==(1,2043,5120)and x.dtype==torch.float16 and bool(torch.isfinite(x).all())
            x=x.squeeze(0).contiguous().to(flags.device);x_before=bits_sha(x)
            original=F.linear(x,w,b)if original_module is None else original_module(x)[0]
            repeat=F.linear(x,w,b)if original_module is None else original_module(x)[0]
            assert bits_sha(original)==bits_sha(F.linear(x,w,b))
            assert bits_sha(original)==bits_sha(repeat)
            row={'input':stats(x),'original_unmodified_Half_projection':stats(original),'original_repeat_bits_equal':True,
                 'range_controls':{},'controls_are_NOT_original_four_stage_reference':True}
            original_cpu=original.cpu();original_finite=torch.isfinite(original_cpu)
            for exponent in(1,2,3,4):
                scale=2**exponent;factor=torch.tensor(1/scale,dtype=torch.float16,device=flags.device)
                scaled_x=x*factor;scaled_b=b*factor
                projected=F.linear(scaled_x,w,scaled_b)
                restored=projected.float()*scale
                projected_cpu=projected.cpu();restored_cpu=restored.cpu()
                exact_input=(scaled_x.float()*scale==x.float())
                exact_bias=(scaled_b.float()*scale==b.float())
                same=original_finite&torch.isfinite(restored_cpu)
                diff=(restored_cpu[same].double()-original_cpu[same].double())
                denom=original_cpu[same].double().square().mean().sqrt().item()if bool(same.any())else None
                recast=restored_cpu.to(torch.float16)
                matching_inf_threshold=(torch.isposinf(original_cpu)&torch.isposinf(recast))|(torch.isneginf(original_cpu)&torch.isneginf(recast))
                row['range_controls'][str(scale)]={'scaled_Half_projection':stats(projected),'Float_rescaled_diagnostic':stats(restored),
                    'Half_input_values_changed_by_divide_restore':int((~exact_input).sum()),
                    'Half_bias_values_changed_by_divide_restore':int((~exact_bias).sum()),
                    'all_original_nonfinite_explained_by_rescaled_Half_overflow':bool(matching_inf_threshold.sum()==(~original_finite).sum())and bool(torch.isfinite(restored_cpu).all()),
                    'finite_original_elements_compared':int(same.sum()),
                    'finite_original_Half_recast_bits_equal':bool(torch.equal(original_cpu[same].view(torch.uint16),recast[same].view(torch.uint16))),
                    'finite_elements_relativeRMS':diff.square().mean().sqrt().item()/max(denom,1e-30)if denom is not None else None,
                    'finite_elements_max_abs_error':diff.abs().max().item()if diff.numel()else None,
                    'not_an_oracle_or_model_fix':True}
            assert bits_sha(x)==x_before and file_sha(pin['path'])==pin['file_SHA256']
            result['cases'][name]=row
    assert {name:bits_sha(p)for name,p in params.items()}==parameter_before
    result['original_parameters_inputs_and_capture_files_unchanged']=True;result['completed']=True
    with filename.open('x')as f:f.write(json.dumps(result,indent=2)+'\n')
    if original_module is not None:
        from vllm_omni.diffusion.distributed.parallel_state import destroy_model_parallel
        destroy_model_parallel()
        if torch.distributed.is_initialized():torch.distributed.destroy_process_group()
    print(json.dumps({'completed':True,'receipt':str(filename),'original_Half_overflow_counts':{name:row['original_unmodified_Half_projection']for name,row in result['cases'].items()}},indent=2))


if __name__=='__main__':main()
