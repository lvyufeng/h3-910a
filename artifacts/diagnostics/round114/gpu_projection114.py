#!/usr/bin/env python3
"""4090 original FP16/BF16 projection references, exact same weights and captured inputs."""
from pathlib import Path
import hashlib,json


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def info(t):
    import torch
    x=t.detach().cpu().contiguous();finite=torch.isfinite(x)
    return {'dtype':str(x.dtype),'shape':list(x.shape),'sha256':hashlib.sha256(x.view(torch.uint8).numpy().tobytes()).hexdigest(),
            'finite':bool(finite.all()),'NaN_count':int(torch.isnan(x).sum()),'Inf_count':int(torch.isinf(x).sum()),
            'max_abs_finite':float(x[finite].abs().max())if bool(finite.any())else None}


def metric(candidate,reference):
    import torch
    a=candidate.to('cuda:0');b=reference.to('cuda:0');finite=torch.isfinite(a)&torch.isfinite(b)
    aa=a[finite].double();bb=b[finite].double()
    return {'candidate':info(candidate),'reference':info(reference),'compared_finite_elements':int(finite.sum()),
            'excluded_nonfinite_reference_elements':int((~torch.isfinite(b)).sum()),
            'all_elements_comparable':bool(finite.all()),
            'finite_relativeRMS':float((aa-bb).square().mean().sqrt()/bb.square().mean().sqrt())if aa.numel()else None,
            'finite_max_abs_error':float((aa-bb).abs().max())if aa.numel()else None,
            'BF16_reference_not_cast_to_FP16':reference.dtype==torch.bfloat16,
            'no_numerical_tolerance_or_pass':True}


def main():
    import torch
    from safetensors import safe_open
    from safetensors.torch import save_file
    root=Path('/tmp/h3_gpu114_01');data=root/'projection';dest=root/'GPU_projection114.json';assert not dest.exists()
    assert torch.cuda.is_available()and torch.cuda.is_bf16_supported()
    assert torch.cuda.get_device_name(0)=='NVIDIA GeForce RTX 4090'
    torch.backends.cuda.matmul.allow_tf32=False
    torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction=False
    torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction=False
    manifest=json.loads((data/'package_manifest.json').read_text());assert manifest['completed']and manifest['kind']=='projection'
    raw={}
    for name,row in manifest['parameters'].items():
        p=data/row['packed_file'];assert sha(p)==manifest['files'][p.name]['sha256']
        with safe_open(str(p),framework='pt',device='cpu')as f:t=f.get_tensor(name)
        assert info(t)['sha256']==row['original_BF16_SHA256']and t.dtype==torch.bfloat16
        raw[name]=t
    npu_manifest=json.loads((data/'NPU_projection_manifest.json').read_text())
    assert sha(data/'NPU_original_projection.safetensors')==npu_manifest['file_SHA256']
    with safe_open(str(data/'NPU_original_projection.safetensors'),framework='pt',device='cpu')as f:npu={name:f.get_tensor(name)for name in f.keys()}
    result={'scope':__doc__,'completed':False,'torch':torch.__version__,'cuda':torch.version.cuda,'hardware':'RTX4090',
            'original_BF16_direct_weights_not_FP16_roundtrip':True,'TF32_and_reduced_precision_reduction_disabled':True,
            'no_clamp_Float32_Linear_teacher_or_fake_norm':True,'not_full_encoder_precision_acceptance':True,
            'whole_goal_complete':False,'numerical_tolerance':None,'cases':{}}
    saved={}
    with torch.inference_mode():
        for label,dtype in(('FP16',torch.float16),('BF16',torch.bfloat16)):
            params={}
            for name,t in raw.items():
                converted=t.to(dtype=dtype)
                if label=='FP16':assert info(converted)['sha256']==manifest['parameters'][name]['FP16_converted_SHA256']
                params[name]=converted.to('cuda:0')
            result['cases'][label]={}
            for name,pin in manifest['precision_inputs'].items():
                p=data/pin['relative_path'];assert sha(p)==pin['file_SHA256']
                source=torch.load(p,map_location='cpu',weights_only=True)[pin['tensor_key']]
                assert info(source)['sha256']==pin['tensor_SHA256']and source.dtype==torch.float16
                x=source.squeeze(0).to('cuda:0',dtype=dtype)
                y=torch.nn.functional.linear(x,params['condition_proj.weight'],params['condition_proj.bias'])
                repeat=torch.nn.functional.linear(x,params['condition_proj.weight'],params['condition_proj.bias'])
                assert torch.equal(y.view(torch.uint16),repeat.view(torch.uint16))
                assert info(npu[name])['sha256']==npu_manifest['cases'][name]['sha256']
                row={'GPU_output':info(y),'repeat_bits_equal':True,'source_input':info(source),'actual_input_dtype':str(x.dtype),
                     'NPU_FP16_vs_GPU':metric(npu[name],y),
                     'GPU_Inf_NPU_saturated_to_maxHalf_count':int((torch.isinf(y)&(npu[name].to('cuda:0').abs()==65504)).sum()),
                     'not_a_full_encoder_result':True}
                result['cases'][label][name]=row;saved[label+'_'+name]=y.cpu().contiguous()
    output=root/'GPU_projection_outputs114.safetensors';save_file(saved,str(output));result['output_file_SHA256']=sha(output)
    result['completed']=True
    with dest.open('x')as f:f.write(json.dumps(result,indent=2)+'\n')
    print(json.dumps({'completed':True,'GPU_reference_projection':{label:{name:row['GPU_output']for name,row in cases.items()}for label,cases in result['cases'].items()}},indent=2))


if __name__=='__main__':main()
