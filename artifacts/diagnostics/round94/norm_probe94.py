#!/usr/bin/env python3
"""Actual Half raw24 RMSNorm5120 and Float-rsqrt Newton controls;no Linear teacher."""
from pathlib import Path
import json
ROOT=Path('/tmp/h3_round94_norm01')


def main():
    import torch
    import torch_npu
    import full_row_fp32_fixed84 as previous
    from safetensors import safe_open
    from transformers import Qwen3VLConfig
    from transformers.models.qwen3_vl import modeling_qwen3_vl as hf
    torch.set_num_threads(2);torch.npu.set_device(0);previous.ROOT=ROOT
    r={'scope':__doc__,'completed':False,'not_full_encoder_timing':True,'no_Linear_teacher_or_BF_outputcasts':True,
       'original_Half_gamma_and_Half_inputs':True,'no_CANN_compile':True,'CPU93_concurrent_no_speed_claim':True,
       'whole_goal_complete':False,'full_encoder_precision_accepted':False,'numerical_tolerance':None,
       'source_SHA256':previous.file_sha(Path(__file__)),'cases':{}}
    try:
        assert previous.file_sha(Path(hf.__file__))=='fcb5571ef95e4a6679af866ef9ef4dfa5cee017f9f91453daf28c7804c7a50fe'
        candidate=previous.load_candidate(r);reference=json.loads((previous.CPU_ROOT/'reference.json').read_text())
        assert previous.file_sha(previous.CPU_ROOT/'reference.json')=='44acac61f9b4d9bc8afc91632392502ccaa29dee7bfab951e2b15902a0be74c9'
        cfg=Qwen3VLConfig.from_pretrained(str(previous.MODEL),trust_remote_code=False,local_files_only=True)
        index=json.loads((previous.MODEL/'model.safetensors.index.json').read_text())['weight_map']
        weight_key='model.language_model.layers.25.input_layernorm.weight'
        with safe_open(str(previous.MODEL/index[weight_key]),framework='pt',device='cpu')as f:gamma=f.get_tensor(weight_key).to(torch.float16)
        assert previous.tensor_info(gamma)['sha256']==reference['parameter_conversion'][weight_key[len('model.'):]]['reference_sha256']
        CPU=hf.Qwen3VLTextRMSNorm(5120,eps=cfg.text_config.rms_norm_eps)
        CPU.weight=torch.nn.Parameter(gamma.clone(),requires_grad=False)
        NPU=candidate.MiniMaxH3Qwen3VLRMSNorm(5120,eps=cfg.text_config.rms_norm_eps,dtype=torch.float16).to('npu:0')
        with torch.no_grad():NPU.weight.copy_(gamma.to('npu:0'))
        assert CPU.weight.dtype==NPU.weight.dtype==torch.float16
        inputs={'CPU_raw24':torch.load(previous.CPU_ROOT/'FP16_layer24.pt',map_location='cpu',weights_only=True)}
        bundle=torch.load('/tmp/h3_round92_case02/NPU_raw24_inputs92.pt',map_location='cpu',weights_only=True)
        inputs['corrected_vision_native_text']=bundle['inputs']['corrected_vision_native_text']
        before=previous.tensor_info(NPU.weight)['sha256']
        for name,cpu in inputs.items():
            value=cpu.to('npu:0');input_SHA=previous.tensor_info(value)['sha256']
            def refined(iterations):
                xf=value.float();variance=xf.square().mean(-1,keepdim=True)
                v=variance+NPU.variance_epsilon;y=torch.rsqrt(v)
                for _ in range(iterations):y=y*(1.5-0.5*v*y*y)
                return (xf*y).to(torch.float16)*NPU.weight
            with torch.inference_mode():
                gold=CPU(cpu);v_CPU=cpu.float().square().mean(-1,keepdim=True)+CPU.variance_epsilon
                v_NPU=value.float().square().mean(-1,keepdim=True)+NPU.variance_epsilon
                y_CPU=torch.rsqrt(v_CPU);y_NPU=torch.rsqrt(v_NPU)
            r['cases'][name]={'variance_vs_CPU':previous.compare(v_NPU,v_CPU),'rsqrt_on_own_variance_vs_CPU':previous.compare(y_NPU,y_CPU),
                             'rsqrt_on_exact_CPU_variance':previous.compare(torch.rsqrt(v_CPU.to('npu:0')),y_CPU),'modes':{}}
            for mode,it in(('native',None),('Newton1',1),('Newton2',2)):
                with torch.inference_mode():
                    out=NPU.forward_npu(value)if it is None else refined(it)
                    repeat=NPU.forward_npu(value)if it is None else refined(it)
                same=previous.compare(repeat,out);assert same['raw_bits_equal']
                m=previous.compare(out,gold);assert m['finite']
                r['cases'][name]['modes'][mode]={'vs_original_CPU_Half_norm':m,'repeat_bit_identical':True}
            assert input_SHA==previous.tensor_info(value)['sha256']
        assert before==previous.tensor_info(NPU.weight)['sha256']
        r['input_and_Half_gamma_unchanged']=True;r['completed']=True
    except BaseException as error:r['failure']={'type':type(error).__name__,'message':str(error)};raise
    finally:
        with(ROOT/'norm94.json').open('x')as f:f.write(json.dumps(r,indent=2)+'\n')
        print(json.dumps({'completed':r['completed'],'cases':r['cases']},indent=2),flush=True)


if __name__=='__main__':main()
