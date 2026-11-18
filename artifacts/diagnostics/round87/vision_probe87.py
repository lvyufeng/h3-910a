#!/usr/bin/env python3
"""Real vision FP32-bias/one-Half-cast A/B,compared to unchanged CPUFP16 gold."""
from pathlib import Path
import hashlib
import json
import time
import statistics
ROOT=Path('/tmp/h3_round87_vision01')


def main():
    import torch
    import torch_npu
    import full_row_fp32_fixed84 as previous
    from evidence_io87 import EvidenceWriter
    from safetensors import safe_open
    from transformers import Qwen3VLConfig
    from runtime5gm83 import Runtime5GM
    from vision_adapter87 import VisionBiasAdapter87
    torch.set_num_threads(2);torch.npu.set_device(0);previous.ROOT=ROOT
    writer=EvidenceWriter(ROOT/'vision87.json')
    r={'scope':__doc__,'completed':False,'source_sha256':previous.file_sha(Path(__file__)),
       'production_changed':False,'full_encoder_precision_accepted':False,'whole_goal_complete':False,
       'numerical_tolerance':None,'not_full_encoder_timing':True,'cases':{}}
    adapter=None;handles=[]
    try:
        proof=previous.load_cpu_gate(r);candidate=previous.load_candidate(r)
        assert previous.file_sha(previous.MODEL/'config.json')==previous.CPU_CONFIG_SHA256
        cfg=Qwen3VLConfig.from_pretrained(str(previous.MODEL),local_files_only=True).vision_config
        assert(cfg.hidden_size,cfg.intermediate_size,cfg.out_hidden_size)==(1152,4304,5120)
        with torch.device('meta'):vision=candidate.MiniMaxH3Qwen3VLVisionModel(cfg)
        wanted=set(vision.state_dict());index=json.loads((previous.MODEL/'model.safetensors.index.json').read_text())['weight_map'];state={}
        for shard in sorted(set(index.values())):
            keys=[name for name,s in index.items()if s==shard and name.startswith('model.visual.')and name[len('model.visual.'):]in wanted]
            if not keys:continue
            with safe_open(str(previous.MODEL/shard),framework='pt',device='cpu')as h:
                for name in keys:
                    value=h.get_tensor(name).to(torch.float16);key=name[len('model.visual.'):]
                    assert previous.tensor_info(value)['sha256']==proof['parameter_conversion']['visual.'+key]['reference_sha256']
                    state[key]=value
        assert set(state)==wanted
        vision.load_state_dict(state,strict=True,assign=True);del state
        vision.to(device='npu:0');vision.eval()
        vision.rotary_pos_emb=candidate.MiniMaxH3Qwen3VLVisionRotaryEmbedding(cfg.hidden_size//cfg.num_heads//2)
        assert all(p.dtype==torch.float16 for p in vision.parameters())
        r['original_parameter_tensors']=len(list(vision.parameters()))
        before=previous.parameters_sha(vision)
        payload=torch.load(previous.FIXTURE,map_location='cpu',weights_only=True)
        pixels=payload['pixel_values'].to('npu:0');grids=payload['image_grid_thw'].to('npu:0')
        pixels_sha=previous.tensor_info(pixels)['sha256']
        oracle=torch.load(previous.CPU_ROOT/'FP16_vision.pt',map_location='cpu',weights_only=True)
        adapter=VisionBiasAdapter87(candidate,Runtime5GM(0),ROOT/'kernels');adapter.prime(vision)
        r['adapter_primed']=adapter.snapshot();r['kernel_compile_and_weight_cache_excluded_from_warm']=True
        captured={};leaf_names=('blocks.0.attn.qkv','blocks.0.mlp.linear_fc1','blocks.0.mlp.linear_fc2')
        modules=dict(vision.named_modules())
        def capture(name):
            def hook(module,args):captured[name]=args[0].detach().clone()
            return hook
        first_native=None
        for name,enabled in(('native',False),('FP32_bias_one_Half',True)):
            adapter.set_enabled(enabled);layers=[]
            def layer_post(index):
                def hook(module,args,out):
                    info=previous.tensor_info(out);assert info['finite'];layers.append({'layer':index,'output':info})
                return hook
            if not enabled:
                for leaf in leaf_names:handles.append(modules[leaf].register_forward_pre_hook(capture(leaf)))
            for i,layer in enumerate(vision.blocks):handles.append(layer.register_forward_hook(layer_post(i)))
            try:
                with torch.inference_mode():merged,deepstack=vision(pixels,grids)
            finally:
                for handle in handles:handle.remove()
                handles.clear()
            assert len(layers)==27
            case={'merged_vs_CPU':previous.compare(merged,oracle['merged']),
                  'DeepStack_vs_CPU':[previous.compare(a,b)for a,b in zip(deepstack,oracle['deepstack'])],
                  'raw27_layers':layers,'warm_s':[],'repeat_bits':[]}
            assert case['merged_vs_CPU']['finite']and all(x['finite']for x in case['DeepStack_vs_CPU'])
            with torch.inference_mode():
                for _ in range(3):
                    torch.npu.synchronize();start=time.perf_counter();out,ds=vision(pixels,grids);torch.npu.synchronize()
                    case['warm_s'].append(time.perf_counter()-start)
                    check={'merged':previous.compare(out,merged),'deepstack':[previous.compare(a,b)for a,b in zip(ds,deepstack)]}
                    assert check['merged']['raw_bits_equal']and all(x['raw_bits_equal']for x in check['deepstack'])
                    case['repeat_bits'].append(check)
            case['median_resident_vision_s']=statistics.median(case['warm_s']);case['adapter']=adapter.snapshot()
            r['cases'][name]=case
            if not enabled:
                first_native=(merged.detach().cpu(),[x.detach().cpu()for x in deepstack])
                # Actual same-operand CPU Half Linear leaf checks,not another full teacher.
                r['actual_biased_leaf']={}
                with torch.inference_mode():
                    for name in leaf_names:
                        module=modules[name];input_=captured[name]
                        controls=adapter.controls(module,input_)
                        cpu_gold=torch.nn.functional.linear(input_.cpu(),module.weight.detach().cpu(),module.bias.detach().cpu())
                        adapter.set_enabled(False);native=module(input_)
                        adapter.set_enabled(True);custom=module(input_);repeat=module(input_)
                        assert previous.compare(repeat,custom)['raw_bits_equal']
                        r['actual_biased_leaf'][name]={'native_vs_CPU_Half':previous.compare(native,cpu_gold),
                                                     'custom_vs_CPU_Half':previous.compare(custom,cpu_gold),
                                                     'padding_controls':controls,'same_original_Half_parameters':True}
                adapter.set_enabled(False)
        adapter.set_enabled(False)
        with torch.inference_mode():out,ds=vision(pixels,grids)
        r['native_recovery']={'merged':previous.compare(out,first_native[0]),
                              'deepstack':[previous.compare(a,b)for a,b in zip(ds,first_native[1])]}
        assert r['native_recovery']['merged']['raw_bits_equal']and all(x['raw_bits_equal']for x in r['native_recovery']['deepstack'])
        r['original_parameters_unchanged']=before==previous.parameters_sha(vision);assert r['original_parameters_unchanged']
        r['original_pixels_unchanged']=pixels_sha==previous.tensor_info(pixels)['sha256'];assert r['original_pixels_unchanged']
        r['CPU_reference_SHA_unchanged']=previous.file_sha(previous.CPU_ROOT/'reference.json')=='44acac61f9b4d9bc8afc91632392502ccaa29dee7bfab951e2b15902a0be74c9'
        assert r['CPU_reference_SHA_unchanged'];r['completed']=True
    except BaseException as error:
        r['failure']={'type':type(error).__name__,'message':str(error)};raise
    finally:
        for handle in handles:handle.remove()
        if adapter is not None:r['adapter_final']=adapter.snapshot();adapter.uninstall()
        writer.write(r)


if __name__=='__main__':main()
