#!/usr/bin/env python3
"""Real head72 unmasked vision SDPA and full27-layer A/B on unchanged CPUFP16 gold."""
from pathlib import Path
import json
import hashlib
import time
import statistics
ROOT=Path('/tmp/h3_round88_vision01')


def main():
    import torch
    import torch_npu
    import full_row_fp32_fixed84 as previous
    from evidence_io88 import EvidenceWriter
    from safetensors import safe_open
    from transformers import Qwen3VLConfig
    from torch.nn.attention import sdpa_kernel,SDPBackend
    from runtime5gm83 import Runtime5GM
    from vision_adapter87 import VisionBiasAdapter87
    from accurate_vision_attention88 import AccurateVisionAttention88
    torch.set_num_threads(2);torch.npu.set_device(0);previous.ROOT=ROOT
    writer=EvidenceWriter(ROOT/'vision88.json')
    r={'scope':__doc__,'completed':False,'source_sha256':previous.file_sha(Path(__file__)),
       'not_full_encoder_timing':True,'full_encoder_precision_accepted':False,'whole_goal_complete':False,
       'numerical_tolerance':None,'cases':{}}
    bias=None;engine=None;handles=[]
    try:
        proof=previous.load_cpu_gate(r);candidate=previous.load_candidate(r)
        assert previous.file_sha(previous.MODEL/'config.json')==previous.CPU_CONFIG_SHA256
        cfg=Qwen3VLConfig.from_pretrained(str(previous.MODEL),local_files_only=True).vision_config
        assert(cfg.hidden_size,cfg.num_heads,cfg.intermediate_size)==(1152,16,4304)
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
        before=previous.parameters_sha(vision)
        payload=torch.load(previous.FIXTURE,map_location='cpu',weights_only=True)
        pixels=payload['pixel_values'].to('npu:0');grids=payload['image_grid_thw'].to('npu:0')
        pixels_sha=previous.tensor_info(pixels)['sha256']
        oracle=torch.load(previous.CPU_ROOT/'FP16_vision.pt',map_location='cpu',weights_only=True)
        bias=VisionBiasAdapter87(candidate,Runtime5GM(0),ROOT/'bias_kernels');bias.prime(vision);bias.set_enabled(False)
        original=candidate._vision_sdpa_bf16_compatible;capture={};native_calls=0
        def capture_sdpa(q,k,v):
            nonlocal native_calls
            out=original(q,k,v)
            epoch=native_calls//54;image=native_calls%54
            if epoch<2 and image<2:
                assert q.shape==k.shape==v.shape==(1,16,4032,72)
                assert q.dtype==k.dtype==v.dtype==torch.float16
                capture[(epoch,image)]={name:x.detach().cpu().contiguous()for name,x in(('q',q),('k',k),('v',v),('native_out',out))}
            native_calls+=1
            return out
        candidate._vision_sdpa_bf16_compatible=capture_sdpa
        engine=AccurateVisionAttention88(candidate,Runtime5GM(0),ROOT/'attention_kernels');engine.prepare();engine.set_enabled(False)
        r['bias_primed']=bias.snapshot();r['engine_prepared']=engine.snapshot()
        first_native=None
        for name,use_bias,use_attention in(('native',False,False),('bias_only',True,False),
                                          ('accurate_PV_only',False,True),('accurate_PV_and_bias',True,True)):
            bias.set_enabled(use_bias);engine.set_enabled(use_attention);layers=[]
            def layer_hook(i):
                def hook(module,args,out):
                    info=previous.tensor_info(out);assert info['finite'];layers.append({'layer':i,'output':info})
                return hook
            for i,layer in enumerate(vision.blocks):handles.append(layer.register_forward_hook(layer_hook(i)))
            try:
                with torch.inference_mode():merged,deepstack=vision(pixels,grids)
            finally:
                for h in handles:h.remove()
                handles.clear()
            assert len(layers)==27
            case={'merged_vs_CPU':previous.compare(merged,oracle['merged']),
                  'DeepStack_vs_CPU':[previous.compare(a,b)for a,b in zip(deepstack,oracle['deepstack'])],
                  'raw27_layers':layers,'warm_s':[],'repeat_bits':[]}
            with torch.inference_mode():
                for _ in range(3):
                    torch.npu.synchronize();start=time.perf_counter();out,ds=vision(pixels,grids);torch.npu.synchronize()
                    case['warm_s'].append(time.perf_counter()-start)
                    repeated={'merged':previous.compare(out,merged),'deepstack':[previous.compare(a,b)for a,b in zip(ds,deepstack)]}
                    assert repeated['merged']['raw_bits_equal']and all(a['raw_bits_equal']for a in repeated['deepstack'])
                    case['repeat_bits'].append(repeated)
            case['resident_vision_median_s']=statistics.median(case['warm_s']);case['engine']=engine.snapshot();case['bias']=bias.snapshot()
            r['cases'][name]=case
            if name=='native':
                first_native=(merged.detach().cpu(),[x.detach().cpu()for x in deepstack])
                assert set(capture)=={(0,0),(0,1),(1,0),(1,1)}
                r['real_vision_SDPA_leaf']=[]
                engine.set_enabled(True)
                for image in range(2):
                    operands=capture[(0,image)]
                    assert all(previous.compare(operands[k],capture[(1,image)][k])['raw_bits_equal']for k in operands)
                    q,k,v=[operands[x]for x in('q','k','v')]
                    with torch.inference_mode(),sdpa_kernel(SDPBackend.MATH):
                        gold=torch.nn.functional.scaled_dot_product_attention(q,k,v,dropout_p=0.,is_causal=False)
                        gold_repeat=torch.nn.functional.scaled_dot_product_attention(q,k,v,dropout_p=0.,is_causal=False)
                    assert previous.compare(gold_repeat,gold)['raw_bits_equal']
                    qn,kn,vn=[x.to('npu:0')for x in(q,k,v)]
                    padding=engine.padding_selfcheck(qn,kn,vn)
                    assert all(padding[x]for x in('q_prefix_exact','q_width_tail_zero','k_key_tail_zero','k_prefix_exact','v_prefix_exact','v_width_tail_zero','packed_k_head_distinct'))
                    before_leaf={name:previous.tensor_info(x)['sha256']for name,x in(('q',qn),('k',kn),('v',vn))}
                    with torch.inference_mode():
                        custom=candidate._vision_sdpa_bf16_compatible(qn,kn,vn)
                        custom_repeat=candidate._vision_sdpa_bf16_compatible(qn,kn,vn)
                    repeat=previous.compare(custom_repeat,custom);assert repeat['raw_bits_equal']
                    leaf_unchanged=before_leaf=={name:previous.tensor_info(x)['sha256']for name,x in(('q',qn),('k',kn),('v',vn))}
                    assert leaf_unchanged
                    # Float leaf intermediates are diagnostics,not a replacement teacher.
                    scale=72**-.5
                    cs=torch.matmul(q.float()[...,:512,:],k.float().transpose(-2,-1))*scale
                    cp=torch.softmax(cs,-1);cv=torch.matmul(cp,v.float())
                    nv=torch.matmul(cp.to('npu:0'),vn.float())
                    r['real_vision_SDPA_leaf'].append({'image':image,'native_vs_CPU_Half_MATH':previous.compare(operands['native_out'],gold),
                                                      'custom_vs_CPU_Half_MATH':previous.compare(custom,gold),'repeat_bits':repeat,
                                                      'native_Float_PV_with_exact_CPU_P_vs_CPU_Float_PV':previous.compare(nv,cv),
                                                      'original_head_dim':72,'no_causal_mask':True,'padding_controls':padding,'original_QKV_unchanged':leaf_unchanged,'original_qkv_info':{x:previous.tensor_info(operands[x])for x in('q','k','v')}})
                    operands['CPU_Half_MATH_output']=gold
                capture_path=ROOT/'FrozenVisionQKV88.pt';assert not capture_path.exists()
                torch.save({'images':[capture[(0,0)],capture[(0,1)]],'original_parameter_SHA256':before},capture_path)
                r['frozen_capture_SHA256']=previous.file_sha(capture_path)
                engine.set_enabled(False)
        bias.set_enabled(False);engine.set_enabled(False)
        with torch.inference_mode():out,ds=vision(pixels,grids)
        r['native_recovery']={'merged':previous.compare(out,first_native[0]),'deepstack':[previous.compare(a,b)for a,b in zip(ds,first_native[1])]}
        assert r['native_recovery']['merged']['raw_bits_equal']and all(x['raw_bits_equal']for x in r['native_recovery']['deepstack'])
        r['original_parameters_unchanged']=before==previous.parameters_sha(vision);assert r['original_parameters_unchanged']
        r['original_pixels_unchanged']=pixels_sha==previous.tensor_info(pixels)['sha256'];assert r['original_pixels_unchanged']
        r['CPU_reference_SHA_unchanged']=previous.file_sha(previous.CPU_ROOT/'reference.json')=='44acac61f9b4d9bc8afc91632392502ccaa29dee7bfab951e2b15902a0be74c9'
        assert r['CPU_reference_SHA_unchanged'];r['completed']=True
    except BaseException as error:
        r['failure']={'type':type(error).__name__,'message':str(error)};raise
    finally:
        for h in handles:h.remove()
        if engine is not None:r['engine_final']=engine.snapshot();engine.uninstall()
        if bias is not None:r['bias_final']=bias.snapshot();bias.uninstall()
        writer.write(r)


if __name__=='__main__':main()
