#!/usr/bin/env python3
"""Actual original-Half full vision current fixes plus single-launch Float flash; not full text timer."""
from pathlib import Path
import json
import os
import time
import statistics
ROOT=Path(os.environ.get('R99_ROOT','/tmp/h3_round99_vision02'))


def main():
    import torch
    import torch_npu
    import full_row_fp32_fixed84 as previous
    from evidence_io99 import EvidenceWriter
    from safetensors import safe_open
    from transformers import Qwen3VLConfig
    from runtime5gm83 import Runtime5GM
    from vision_adapter87 import VisionBiasAdapter87
    from patch_bias89 import PatchBiasAdapter89
    from erf_gelu_fixed90 import ErfGeluAdapter90
    from accurate_vision_attention88 import AccurateVisionAttention88
    torch.set_num_threads(2);torch.npu.set_device(0);previous.ROOT=ROOT
    writer=EvidenceWriter(ROOT/'vision99.json')
    r={'scope':__doc__,'completed':False,'source_SHA256':previous.file_sha(Path(__file__)),
       'cases':{},'full_encoder_precision_accepted':False,'whole_goal_complete':False,'numerical_tolerance':None,
       'not_full_encoder_timing':True,'no_CPU_precomputed_entry_or_image_conditioning_in_fullvision':True,
       'no_FloatLinear_weights_or_BF_gold_outputcasts':True}
    bias=None;patch=None;erf=None;engine=None;handles=[]
    try:
        expected=json.loads((ROOT/'expected_inputs99.json').read_text())
        assert previous.file_sha(Path(__file__).parent/'accurate_vision_attention88.py')==expected['R88_current_attention_source_SHA256']
        proof=previous.load_cpu_gate(r);candidate=previous.load_candidate(r)
        assert previous.file_sha(previous.FIXTURE)==previous.FIXTURE_SHA256
        assert previous.file_sha(previous.MODEL/'config.json')==previous.CPU_CONFIG_SHA256
        # Frozen original attention leaf is validated before combining fullvision fixes.
        frozen_path=Path('/tmp/h3_round88_vision01/FrozenVisionQKV88.pt')
        assert previous.file_sha(frozen_path)==expected['frozenQKVfileSHA256']
        frozen=torch.load(frozen_path,map_location='cpu',weights_only=True)
        from flash_engine99 import FlashVisionAttention99
        engine=FlashVisionAttention99(candidate,Runtime5GM(0),ROOT/'attention_kernels')
        old_engine=False
        r['engine_selected']='single_launch_Float_flash99'
        r['attention_prepare']=engine.prepare();engine.set_enabled(True)
        r['frozen_original_leaves']=[]
        for image in range(2):
            item=frozen['images'][image];args=[item[name].to('npu:0')for name in('q','k','v')]
            before_leaf=[previous.tensor_info(x)['sha256']for x in args]
            with torch.inference_mode():out=engine.forward(*args);repeat=engine.forward(*args)
            assert out.shape==repeat.shape==(1,16,4032,72)
            assert previous.compare(repeat,out)['raw_bits_equal']
            if old_engine:assert previous.tensor_info(out)['sha256']==expected['R88_precise_leaf_output_SHAs'][image]
            assert before_leaf==[previous.tensor_info(x)['sha256']for x in args]
            clocks=[]
            for _ in range(5):
                torch.npu.synchronize();start=time.perf_counter()
                with torch.inference_mode():timed=engine.forward(*args)
                torch.npu.synchronize();clocks.append(time.perf_counter()-start)
                assert previous.compare(timed,out)['raw_bits_equal']
            r['frozen_original_leaves'].append({'image':image,'CPU_Half_MATH':previous.compare(out,item['CPU_Half_MATH_output']),
                                               'old_R88_output_SHA':expected['R88_precise_leaf_output_SHAs'][image],
                                               'same_old_R88_bits':previous.tensor_info(out)['sha256']==expected['R88_precise_leaf_output_SHAs'][image],
                                               'all_repeats_and_original_QKV_SHAs_unchanged':True,'real_leaf_warm_s':clocks,
                                               'real_leaf_median_s':statistics.median(clocks)})
        engine.set_enabled(False)
        cfg=Qwen3VLConfig.from_pretrained(str(previous.MODEL),local_files_only=True).vision_config
        assert(cfg.hidden_size,cfg.intermediate_size,cfg.num_heads)==(1152,4304,16)
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
        vision.to(device='npu:0');vision.eval();vision.rotary_pos_emb=candidate.MiniMaxH3Qwen3VLVisionRotaryEmbedding(cfg.hidden_size//cfg.num_heads//2)
        assert all(p.dtype==torch.float16 for p in vision.parameters());before=previous.parameters_sha(vision)
        payload=torch.load(previous.FIXTURE,map_location='cpu',weights_only=True)
        pixels=payload['pixel_values'].to('npu:0');grids=payload['image_grid_thw'].to('npu:0');pixel_sha=previous.tensor_info(pixels)['sha256']
        oracle=torch.load(previous.CPU_ROOT/'FP16_vision.pt',map_location='cpu',weights_only=True)
        bias=VisionBiasAdapter87(candidate,Runtime5GM(0),ROOT/'bias_kernels');bias.prime(vision)
        patch=PatchBiasAdapter89(vision,Runtime5GM(0),ROOT/'bias_kernels')
        erf=ErfGeluAdapter90(vision,Runtime5GM(0),ROOT/'erf_kernels');erf.prepare()
        first_native=None
        for name,fixed,attn in(('native',False,False),('current_patch_bias_erf',True,False),
                               ('current_patch_bias_erf_flash99_attention',True,True)):
            bias.set_enabled(fixed);patch.set_enabled(fixed);erf.set_enabled(fixed);engine.set_enabled(attn);layers=[]
            def layer_post(i):
                def hook(module,args,out):
                    info=previous.tensor_info(out);assert info['finite'];layers.append({'layer':i,'output':info})
                return hook
            for i,layer in enumerate(vision.blocks):handles.append(layer.register_forward_hook(layer_post(i)))
            try:
                with torch.inference_mode():merged,ds=vision(pixels,grids)
            finally:
                for h in handles:h.remove()
                handles.clear()
            assert len(layers)==27
            case={'original_Half_patch_and_bias_and_erf':fixed,'accurate_attention_enabled':attn,
                  'merged_vs_CPU':previous.compare(merged,oracle['merged']),
                  'DeepStack_vs_CPU':[previous.compare(a,b)for a,b in zip(ds,oracle['deepstack'])],
                  'raw27_layers':layers,'warm_s':[],'repeat_bits':[]}
            assert case['merged_vs_CPU']['finite']and all(a['finite']for a in case['DeepStack_vs_CPU'])
            with torch.inference_mode():
                for _ in range(5):
                    torch.npu.synchronize();start=time.perf_counter();out,out_ds=vision(pixels,grids);torch.npu.synchronize()
                    case['warm_s'].append(time.perf_counter()-start)
                    repeat={'merged':previous.compare(out,merged),'deepstack':[previous.compare(a,b)for a,b in zip(out_ds,ds)]}
                    assert repeat['merged']['raw_bits_equal']and all(a['raw_bits_equal']for a in repeat['deepstack'])
                    case['repeat_bits'].append(repeat)
            case['resident_vision_median_s']=statistics.median(case['warm_s']);case['engine']=engine.snapshot()
            expected_stream=json.loads((ROOT/'expected_streaming99.json').read_text())
            assert case['merged_vs_CPU']['npu']['sha256']==expected_stream[name]['merged_SHA256']
            assert [x['output']['sha256']for x in layers]==expected_stream[name]['raw27_SHA256']
            assert [x['npu']['sha256']for x in case['DeepStack_vs_CPU']]==expected_stream[name]['DeepStack_SHA256']
            case['exact_streaming_baseline_all27_DS_merger_SHAs_unchanged']=True
            r['cases'][name]=case;writer.write(r)
            if name=='native':
                assert previous.tensor_info(merged)['sha256']==expected['R90_native_vision_merged_SHA256']
                first_native=(merged.detach().cpu(),[x.detach().cpu()for x in ds])
            if name=='current_patch_bias_erf':assert previous.tensor_info(merged)['sha256']==expected['R90_current_joint_vision_merged_SHA256']
        bias.set_enabled(False);patch.set_enabled(False);erf.set_enabled(False);engine.set_enabled(False)
        with torch.inference_mode():out,ds=vision(pixels,grids)
        r['native_recovery']={'merged':previous.compare(out,first_native[0]),'deepstack':[previous.compare(a,b)for a,b in zip(ds,first_native[1])]}
        assert r['native_recovery']['merged']['raw_bits_equal']and all(a['raw_bits_equal']for a in r['native_recovery']['deepstack'])
        r['original_parameters_unchanged']=before==previous.parameters_sha(vision);assert r['original_parameters_unchanged']
        r['pixels_unchanged']=pixel_sha==previous.tensor_info(pixels)['sha256'];assert r['pixels_unchanged']
        r['CPU_reference_SHA_unchanged']=previous.file_sha(previous.CPU_ROOT/'reference.json')=='44acac61f9b4d9bc8afc91632392502ccaa29dee7bfab951e2b15902a0be74c9'
        assert r['CPU_reference_SHA_unchanged'];r['completed']=True
    except BaseException as error:r['failure']={'type':type(error).__name__,'message':str(error)};raise
    finally:
        for h in handles:h.remove()
        if engine is not None:r['engine_final']=engine.snapshot();engine.uninstall()
        if erf is not None:r['TIK_erf_final']=erf.snapshot();erf.uninstall()
        if patch is not None:r['patch_final']=patch.snapshot();patch.uninstall()
        if bias is not None:r['bias_final']=bias.snapshot();bias.uninstall()
        writer.write(r)


if __name__=='__main__':main()
