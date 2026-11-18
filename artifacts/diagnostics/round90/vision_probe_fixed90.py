#!/usr/bin/env python3
"""Full real27vision A/B for explicit erf and new TIK GELU,originalCPUHalfgold."""
from pathlib import Path
import json
import time
import statistics
ROOT=Path('/tmp/h3_round90_vision02')


def main():
    import torch
    import torch_npu
    import full_row_fp32_fixed84 as previous
    from evidence_io90 import EvidenceWriter
    from safetensors import safe_open
    from transformers import Qwen3VLConfig
    from runtime5gm83 import Runtime5GM
    from vision_adapter87 import VisionBiasAdapter87
    from patch_bias89 import PatchBiasAdapter89
    from torch_erf_adapter90 import TorchErfAdapter90
    from erf_gelu_fixed90 import ErfGeluAdapter90
    torch.set_num_threads(2);torch.npu.set_device(0);previous.ROOT=ROOT
    writer=EvidenceWriter(ROOT/'vision90.json')
    r={'scope':__doc__,'completed':False,'source_SHA256':previous.file_sha(Path(__file__)),
       'cases':{},'full_encoder_precision_accepted':False,'whole_goal_complete':False,
       'numerical_tolerance':None,'not_full_encoder_timing':True,'no_CPU_precomputed_entry_in_real_path':True,
       'R88_slow_attention_and_R89_redundant_Float_nonlinears_disabled':True}
    bias=None;patch=None;explicit=None;erf=None;handles=[]
    try:
        preflight=json.loads((ROOT/'gelu_preflight90.json').read_text());kernel_proof=json.loads((ROOT/'kernel_probe90.json').read_text())
        assert preflight['completed']and kernel_proof['completed']
        assert all(c['repeat_bits']['raw_bits_equal']and c['input_SHA_unchanged']for c in kernel_proof['cases'].values())
        proof=previous.load_cpu_gate(r);candidate=previous.load_candidate(r)
        assert previous.file_sha(previous.FIXTURE)=='ed099e792982f9cd37b67f2cff24443008683d0cc819d1decacabf55de1f4ebc'
        assert previous.file_sha(previous.MODEL/'config.json')==previous.CPU_CONFIG_SHA256
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
        vision.to(device='npu:0');vision.eval()
        vision.rotary_pos_emb=candidate.MiniMaxH3Qwen3VLVisionRotaryEmbedding(cfg.hidden_size//cfg.num_heads//2)
        assert all(p.dtype==torch.float16 for p in vision.parameters());before=previous.parameters_sha(vision)
        payload=torch.load(previous.FIXTURE,map_location='cpu',weights_only=True)
        pixels=payload['pixel_values'].to('npu:0');grids=payload['image_grid_thw'].to('npu:0');pixel_sha=previous.tensor_info(pixels)['sha256']
        oracle=torch.load(previous.CPU_ROOT/'FP16_vision.pt',map_location='cpu',weights_only=True)
        bias=VisionBiasAdapter87(candidate,Runtime5GM(0),ROOT/'bias_kernels');bias.prime(vision);bias.set_enabled(False)
        patch=PatchBiasAdapter89(vision,Runtime5GM(0),ROOT/'bias_kernels');patch.set_enabled(False)
        explicit=TorchErfAdapter90(vision)
        cpu_mode_supported=all(preflight['leafs'][name]['NPU_explicit_erf_supported']for name in('deepstack_merger_list.0.act_fn','merger.act_fn'))
        first_native=None
        def run_case(name,use_bias,use_explicit,use_tik):
            nonlocal first_native
            bias.set_enabled(use_bias);patch.set_enabled(use_bias)
            if explicit is not None and explicit.installed:explicit.set_enabled(use_explicit)
            if erf is not None:erf.set_enabled(use_tik)
            layers=[]
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
            case={'patch_and_linear_onecast_bias':use_bias,'explicit_Torch_erf':use_explicit,'TIK_erf_enabled':use_tik,
                  'merged_vs_CPU':previous.compare(merged,oracle['merged']),
                  'DeepStack_vs_CPU':[previous.compare(a,b)for a,b in zip(ds,oracle['deepstack'])],
                  'raw27_layers':layers,'warm_s':[],'repeat_bits':[]}
            assert case['merged_vs_CPU']['finite']and all(a['finite']for a in case['DeepStack_vs_CPU'])
            with torch.inference_mode():
                for _ in range(3):
                    torch.npu.synchronize();start=time.perf_counter();out,out_ds=vision(pixels,grids);torch.npu.synchronize()
                    case['warm_s'].append(time.perf_counter()-start)
                    repeat={'merged':previous.compare(out,merged),'deepstack':[previous.compare(a,b)for a,b in zip(out_ds,ds)]}
                    assert repeat['merged']['raw_bits_equal']and all(a['raw_bits_equal']for a in repeat['deepstack'])
                    case['repeat_bits'].append(repeat)
            case['resident_vision_median_s']=statistics.median(case['warm_s'])
            case['patch']=patch.snapshot();case['bias']=bias.snapshot()
            if explicit is not None:case['explicit_erf']=explicit.snapshot()
            if erf is not None:case['TIK_erf']=erf.snapshot()
            r['cases'][name]=case;writer.write(r)
            if name=='native':first_native=(merged.detach().cpu(),[x.detach().cpu()for x in ds])
        run_case('native',False,False,False)
        run_case('patch_and_linear_bias',True,False,False)
        if cpu_mode_supported:
            run_case('Torch_explicit_erf_only',False,True,False)
            run_case('Torch_explicit_erf_patch_and_bias',True,True,False)
        else:r['explicit_Torch_erf_full_cases_skipped']='unsupported in realleafpreflight'
        r['explicit_erf_final']=explicit.snapshot();explicit.uninstall()
        erf=ErfGeluAdapter90(vision,Runtime5GM(0),ROOT/'erf_kernels');r['erf_prepared']=erf.prepare()
        run_case('TIK_erf_only',False,False,True)
        run_case('TIK_erf_patch_and_bias',True,False,True)
        bias.set_enabled(False);patch.set_enabled(False);erf.set_enabled(False)
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
        if erf is not None:r['TIK_erf_final']=erf.snapshot();erf.uninstall()
        if explicit is not None:explicit.uninstall()
        if patch is not None:r['patch_final']=patch.snapshot();patch.uninstall()
        if bias is not None:r['bias_final']=bias.snapshot();bias.uninstall()
        writer.write(r)


if __name__=='__main__':main()
