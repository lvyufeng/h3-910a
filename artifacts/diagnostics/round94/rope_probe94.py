#!/usr/bin/env python3
"""Real HF CPU/NPU text Half and vision Float RoPE/trig audit;no Linear weights."""
from pathlib import Path
import json
from types import SimpleNamespace
ROOT=Path('/tmp/h3_round94_rope01')


def main():
    import torch
    import torch_npu
    import full_row_fp32_fixed84 as previous
    from transformers import Qwen3VLConfig
    from transformers.models.qwen3_vl import modeling_qwen3_vl as hf
    torch.set_num_threads(2);torch.npu.set_device(0);previous.ROOT=ROOT
    r={'scope':__doc__,'completed':False,'not_full_encoder_timing':True,'no_Linear_weight_or_teacher':True,
       'no_CANN_kernel_build':True,'R93_CPU_control_concurrent_no_speed_comparison':True,
       'full_encoder_precision_accepted':False,'whole_goal_complete':False,'numerical_tolerance':None,
       'source_SHA256':previous.file_sha(Path(__file__))}
    try:
        assert previous.file_sha(Path(hf.__file__))=='fcb5571ef95e4a6679af866ef9ef4dfa5cee017f9f91453daf28c7804c7a50fe'
        assert previous.file_sha(previous.CPU_ROOT/'reference.json')=='44acac61f9b4d9bc8afc91632392502ccaa29dee7bfab951e2b15902a0be74c9'
        assert previous.file_sha(previous.MODEL/'config.json')==previous.CPU_CONFIG_SHA256
        candidate=previous.load_candidate(r)
        cfg=Qwen3VLConfig.from_pretrained(str(previous.MODEL),trust_remote_code=False,local_files_only=True)
        mm=torch.load(previous.CPU_ROOT/'FP16_multimodal_inputs.pt',map_location='cpu',weights_only=True)
        hidden=torch.load(previous.CPU_ROOT/'FP16_layer24.pt',map_location='cpu',weights_only=True)
        positions=mm['position_ids'];position_SHA=previous.tensor_info(positions)['sha256']
        assert positions.shape==(3,1,2043)and positions.dtype==torch.int64
        CPUrot=hf.Qwen3VLTextRotaryEmbedding(cfg.text_config,device='cpu')
        NPUrot=candidate.MiniMaxH3Qwen3VLTextRotaryEmbedding(cfg.text_config).to('npu:0')
        assert sum(p.numel()for p in CPUrot.parameters())==0 and sum(p.numel()for p in NPUrot.parameters())==0
        r['text_inv_freq_CPU_vs_candidate']=previous.compare(NPUrot._get_inv_freq(torch.device('cpu')),CPUrot.inv_freq)
        assert r['text_inv_freq_CPU_vs_candidate']['raw_bits_equal']
        with torch.inference_mode():
            cpu_cos,cpu_sin=CPUrot(hidden,positions)
            npu_hidden=hidden.to('npu:0');npu_positions=positions.to('npu:0')
            native_cos,native_sin=NPUrot(npu_hidden,npu_positions)
            repeat_cos,repeat_sin=NPUrot(npu_hidden,npu_positions)
        r['text_actual_Half']={'cos_vs_CPU':previous.compare(native_cos,cpu_cos),'sin_vs_CPU':previous.compare(native_sin,cpu_sin),
                              'cos_repeat':previous.compare(repeat_cos,native_cos),'sin_repeat':previous.compare(repeat_sin,native_sin),
                              'existing_R79_Float_direct_frequency_branch_used':True}
        assert r['text_actual_Half']['cos_repeat']['raw_bits_equal']and r['text_actual_Half']['sin_repeat']['raw_bits_equal']
        # Vision Float inverse frequencies and real positions,not guessed/synthetic q/k.
        grid=torch.tensor([[1,48,84],[1,48,84]],dtype=torch.long)
        CPUvision=SimpleNamespace(spatial_merge_size=2,rotary_pos_emb=hf.Qwen3VLVisionRotaryEmbedding(36))
        NPUvision=SimpleNamespace(spatial_merge_size=2,rotary_pos_emb=candidate.MiniMaxH3Qwen3VLVisionRotaryEmbedding(36))
        with torch.inference_mode():
            CPUfreq=hf.Qwen3VLVisionModel.rot_pos_emb(CPUvision,grid)
            NPUfreq=candidate.MiniMaxH3Qwen3VLVisionModel.rot_pos_emb(NPUvision,grid.to('npu:0'))
            assert CPUfreq.shape==NPUfreq.shape==(8064,36)
            r['vision_frequency_vs_CPU']=previous.compare(NPUfreq,CPUfreq)
            assert r['vision_frequency_vs_CPU']['raw_bits_equal']
            CPUemb=torch.cat((CPUfreq,CPUfreq),dim=-1);NPUemb=torch.cat((NPUfreq,NPUfreq),dim=-1)
            CPUvc,CPUvs=CPUemb.cos(),CPUemb.sin();NPUvc,NPUvs=NPUemb.cos(),NPUemb.sin()
            rc,rs=NPUemb.cos(),NPUemb.sin()
        r['vision_actual_Float']={'cos_vs_CPU':previous.compare(NPUvc,CPUvc),'sin_vs_CPU':previous.compare(NPUvs,CPUvs),
                                 'cos_repeat':previous.compare(rc,NPUvc),'sin_repeat':previous.compare(rs,NPUvs)}
        assert r['vision_actual_Float']['cos_repeat']['raw_bits_equal']and r['vision_actual_Float']['sin_repeat']['raw_bits_equal']
        r['original_positions_SHA_unchanged']=position_SHA==previous.tensor_info(positions)['sha256']
        assert r['original_positions_SHA_unchanged']
        r['original_HF_candidate_CPUgold_unchanged']=previous.file_sha(Path(hf.__file__))=='fcb5571ef95e4a6679af866ef9ef4dfa5cee017f9f91453daf28c7804c7a50fe'and previous.file_sha(ROOT/'candidate_encoder79.py')==previous.CANDIDATE_SHA256 and previous.file_sha(previous.CPU_ROOT/'reference.json')=='44acac61f9b4d9bc8afc91632392502ccaa29dee7bfab951e2b15902a0be74c9'
        assert r['original_HF_candidate_CPUgold_unchanged'];r['completed']=True
    except BaseException as error:r['failure']={'type':type(error).__name__,'message':str(error)};raise
    finally:
        with(ROOT/'rope94.json').open('x')as f:f.write(json.dumps(r,indent=2)+'\n')
        print(json.dumps({'completed':r['completed'],'text':r.get('text_actual_Half'),'vision':r.get('vision_actual_Float')},indent=2))


if __name__=='__main__':main()
