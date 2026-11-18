#!/usr/bin/env python3
"""Recompute original HF CPU Half vision entry; verify original raw block0 SHA."""
from pathlib import Path
import hashlib
import time
HF_SOURCE_SHA256='fcb5571ef95e4a6679af866ef9ef4dfa5cee017f9f91453daf28c7804c7a50fe'


def recompute(cfg,state,payload,proof,compare,tensor_info):
    import torch
    from torch.nn.attention import sdpa_kernel,SDPBackend
    from transformers.models.qwen3_vl import modeling_qwen3_vl as hf
    actual=hashlib.sha256(Path(hf.__file__).read_bytes()).hexdigest()
    assert actual==proof['HF_modeling_source_sha256']==HF_SOURCE_SHA256
    assert payload['pixel_values'].shape==(8064,1536)and payload['pixel_values'].dtype==torch.float16
    assert payload['image_grid_thw'].tolist()==[[1,48,84],[1,48,84]]
    assert all(x.dtype==torch.float16 and x.device.type=='cpu'for x in state.values())
    saved_threads=torch.get_num_threads();torch.set_num_threads(proof['CPU_num_threads'])
    assert torch.backends.mkldnn.enabled==proof['CPU_mkldnn_enabled']
    try:
        cfg.dtype=torch.float16;cfg._attn_implementation='sdpa'
        with torch.device('meta'):cpu=hf.Qwen3VLVisionModel(cfg)
        cpu.load_state_dict(state,strict=True,assign=True)
        cpu.rotary_pos_emb=hf.Qwen3VLVisionRotaryEmbedding(cfg.hidden_size//cfg.num_heads//2)
        cpu.eval()
        assert all(p.dtype==torch.float16 and p.device.type=='cpu'and not p.is_meta for p in cpu.parameters())
        initial=[];parts=[];start=time.perf_counter()
        with torch.inference_mode():
            for _ in range(2):
                patch=cpu.patch_embed(payload['pixel_values'])
                pos=cpu.fast_pos_embed_interpolate(payload['image_grid_thw']).to(patch.dtype)
                initial.append(patch+pos);parts.append((patch,pos))
            assert compare(initial[0],initial[1])['raw_bits_equal']
            assert all(compare(x,y)['raw_bits_equal']for x,y in zip(parts[0],parts[1]))
            position_ids=hf.get_vision_position_ids(payload['image_grid_thw'],cfg.spatial_merge_size)
            cu_seqlens=hf.get_vision_cu_seqlens(payload['image_grid_thw'])
            rotary=cpu.rotary_pos_emb(position_ids).reshape(initial[0].shape[0],-1)
            doubled=torch.cat((rotary,rotary),dim=-1);position_embeddings=(doubled.cos(),doubled.sin())
            with sdpa_kernel(SDPBackend.MATH):
                block0=cpu.blocks[0](initial[0],cu_seqlens=cu_seqlens,position_embeddings=position_embeddings)
            observed=tensor_info(block0)
            expected=proof['vision_layers'][0]['output']
            assert observed['sha256']==expected['sha256'],('recomputedCPUentrydoesnotreproduceoriginalR78block0SHA: '+observed['sha256']+' vs '+expected['sha256'])
        receipt={'scope':__doc__,'HF_source_SHA256':actual,'CPU_threads':torch.get_num_threads(),
                 'all_original_CPU_parameters_Half':True,'parameter_tensors':len(list(cpu.parameters())),
                 'CPU_entry_repeat_bits':True,'CPU_patch_repeat_bits':True,'CPU_position_repeat_bits':True,
                 'CPU_block0_matches_original_R78_output_SHA':True,'CPU_block0':observed,
                 'initial':tensor_info(initial[0]),'patch':tensor_info(parts[0][0]),'positions':tensor_info(parts[0][1]),
                 'CPU_recompute_wall_s_excluded_from_NPU_warm':time.perf_counter()-start,
                 'no_FP32_linear_weight_teacher':True,'no_BF16_gold_cast':True,'numerical_tolerance':None}
        return{'entry':initial[0],'patch':parts[0][0],'positions':parts[0][1],'block0':block0},receipt
    finally:torch.set_num_threads(saved_threads)
