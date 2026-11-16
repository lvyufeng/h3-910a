#!/usr/bin/env python3
"""Verify new instance-local service integration on the real captured VAE tile."""
import hashlib
import json
import os
from pathlib import Path
os.environ.update(VLLM_OMNI_910A_VAE_FINITE_SELECT='1', VLLM_OMNI_910A_VAE_QUERY_CHUNK='256',
                  VLLM_OMNI_910A_NO_FUSED_NORM='1', VLLM_OMNI_910A_NO_FUSED_ACT='1', VLLM_OMNI_910A_ATTN_CHUNK='128')
import torch
import torch_npu
from vllm_omni.platforms import current_omni_platform
from vllm_omni.diffusion.models.minimax_h3.vae import MiniMaxH3VideoVAE
from vllm_omni.diffusion.models.minimax_h3.h3_910a_vae import mode
from h3_evidence64 import EvidenceWriter
ROOT = Path('/tmp/h3_round64_service01')
CAP = Path('/tmp/h3_round63_decode01/capture/rank0')
writer = EvidenceWriter(ROOT / 'smoke.json')
r = {'cases': [], 'source_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
try:
    current_omni_platform.set_device(torch.device('npu', 0))
    torch.set_num_threads(8)
    global_sdpa = torch.nn.functional.scaled_dot_product_attention
    previous = torch.get_default_dtype()
    torch.set_default_dtype(torch.float16)
    try:
        vae = MiniMaxH3VideoVAE('/home/mseco/models/MiniMax-H3-native-fp16-audit53/FL2VA/video_vae',
                              device=torch.device('npu', 0), load_device=torch.device('cpu'), decode_only=True, trust_remote_code=True)
    finally:
        torch.set_default_dtype(previous)
    vae.decoder_component.load_to_device()
    x = torch.load(CAP / 'decoder_tile_input.pt', weights_only=True).to('npu').contiguous()
    expected = torch.load(CAP / 'decoder_tile_output.pt', weights_only=True)
    with torch.inference_mode(), torch.autocast('npu', dtype=torch.float16):
        for active in (False, True, True, False):
            with mode(vae, active):
                output = vae.model.decoder(x)
                torch.npu.synchronize()
                cpu = output.cpu().contiguous()
                exact = torch.equal(cpu.view(torch.uint8), expected.contiguous().view(torch.uint8))
                r['cases'].append({'active': active, 'elements': cpu.numel(), 'raw_bits_equal_to_capture': exact})
                assert exact
    r['global_sdpa_unchanged'] = torch.nn.functional.scaled_dot_product_attention is global_sdpa
    r['local_call_count'] = vae._h3_910a_finite_select.calls
    r['soc'] = vae._h3_910a_finite_select.soc
    assert r['global_sdpa_unchanged'] and r['local_call_count'] == 72
    r['completed'] = True
except BaseException as exc:
    r['failure'] = {'type': type(exc).__name__, 'message': str(exc)}
    raise
finally:
    writer.write(r)
    print(json.dumps(r, indent=2), flush=True)
