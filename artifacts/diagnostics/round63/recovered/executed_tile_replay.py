#!/usr/bin/env python3
"""Same-input complete native VAE tile replay and exact FP16-linear materialization A/B.

This measures one actual 7×16×16 latent tile through the36-layer decoder, not
native whole-video tiling/gathers/offload or original-BF16 quality acceptance.
"""
import collections
import gc
import json
import os
from pathlib import Path
import statistics
import time
import torch
import torch_npu

ROOT=Path('/tmp/h3_round63_decode01');CAP=ROOT/'capture'/'rank0'
MODEL=Path('/home/mseco/models/MiniMax-H3-native-fp16-audit53/FL2VA/video_vae')
OUT=ROOT/'tile_replay.json'
if OUT.exists():raise RuntimeError('Refuse VAE tile replay proof overwrite')
os.environ.pop('VLLM_OMNI_910A_VAE_AUDIT_DIR',None)
torch.npu.set_device(0);torch.set_num_threads(8)
from vllm_omni.diffusion.models.minimax_h3.vae import MiniMaxH3VideoVAE

r={'cases':{},'scope':'Actual same-input complete36-layer decoder tile, no post-quant conv/tile gather/temporal blend/pixel quantizer/offload; not whole video or official warmed baseline.'}


def metric(a,b):
    x,y=a.detach().cpu().contiguous(),b.detach().cpu().contiguous();e=x.double()-y.double()
    return {'raw_bits_equal':torch.equal(x.view(torch.uint8),y.view(torch.uint8)),'finite':bool(torch.isfinite(x).all() and torch.isfinite(y).all()),
            'elements':x.numel(),'max_abs_error':float(e.abs().max()),'relative_rms':float(e.square().mean().sqrt()/y.double().square().mean().sqrt().clamp_min(1e-30))}


def once(decoder,x):
    with torch.autocast('npu',dtype=torch.float16,enabled=True):return decoder(x)


def measured(decoder,x):
    torch.npu.synchronize();t=time.perf_counter();first=once(decoder,x);torch.npu.synchronize();first_s=time.perf_counter()-t
    # Each context is fresh, like separate decode entries. The nested group
    # keeps one context alive across identical tile forwards, like tiling.
    fresh=[];fresh_out=[]
    for _ in range(3):
        torch.npu.synchronize();t=time.perf_counter();out=once(decoder,x);torch.npu.synchronize();fresh.append(time.perf_counter()-t);fresh_out.append(out.cpu().contiguous())
    same=[];same_out=[]
    with torch.autocast('npu',dtype=torch.float16,enabled=True):
        decoder(x);torch.npu.synchronize()
        for _ in range(3):
            t=time.perf_counter();out=decoder(x);torch.npu.synchronize();same.append(time.perf_counter()-t);same_out.append(out.cpu().contiguous())
    return same_out[-1],{'first_s':first_s,'fresh_context_groups_s':fresh,'same_context_groups_s':same,
                       'median_fresh_s':statistics.median(fresh),'median_same_context_s':statistics.median(same),
                       'first_vs_repeat':metric(first,same_out[0]),'fresh_repeats_equal':all(metric(z,fresh_out[0])['raw_bits_equal'] for z in fresh_out),
                       'same_repeats_equal':all(metric(z,same_out[0])['raw_bits_equal'] for z in same_out),'fresh_same_output':metric(fresh_out[-1],same_out[-1])}


try:
    metadata=json.loads((CAP/'decode_profile.json').read_text())
    assert metadata['autocast_enabled'] and metadata['autocast_dtype']=='torch.float16'
    assert metadata['parameter_inference_tensor_counts']=={'False':439}
    # Actual production parameters are ordinary FP32 leaves, not inference
    # tensors. Construct outside inference mode to preserve cache eligibility.
    torch.set_default_dtype(torch.float32)
    t=time.perf_counter()
    vae=MiniMaxH3VideoVAE(str(MODEL),device=torch.device('npu',0),load_device=torch.device('cpu'),decode_only=True,trust_remote_code=True)
    vae.remote.to(torch.device('npu',0));torch.npu.synchronize()
    decoder=vae.model.decoder.eval();r['load_and_device_s']=time.perf_counter()-t
    r['parameter_bytes']=sum(p.numel()*p.element_size() for p in decoder.parameters())
    r['parameter_dtypes']=dict(collections.Counter(str(p.dtype) for p in decoder.parameters()))
    r['parameter_inference_counts']=dict(collections.Counter(str(p.is_inference()) for p in decoder.parameters()))
    assert r['parameter_bytes']==metadata['decoder_parameter_bytes'] and r['parameter_inference_counts']=={'False':439}
    x=torch.load(CAP/'decoder_tile_input.pt',weights_only=True).to('npu').contiguous()
    actual=torch.load(CAP/'decoder_tile_output.pt',weights_only=True)
    r['shape'],r['dtype']=list(x.shape),str(x.dtype)
    baseline=None
    with torch.inference_mode():
        out,receipt=measured(decoder,x);receipt['vs_actual_capture']=metric(out,actual)
        baseline=out;r['cases']['fp32_storage']=receipt;OUT.write_text(json.dumps(r,indent=2)+'\n');print(json.dumps({'mode':'fp32_storage',**receipt}),flush=True)
        # Persist the same FP16 values that the existing NPU autocast consumes;
        # retain FP32 norm/scale/register/output parameters. No alternate GEMM
        # accumulation or whole-model precision policy is introduced.
        linears=[linear for block in decoder.transformer_blocks for linear in (block.attn.to_qkv,block.attn.to_out,block.ff.w1,block.ff.w2)]
        assert len(linears)==144 and all(m.weight.dtype==torch.float32 for m in linears)
        backups=[(m,m.weight.detach().cpu().clone(),m.bias.detach().cpu().clone() if m.bias is not None else None) for m in linears]
        torch.clear_autocast_cache();torch.npu.synchronize();t=time.perf_counter()
        for m in linears:m.to(dtype=torch.float16)
        torch.npu.synchronize();r['fp16_materialize_s']=time.perf_counter()-t
        r['fp16_parameter_bytes']=sum(p.numel()*p.element_size() for p in decoder.parameters())
        out,receipt=measured(decoder,x);receipt['vs_actual_capture']=metric(out,actual);receipt['vs_fp32_storage']=metric(out,baseline)
        r['cases']['fp16_block_linears']=receipt;OUT.write_text(json.dumps(r,indent=2)+'\n');print(json.dumps({'mode':'fp16_block_linears',**receipt}),flush=True)
        torch.clear_autocast_cache()
        for m,w,b in backups:
            m.to(dtype=torch.float32);m.weight.copy_(w.to(m.weight.device))
            if b is not None:m.bias.copy_(b.to(m.bias.device))
        torch.npu.synchronize()
        restored=once(decoder,x);torch.npu.synchronize()
        r['restored_fp32_output']=metric(restored,baseline)
        assert r['restored_fp32_output']['raw_bits_equal']
    r['completed']=True
except BaseException as exc:r['failure']={'type':type(exc).__name__,'message':str(exc)};raise
finally:
    OUT.write_text(json.dumps(r,indent=2)+'\n');print('Evidence:',OUT,flush=True)
