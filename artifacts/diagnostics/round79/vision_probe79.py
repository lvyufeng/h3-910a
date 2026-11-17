#!/usr/bin/env python3
"""Scoped 910A FP16 vision probe against fresh independent CPU FP16 vision tensors.

Single-rank vision-only candidate;native FP16 weights/linear/Conv/LayerNorm,
original vision-RoPE FP32 and explicit FP32 QK/softmax/PV. No BF16 I/O,
no production edits,old BF16 teacher,global functional patch or full-text claim.
CPU MATH and NPU attention reduction/order differences are measured,not waived.
This stage probe is not full encoder<=2s or whole49-step lossless acceptance.
"""
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import signal
import sys
import time
import torch
import torch_npu
from torch import nn
from safetensors import safe_open
from transformers import Qwen3VLConfig
from evidence_io79 import EvidenceWriter

ROOT=Path('/tmp/h3_round79_fp16_vision01')
MODEL=Path('/home/mseco/models/MiniMax-H3/FL2VA/text_encoder')
REFERENCE_ROOT=Path('/tmp/h3_round78_hf_fp1601')
INPUT=Path('/tmp/h3_round54_capture01/encoder_input.pt')
writer=EvidenceWriter(ROOT/'vision.json')
r={'scope':__doc__,'source_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),'device':'npu:0','parameter_dtype':'FP16',
   'cases':[],'layers':[],'production_changed':False,'CPU_full_reference_repeat_verified':False,'full_encoder_precision_accepted':False,'whole_goal_complete':False}
handles=[]


def file_sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb')as f:
        for block in iter(lambda:f.read(8*1024*1024),b''):h.update(block)
    return h.hexdigest()


def tensor_info(value):
    data=value.detach().cpu().contiguous()
    return {'shape':list(data.shape),'dtype':str(data.dtype),'finite':bool(torch.isfinite(data.float()).all()),
            'max_abs':float(data.float().abs().max()),'sha256':hashlib.sha256(memoryview(data.view(torch.uint8).numpy())).hexdigest()}


def metric(value,reference):
    a=value.detach().cpu().contiguous();b=reference.detach().cpu().contiguous()
    assert a.shape==b.shape and a.dtype==b.dtype==torch.float16
    delta=a.double()-b.double();finite=bool(torch.isfinite(a.float()).all()and torch.isfinite(b.float()).all())
    return {'raw_bits_equal':torch.equal(a.view(torch.uint8),b.view(torch.uint8)),'elements':a.numel(),'finite':finite,
            'max_abs_error':float(delta.abs().max())if finite else None,
            'relative_rms':float(delta.square().mean().sqrt()/b.double().square().mean().sqrt().clamp_min(1e-30))if finite else None,
            'RMSE':float(delta.square().mean().sqrt())if finite else None}


try:
    signal.signal(signal.SIGTERM,lambda sig,frame:(_ for _ in()).throw(SystemExit('FP16visionprobe interrupted')))
    torch.set_num_threads(2);torch.npu.set_device(0)
    for name in('VLLM_OMNI_910A_TE_DTYPE','VLLM_OMNI_910A_TEXT_FP32_LINEAR','VLLM_OMNI_910A_TEXT_PAD_KV','VLLM_OMNI_910A_VISION_FP32'):
        r.setdefault('environment',{})[name]=os.environ.get(name)
    assert os.environ.get('VLLM_OMNI_910A_TE_DTYPE')=='half'
    assert os.environ.get('VLLM_OMNI_910A_TEXT_FP32_LINEAR','0')=='0'and os.environ.get('VLLM_OMNI_910A_TEXT_PAD_KV','0')=='0'
    assert file_sha(INPUT)=='ed099e792982f9cd37b67f2cff24443008683d0cc819d1decacabf55de1f4ebc'
    proof=json.loads((REFERENCE_ROOT/'reference.json').read_text())
    assert proof['source_sha256']=='50165e77adb393b4573fbe0fdb29c62a6ddb14b8f39082d0ea67e904848010ea'
    assert proof['parameter_dtypes']=={'torch.float16':903}
    oracle_path=REFERENCE_ROOT/'FP16_vision.pt';assert file_sha(oracle_path)==proof['stages']['FP16_vision']['file_sha256']=='c01bbe180f454fe44b182bd9307491ef0f4c342b6c2ff6be08d1e3453d8c312a'
    reference=torch.load(oracle_path,map_location='cpu',weights_only=True)
    r['CPU_reference_file_sha256']=file_sha(oracle_path);r['CPU_reference_tensors']={name:[tensor_info(x)for x in value]if isinstance(value,list)else tensor_info(value)for name,value in reference.items()}
    source=ROOT/'candidate_encoder79.py';manifest=json.loads((ROOT/'candidate_manifest79.json').read_text())
    assert file_sha(source)==manifest['candidate_sha256']
    r['candidate_source_sha256']=file_sha(source);r['candidate_manifest_sha256']=file_sha(ROOT/'candidate_manifest79.json')
    spec=importlib.util.spec_from_file_location('candidate_encoder79',source);candidate=importlib.util.module_from_spec(spec);sys.modules[spec.name]=candidate;spec.loader.exec_module(candidate)
    config=Qwen3VLConfig.from_pretrained(str(MODEL),local_files_only=True).vision_config
    with torch.device('meta'):vision=candidate.MiniMaxH3Qwen3VLVisionModel(config)
    wanted=set(vision.state_dict());index=json.loads((MODEL/'model.safetensors.index.json').read_text())['weight_map'];state={}
    expected_source_tensors=proof['parameter_conversion']
    for shard in sorted(set(index.values())):
        keys=[key for key,filename in index.items()if filename==shard and key.startswith('model.visual.')and key[len('model.visual.'):]in wanted]
        if not keys:continue
        with safe_open(str(MODEL/shard),framework='pt',device='cpu')as f:
            for key in keys:
                value=f.get_tensor(key);assert value.dtype==torch.bfloat16
                value=value.to(torch.float16);name=key[len('model.visual.'):]
                assert tensor_info(value)['sha256']==expected_source_tensors['visual.'+name]['reference_sha256']
                state[name]=value
    assert set(state)==wanted
    vision.load_state_dict(state,strict=True,assign=True);del state
    vision=vision.to(device='npu:0');vision.eval()
    assert all(p.dtype==torch.float16 and p.device.type in('npu','privateuseone')for p in vision.parameters())
    # The rotary attribute isn't a buffer and must be recreated outside the meta
    # context to preserve original F32 construction instead of a meta tensor.
    vision.rotary_pos_emb=candidate.MiniMaxH3Qwen3VLVisionRotaryEmbedding(config.hidden_size//config.num_heads//2)
    r['parameter_tensors']=len(wanted);r['parameter_bytes']=sum(p.numel()*p.element_size()for p in vision.parameters())
    r['attention_policy']='DeclaredvisionF32QK/softmax/PVwithF16QKVandoutput;noR73BF16codec'
    input_payload=torch.load(INPUT,map_location='cpu',weights_only=True);pixels=input_payload['pixel_values'].to('npu:0');grids=input_payload['image_grid_thw'].to('npu:0')
    assert pixels.dtype==torch.float16
    r['actual_input_pixels']=tensor_info(pixels);r['actual_input_grid']=grids.cpu().tolist()
    def layer_post(index):
        def hook(module,args,out):
            row={'layer':index,'output':tensor_info(out)};r['layers'].append(row)
            assert row['output']['finite']and row['output']['dtype']=='torch.float16'
            writer.write(r)
        return hook
    for index,layer in enumerate(vision.blocks):handles.append(layer.register_forward_hook(layer_post(index)))
    with torch.inference_mode():
        torch.npu.synchronize();started=time.perf_counter();merged,deepstack=vision(pixels,grids);torch.npu.synchronize()
    r['excluded_diagnostic_first_wall_s']=time.perf_counter()-started
    r['CPU_FP16_comparison']={'merged':metric(merged,reference['merged']),'deepstack':[metric(a,b)for a,b in zip(deepstack,reference['deepstack'])]}
    assert len(deepstack)==len(reference['deepstack'])==3 and len(r['layers'])==27
    assert r['CPU_FP16_comparison']['merged']['finite']and all(x['finite']for x in r['CPU_FP16_comparison']['deepstack'])
    for handle in handles:handle.remove()
    handles.clear()
    with torch.inference_mode():
        for index in range(3):
            torch.npu.synchronize();started=time.perf_counter();a,b=vision(pixels,grids);torch.npu.synchronize()
            row={'call':index,'resident_vision_s':time.perf_counter()-started,'merged_repeat':metric(a,merged),'deepstack_repeat':[metric(x,y)for x,y in zip(b,deepstack)]}
            assert row['merged_repeat']['raw_bits_equal']and all(x['raw_bits_equal']for x in row['deepstack_repeat'])
            r['cases'].append(row);writer.write(r)
    r['completed']=True;r['vision_FP16_finite_and_repeat_stable']=True
    # Numerical acceptance/tolerances are intentionally not self-selected here.
    r['full_encoder_precision_accepted']=False
except BaseException as error:r['failure']={'type':type(error).__name__,'message':str(error)};raise
finally:
    for handle in handles:handle.remove()
    writer.write(r)
