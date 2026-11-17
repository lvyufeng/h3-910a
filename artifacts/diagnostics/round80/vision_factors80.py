#!/usr/bin/env python3
"""Same-model FP16 vision bias-rounding/LayerNorm factors; CPU teacher unchanged.

Compare native-before,FP32-biased-linear factor,plus stable LayerNorm factor,
and native-restored. FP32 weights are diagnostic caches derived from the fixed
FP16 parameters,not new weights or a production change. A later fused FP16
GEMM must retain FP32 accumulators through bias to remove these caches.
"""
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
import time
import torch
import torch_npu
from torch import nn
from torch.nn import functional as F
from safetensors import safe_open
from transformers import Qwen3VLConfig
from evidence_io80 import EvidenceWriter

ROOT=Path('/tmp/h3_round80_vision_factors01')
MODEL=Path('/home/mseco/models/MiniMax-H3/FL2VA/text_encoder')
CPU=Path('/tmp/h3_round78_hf_fp1601')
INPUT=Path('/tmp/h3_round54_capture01/encoder_input.pt')
writer=EvidenceWriter(ROOT/'vision_factors.json')
phase='native_before';cache={}
r={'scope':__doc__,'source_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),'phases':{},'production_changed':False,'CPU_teacher_changed':False,'full_encoder_precision_accepted':False,'whole_goal_complete':False}


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb')as f:
        for block in iter(lambda:f.read(8*1024*1024),b''):h.update(block)
    return h.hexdigest()


def raw(x):
    return hashlib.sha256(memoryview(x.detach().cpu().contiguous().view(torch.uint8).numpy())).hexdigest()


def params_sha(model):
    h=hashlib.sha256()
    for name,p in model.named_parameters():h.update(name.encode());h.update(memoryview(p.detach().cpu().contiguous().view(torch.uint8).numpy()))
    return h.hexdigest()


def metric(a,b):
    a=a.detach().cpu().contiguous();b=b.detach().cpu().contiguous();assert a.shape==b.shape and a.dtype==b.dtype==torch.float16
    e=a.double()-b.double();finite=bool(torch.isfinite(a.float()).all()and torch.isfinite(b.float()).all())
    return {'finite':finite,'raw_bits_equal':torch.equal(a.view(torch.uint8),b.view(torch.uint8)),'max_abs_error':float(e.abs().max())if finite else None,'relative_RMS':float(e.square().mean().sqrt()/b.double().square().mean().sqrt().clamp_min(1e-30))if finite else None,'elements':a.numel()}


def compare(out,ref):
    return{'merged':metric(out[0],ref['merged']),'deepstack':[metric(a,b)for a,b in zip(out[1],ref['deepstack'])]}


try:
    torch.set_num_threads(2);torch.npu.set_device(0)
    proof=json.loads((CPU/'reference.json').read_text());assert proof['completed']and proof['repeat_pre_final_norm_bits_equal']
    assert sha(CPU/'FP16_vision.pt')==proof['stages']['FP16_vision']['file_sha256']=='c01bbe180f454fe44b182bd9307491ef0f4c342b6c2ff6be08d1e3453d8c312a'
    reference=torch.load(CPU/'FP16_vision.pt',map_location='cpu',weights_only=True)
    source=ROOT/'candidate_encoder79.py';assert sha(source)=='2141b29adc9ecc601d7fa2599e6ba56351adfbdaa5f16acb62c00d95cee3cf78'
    spec=importlib.util.spec_from_file_location('vision_candidate80',source);candidate=importlib.util.module_from_spec(spec);sys.modules[spec.name]=candidate;spec.loader.exec_module(candidate)
    config=Qwen3VLConfig.from_pretrained(str(MODEL),local_files_only=True).vision_config
    with torch.device('meta'):vision=candidate.MiniMaxH3Qwen3VLVisionModel(config)
    wanted=set(vision.state_dict());index=json.loads((MODEL/'model.safetensors.index.json').read_text())['weight_map'];state={}
    for shard in sorted(set(index.values())):
        keys=[key for key,path in index.items()if path==shard and key.startswith('model.visual.')and key[len('model.visual.'):]in wanted]
        if not keys:continue
        with safe_open(str(MODEL/shard),framework='pt',device='cpu')as f:
            for key in keys:
                original=f.get_tensor(key);assert original.dtype==torch.bfloat16
                value=original.to(torch.float16);name=key[len('model.visual.'):]
                assert raw(value)==proof['parameter_conversion']['visual.'+name]['reference_sha256'];state[name]=value
    assert set(state)==wanted;vision.load_state_dict(state,strict=True,assign=True);del state
    vision=vision.to('npu:0');vision.eval();vision.rotary_pos_emb=candidate.MiniMaxH3Qwen3VLVisionRotaryEmbedding(config.hidden_size//config.num_heads//2)
    assert all(p.dtype==torch.float16 for p in vision.parameters())
    r['parameters_before']=params_sha(vision);r['candidate_sha256']=sha(source)
    inputs=torch.load(INPUT,map_location='cpu',weights_only=True);assert sha(INPUT)=='ed099e792982f9cd37b67f2cff24443008683d0cc819d1decacabf55de1f4ebc'
    pixels=inputs['pixel_values'].to('npu:0');grids=inputs['image_grid_thw'].to('npu:0');before_input=raw(pixels)
    original_linear=candidate._NpuBF16CompatibleLinear.forward;original_norm=candidate._NpuBF16CompatibleLayerNorm.forward
    def linear(self,x):
        if phase in('fp32_bias','fp32_bias_stable_norm')and x.dtype==torch.float16 and self.bias is not None:
            key=(self.weight.data_ptr(),self.bias.data_ptr())
            if key not in cache:cache[key]=(self.weight.float(),self.bias.float())
            w,b=cache[key];return F.linear(x.float(),w,b).to(x.dtype)
        return original_linear(self,x)
    def norm(self,x):
        if phase=='fp32_bias_stable_norm'and x.dtype==torch.float16:return self._stable_bf16_forward(x)
        return original_norm(self,x)
    candidate._NpuBF16CompatibleLinear.forward=linear;candidate._NpuBF16CompatibleLayerNorm.forward=norm
    initial=None
    with torch.inference_mode():
        for name in('native_before','fp32_bias','fp32_bias_stable_norm','native_restored'):
            phase=name
            if name=='native_restored':cache.clear()
            torch.npu.synchronize();started=time.perf_counter();first=vision(pixels,grids);torch.npu.synchronize()
            diag=time.perf_counter()-started
            if initial is None:initial=(first[0].detach().clone(),[v.detach().clone()for v in first[1]])
            checks=compare(first,reference);assert checks['merged']['finite']and all(x['finite']for x in checks['deepstack'])
            rows=[]
            for call in range(3):
                torch.npu.synchronize();started=time.perf_counter();out=vision(pixels,grids);torch.npu.synchronize();seconds=time.perf_counter()-started
                repeat={'merged':metric(out[0],first[0]),'deepstack':[metric(a,b)for a,b in zip(out[1],first[1])]}
                rows.append({'call':call,'resident_vision_s':seconds,'repeat':repeat})
            r['phases'][name]={'excluded_first_s':diag,'CPU_FP16_comparison':checks,'calls':rows,'FP32_cache_bytes':sum(w.numel()*w.element_size()+b.numel()*b.element_size()for w,b in cache.values())}
            if name=='native_restored':
                restored={'merged':metric(first[0],initial[0]),'deepstack':[metric(a,b)for a,b in zip(first[1],initial[1])]};r['native_restoration']=restored
                assert restored['merged']['raw_bits_equal']and all(x['raw_bits_equal']for x in restored['deepstack'])
            writer.write(r)
    candidate._NpuBF16CompatibleLinear.forward=original_linear;candidate._NpuBF16CompatibleLayerNorm.forward=original_norm
    r['parameters_after']=params_sha(vision);r['input_unchanged']=before_input==raw(pixels)
    assert r['parameters_before']==r['parameters_after']and r['input_unchanged']
    r['completed']=True
except BaseException as error:r['failure']={'type':type(error).__name__,'message':str(error)};raise
finally:writer.write(r)
