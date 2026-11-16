#!/usr/bin/env python3
"""Same actual layer0 normalized input/weights: repeat narrow FP32 projections.

No collectives, attention or new fused kernels. Compare input casts/resident
operands, explicit MM, dimensions and deterministic mode; retain raw hashes.
"""
import hashlib
import json
from pathlib import Path
import statistics
import time
import torch
import torch_npu
from torch.nn import functional as F
from safetensors import safe_open
from transformers import Qwen3VLConfig

ROOT = Path('/tmp/h3_round59_repeat01')
MODEL = Path('/home/mseco/models/MiniMax-H3/FL2VA/text_encoder')
OUT = ROOT / 'projection_repeat.json'
if OUT.exists():
    raise RuntimeError('Refuse projection repeat evidence overwrite')
torch.npu.set_device(0)
torch.set_num_threads(16)
r = {'cases': [], 'scope': 'Actual rank0 layer0 Q/K/V FP32 linear repeats only, same normalized input; no attention/HCCL/full quality attribution'}


def digest(t):
    cpu = t.detach().cpu().contiguous()
    return hashlib.sha256(memoryview(cpu.view(torch.uint8).numpy())).hexdigest()


def metric(a, b):
    a, b = a.float().double(), b.float().double()
    e = a-b
    return {'max_abs_error': float(e.abs().max()), 'relative_rms': float(e.square().mean().sqrt()/b.square().mean().sqrt().clamp_min(1e-30))}


try:
    with torch.inference_mode():
        index = json.loads((MODEL/'model.safetensors.index.json').read_text())['weight_map']
        config = Qwen3VLConfig.from_pretrained(str(MODEL)).text_config
        def weight(key):
            with safe_open(str(MODEL/index[key]), framework='pt', device='cpu') as f:
                return f.get_tensor(key).bfloat16()
        prefix = 'model.language_model.layers.0.'
        gc = weight(prefix+'input_layernorm.weight')
        xc = torch.load(Path('/tmp/h3_round54_capture01/text_inputs.pt'), weights_only=True)['inputs_embeds']
        x0, gamma = xc.to('npu'), gc.to('npu')
        xf = x0.float()
        normalized = (xf*torch.rsqrt(xf.square().mean(-1, keepdim=True)+config.rms_norm_eps)).bfloat16()
        x = (normalized.float()*gamma.float()).bfloat16()
        expected = json.loads((ROOT/'repeat_audit_rank0.json').read_text())
        normal_hash = next(row['full_sha256'] for row in expected['stage_rows'] if row['phase']=='hook0' and row['key']=='layer0_norm1')
        r['normalized_input_sha256'], r['matches_actual_audit_norm_input'] = digest(x), digest(x)==normal_hash
        assert r['matches_actual_audit_norm_input']
        r['context'] = {'autocast_enabled': torch.is_autocast_enabled('npu'), 'matmul_allow_hf32': torch.npu.matmul.allow_hf32}
        xf = x.float().contiguous(); x_cpu = x.cpu().float()
        inputs = {'normalized_input': x.cpu()}
        for label, n in (('q', 1024), ('k',128), ('v',128)):
            wc = weight(prefix+'self_attn.'+label+'_proj.weight')[:n].contiguous()
            w = wc.to('npu'); wf = w.float().contiguous()
            teacher = F.linear(x_cpu, wc.float())
            inputs[label+'_weight'] = wc
            padded = {p: F.pad(wf,(0,0,0,p-n)).contiguous() for p in (256,512,1024) if p>=n}
            xp = F.pad(xf,(0,0,0,5)).contiguous()
            variants = [('original_recast_linear', lambda: F.linear(x.float(),w.float())),
                        ('resident_fp32_linear', lambda: F.linear(xf,wf)),
                        ('explicit_resident_mm', lambda: torch.mm(xf.reshape(-1,xf.shape[-1]),wf.T).reshape(1,-1,n)),
                        ('padded_rows_2048',lambda: F.linear(xp,wf)[:,:2043].contiguous())]
            variants += [(f'padded_outputs_{p}', lambda p=p: F.linear(xf,padded[p])[...,:n].contiguous()) for p in padded]
            for name, call in variants:
                call(); torch.npu.synchronize()
                values, times, hashes, hashes_bf16 = [], [], [], []
                for _ in range(20):
                    t=time.perf_counter(); out=call(); torch.npu.synchronize(); times.append((time.perf_counter()-t)*1000)
                    cpu=out.cpu().contiguous(); values.append(cpu); hashes.append(digest(cpu)); hashes_bf16.append(digest(cpu.bfloat16()))
                row={'projection':label,'variant':name,'output_shape':list(values[0].shape),'median_ms':statistics.median(times),
                     'unique_fp32_outputs':len(set(hashes)),'unique_bf16_outputs':len(set(hashes_bf16)),
                     'fp32_repeat_hashes':hashes,'bf16_repeat_hashes':hashes_bf16,
                     'first_vs_cpu_fp32':metric(values[0],teacher),'last_vs_first':metric(values[-1],values[0])}
                r['cases'].append(row); print(json.dumps(row),flush=True)
                OUT.write_text(json.dumps(r,indent=2)+'\n')
            original = torch.are_deterministic_algorithms_enabled()
            try:
                torch.use_deterministic_algorithms(True)
                values=[];times=[]
                for _ in range(20):
                    t=time.perf_counter();out=F.linear(xf,wf);torch.npu.synchronize();times.append((time.perf_counter()-t)*1000);values.append(out.cpu().contiguous())
                row={'projection':label,'variant':'deterministic_resident_linear','unique_fp32_outputs':len({digest(v) for v in values}),
                     'unique_bf16_outputs':len({digest(v.bfloat16()) for v in values}),'median_ms':statistics.median(times),
                     'first_vs_cpu_fp32':metric(values[0],teacher),'last_vs_first':metric(values[-1],values[0])}
            except Exception as exc:
                row={'projection':label,'variant':'deterministic_resident_linear','failure':{'type':type(exc).__name__,'message':str(exc)}}
            finally:
                torch.use_deterministic_algorithms(original)
            r['cases'].append(row); print(json.dumps(row),flush=True)
        target=ROOT/'evidence/projection_actual_inputs.pt'
        if target.exists():raise RuntimeError('Refuse actual projection input overwrite')
        assert sum(v.numel()*v.element_size() for v in inputs.values())<512*1024*1024
        with target.open('xb') as f:torch.save(inputs,f)
        r['normalized_input_unchanged']=digest(x)==normal_hash
        r['completed']=True
except BaseException as exc:
    r['failure']={'type':type(exc).__name__,'message':str(exc)}
    raise
finally:
    OUT.write_text(json.dumps(r,indent=2)+'\n');print('Evidence:',OUT,flush=True)
