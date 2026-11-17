#!/usr/bin/env python3
"""Actual vision Linear output-column geometry: full K,strict BF16 output gate.

Single-card serial partition probe,NOT TP8/whole-encoder acceleration. Every
FP32 private-intermediate difference is reported separately from the original
BF16 projection interface; no GEMM precision flag,normalization or bias-fusion
change. Actual inputs/weights/bias/full output are from an excluded current
experimental full encoder forward,not a CPU/HF surrogate activation corpus.
"""
import hashlib
import json
import os
from pathlib import Path
import statistics
import time
import torch
import torch_npu
from torch.nn import functional as F
from safetensors import safe_open
from column_plan75 import WIDTHS,INPUTS,indices,padded_width
from evidence_io75 import EvidenceWriter

ROOT=Path('/tmp/h3_round75_columns02');CAP=Path('/tmp/h3_round75_capture01')
MODEL=Path('/home/mseco/models/MiniMax-H3/FL2VA/text_encoder')
writer=EvidenceWriter(ROOT/'column_probe01.json');torch.set_num_threads(8)
torch.npu.set_device(int(os.environ.get('H3_PROBE_DEVICE','0')))
r={'scope':__doc__,'cases':[],'device_index':torch.npu.current_device(),'source_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
   'math_flags_before':{'matmul_allow_hf32':torch.npu.matmul.allow_hf32,'conv_allow_hf32':torch.npu.conv.allow_hf32},
   'actual_capture_root':str(CAP),'single_card_serial_not_TP8':True,'whole_encoder_acceleration_proven':False,'official_goal_complete':False}
assert not r['math_flags_before']['matmul_allow_hf32']


def raw(x):return hashlib.sha256(memoryview(x.detach().cpu().contiguous().view(torch.uint8).numpy())).hexdigest()


def metric(a,b):
    assert a.shape==b.shape and a.dtype==b.dtype
    x,y=a.detach().cpu().contiguous().reshape(-1),b.detach().cpu().contiguous().reshape(-1)
    different=0;err=0.;ref=0.;maximum=0.;finite=True
    word=torch.int32 if a.dtype==torch.float32 else torch.int16
    for s in range(0,x.numel(),1048576):
        xx,yy=x[s:s+1048576],y[s:s+1048576];different+=int((xx.view(word)!=yy.view(word)).sum())
        xd,yd=xx.double(),yy.double();valid=bool(torch.isfinite(xd).all()and torch.isfinite(yd).all());finite=finite and valid
        if valid:
            e=xd-yd;err+=float(e.square().sum());ref+=float(yd.square().sum());maximum=max(maximum,float(e.abs().max()))
    return {'shape':list(a.shape),'dtype':str(a.dtype),'elements':a.numel(),'different_words':different,'raw_bits_equal':different==0,
            'finite':finite,'max_abs_error':maximum if finite else None,'relative_rms':(err/max(ref,1e-30))**.5 if finite else None}


def layout(x):return{'shape':list(x.shape),'stride':list(x.stride()),'dtype':str(x.dtype),'physical_format':torch_npu.get_npu_format(x)}


def measure(fn):
    torch.npu.synchronize();start=time.perf_counter();out=fn();torch.npu.synchronize()
    return out,time.perf_counter()-start


try:
    index=json.loads((MODEL/'model.safetensors.index.json').read_text())['weight_map']
    r['checkpoint_index_sha256']=hashlib.sha256((MODEL/'model.safetensors.index.json').read_bytes()).hexdigest()
    with torch.inference_mode():
        for name in('qkv','proj','fc1','fc2'):
            meta=json.loads((CAP/(name+'.json')).read_text());file=CAP/(name+'.pt')
            assert file.stat().st_size<128*1024*1024 and hashlib.sha256(file.read_bytes()).hexdigest()==meta['file_sha256']
            # Own capture already verified by file SHA; allow only the installed
            # numeric Format enum,not arbitrary pickle globals or weights_only=False.
            from torch_npu.npu._format import Format
            with torch.serialization.safe_globals([Format]):
                payload=torch.load(file,map_location='cpu',weights_only=True)
            for key in('x','w','b','y'):
                value=payload[key];assert value.dtype==torch.bfloat16 and raw(value)==meta['tensors'][key]['sha256']and bool(torch.isfinite(value.float()).all())
            assert payload['x'].shape==(8064,INPUTS[name])and payload['w'].shape==(WIDTHS[name],INPUTS[name])and payload['y'].shape==(8064,WIDTHS[name])
            assert meta['input_layout']['physical_format']in(0,2)and meta['source_input_and_weight_readonly']
            for role,suffix in(('w','weight'),('b','bias')):
                key=meta['checkpoint_module']+'.'+suffix
                with safe_open(str(MODEL/index[key]),framework='pt',device='cpu')as f:original=f.get_tensor(key)
                assert original.dtype==torch.bfloat16 and raw(original)==raw(payload[role])
            x=payload['x'].npu().float();w=payload['w'].npu().float();b=payload['b'].npu().float()
            before={'x':raw(x),'w':raw(w),'b':raw(b)}
            assert torch_npu.get_npu_format(x)in(0,2)
            def full():
                fp32=F.linear(x,w,b);return fp32,fp32.to(torch.bfloat16)
            (base32,base16),excluded=measure(full)
            row={'name':name,'input_shape':list(x.shape),'full_weight_shape':list(w.shape),'capture_file_sha256':meta['file_sha256'],
                 'full_replay_vs_captured':metric(base16,payload['y']),'full_output_layout':layout(base16),'captured_output_layout':meta['output_layout'],
                 'first_full_s_excluded_compile':excluded,'modes':{}}
            assert row['full_replay_vs_captured']['raw_bits_equal']and row['full_replay_vs_captured']['finite']
            cpu32=base32.cpu();cpu16=base16.cpu()
            (repeat32,repeat16),_=measure(full)
            row['native_repeat_fp32']=metric(repeat32,cpu32);row['native_repeat_bf16']=metric(repeat16,cpu16)
            assert row['native_repeat_fp32']['raw_bits_equal']and row['native_repeat_bf16']['raw_bits_equal']
            for aligned in((False,True)if name=='fc1'else(False,)):
                label='zero_pad_'+str(padded_width(name,True))if aligned else'native_width'
                assembled32=torch.empty_like(cpu32);assembled16=torch.empty_like(cpu16);parts=[]
                for rank in range(8):
                    idx=indices(name,rank);valid=len(idx);width=padded_width(name,aligned)
                    idxcpu=torch.tensor(idx,dtype=torch.int64)
                    wp=payload['w'].index_select(0,idxcpu).float();bp=payload['b'].index_select(0,idxcpu).float()
                    assert bool(torch.equal(wp,payload['w'].float().index_select(0,idxcpu)))and bool(torch.equal(bp,payload['b'].float().index_select(0,idxcpu)))
                    if width>valid:
                        wp=F.pad(wp,(0,0,0,width-valid));bp=F.pad(bp,(0,width-valid))
                        assert not bool(wp[valid:].any())and not bool(bp[valid:].any())
                    wp,bp=wp.npu(),bp.npu()
                    def part():
                        z=F.linear(x,wp,bp);return z,z.to(torch.bfloat16)
                    (y32,y16),first_s=measure(part)
                    c32,c16=y32.cpu(),y16.cpu();actual32,actual16=c32[:,:valid],c16[:,:valid]
                    target32=cpu32.index_select(1,idxcpu);target16=cpu16.index_select(1,idxcpu)
                    p={'partition':rank,'valid_width':valid,'gemm_output_width':width,'full_K_preserved':INPUTS[name],
                       'indices':idx,'index_sha256':hashlib.sha256(json.dumps(idx).encode()).hexdigest(),
                       'fp32_vs_full_columns':metric(actual32,target32),'bf16_vs_full_columns':metric(actual16,target16),
                       'output_layout':layout(y16),'first_s_excluded_compile':first_s}
                    if width>valid:
                        p['padded_fp32_output_zero']=bool((c32[:,valid:]==0).all());p['padded_bf16_output_zero']=bool((c16[:,valid:]==0).all())
                        assert p['padded_fp32_output_zero']and p['padded_bf16_output_zero']
                    warm=[];full_warm=[];repeat_proofs=[]
                    for sample in range(3):
                        _,native_s=measure(full);(again32,again16),part_s=measure(part)
                        full_warm.append(native_s);warm.append(part_s)
                        repeat_proofs.append({'fp32':metric(again32,y32),'bf16':metric(again16,y16)})
                    p['warm_part_s']=warm;p['same_pair_native_s']=full_warm;p['standalone_part_median_s']=statistics.median(warm)
                    p['full_median_same_pair_s']=statistics.median(full_warm);p['repeat_proofs']=repeat_proofs
                    assert all(proof[role]['raw_bits_equal']for proof in repeat_proofs for role in('fp32','bf16'))
                    assembled32[:,idxcpu]=actual32;assembled16[:,idxcpu]=actual16
                    parts.append(p)
                row['modes'][label]={'parts':parts,'all_fp32_private_parts_bits_equal':all(p['fp32_vs_full_columns']['raw_bits_equal']for p in parts),
                                     'all_bf16_interface_parts_bits_equal':all(p['bf16_vs_full_columns']['raw_bits_equal']for p in parts),
                                     'reconstructed_fp32':metric(assembled32,cpu32),'reconstructed_bf16':metric(assembled16,cpu16),
                                     'reconstruction_layout_cpu':{'shape':list(assembled16.shape),'stride':list(assembled16.stride()),'dtype':str(assembled16.dtype)}}
                row['modes'][label]['stable_bf16_output_interface_accepted']=row['modes'][label]['reconstructed_bf16']['raw_bits_equal']and row['modes'][label]['reconstructed_bf16']['finite']
                print(json.dumps({'name':name,'mode':label,'fp32_bits_equal':row['modes'][label]['all_fp32_private_parts_bits_equal'],
                                  'bf16_bits_equal':row['modes'][label]['all_bf16_interface_parts_bits_equal'],
                                  'reconstruction_bf16':row['modes'][label]['reconstructed_bf16']}),flush=True)
            if name=='qkv':
                wrong=torch.cat([cpu16.index_select(1,torch.tensor(indices(name,rank)))for rank in range(8)],dim=1)
                row['wrong_simple_concatenation_counterexample']=metric(wrong,cpu16)
                assert not row['wrong_simple_concatenation_counterexample']['raw_bits_equal']
            row['input_weight_bias_readonly']=before=={'x':raw(x),'w':raw(w),'b':raw(b)};assert row['input_weight_bias_readonly']
            r['cases'].append(row);writer.write(r)
            del base32,base16,repeat32,repeat16,cpu32,cpu16,x,w,b,payload
    r['math_flags_after']={'matmul_allow_hf32':torch.npu.matmul.allow_hf32,'conv_allow_hf32':torch.npu.conv.allow_hf32}
    assert r['math_flags_before']==r['math_flags_after']
    r['all_tested_bf16_column_interfaces_exact']=all(mode['stable_bf16_output_interface_accepted']for row in r['cases']for mode in row['modes'].values())
    r['completed']=True
except BaseException as e:
    r['failure']={'type':type(e).__name__,'message':str(e)}
    r['partial_case']=row if 'row'in globals()else None
    r['partial_partition']=p if 'p'in globals()else None
    raise
finally:writer.write(r)
