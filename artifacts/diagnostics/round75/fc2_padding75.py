#!/usr/bin/env python3
"""Diagnose fc2 output-column shape instability with explicit zero-row padding.

The N=144 current-real-input counterexample is preserved separately. Each new
output width keeps every K=4304 original value and fused bias;extra weight/bias
rows are zero and discarded. No precision/deterministic-runtime global switch,
no numerical repair,and no prior failed shape promoted to a pass. Standalone
single-card geometry is not full TP8 or encoder speed/quality proof.
"""
import hashlib
import json
import os
from pathlib import Path
import torch
import torch_npu
from torch.nn import functional as F
from torch_npu.npu._format import Format
from column_plan75 import indices
from proof_helpers75 import metric,raw,measure
from evidence_io75 import EvidenceWriter

ROOT=Path('/tmp/h3_round75_fc2pads01');CAP=Path('/tmp/h3_round75_capture01')
writer=EvidenceWriter(ROOT/'fc2_padding01.json');torch.set_num_threads(8)
torch.npu.set_device(int(os.environ.get('H3_PROBE_DEVICE','0')))
r={'scope':__doc__,'widths':{},'source_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
   'math_flags_before':{'matmul_allow_hf32':torch.npu.matmul.allow_hf32,'conv_allow_hf32':torch.npu.conv.allow_hf32},
   'whole_encoder_acceleration_proven':False,'official_goal_complete':False}


try:
    meta=json.loads((CAP/'fc2.json').read_text());file=CAP/'fc2.pt'
    assert hashlib.sha256(file.read_bytes()).hexdigest()==meta['file_sha256']
    with torch.serialization.safe_globals([Format]):payload=torch.load(file,map_location='cpu',weights_only=True)
    assert payload['x'].shape==(8064,4304)and payload['w'].shape==(1152,4304)
    assert not r['math_flags_before']['matmul_allow_hf32']
    r['fixture_sha256']=meta['file_sha256'];r['source_inputs_and_weights']={key:raw(payload[key])for key in('x','w','b','y')}
    with torch.inference_mode():
        x=payload['x'].npu().float();w=payload['w'].npu().float();b=payload['b'].npu().float()
        def full():
            y=F.linear(x,w,b);return y,y.to(torch.bfloat16)
        (base32,base16),_=measure(full);base32cpu,base16cpu=base32.cpu(),base16.cpu()
        r['full_replay_bf16_vs_captured']=metric(base16,payload['y']);assert r['full_replay_bf16_vs_captured']['raw_bits_equal']
        for width in(160,192,256,512):
            rows=[];assembled32=torch.empty_like(base32cpu);assembled16=torch.empty_like(base16cpu)
            for rank in range(8):
                idxcpu=torch.tensor(indices('fc2',rank));valid=144
                wp=F.pad(payload['w'].index_select(0,idxcpu).float(),(0,0,0,width-valid))
                bp=F.pad(payload['b'].index_select(0,idxcpu).float(),(0,width-valid))
                assert bool((wp[valid:]==0).all())and bool((bp[valid:]==0).all())
                wp,bp=wp.npu(),bp.npu()
                def candidate():
                    y=F.linear(x,wp,bp);return y,y.to(torch.bfloat16)
                (first32,first16),first_s=measure(candidate);cpu32,cpu16=first32.cpu(),first16.cpu()
                row={'partition':rank,'gemm_output_width':width,'valid_width':valid,'K':4304,
                     'fp32_vs_full':metric(cpu32[:,:valid],base32cpu.index_select(1,idxcpu)),
                     'bf16_vs_full':metric(cpu16[:,:valid],base16cpu.index_select(1,idxcpu)),
                     'first_s_excluded_compile':first_s,'padding_outputs_zero':bool((cpu32[:,valid:]==0).all()and(cpu16[:,valid:]==0).all()),'repeats':[]}
                assert row['padding_outputs_zero']
                for sample in range(3):
                    (again32,again16),seconds=measure(candidate)
                    row['repeats'].append({'seconds':seconds,'fp32':metric(again32,first32),'bf16':metric(again16,first16)})
                row['all_repeat_bf16_bits_equal']=all(p['bf16']['raw_bits_equal']for p in row['repeats'])
                row['all_repeat_fp32_bits_equal']=all(p['fp32']['raw_bits_equal']for p in row['repeats'])
                assembled32[:,idxcpu]=cpu32[:,:valid];assembled16[:,idxcpu]=cpu16[:,:valid]
                rows.append(row)
            r['widths'][str(width)]={'parts':rows,'reconstructed_fp32':metric(assembled32,base32cpu),'reconstructed_bf16':metric(assembled16,base16cpu),
                                    'strict_fp32_private_and_bf16_interface_accepted':all(row['fp32_vs_full']['raw_bits_equal']and row['bf16_vs_full']['raw_bits_equal']and row['all_repeat_bf16_bits_equal']and row['all_repeat_fp32_bits_equal']for row in rows)}
            writer.write(r);print(json.dumps({'width':width,'strict_accepted':r['widths'][str(width)]['strict_fp32_private_and_bf16_interface_accepted'],
                                             'bf16_different_words':r['widths'][str(width)]['reconstructed_bf16']['different_words']}),flush=True)
    r['source_input_weights_unchanged']=r['source_inputs_and_weights']=={key:raw(payload[key])for key in('x','w','b','y')}
    assert r['source_input_weights_unchanged']
    r['math_flags_after']={'matmul_allow_hf32':torch.npu.matmul.allow_hf32,'conv_allow_hf32':torch.npu.conv.allow_hf32}
    assert r['math_flags_before']==r['math_flags_after'];r['completed']=True
except BaseException as e:
    r['failure']={'type':type(e).__name__,'message':str(e)}
    r['partial_width_rows']=rows if 'rows'in globals()else[]
    raise
finally:writer.write(r)
