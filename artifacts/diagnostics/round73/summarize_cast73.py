#!/usr/bin/env python3
"""Full encode_ids stridedcast factors with original attention output proofs."""
from collections import Counter
import hashlib
import json
import math
from pathlib import Path
import statistics
import sys
sys.path.insert(0,str(Path(__file__).resolve().parent.parent/'round70'))
from summarize import exact,validate_identities

ROOT=Path(__file__).resolve().parent


def aggregate(rows,previous):
    if len(rows)!=8 or {row['rank']for row in rows}!=set(range(8))or any(not row.get('completed')or row.get('failure')for row in rows):raise ValueError('Need eight completed genuineNPU encoder A/B ranks')
    rows=sorted(rows,key=lambda row:row['rank']);old={row['rank']:row for row in previous}
    reference=old[0]['cases']['tail']['first_identities']
    r={'scope':'Exact finite strided BF16 cast to native contiguousFP32 BNSD only; allQK/scale/softmax/PV/BF16return unchanged. Fullresidentencode_ids originalFP16pixels/originalBF16weights and nativeCPUoutput; upstreamHTTPprocessor/broadcast/load/offload excluded. NaNpayload counterexample retained, not a genericdtype or HFlossless proof.',
       'cases':{},'complete_output_identity_accepted':False,'complete_encode_ids_under2s_accepted':False,
       'whole_service_text_under2s_proven':False,'official_goal_complete':False,'production_deployed':False,
       'actual_cast_checks':0,'actual_attention_output_checks':0}
    identities,proofs=[],[]
    for row in rows:
        prev=old[row['rank']]
        if row['parameters_before']!=row['parameters_after']or row['parameters_before']!=prev['parameters_before']or row['actual_inputs_before']!=row['actual_inputs_after']or row['actual_inputs_before']!=prev['actual_inputs_before']:raise ValueError('Original input/parameter bytes changed')
        for field in('environment','math_flags_before','math_flags_after','checkpoint_root','checkpoint_index_sha256','capture_file_sha256','parameter_dtypes','loaded_parameter_tensors','component_parameter_bytes'):
            if row[field]!=prev[field]:raise ValueError('Actual source/precision configuration changed')
        if not row['attention_restored']or not row['vision_rope_restored']or not row['vision_wrappers_restored']:raise ValueError('Original vision state not restored')
        if not exact(row['first_vs_round70_fresh_reference'],10460160):raise ValueError('Full native attention baseline differs from reference')
    for mode in('native','cast','restored'):
        cases=[row['cases'][mode]for row in rows]
        for case in cases:
            if case['mode']!=mode or len(case['warm_calls'])!=3:raise ValueError('Wrong actual mode/warm repeat count')
            checks=case['actual_attention_checks']
            if mode=='cast':
                if len(checks)!=54 or Counter(x['call']for x in checks)!=Counter({i:1 for i in range(54)}):raise ValueError('Missing actual complete attention calls')
                for check in checks:
                    if check['layer']!=check['call']//2 or check['image_segment']!=check['call']%2 or not exact(check['attention_output'],4644864)or not check['input_values_readonly']or check['output_layout']!=check['original_output_layout']:raise ValueError('Actual full attention bit/layout/readonly proof failed')
                    casts=check['input_cast_proofs']
                    if len(casts)!=3 or Counter(x['operand']for x in casts)!={'q':1,'k':1,'v':1}:raise ValueError('Missing all3actualinputcast proofs')
                    for cast in casts:
                        if not exact(cast['proof'],4644864)or cast['candidate_layout']!=cast['native_layout']:raise ValueError('Actual stridedFP32castbits/layout changed')
                        pitch=3456 if cast['operand']=='v'else 1152
                        if cast['input_layout']['shape']!=[1,16,4032,72]or cast['input_layout']['stride'][1:]!=[72,pitch,1]or cast['input_layout']['dtype']!='torch.bfloat16':raise ValueError('Actual strided cast input geometry changed')
                        if cast['candidate_layout']['shape']!=[1,16,4032,72]or cast['candidate_layout']['stride']!=[4644864,290304,72,1]or cast['candidate_layout']['dtype']!='torch.float32':raise ValueError('Native contiguousBNSD output changed')
                        r['actual_cast_checks']+=1
                    r['actual_attention_output_checks']+=1
            elif checks:raise ValueError('Native/restored unexpectedly containscandidatechecks')
            validate_identities(case['first_identities']);identities.append(case['first_identities']);proofs.append(case['first_vs_baseline'])
            for call in case['warm_calls']:
                if call['actual_attention_calls']!=54 or not math.isfinite(call['seconds'])or call['seconds']<=0:raise ValueError('Actual attention call/timer changed')
                validate_identities(call['identities']);identities.append(call['identities']);proofs.append(call['vs_baseline'])
                if call['output']!=call['identities']['complete_output']:raise ValueError('WrongCPUoutputproof')
            if case['vision_shadow_cache']!={'entries':232,'cached_fp32_bytes':2362750400,'misses':232,'hits':696,'invalidations':0,'uncacheable_bypasses':0,'max_bytes':2362750400}:raise ValueError('Fixed vision shadow changed')
            if case['text_shadow_cache']['entries']!=200 or case['text_shadow_cache']['cached_fp32_bytes']!=11927552000 or case['existing_padding_cache']!={'entries':100,'cached_fp32_bytes':2097152000}:raise ValueError('Fixedtextcachechanged')
        maxima=[max(case['warm_calls'][i]['seconds']for case in cases)for i in range(3)]
        r['cases'][mode]={'same_call_rank_max_s':maxima,'median_complete_encode_ids_s':statistics.median(maxima),
                          'first_max_s_excluded_compile_doublecompute':max(case['first_s_excluded_doublecompute_compile']for case in cases),
                          'allocated_hbm_bytes_per_rank':[case['hbm_allocated_bytes']for case in cases],
                          'reserved_hbm_bytes_per_rank':[case['hbm_reserved_bytes']for case in cases]}
    r['intermediate_identity_against_round72']={name:all(identity[name]==reference[name]for identity in identities)for name in reference}
    r['complete_outputs_compared']=len(identities)
    r['complete_output_identity_accepted']=all(r['intermediate_identity_against_round72'].values())and all(exact(p,10460160)for p in proofs)
    r['complete_encode_ids_under2s_accepted']=r['complete_output_identity_accepted']and all(v<2 for v in r['cases']['cast']['same_call_rank_max_s'])
    native=r['cases']['native']['median_complete_encode_ids_s'];candidate=r['cases']['cast']['median_complete_encode_ids_s']
    r['speedup_vs_same_run_native']=native/candidate
    r['restored_timing_drift']=r['cases']['restored']['median_complete_encode_ids_s']/native-1
    return r


def summarize(root=ROOT):
    rows=[json.loads((root/'ranks'/f'cast_rank{i}.json').read_text())for i in range(8)]
    previous=[json.loads((root.parent/'round72'/'ranks'/f'rope_rank{i}.json').read_text())for i in range(8)]
    for row in rows:
        if set(row['sources'])!={'cast_attention_fixed_tp8.py','cast_attention73.py','candidate_padded_text_encoder.py','evidence_io73.py'}:raise ValueError('Missing actual executedsources')
        for name,digest in row['sources'].items():
            path=root/name
            if name=='candidate_padded_text_encoder.py':path=root.parent/'round65'/'source'/name
            elif name=='evidence_io73.py':path=root.parent/'round63'/'evidence_io63.py'
            if hashlib.sha256(path.read_bytes()).hexdigest()!=digest:raise ValueError('Executedsourcefingerprintdiffers')
    return aggregate(rows,previous)


if __name__=='__main__':
    r=summarize();(ROOT/'summary.json').write_text(json.dumps(r,indent=2)+'\n');print(json.dumps(r,indent=2))
