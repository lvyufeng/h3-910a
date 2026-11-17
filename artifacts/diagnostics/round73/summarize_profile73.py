#!/usr/bin/env python3
"""Actual vision attention inner-profile reconciliation; nested levels stay separate."""
from collections import Counter
import hashlib
import json
import math
from pathlib import Path
import statistics
import sys
sys.path.insert(0,str(Path(__file__).resolve().parent.parent/'round70'))
from summarize import exact,validate_identities,tensor_contract

ROOT=Path(__file__).resolve().parent
CHILDREN=('q_cast','q_contiguous','k_cast','k_contiguous','v_cast','v_contiguous','qk','scale','softmax','pv','output_cast')


def vector(records):
    counts=Counter(row['name'] for row in records)
    expected={'parent.vision':1,'parent.attention':54,**{'child.'+name:54 for name in CHILDREN}}
    if counts!=expected:raise ValueError('Expected649 actual nested vision attention events')
    sums={};host={}
    for row in records:
        if any(not math.isfinite(value) or value<0 for value in (row['current_stream_elapsed_ms'],row['host_dispatch_s'])):raise ValueError('Invalid event/host duration')
        name=row['name'];sums[name]=sums.get(name,0.)+row['current_stream_elapsed_ms']/1000
        host[name]=host.get(name,0.)+row['host_dispatch_s']
    children=sum(sums['child.'+name] for name in CHILDREN)
    return {'current_stream_category_s':sums,'host_dispatch_category_s':host,'child_sum_s':children,
            'attention_parent_minus_children_s':sums['parent.attention']-children,
            'vision_parent_minus_attention_s':sums['parent.vision']-sums['parent.attention'],
            'note':'Inclusive vision/attention parents are never summed with children; host intervals are distinct and not added to device durations. Gaps unassigned, not clamped.'}


def aggregate(rows,previous):
    if len(rows)!=8 or {row['rank'] for row in rows}!=set(range(8)) or any(not row.get('completed') or row.get('failure') for row in rows):raise ValueError('Need eight completed genuineNPU profiles')
    rows=sorted(rows,key=lambda row:row['rank']);old={row['rank']:row for row in previous}
    reference=old[0]['cases']['tail']['first_identities']
    ids=[];proofs=[];r={'scope':'Unchanged F32 vision attention transcription clean/profile/clean controls with original-source output checks on4capturedcalls. Full encode_ids CPUoutput included; extra event reads/proofs and first captures excluded. Not independentHF/officialquality. Original/wrapped/restored controls are separate.',
                       'phases':{},'complete_output_identity_accepted':False,'whole_service_text_under2s_proven':False,'official_goal_complete':False,'production_deployed':False}
    for row in rows:
        p=old[row['rank']]
        if row['parameters_before']!=row['parameters_after'] or row['parameters_before']!=p['parameters_before'] or row['actual_inputs_before']!=row['actual_inputs_after'] or row['actual_inputs_before']!=p['actual_inputs_before']:raise ValueError('Input/parameter bytes changed')
        for field in ('environment','math_flags_before','math_flags_after','checkpoint_root','checkpoint_index_sha256','component_parameter_bytes','parameter_dtypes','loaded_parameter_tensors'):
            if row[field]!=p[field]:raise ValueError('Original precision/model provenance changed')
        if not row['vision_rope_restored'] or not row['vision_wrappers_restored'] or not row['attention_stage_wrappers_restored']:raise ValueError('Isolated wrappers did not restore')
        if not exact(row['first_vs_round70_fresh_reference'],10460160):raise ValueError('Profile first whole reference mismatch')
        validate_identities(row['first_identities']);ids.append(row['first_identities'])
        proofs.append(row['first_vs_round70_fresh_reference'])
        if row['vision_shadow_cache']!={'entries':232,'cached_fp32_bytes':2362750400,'misses':232,'hits':2088,'invalidations':0,'uncacheable_bypasses':0,'max_bytes':2362750400}:raise ValueError('Fixed vision shadow cache changed')
        if row['text_shadow_cache']['entries']!=200 or row['text_shadow_cache']['cached_fp32_bytes']!=11927552000 or row['existing_padding_cache']!={'entries':100,'cached_fp32_bytes':2097152000}:raise ValueError('Fixed text cache changed')
    for phase in ('clean_before','profile','clean_after'):
        for row in rows:
            calls=row['cases'][phase]
            if len(calls)!=3:raise ValueError('Expectedthree clean/profile/clean calls')
            for call in calls:
                if call['actual_attention_calls']!=54 or not math.isfinite(call['seconds']) or call['seconds']<=0:raise ValueError('Wrong actual attention calls/timer')
                validate_identities(call['identities']);ids.append(call['identities']);proofs.append(call['vs_baseline'])
                if call['output']!=call['identities']['complete_output']:raise ValueError('Wrong CPUoutputreceipt')
                if phase=='profile':vector(call['event_records'])
                elif 'event_records'in call:raise ValueError('Clean call contains events')
        maxima=[max(row['cases'][phase][i]['seconds'] for row in rows) for i in range(3)]
        r['phases'][phase]={'same_call_rank_max_s':maxima,'median_complete_encode_ids_s':statistics.median(maxima)}
    vectors=[]
    for i in range(3):
        ranks=[{'rank':row['rank'],'complete_wall_s':row['cases']['profile'][i]['seconds'],**vector(row['cases']['profile'][i]['event_records'])} for row in rows]
        chosen=max(ranks,key=lambda row:row['complete_wall_s'])
        vectors.append({'call':i,'slowest_complete_wall_rank':chosen['rank'],'representative':chosen,'all_ranks':ranks})
    r['profile_vectors']=vectors
    r['representative_median_call']=sorted(vectors,key=lambda row:row['representative']['complete_wall_s'])[1]
    r['profile_perturbation_vs_clean']=r['phases']['profile']['median_complete_encode_ids_s']/r['phases']['clean_before']['median_complete_encode_ids_s']-1
    r['after_drift_vs_before']=r['phases']['clean_after']['median_complete_encode_ids_s']/r['phases']['clean_before']['median_complete_encode_ids_s']-1
    r['intermediate_identity_against_round72']={name:all(value[name]==reference[name] for value in ids) for name in reference}
    r['complete_outputs_compared']=len(ids)
    r['complete_output_identity_accepted']=all(r['intermediate_identity_against_round72'].values())and all(exact(proof,10460160) for proof in proofs)
    r['complete_encode_ids_under2s_accepted']=r['complete_output_identity_accepted']and all(value<2 for value in r['phases']['clean_before']['same_call_rank_max_s'])
    r['profile_events_verified']=649*3*8
    return r


def summarize(root=ROOT):
    rows=[json.loads((root/'profile'/f'attention_rank{rank}.json').read_text()) for rank in range(8)]
    previous=[json.loads((root.parent/'round72'/'ranks'/f'rope_rank{rank}.json').read_text()) for rank in range(8)]
    for row in rows:
        if set(row['sources'])!={'attention_profile_tp8.py','attention_stages73.py','candidate_padded_text_encoder.py','evidence_io73.py'}:raise ValueError('Missing exactexecutedsource fingerprints')
        for name,digest in row['sources'].items():
            path=root/name
            if name=='candidate_padded_text_encoder.py':path=root.parent/'round65'/'source'/name
            elif name=='evidence_io73.py':path=root.parent/'round63'/'evidence_io63.py'
            if hashlib.sha256(path.read_bytes()).hexdigest()!=digest:raise ValueError('Executedsourcehash differs')
    r=aggregate(rows,previous)
    metas=[json.loads((root/'profile'/f'attention_call{index:02d}.json').read_text()) for index in (0,17,33,53)]
    for meta,index in zip(metas,(0,17,33,53)):
        if meta['call']!=index or meta['layer']!=index//2 or meta['image_segment']!=index%2 or not meta['same_input_original_bits_equal']or meta['file_bytes']>=64*1024*1024:raise ValueError('Wrong actual capturedattention fixture')
        for name in ('query','key','value','output'):tensor_contract(meta['tensor_identities'][name],[1,16,4032,72],'torch.bfloat16',9289728)
    if sum(meta['file_bytes']for meta in metas)>=256*1024*1024:raise ValueError('Capture total budget exceeded')
    r['captured_fixture_bytes']=sum(meta['file_bytes']for meta in metas)
    return r


if __name__=='__main__':
    r=summarize();(ROOT/'profile_summary.json').write_text(json.dumps(r,indent=2)+'\n')
    print(json.dumps({key:value for key,value in r.items()if key not in ('profile_vectors','representative_median_call')},indent=2))
    print(json.dumps(r['representative_median_call']['representative'],indent=2))
