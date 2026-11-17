#!/usr/bin/env python3
"""Original/wrapped/profile/restored controls of unchanged vision attention transcription."""
import hashlib
import json
from pathlib import Path
import statistics
from summarize_profile73 import vector
import sys
sys.path.insert(0,str(Path(__file__).resolve().parent.parent/'round70'))
from summarize import exact,validate_identities

ROOT=Path(__file__).resolve().parent


def summarize(root=ROOT):
    rows=[json.loads((root/'controls'/f'attention_rank{i}.json').read_text())for i in range(8)]
    previous=[json.loads((root/'profile'/f'attention_rank{i}.json').read_text())for i in range(8)]
    if any(not row.get('completed')or row.get('failure')for row in rows):raise ValueError('Failed original/wrapped controls')
    reference=previous[0]['first_identities'];ids=[];proofs=[]
    r={'scope':'Independent original-function/wrapped-transcription/profile/restored controls. First432same-input attention outputs verify transcription. Not kernel-speed/quality acceptance or a substitute for the separate actualcastA/B.',
       'phases':{},'actual_transcription_checks':0,'complete_output_identity_accepted':False,'official_goal_complete':False,'whole_service_text_under2s_proven':False}
    for row,old in zip(rows,previous):
        if row['rank']!=old['rank']or row['parameters_before']!=row['parameters_after']or row['parameters_before']!=old['parameters_before']or row['actual_inputs_before']!=row['actual_inputs_after']or row['actual_inputs_before']!=old['actual_inputs_before']:raise ValueError('Input/parameter/rank provenance changed')
        for field in('environment','math_flags_before','math_flags_after','checkpoint_root','checkpoint_index_sha256','parameter_dtypes','loaded_parameter_tensors'):
            if row[field]!=old[field]:raise ValueError('Original math/model configuration changed')
        if not row['vision_rope_restored']or not row['vision_wrappers_restored']or not row['attention_stage_wrappers_restored']:raise ValueError('Independent control wrappers not restored')
        if row['vision_shadow_cache']!={'entries':232,'cached_fp32_bytes':2362750400,'misses':232,'hits':3016,'invalidations':0,'uncacheable_bypasses':0,'max_bytes':2362750400}:raise ValueError('Fixed vision cache changed across14controlforwards')
        if row['text_shadow_cache']['entries']!=200 or row['text_shadow_cache']['cached_fp32_bytes']!=11927552000 or row['existing_padding_cache']!={'entries':100,'cached_fp32_bytes':2097152000}:raise ValueError('Fixedtextcachechanged')
        if set(row['sources'])!={'attention_control_tp8.py','attention_stages73.py','attention_controls73.py','candidate_padded_text_encoder.py','evidence_io73.py'}:raise ValueError('Missing actualcontrolsourcefingerprints')
        for name in('first_identities','transcription_first_identities'):
            validate_identities(row[name]);ids.append(row[name])
        proofs.extend((row['first_vs_round70_fresh_reference'],row['transcription_complete_output']))
        checks=row['transcription_checks']
        if len(checks)!=54 or {check['call']for check in checks}!=set(range(54)):raise ValueError('Missing actual54attentiontranscription comparisons')
        for check in checks:
            if not exact(check['proof'],4644864)or check['output_layout']!=check['original_output_layout']or check['layer']!=check['call']//2 or check['image_segment']!=check['call']%2:raise ValueError('Actual attention transcription changed')
        r['actual_transcription_checks']+=len(checks)
        for name,digest in row['sources'].items():
            path=root/name
            if name=='candidate_padded_text_encoder.py':path=root.parent/'round65'/'source'/name
            elif name=='evidence_io73.py':path=root.parent/'round63'/'evidence_io63.py'
            if hashlib.sha256(path.read_bytes()).hexdigest()!=digest:raise ValueError('Actualexecutedcontrols sourcehashchanged')
    for phase in('original','wrapped','profile','restored'):
        for row in rows:
            if len(row['cases'][phase])!=3:raise ValueError('Needthreeactualwarmcontrolcalls')
            for call in row['cases'][phase]:
                validate_identities(call['identities']);ids.append(call['identities']);proofs.append(call['vs_baseline'])
                if call['output']!=call['identities']['complete_output']:raise ValueError('WrongCPUoutputproof')
                if phase=='profile':vector(call['event_records'])
                elif 'event_records'in call:raise ValueError('Cleanoriginal/wrapped contains profileevents')
        values=[max(row['cases'][phase][i]['seconds']for row in rows)for i in range(3)]
        r['phases'][phase]={'same_call_rank_max_s':values,'median_complete_encode_ids_s':statistics.median(values)}
    native=r['phases']['original']['median_complete_encode_ids_s'];wrapped=r['phases']['wrapped']['median_complete_encode_ids_s']
    r['wrapper_perturbation_vs_original']=wrapped/native-1
    r['profile_perturbation_vs_wrapped']=r['phases']['profile']['median_complete_encode_ids_s']/wrapped-1
    r['restored_drift_vs_original']=r['phases']['restored']['median_complete_encode_ids_s']/native-1
    r['complete_outputs_compared']=len(ids)
    r['all_intermediate_identity']=all(all(item[name]==reference[name]for item in ids)for name in reference)
    r['complete_output_identity_accepted']=r['all_intermediate_identity']and all(exact(p,10460160)for p in proofs)
    r['profile_events_verified']=649*3*8
    return r


if __name__=='__main__':
    r=summarize();(ROOT/'control_summary.json').write_text(json.dumps(r,indent=2)+'\n');print(json.dumps(r,indent=2))
