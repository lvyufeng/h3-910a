#!/usr/bin/env python3
"""Fresh full encoder evidence and its separate FP16 video-conditioning handoff."""
import hashlib
import json
from pathlib import Path
import statistics
import sys
sys.path.insert(0,str(Path(__file__).resolve().parent.parent/'round70'))
from summarize import exact,validate_identities

ROOT=Path(__file__).resolve().parent


def summarize(root=ROOT):
    rows=[json.loads((root/'encoder'/f'encoder_rank{i}.json').read_text())for i in range(8)]
    old=[json.loads((root.parent/'round73'/'ranks'/f'cast_rank{i}.json').read_text())for i in range(8)]
    ref=old[0]['cases']['cast']['first_identities'];checked=0
    for row,previous in zip(rows,old):
        if not row.get('completed')or row.get('failure')or row['rank']!=previous['rank']:raise ValueError('Failed fresh optimized complete encoder')
        if row['parameters_before']!=row['parameters_after']or row['parameters_before']!=previous['parameters_before']or row['actual_inputs_before']!=row['actual_inputs_after']or row['actual_inputs_before']!=previous['actual_inputs_before']:raise ValueError('Fresh encoder input/original weights changed')
        if row['math_flags_before']!=row['math_flags_after']or row['math_flags_before']!=previous['math_flags_before']or row['environment']!=previous['environment']:raise ValueError('Fresh encoder precision configuration changed')
        if not row['vision_rope_restored']or not row['attention_restored']or not row['vision_wrappers_restored']or row['vision_shadow_bytes_after_clear']:raise ValueError('Fresh encoder cleanup did not restore wrappers/shadows')
        if row['first_identities']!=ref or not exact(row['first_output_vs_fixed_reference'],10460160):raise ValueError('Fresh encoder first output/intermediate identity changed')
        checked+=1
        if len(row['warm_calls'])!=3:raise ValueError('Require three warm complete-encoder calls')
        for call in row['warm_calls']:
            if call['identities']!=ref or not exact(call['full_output_vs_reference'],10460160):raise ValueError('Fresh encoder warm output/intermediate identity changed')
            validate_identities(call['identities']);checked+=1
        for name,digest in row['sources'].items():
            path=root/name
            if name=='candidate_padded_text_encoder.py':path=root.parent/'round65'/'source'/name
            elif name=='evidence_io76.py':path=root.parent/'round63'/'evidence_io63.py'
            if hashlib.sha256(path.read_bytes()).hexdigest()!=digest:raise ValueError('Fresh encoder executed source changed')
    times=[max(row['warm_calls'][i]['seconds']for row in rows)for i in range(3)]
    return {'scope':'Fresh realembedding/vision/Mrope/DeepStack/50textlayers/currentexperimentalfullencode_ids;originalBF16storage,FP32+integerfallbackcompute on910A,NOTnativeBF16compute. Condition output later supplied externally to actualvideo request,no liveencodercostinside that request.',
            'warm_same_call_rank_max_s':times,'median_complete_resident_encode_ids_s':statistics.median(times),
            'complete_output_identity_checks':checked,'all_output_and_intermediate_identities_equal':True,
            'conditioning_BF16_sha256':rows[0]['conditioning_hidden_bf16_sha256'],'conditioning_FP16_consumed_sha256':rows[0]['conditioning_hidden_fp16_sha256'],
            'conditioning_file_sha256':rows[0]['conditioning_file_sha256'],'conditioning_file_bytes':rows[0]['conditioning_file_bytes'],
            'live_encoder_generation_end_to_end_cost_proven':False,'nativeBF16compute_used':False,'official_goal_complete':False}


if __name__=='__main__':
    r=summarize();(ROOT/'encoder_summary.json').write_text(json.dumps(r,indent=2)+'\n');print(json.dumps(r,indent=2))
