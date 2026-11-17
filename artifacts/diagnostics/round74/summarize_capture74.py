#!/usr/bin/env python3
"""Validate current experimental capture without treating excluded forward as timing."""
import hashlib
import json
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parent.parent/'round70'))
from summarize import exact,validate_identities

ROOT=Path(__file__).resolve().parent


def summarize(root=ROOT):
    rows=[json.loads((root/'capture'/f'capture_rank{i}.json').read_text())for i in range(8)]
    previous=[json.loads((root.parent/'round73'/'ranks'/f'cast_rank{i}.json').read_text())for i in range(8)]
    if any(not row.get('completed')or row.get('failure')for row in rows):raise ValueError('Failed currentexperimental capture')
    reference=previous[0]['cases']['cast']['first_identities']
    for row,old in zip(rows,previous):
        if row['rank']!=old['rank']or row['parameters_before']!=row['parameters_after']or row['parameters_before']!=old['parameters_before']:raise ValueError('Captured model/rank changed')
        if row['actual_inputs_before']!=row['actual_inputs_after']or row['actual_inputs_before']!=old['actual_inputs_before']:raise ValueError('Original actual inputs changed')
        if not row['vision_wrappers_restored']or not row['attention_restored']or not row['vision_rope_restored']or row['vision_shadow_bytes_after_clear']!=0:raise ValueError('Capture wrappers/cache not restored')
        for field in('environment','math_flags_before','math_flags_after','checkpoint_root','checkpoint_index_sha256','parameter_dtypes','component_parameter_bytes'):
            if row[field]!=old[field]:raise ValueError('Actual computation/model/source boundary changed')
        validate_identities(row['actual_identities'])
        if row['actual_identities']!=reference or not exact(row['full_output_vs_round70'],10460160):raise ValueError('Capture perturbed complete encoder output')
        for name,digest in row['sources'].items():
            path=root/name
            if name=='candidate_padded_text_encoder.py':path=root.parent/'round65'/'source'/name
            elif name=='evidence_io74.py':path=root.parent/'round63'/'evidence_io63.py'
            if hashlib.sha256(path.read_bytes()).hexdigest()!=digest:raise ValueError('Executed capture source differs')
        if row['rank']!=0 and row['captured_layers']:raise ValueError('Onlyrank0selectedlayersshouldbesaved')
    captures=rows[0]['captured_layers']
    if [item['layer']for item in captures]!=[0,1,2,9,24,49]:raise ValueError('Selected current layer coverage incomplete')
    for item in captures:
        expected=json.loads((root/'capture'/f"text_layer{item['layer']:02d}.json").read_text())
        if item!=expected or not item['finite']or item['shape']!=[1,2043,5120]or item['dtype']!='torch.bfloat16'or item['bytes']!=20920320:raise ValueError('Selected current full layer proof failed')
        if not item['boundary'].startswith('Raw text block output BEFORE'):raise ValueError('DeepStack comparison boundary wrong')
    if captures[-1]['sha256']!=reference['complete_output']['sha256']:raise ValueError('Final layer pre-norm identity changed')
    return {'scope':'Eight excluded completeforward identity controls,sixfullcurrentR73experimentalblockoutputsPREloopDeepStack at0/1/2. Not clean warmed performance samples,not HF reference acceptance.',
            'complete_outputs_verified':8,'current_block_fixtures':6,'selected_layers':[item['layer']for item in captures],
            'total_tensor_bytes':sum(item['bytes']for item in captures),'total_fixture_bytes':sum(item['file_bytes']for item in captures),
            'complete_output_sha256':reference['complete_output']['sha256'],'all_current_intermediate_identity':True,
            'original_input_and_parameter_bytes_unchanged':True,'capture_wrappers_restored':True,'official_goal_complete':False}


if __name__=='__main__':
    r=summarize();(ROOT/'capture_summary.json').write_text(json.dumps(r,indent=2)+'\n');print(json.dumps(r,indent=2))
