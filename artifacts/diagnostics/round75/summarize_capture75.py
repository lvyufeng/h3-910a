#!/usr/bin/env python3
"""Validate current real vision Linear captures without labelling excluded I/O as speed."""
import hashlib
import json
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parent.parent/'round70'))
from summarize import exact,validate_identities
from column_plan75 import WIDTHS,INPUTS

ROOT=Path(__file__).resolve().parent


def summarize(root=ROOT):
    rows=[json.loads((root/'capture'/f'capture_rank{i}.json').read_text())for i in range(8)]
    previous=[json.loads((root.parent/'round73'/'ranks'/f'cast_rank{i}.json').read_text())for i in range(8)]
    reference=previous[0]['cases']['cast']['first_identities']
    for row,old in zip(rows,previous):
        if row.get('failure')or not row.get('completed')or row['rank']!=old['rank']:raise ValueError('Failed currentLinear capture')
        if row['parameters_before']!=row['parameters_after']or row['parameters_before']!=old['parameters_before']or row['actual_inputs_before']!=row['actual_inputs_after']or row['actual_inputs_before']!=old['actual_inputs_before']:raise ValueError('Real originalmodel/input changed')
        if not exact(row['full_output_vs_round70'],10460160)or row['actual_identities']!=reference:raise ValueError('Capture perturbed current fulloutput/intermediate identity')
        validate_identities(row['actual_identities'])
        if not row['vision_wrappers_restored']or not row['attention_restored']or not row['vision_rope_restored']or row['vision_shadow_bytes_after_clear']:raise ValueError('Capture not restored')
        for name,digest in row['sources'].items():
            path=root/name
            if name=='candidate_padded_text_encoder.py':path=root.parent/'round65'/'source'/name
            elif name=='evidence_io75.py':path=root.parent/'round63'/'evidence_io63.py'
            if hashlib.sha256(path.read_bytes()).hexdigest()!=digest:raise ValueError('ExecutedLinear capture source changed')
        if row['rank']and row['captured_linears']:raise ValueError('Onlyrank0shouldsavefouractualLinearfixtures')
    captured=rows[0]['captured_linears']
    if {item['name']for item in captured}!=set(WIDTHS)or len(captured)!=4:raise ValueError('Actual fourLinear coverage missing')
    for item in captured:
        name=item['name'];manifest=json.loads((root/'capture'/(name+'.json')).read_text())
        if manifest!=item or not item['source_input_and_weight_readonly']or item['file_bytes']>=128*1024*1024:raise ValueError('Badactualfixturemanifest')
        for role,shape in(('x',[8064,INPUTS[name]]),('w',[WIDTHS[name],INPUTS[name]]),('b',[WIDTHS[name]]),('y',[8064,WIDTHS[name]])):
            proof=item['tensors'][role]
            if proof['shape']!=shape or proof['dtype']!='torch.bfloat16'or not proof['finite']:raise ValueError('Truncated/nonfinite/alteredactualLinearoperand')
        if item['input_layout']['physical_format']not in(0,2)or item['output_layout']['physical_format']not in(0,2):raise ValueError('NotNDactualruntime')
    total=sum(item['file_bytes']for item in captured)
    if total>=384*1024*1024:raise ValueError('Actualfixturebudgetexceeded')
    return {'scope':'Current experimental L0qkv/proj/fc1/fc2 complete real activation/originalBF16weight/bias/BF16output fixtures;currentfullencoder8identitycontrols unchanged. MergerLinear/laterblocks untested;notHPindependentquality or speedproof.',
            'complete_output_identity_accepted':True,'complete_outputs_verified':8,'actual_linear_fixtures':4,'total_fixture_bytes':total,
            'complete_output_sha256':reference['complete_output']['sha256'],'production_modified':False,'whole_encoder_acceleration_proven':False,'official_goal_complete':False}


if __name__=='__main__':
    r=summarize();(ROOT/'capture_summary.json').write_text(json.dumps(r,indent=2)+'\n');print(json.dumps(r,indent=2))
