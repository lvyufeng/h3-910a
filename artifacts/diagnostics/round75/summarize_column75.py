#!/usr/bin/env python3
"""Retained FC2 counterexample plus separately tested zero-padding recovery;no fullTPclaim."""
import hashlib
import json
from pathlib import Path
import statistics
from column_plan75 import WIDTHS,INPUTS,indices,padded_width

ROOT=Path(__file__).resolve().parent


def exact(p,shape,dtype):
    import math
    return p['shape']==shape and p['dtype']==dtype and p['elements']==math.prod(shape)and p['finite']and p['raw_bits_equal']and p['different_words']==0 and p['max_abs_error']==0 and p['relative_rms']==0


def summarize(root=ROOT):
    failed=json.loads((root/'column_determinism_failure02.json').read_text())
    if failed.get('completed')or failed['failure']['type']!='AssertionError'or len(failed['cases'])!=3:raise ValueError('Original FC2 repeat failure must remain rejected')
    expected=hashlib.sha256((root/'column_probe_safe75.py').read_bytes()).hexdigest()
    if failed['source_sha256']!=expected:raise ValueError('Rejected geometry source changed')
    result={'scope':'ActualL0singleNPUserialcolumngeometryonly;keepsfullK,fusedbias,originalinput/modelbytes. CurrentBF16outputinterface/privateFP32proofs;notall116Linears/headparallel/TP8/wholeencoder/qualityacceptance.',
            'tested_successful_native_modules':{},'rejected_fc2_144':{},'separate_fc2_zero_padding':{},
            'unmodified_144_fc2_family_accepted':False,'all_later_layers_and_mergers_covered':False,
            'whole_encoder_acceleration_proven':False,'official_goal_complete':False,'production_deployed':False}
    for row in failed['cases']:
        name=row['name'];outshape=[8064,WIDTHS[name]]
        if name not in('qkv','proj','fc1')or not row['input_weight_bias_readonly']or not exact(row['full_replay_vs_captured'],outshape,'torch.bfloat16'):raise ValueError('Wrong actual fullLinear replay')
        expected_modes={'native_width','zero_pad_544'}if name=='fc1'else{'native_width'}
        if set(row['modes'])!=expected_modes:raise ValueError('Mode labels do not match tested output widths')
        for mode,value in row['modes'].items():
            if len(value['parts'])!=8 or not exact(value['reconstructed_fp32'],outshape,'torch.float32')or not exact(value['reconstructed_bf16'],outshape,'torch.bfloat16'):raise ValueError('Native-width successful reconstruction not exact')
            for p in value['parts']:
                target=indices(name,p['partition']);shape=[8064,len(target)]
                if p['indices']!=target or p['full_K_preserved']!=INPUTS[name]or p['valid_width']!=len(target)or p['gemm_output_width']!=padded_width(name,mode!='native_width')or not exact(p['fp32_vs_full_columns'],shape,'torch.float32')or not exact(p['bf16_vs_full_columns'],shape,'torch.bfloat16'):raise ValueError('Output-column index/K/width/proof changed')
                if len(p['repeat_proofs'])!=3 or not all(exact(p['repeat_proofs'][i][role],[8064,p['gemm_output_width']],dtype)for i in range(3)for role,dtype in(('fp32','torch.float32'),('bf16','torch.bfloat16'))):raise ValueError('Private/output repeat instability')
            result['tested_successful_native_modules'].setdefault(name,{})[mode]={'FP32_and_BF16_full_reconstruction_exact':True,'parts':8,
                    'standalone_part_median_s':[p['standalone_part_median_s']for p in value['parts']],
                    'same_pair_full_reference_median_s':[p['full_median_same_pair_s']for p in value['parts']]}
    p=failed['partial_partition']
    if p['valid_width']!=144 or p['gemm_output_width']!=144 or p['full_K_preserved']!=4304 or p['partition']!=0 or p['bf16_vs_full_columns']['raw_bits_equal']:raise ValueError('OriginalFC2counterexamplewaslost')
    if not any(not proof['bf16']['raw_bits_equal']for proof in p['repeat_proofs']):raise ValueError('OriginalBF16repeatfailurelost')
    result['rejected_fc2_144']={'partition':p['partition'],'BF16_vs_full':p['bf16_vs_full_columns'],'FP32_vs_full':p['fp32_vs_full_columns'],
                               'repeat_proofs':p['repeat_proofs'],'acceptance_rejected':True}
    padding=json.loads((root/'fc2_padding01.json').read_text())
    if padding['fixture_sha256']!=failed['partial_case']['capture_file_sha256']:raise ValueError('Padding recovery used a different FC2 fixture than the retained failure')
    fixture=json.loads((root/'capture'/'fc2.json').read_text())
    if padding['fixture_sha256']!=fixture['file_sha256']or padding['source_inputs_and_weights']!={key:fixture['tensors'][key]['sha256']for key in('x','w','b','y')}:raise ValueError('Padding source original activation/weight/bias/output identity changed')
    if not padding.get('completed')or padding.get('failure')or padding['math_flags_before']!=padding['math_flags_after']or not padding['source_input_weights_unchanged']:raise ValueError('Separatepadprobefailed')
    if padding['source_sha256']!=hashlib.sha256((root/'fc2_padding75.py').read_bytes()).hexdigest():raise ValueError('Separatepadprobesourcechanged')
    if set(padding['widths'])!={'160','192','256','512'}:raise ValueError('Incomplete recovery width coverage')
    for width,value in padding['widths'].items():
        if len(value['parts'])!=8 or not exact(value['reconstructed_fp32'],[8064,1152],'torch.float32')or not exact(value['reconstructed_bf16'],[8064,1152],'torch.bfloat16'):raise ValueError('Recoveryfullmatrixnotexact')
        for part in value['parts']:
            if part['gemm_output_width']!=int(width)or part['valid_width']!=144 or part['K']!=4304 or not part['padding_outputs_zero']:raise ValueError('ChangedK/nonzeroPadding')
            if not exact(part['fp32_vs_full'],[8064,144],'torch.float32')or not exact(part['bf16_vs_full'],[8064,144],'torch.bfloat16'):raise ValueError('Recoverypartnotexact')
            if len(part['repeats'])!=3 or not all(exact(repeat[role],[8064,int(width)],dtype)for repeat in part['repeats']for role,dtype in(('fp32','torch.float32'),('bf16','torch.bfloat16'))):raise ValueError('Recoveryrepeatnotstable')
        result['separate_fc2_zero_padding'][width]={'FP32_and_BF16_full_reconstruction_exact':True,'repeat_stable':True,
                                                  'standalone_part_median_s':[statistics.median([p['seconds']for p in part['repeats']])for part in value['parts']]}
    result['smallest_tested_strictly_exact_fc2_width']=min(int(width)for width in result['separate_fc2_zero_padding'])
    return result


if __name__=='__main__':
    r=summarize();(ROOT/'column_summary.json').write_text(json.dumps(r,indent=2)+'\n');print(json.dumps(r,indent=2))
