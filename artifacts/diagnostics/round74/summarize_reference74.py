#!/usr/bin/env python3
"""Strict independent-reference gate; incomplete CPU progress cannot pass."""
import hashlib
import json
from pathlib import Path
from summarize_capture74 import summarize as capture_summary

ROOT=Path(__file__).resolve().parent


def finite_full(proof,shape,dtype):
    import math
    count=math.prod(shape)
    return proof['shape']==shape and proof['dtype']==dtype and proof['elements']==count and proof['finite'] and proof['max_abs_error']is not None and proof['relative_rms']is not None


def aggregate(first,repeat,modal,vision_repeat,capture,stability=None):
    if not first.get('completed')or not first.get('full_multimodal_reference_completed')or first.get('failure'):raise ValueError('Independent full multimodal BF16 HF reference is not complete')
    if not repeat.get('completed')or repeat.get('failure')or not modal.get('completed')or modal.get('failure'):raise ValueError('Reference repeat/modal comparison incomplete')
    if not vision_repeat.get('completed')or not vision_repeat.get('all_full_vision_outputs_bits_equal'):raise ValueError('Independent vision reference stability missing')
    if first['shared_native_visual_features_used']or first['fp32_linear_adapter_used']or repeat['fp32_linear_adapter_used']or not first['reference_is_original_bf16_HF']:raise ValueError('Shared experimental features/altered teacher math is not independent BF16 HF')
    if first['parameters_before']!=first['parameters_after']or first['actual_inputs_before']!=first['actual_inputs_after']:raise ValueError('Reference model/input bytes changed')
    if first['loaded_parameter_tensors']!=903 or first['complete_parameter_bytes']!=51506202080 or first['parameter_dtypes']!={'torch.bfloat16':903}or first['linear_input_dtypes_observed']!=['torch.bfloat16']:raise ValueError('Original unsharded BF16 reference weights/operators changed')
    if first['original_config_language_layers']!=64 or len(first['language_layers'])!=50 or {x['layer']for x in first['language_layers']}!=set(range(50)):raise ValueError('Selected50 HF language layer coverage incomplete')
    if len(repeat['layers'])!=50 or {x['layer']for x in repeat['layers']}!=set(range(50)):raise ValueError('HF reference repeat not all50 layers')
    if not repeat.get('original_hf_finalnorm_retained')or not modal['final_norm_difference_is_excluded_from_h3_comparison']:raise ValueError('Wrong H3selectedhidden/finalnorm boundary')
    selected=repeat['selected']
    if len(selected)!=6 or {x['layer']for x in selected}!={0,1,2,9,24,49}:raise ValueError('Fresh experimental/HF layer comparison coverage incomplete')
    for row in selected:
        for mode,shape in(('full',[1,2043,5120]),('text',[27,5120]),('visual',[2016,5120])):
            if not finite_full(row[mode],shape,'torch.bfloat16'):raise ValueError('Partial/nonfinite selected-layer comparison')
    if not stability or not stability.get('completed')or stability.get('failure')or not stability.get('posthoc_full_reference_repeat_accepted'):raise ValueError('Missing authoritative post-completion reference stability proof')
    if repeat['independent_HF_multimodal_input_sha256']!=first['stages']['hf_multimodal_inputs']['file_sha256']:raise ValueError('Repeat does not consume the independently generated HF multimodal input')
    stability_proof=stability['pre_final_norm_repeat']
    if not finite_full(stability_proof,[1,2043,5120],'torch.bfloat16')or not stability_proof['raw_bits_equal']or stability_proof['max_abs_error']!=0:raise ValueError('Original BF16 language reference is not repeat-stable')
    derived_stability=[]
    for a,b in zip(first['language_layers'],repeat['layers']):
        if a['layer']!=b['layer']or a['output']['sha256']!=b['sha256']or not a['output']['finite']or not b['finite']:raise ValueError('HF same-context repeat layer identity changed')
        derived_stability.append({'layer':a['layer'],'first_sha256':a['output']['sha256'],'repeat_sha256':b['sha256'],'finite':a['output']['finite']and b['finite']})
    if stability['all50_layer_raw_identities']!=derived_stability:raise ValueError('Published stability receipt does not match authoritative fifty-layer identities')
    if modal['token_counts']!={'total':2043,'visual':2016,'text':27}:raise ValueError('Real image/text mask counts changed')
    for mode,shape in(('full',[1,2043,5120]),('text_positions',[27,5120]),('visual_positions',[2016,5120])):
        if not finite_full(modal['comparisons']['selected50'][mode],shape,'torch.bfloat16'):raise ValueError('Independent multimodal final output not fully compared')
    p_first=first['full_multimodal_pre_final_norm_vs_experimental'];p_modal=modal['comparisons']['selected50']['full']
    for key in('dtype','elements','raw_bits_equal','different_bytes','finite','max_abs_error'):
        if p_first[key]!=p_modal[key]:raise ValueError('Final independent-reference comparison boundary differs between producers')
    import math
    if not math.isclose(p_first['relative_rms'],p_modal['relative_rms'],rel_tol=1e-12,abs_tol=1e-12):raise ValueError('Full independent-reference aggregate errors disagree')
    last_repeat=next(item for item in selected if item['layer']==49)
    for repeat_role,modal_role in(('full','full'),('text','text_positions'),('visual','visual_positions')):
        p_repeat=last_repeat[repeat_role];p_modal=modal['comparisons']['selected50'][modal_role]
        for key in('dtype','elements','raw_bits_equal','different_bytes','finite','max_abs_error'):
            if p_repeat[key]!=p_modal[key]:raise ValueError('Final full/text/visual comparison differs between independent producers')
        if not math.isclose(p_repeat['relative_rms'],p_modal['relative_rms'],rel_tol=1e-12,abs_tol=1e-12):raise ValueError('Final modal relative RMS aggregates disagree')
    for name,shape,dtype in(('text_positions',[3,1,2043],'torch.int64'),('visual_mask',[1,2043],'torch.bool')):
        p=modal['comparisons'][name]
        if not finite_full(p,shape,dtype)or not p['raw_bits_equal']:raise ValueError('Independent M-RoPE/visualmask integers differ')
    return {'scope':'Original unchanged-BF16 HF vision+embedding+M-RoPE+threeDeepStack+selected50pre-finalnorm on actual2image2043tokentransportfixture. CPUreferencecompute is not NPU performance,one fixture not officialvideoquality.',
            'independent_full_multimodal_reference_completed':True,'original_bf16_reference_language_repeat_bits_stable':True,
            'original_bf16_reference_vision_vs_retained_independent_reference_bits_stable':True,
            'loaded_HF_parameter_tensors':903,'loaded_HF_parameter_bytes':51506202080,
            'experimental_complete_output_sha256':capture['complete_output_sha256'],
            'embedding_proof':first['embedding_vs_experimental'],'vision_comparisons':first['vision_vs_experimental'],
            'multimodal_input_comparisons':first['multimodal_text_input_vs_experimental'],
            'selected50_full_and_modal_comparisons':modal['comparisons']['selected50'],
            'selected_layer_comparisons':selected,'selected50_outliers':modal['selected50_outliers'],
            'outlier_statistics_scope':'Full-tensor producer-derived error concentration from hash-pinned modal_comparison74.py;not independently recomputed from absent local tensor blobs and never removed from the main metric.',
            'HF_pre_final_norm_output_sha256':repeat['repeat_final_sha256'],
            'selected50_hidden_states_equal_HF':modal['full_multimodal_hf_identity_accepted'],
            'official_quality_accepted':False,'whole_service_under2s_proven':False,'official_goal_complete':False,'production_deployed':False}


def summarize(root=ROOT):
    first=json.loads((root/'hf_reference01.json').read_text());repeat=json.loads((root/'language_repeat01.json').read_text())
    modal=json.loads((root/'modal_comparison01.json').read_text());vision_repeat=json.loads((root/'vision_repeat01.json').read_text())
    for file,row in(('hf_multimodal74.py',first),('hf_language_repeat74.py',repeat),('modal_comparison74.py',modal)):
        if hashlib.sha256((root/file).read_bytes()).hexdigest()!=row['source_sha256']:raise ValueError('Executed independent HF comparison source changed')
    stability=json.loads((root/'reference_stability01.json').read_text())
    if hashlib.sha256((root/'verify_reference_repeat74.py').read_bytes()).hexdigest()!=stability['source_sha256']:raise ValueError('Post-completion stability verifier source changed')
    return aggregate(first,repeat,modal,vision_repeat,capture_summary(root),stability)


if __name__=='__main__':
    r=summarize();(ROOT/'summary.json').write_text(json.dumps(r,indent=2)+'\n');print(json.dumps(r,indent=2))
