#!/usr/bin/env python3
"""Verify completed CPU FP16 reference against pinned official source and repeat receipts.

This verifier is run only after the original remote supervisor completes. It
reads bounded local JSON receipts;no kernels/model re-execution and no NPU
accuracy/performance pass may be inferred from CPU reference completion.
"""
import hashlib
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parent


def verify(reference,manifest,preflight,math_preflight,source_sha,preflight_sha,math_sha):
    assert reference['completed']and reference['full_multimodal_reference_completed']
    assert reference['source_sha256']==source_sha
    assert reference['preflight_file_sha256']==preflight_sha and reference['math_preflight_file_sha256']==math_sha
    assert reference['reference_is_independent_HF_CPU_FP16']and not reference['FP32_linear_adapter_used']
    assert not reference['BF16_output_agreement_required']and not reference['old_BF16_outputs_reused_or_cast_as_gold']
    assert reference['loaded_parameter_tensors']==903 and reference['parameter_dtypes']=={'torch.float16':903}
    assert reference['complete_parameter_bytes']==51506202080
    assert reference['original_selected_parameter_sha256']=='a69676bac5e6f3a513ce44080d29c73974553b6e0a327ea272e47b600d84e46e'
    assert reference['parameters_before']==reference['parameters_after']and reference['actual_inputs_before']==reference['actual_inputs_after']
    assert reference['input_file_sha256']=='ed099e792982f9cd37b67f2cff24443008683d0cc819d1decacabf55de1f4ebc'
    assert reference['Linear_input_dtypes_observed']==['torch.float16']and reference['Linear_output_dtypes_observed']==['torch.float16']
    assert reference['pre_final_norm_matches_layer49_identity']and reference['repeat_pre_final_norm_bits_equal']and reference['repeat_post_final_norm_bits_equal']
    assert preflight['completed']and preflight['all_required_preflight_operators_supported']
    assert math_preflight['completed']and math_preflight['supported']and math_preflight['backend']=='SDPBackend.MATH'
    assert reference['versions']==preflight['versions']or(reference['versions']['torch']==preflight['versions']['torch']and reference['versions']['transformers']==preflight['versions']['transformers'])
    shards={}
    for name,item in reference['checkpoint_shards'].items():
        assert item==manifest['lfs_shards'][name],name
        shards[name]=True
    assert len(reference['language_layers'])==50 and len(reference['vision_layers'])==27
    for kind in('language','vision'):
        for index,row in enumerate(reference[kind+'_layers']):
            assert row['layer']==index and row['output']['dtype']=='torch.float16'and row['output']['finite']
    for name,item in reference['parameter_conversion'].items():
        assert item['original_dtype']=='torch.bfloat16'and item['reference_dtype']=='torch.float16'and item['reference_finite'],name
    assert not reference['NPU_precision_accepted']and not reference['whole_goal_complete']
    return {'CPU_FP16_reference_verified':True,'official_source_revision':manifest['revision'],'used_checkpoint_shards_match_official_LFS_SHA256':shards,
            'parameter_tensors':903,'parameter_dtype':'FP16','original_operator_FP32_islands_declared':reference['FP32_islands'],
            'vision_layers':27,'text_layers':50,'pre_final_norm_repeat_bit_equal':True,'NPU_precision_accepted':False,'whole_goal_complete':False}


if __name__=='__main__':
    path=ROOT/(sys.argv[1]if len(sys.argv)>1 else'reference.json')
    reference=json.loads(path.read_text());manifest=json.loads((ROOT/'official_text_manifest_receipt78.json').read_text())
    preflight=json.loads((ROOT/'preflight.json').read_text());math_preflight=json.loads((ROOT/'math_preflight.json').read_text())
    sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
    result=verify(reference,manifest,preflight,math_preflight,sha(ROOT/'hf_cpu_fp16_reference78.py'),sha(ROOT/'preflight.json'),sha(ROOT/'math_preflight.json'))
    result['reference_receipt_sha256']=sha(path)
    dest=ROOT/'final_CPU_reference_verification78.json'
    with dest.open('x')as f:f.write(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2))
