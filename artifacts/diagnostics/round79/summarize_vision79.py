#!/usr/bin/env python3
"""Validate actual vision evidence and summarize only measured FP16 stage results."""
import hashlib
import json
from pathlib import Path
import statistics

ROOT=Path(__file__).resolve().parent
probe=json.loads((ROOT/'vision.json').read_text());manifest=json.loads((ROOT/'candidate'/'build_manifest79.json').read_text())
source=json.loads((ROOT/'CPU_source_verification79.json').read_text());official=json.loads((ROOT.parent/'round78'/'official_text_manifest_receipt78.json').read_text())
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
assert probe['completed']and probe['vision_FP16_finite_and_repeat_stable']
assert probe['source_sha256']==sha(ROOT/'vision_probe79.py')
assert probe['candidate_source_sha256']==manifest['candidate_sha256']==sha(ROOT/'candidate'/'candidate_encoder79.py')
assert probe['candidate_manifest_sha256']==sha(ROOT/'candidate'/'build_manifest79.json')
assert probe['CPU_reference_file_sha256']==source['CPU_first_vision_tensors']['file_sha256']
assert probe['parameter_tensors']==351 and probe['parameter_bytes']==1190533600
assert probe['actual_input_pixels']['dtype']=='torch.float16'and probe['actual_input_grid']==[[1,48,84],[1,48,84]]
assert len(probe['layers'])==27 and all(row['layer']==index and row['output']['dtype']=='torch.float16'and row['output']['finite']for index,row in enumerate(probe['layers']))
assert len(probe['cases'])==3
for row in probe['cases']:
    assert row['merged_repeat']['raw_bits_equal']and all(p['raw_bits_equal']for p in row['deepstack_repeat'])
for name,value in source['used_checkpoint_shards_verified_against_official_LFS'].items():assert value==official['lfs_shards'][name]
assert not probe['full_encoder_precision_accepted']and not probe['whole_goal_complete']
metric=probe['CPU_FP16_comparison']
result={'scope':__doc__,'actual_NPU_vision_probe_completed':True,'parameter_dtype':'FP16','vision_layers':27,'fresh_CPU_FP16_source_revision':source['official_revision'],
        'used_CPU_checkpoint_shards_match_official_LFS':True,'vision_parameter_conversion_hashes_checked_by_probe':True,'all_full_vision_features_finite':True,'three_NPU_repeats_bit_equal':True,
        'resident_single_rank_vision_s':[row['resident_vision_s']for row in probe['cases']],
        'resident_single_rank_vision_median_s':statistics.median(row['resident_vision_s']for row in probe['cases']),
        'excluded_diagnostic_cold_and_checks_s':probe['excluded_diagnostic_first_wall_s'],
        'CPU_FP16_final_vision_relative_RMS':metric['merged']['relative_rms'],'CPU_FP16_final_vision_max_abs_error':metric['merged']['max_abs_error'],
        'CPU_FP16_DeepStack_relative_RMS':[x['relative_rms']for x in metric['deepstack']],
        'scope_qualifications':['Visiononly,notcomplete50layerencoder','SingleNPU,not8rankcompleteencodermax','ConcurrentindependentCPUreferencejobwasrunning;notcrossroundcontrolledspeedA/B','No CPU or BF16 oldwalltime used toinfer NPUacceleration','No thresholdchosenafterdata;precisionnotdeclaredpassed','CPUfullrepeatpending'],
        'production_changed':False,'full_encoder_precision_accepted':False,'full_encoder_2s_target_claimed':False,'whole_goal_complete':False,
        'actual_vision_receipt_sha256':sha(ROOT/'vision.json'),'CPU_source_verification_receipt_sha256':sha(ROOT/'CPU_source_verification79.json')}
with(ROOT/'summary79.json').open('x')as f:f.write(json.dumps(result,indent=2)+'\n')
print(json.dumps(result,indent=2))
