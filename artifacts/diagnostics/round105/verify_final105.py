#!/usr/bin/env python3
"""Read-only final R101-R105 original-source, CPU-FP16 gold, binary and worker audit."""
from pathlib import Path
import hashlib
import json
import os
ROOT=Path('/tmp/h3_round102_full01')
AUDIT=Path('/tmp/h3_round105_audit01')


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    expected=json.loads((ROOT/'expected_sources76.json').read_text())
    r={'scope':__doc__,'production':{},'installed_api_sources':{},'execution_sources':{},'kernel_objects':{},'remaining_workers':[]}
    for category in('production','installed_api_sources'):
        for name,wanted in expected[category].items():
            actual=sha(name);assert actual==wanted,name;r[category][name]={'sha256':actual,'unchanged':True}
    assert len(r['production'])==23 and len(r['installed_api_sources'])==13
    for filename in('expected_execution_sources97.json','expected_execution_sources98.json','expected_execution_sources99_100.json','expected_execution_sources102.json','expected_execution_sources105.json'):
        for name,control in json.loads((AUDIT/filename).read_text()).items():
            actual=sha(name);assert actual==control['sha256'],name
            r['execution_sources'][name]={'sha256':actual,'local_source':control['local_source'],'unchanged':True}
    for name,wanted in(('/tmp/h3_round78_hf_fp1601/reference.json','44acac61f9b4d9bc8afc91632392502ccaa29dee7bfab951e2b15902a0be74c9'),
                       ('/tmp/h3_round78_hf_fp1601/FP16_vision.pt','c01bbe180f454fe44b182bd9307491ef0f4c342b6c2ff6be08d1e3453d8c312a'),
                       ('/tmp/h3_round78_hf_fp1601/FP16_selected50.pt','0137a434148366c3dcbf825970274ad0faa036666561d7aecc0170cf136b3fe4'),
                       ('/tmp/h3_round88_vision01/FrozenVisionQKV88.pt','3fb5d524373c96de48a4be583cee87f171f5e20fdaa8ac57fd4a31cb709ce552'),
                       ('/tmp/libfa910a.so','bd50d7005bf48e9a9a853c01750eaea4eefbb3e0f69abb2e7153aa7139e8b043')):
        actual=sha(name);assert actual==wanted,name;r[name]={'sha256':actual,'unchanged':True}
    reference=json.loads(Path('/tmp/h3_round78_hf_fp1601/reference.json').read_text());hf=Path(reference['HF_modeling_source_path'])
    actual=sha(hf);assert actual==reference['HF_modeling_source_sha256']=='fcb5571ef95e4a6679af866ef9ef4dfa5cee017f9f91453daf28c7804c7a50fe'
    r['original_HF_modeling_source']={'path':str(hf),'sha256':actual,'unchanged':True}
    records=[json.loads((ROOT/'prebuild102.json').read_text())]
    for rank in range(8):
        receipt=json.loads((ROOT/('tp8_rank%d.json'%rank)).read_text());assert receipt['completed']and receipt['rank']==rank
        assert receipt['native_recovery']['raw_bits_equal']and receipt['parameter_and_fixture_SHA_unchanged']and receipt['CPU_reference_SHA_unchanged']
        records.append(receipt)
    controls=('/tmp/h3_round101_smoke02/bc128_smoke101.json',
              '/tmp/h3_round101_leaf01/leaf_compare101.json',
              '/tmp/h3_round101_vision01/vision101.json',
              '/tmp/h3_round103_smoke01/bc256_smoke103.json','/tmp/h3_round103_leaf01/leaf_compare103.json',
              '/tmp/h3_round105_broadcast01/broadcast_float105.json','/tmp/h3_round105_smoke01/bc128_bitcopy_smoke105.json',
              '/tmp/h3_round105_leaf01/leaf_compare105.json')
    for name in controls:
        control=json.loads(Path(name).read_text());assert control['completed'],name;records.append(control)
    import torch
    from best_text87 import tensor_info
    saved=[]
    rank0=json.loads((ROOT/'tp8_rank0.json').read_text())
    for case_name,case in rank0['cases'].items():
        control=case['actual_NPU_output_saved_outside_timer'];path=Path(control['path']).resolve()
        assert path.parent==ROOT and path.is_file()
        actual=sha(path);assert actual==control['file_SHA256']
        payload=torch.load(path,map_location='cpu',weights_only=True)
        assert payload['case']==case_name and payload['selected50'].dtype==torch.float16
        info=tensor_info(payload['selected50'])
        assert info['sha256']==case['selected50_vs_CPU']['npu']['sha256']
        assert payload['reference_receipt_SHA256']=='44acac61f9b4d9bc8afc91632392502ccaa29dee7bfab951e2b15902a0be74c9'
        saved.append({'case':case_name,'path':str(path),'file_SHA256':actual,'tensor_SHA256':info['sha256'],'dtype':'Half','captured_outside_warm_clock_not_used_as_input':True})
    r['actual_saved_outputs']=saved
    def visit(value):
        if isinstance(value,dict):
            if 'obj_path'in value and 'obj_sha256'in value:
                name=value['obj_path'];assert Path(name).is_absolute()
                actual=sha(name);assert actual==value['obj_sha256'],name
                assert value['block_dim']in(1,4,30)
                r['kernel_objects'][name]={'sha256':actual,'block_dim':value['block_dim'],'unchanged':True}
            for child in value.values():visit(child)
        elif isinstance(value,list):
            for child in value:visit(child)
    for record in records:visit(record)
    commands=('bc256_smoke103.py','leaf_compare103.py','broadcast_float105.py','bc128_bitcopy_smoke105.py','leaf_compare105.py','full_encoder102.py','prebuild102.py','bc128_smoke101.py','bc128_smoke_mask_fixed101.py','leaf_compare101.py','vision_probe101.py','full_encoder98.py','prebuild98.py','full_encoder100.py','prebuild100.py','pv_smoke96.py','vision_probe96.py','leaf_compare96.py','leaf_compare_fixed96.py',
              'vision_probe97.py','leaf_compare97.py','softmax_smoke97.py','softmax_smoke_fixed97.py','softmax_smoke_padded_fixed97.py',
              'flash_smoke99.py','flash_smoke_step_fixed99.py','leaf_compare99.py','leaf_compare_guard_fixed99.py','leaf_compare_shift_broadcast99.py',
              'leaf_compare_shift_cast_fixed99.py','leaf_compare_shift_temp_fixed99.py','vision_probe99.py','vision_probe_shift_temp99.py','broadcast_copy99.py')
    for p in Path('/proc').iterdir():
        if not p.name.isdecimal():continue
        try:
            if p.stat().st_uid!=os.getuid():continue
            cmd=(p/'cmdline').read_bytes().replace(b'\x00',b' ').decode(errors='replace')
            if cmd.startswith('/home/mseco/vllm910a/bin/python')and any(name in cmd for name in commands):r['remaining_workers'].append({'pid':int(p.name),'command':cmd})
        except(FileNotFoundError,PermissionError,ProcessLookupError):continue
    assert not r['remaining_workers']
    r['all_production_API_execution_source_originalHF_CPUgold_frozenQKV_launcher_and_recorded_binaries_unchanged']=True
    r['all8_R102_encoder_results_complete_native_recovery_verified']=True
    r['all_relevant_workers_zero']=True
    print(json.dumps(r,indent=2))


if __name__=='__main__':main()
