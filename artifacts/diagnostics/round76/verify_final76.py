#!/usr/bin/env python3
"""CPU-only final source/native-FP16-checkpoint/video provenance and restoration verification."""
import hashlib
import json
from pathlib import Path
from evidence_io76 import EvidenceWriter

ROOT=Path('/tmp/h3_round76_verify01')
writer=EvidenceWriter(ROOT/'final_verification.json')
expected=json.loads((ROOT/'expected_sources.json').read_text())
r={'production_modified':False,'production':{},'installed_api_sources':{},'diagnostics':{},'fixtures':{},'native_fp16_weights':{},
   'local_rehash_scope':'Stream exact remote physical files after both generated requests and restore;actual tensors not downloaded locally are not claimed locally rehashed.',
   'native_BF16_compute_used':False,'official_goal_complete':False}


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb')as f:
        for block in iter(lambda:f.read(8*1024*1024),b''):h.update(block)
    return h.hexdigest()


try:
    for category in('production','installed_api_sources','diagnostics','fixtures','native_fp16_weights'):
        for path,wanted in expected[category].items():
            observed=sha(path);r[category][path]={'sha256':observed,'expected':wanted,'matches':observed==wanted}
    for physical in expected['native_fp16_weights']:
        alias=Path('/tmp/h3_round53_native49_fp16_model/FL2VA/transformer')/Path(physical).name
        assert alias.resolve()==Path(physical).resolve(),'Video runtime alias no longer refers to validated unfused FP16 checkpoint'
    result=json.loads(Path('/tmp/h3_round76_generation02/generation_result.json').read_text())
    restore=json.loads(Path('/tmp/h3_round76_deploy02/restoration.json').read_text())
    assert result['completed']and result['engine_close_called']and not result.get('failure')
    assert restore['restored']and restore['observed_sha256']==restore['baseline_sha256']and restore['only_VAE_decoder_instance_hook']
    assert [case['case']for case in result['cases']]==['optimized','HF_reference_conditioning']
    r['actual_generated_videos']={};r['original_selected_frame_PNGs']={};r['contact_sheets']={}
    for case in result['cases']:
        video=case['video'];p=Path(video['path']);observed=sha(p)
        assert observed==video['sha256']and p.stat().st_size==video['bytes']
        r['actual_generated_videos'][case['case']]={'path':str(p),'bytes':p.stat().st_size,'sha256':observed,'matches_actual_generation_receipt':True,'video_only_export':True}
        for index in(0,12,24,36,48,62,74,86,98,110,123):
            frame=p.parent/(case['case']+f'_frame{index:03d}.png')
            r['original_selected_frame_PNGs'][frame.name]={'bytes':frame.stat().st_size,'sha256':sha(frame)}
        sheet=p.parent/(case['case']+'_contactsheet.png')
        r['contact_sheets'][sheet.name]={'bytes':sheet.stat().st_size,'sha256':sha(sheet)}
    r['different_raw_RGB_outputs']=result['cases'][0]['raw_RGB_frame_sha256']!=result['cases'][1]['raw_RGB_frame_sha256']
    r['original_generation_processes_remaining']={}
    for pid in(3447747,3447783,3447784,3448880,3448881,3448882,3448883,3448884,3448885,3448886,3448887):
        path=Path('/proc')/str(pid)
        if path.is_dir():r['original_generation_processes_remaining'][str(pid)]=(path/'cmdline').read_bytes().replace(b'\0',b' ').decode(errors='replace')
    r['original_generation_processes_gone']=not r['original_generation_processes_remaining']
    r['all_match']=all(info['matches']for category in('production','installed_api_sources','diagnostics','fixtures','native_fp16_weights')for info in r[category].values())
    assert r['all_match']and r['original_generation_processes_gone']
    r['VAE_original_source_restored']=True;r['encoder_and_pipeline_not_modified']=True;r['completed']=True
except BaseException as exc:r['failure']={'type':type(exc).__name__,'message':str(exc)};raise
finally:writer.write(r)
print(json.dumps({'all_match':r['all_match'],'production':len(r['production']),'installed_api_sources':len(r['installed_api_sources']),
                  'diagnostics':len(r['diagnostics']),'fixtures':len(r['fixtures']),'native_fp16_checkpoint_files':len(r['native_fp16_weights']),
                  'generation_processes_gone':r['original_generation_processes_gone']},indent=2))
