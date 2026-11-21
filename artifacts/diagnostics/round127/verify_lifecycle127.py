#!/usr/bin/env python3
"""Verify completed R127 lifecycle/receipt/source/media evidence, no model execution."""
from __future__ import annotations
import hashlib
import json
from pathlib import Path
import argparse
from summarize_lifecycle127 import summarize

ROOT=Path(__file__).resolve().parent
EXECUTED=ROOT/'executed_lifecycle127c_compressed'
SESSION=EXECUTED/'engine_lifecycle127c'


def verify():
    process=json.loads((EXECUTED/'engine_lifecycle127c_process.json').read_text())
    assert process['exit_code']==0 and process['pid']==1947359
    assert process['launcher_pid']==1947358
    for remote,digest in process['source_sha256'].items():
        source=ROOT/Path(remote).name
        assert hashlib.sha256(source.read_bytes()).hexdigest()==digest,('Executed source drift',remote)
    result=summarize(SESSION)
    summary=json.loads((ROOT/'lifecycle_summary127c.json').read_text());assert result==summary
    media=json.loads((ROOT/'native_repeat_media127.json').read_text())
    assert media['completed'] and not media['quality_automatically_accepted'] and len(media['pairs'])==2
    for row in result['cases']:
        assert row['all_8_ranks_native_video_encoder_decoder_and_audio_reload_confirmed']
        assert row['layerwise_prefetch_copy_stream_roles']=={'shared':1016}
        assert row['actual_layerwise_prefetch_calls_total']==20232
        files=list((SESSION/row['label']).glob('audit_complete_pid*127.json'))
        assert len(files)==8
        for path in files:
            receipt=json.loads(path.read_text());assert receipt['success'] and receipt['audit_jsonl_closed'] and receipt['request_index']==row['index']
    cleanup=json.loads((ROOT/'cleanup127c.json').read_text())
    assert cleanup['completed'] and cleanup['private_key_removed'] and not cleanup['private_live_processes']
    assert cleanup['original_sources_and_noise_fingerprints_unchanged'] and cleanup['private_launched_sources_sha_match']
    failed=json.loads((ROOT/'failed_lifecycle127b_summary.json').read_text())
    followup=json.loads((ROOT/'native_cpu_segv_issue.json').read_text())
    assert failed['native_worker_exit_code']==-11 and not failed['root_cause_confirmed']
    assert followup['issue']==14 and followup['unresolved_native_segv']
    return {'scope':__doc__,'verification_passed':True,'actual_full_requests':3,'same8worker_and_model_objects':True,
        'actual_original_DiT50_blocks_each49_step':True,'true_TP2_parameter_bytes_each_rank':33168334080,
        'all_182280_strict_Half_checks_passed':True,'all8ranks_native_stager_reload_each_request':True,
        'actual_copy_stream_shared_not_assumed_independent':True,'private_source_sha_checked':True,
        'native_media_repeat_changes_not_quality_accepted':True,'original_sources_unchanged_cleanup_verified':True,
        'native_CPU_pack_segv_is_still_unresolved':True,'reliability_issue':followup['url'],
        'local_model_or_activation_archive_transfer_required':False,'latency_targets_automatically_completed':False,
        'generation_wall_s':[row['generation_wall_s'] for row in result['cases']],
        'whole_forward_warm_median_s':[row['whole_forward_hook']['warm_steps_1_to_48_median_s'] for row in result['cases']],
        'full_VAE_stage_rank_max_s':[row['cold_or_warm_native_VAE_combined_rank_max_s'] for row in result['cases']]}


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--output',type=Path);args=parser.parse_args()
    result=verify()
    if args.output:
        assert not args.output.exists();args.output.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2))


if __name__=='__main__':main()
