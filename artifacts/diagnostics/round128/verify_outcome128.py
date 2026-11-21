#!/usr/bin/env python3
"""Verify preserved R128 native-only blocked outcome; no inference or tolerance grant."""
from __future__ import annotations
import hashlib
import json
from pathlib import Path
from summarize_formal128 import summarize

ROOT=Path(__file__).resolve().parent


def main():
    executed=ROOT/'executed_formal128a'
    summary=summarize(executed)
    assert summary==json.loads((ROOT/'formal_summary128a.json').read_text())
    assert summary['native_original49_completed_all8'] and summary['generation_strict_checks_observed']==60760
    assert summary['extra_nontrajectory_strict_checks']==3720
    assert summary['candidate_adapter_never_activated_in_model'] and summary['candidate49_not_launched']
    assert summary['original_prediction_preserved_before_sampler26_all8']
    assert summary['source_sha_verified'] and summary['all8_native_master_SHA_unchanged']
    replays=[json.loads(path.read_text()) for path in executed.glob('formal_replay_rank*128.json')]
    assert len(replays)==8 and all(r['completed'] and not r.get('active_async') and not r.get('active_serial') for r in replays)
    for row in replays:
        assert len(row['cases'])==3 and all(c['strict_Half_checks']==155 for c in row['cases'])
        assert row['actual_TP2_runtime']['size']==2
        assert not row['all8ranks_bit_gates_passed'] and not row['candidate_formal_generation_allowed']
        assert not row['leaf_proof'] and not row['byte_proof'] and not row['profiles']
    lifecycle=json.loads((executed/'engine_overlap128a/lifecycle_result127.json').read_text())
    media=lifecycle['requests'][0]['media_verification']
    assert media['decoded_video_frames']==124 and media['width']==1344 and media['height']==768
    assert media['decoded_audio_samples']==165600
    cleanup=json.loads((ROOT/'cleanup128a.json').read_text())
    assert cleanup['completed'] and cleanup['all_private_processes_exited'] and cleanup['private_key_removed']
    assert cleanup['original_sources_and_noise_sha_unchanged'] and cleanup['executed_private_sources_sha_match']
    block=json.loads((ROOT/'issue4_blocked.json').read_text());assert block['blocker_issue']==15 and block['issue4_status']=='Blocked'
    result={'verification_passed':True,'scope':__doc__,'original8x49_native_request_complete':True,
        'original_generation_strict_checks':60760,'extra_replay_checks_not_sampler_steps':3720,'native_media_124frames_valid':True,
        'copy_stream_candidate_never_activated_in_model':True,'candidate_numeric_change_not_claimed':True,
        'original_prediction_preserved_after_replay':True,'native_repeat_root_cause_unknown':True,
        'existing_output_repeatability_not_confused_with_physical_restoration':True,
        'executed_sources_archived_and_sha_verified':True,'failed_driver_receipt_not_replaced_by_retrial':True,
        'private_process_IPC_cleanup_verified':True,'NUMA_probe_canceled_not_executed':True,
        'issue4_blocked_by':block['blocker_url'],'precision_tolerance_not_relaxed':True,'overall_goal_not_complete':True}
    path=ROOT/'verified_outcome128a.json';assert not path.exists();path.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2))


if __name__=='__main__':main()
