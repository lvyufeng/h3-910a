#!/usr/bin/env python3
"""Reconcile full service-stage measurements without resident/E2E confusion."""
import hashlib
import json
from pathlib import Path
import statistics

ROOT = Path(__file__).resolve().parent
LABELS = ('bWarm', 'b1', 'b2', 'b3', 'cWarm', 'c1', 'c2', 'c3', 'paired')


def rank_rows(directory, label):
    rows = [json.loads((directory / ('%s_rank%d.json' % (label, rank))).read_text()) for rank in range(8)]
    if {row['rank'] for row in rows} != set(range(8)):
        raise ValueError('Need eight unique ranks')
    if any(not row.get('completed') or row.get('failure') or row['label'] != label for row in rows):
        raise ValueError('Incomplete/mismatched rank receipt')
    if len({row['timing_source_sha256'] for row in rows}) != 1:
        raise ValueError('Ranks executed mixed timing code')
    return rows


def normal(rows):
    if len(rows) != 8 or any(row['mode'] not in ('baseline', 'candidate') for row in rows):
        raise ValueError('Need eight normal baseline/candidate rank receipts')
    if any(len(row['calls']) != 1 for row in rows):
        raise ValueError('A paired/multi-call request is not normal service latency')
    active = rows[0]['mode'] == 'candidate'
    if any(row['mode'] != rows[0]['mode'] or row['calls'][0]['candidate_enabled'] != active for row in rows):
        raise ValueError('Mixed decode path in one request')
    if active and any(row['controller_calls_delta'] <= 0 for row in rows):
        raise ValueError('Candidate path never executed')
    if not active and any(row['controller_calls_delta'] != 0 for row in rows):
        raise ValueError('Baseline accidentally used the candidate')
    return {'rank_max_wall_s': max(row['calls'][0]['wall_s'] for row in rows),
            'rank_max_stream_s': max(row['calls'][0]['stream_s'] for row in rows),
            'candidate_enabled': active}


def paired(rows):
    expected_modes = [False, True, False, True, False, True, False, True, False]
    outputs, inputs = [], []
    for row in rows:
        if row['mode'] != 'paired' or len(row['calls']) != len(expected_modes):
            raise ValueError('Need exactly the paired complete stage sequence')
        if [call['candidate_enabled'] for call in row['calls']] != expected_modes:
            raise ValueError('Incorrect paired mode ordering')
        if row['input_before'] != row['input_after']:
            raise ValueError('Input was mutated by decode')
        inputs.append(row['input_before'])
        outputs.extend(call['output'] for call in row['calls'])
        outputs.append(row['return_output'])
    if any(value != inputs[0] for value in inputs) or any(value != outputs[0] for value in outputs):
        raise ValueError('Complete same-input video/audio outputs do not match on every rank/call')
    video, audio = outputs[0]['video'], outputs[0]['audio']
    if video['shape'] != [1, 3, 124, 768, 1344] or video['dtype'] != 'torch.uint8' or video['bytes'] != 383975424:
        raise ValueError('Unexpected full-video contract')
    if (len(audio['shape']) != 3 or audio['shape'][:2] != [1, 2] or not 160000 <= audio['shape'][2] <= 176000
            or audio['dtype'] != 'torch.float32' or audio['bytes'] != 8 * audio['shape'][2]):
        raise ValueError('Need nonempty stereo FP32 audio of the actual 5s/32kHz workload, not an equal empty placeholder')
    if any(not row['return_call']['candidate_enabled'] or row['controller_calls_delta'] <= 0 for row in rows):
        raise ValueError('Paired candidate/returned path did not execute')
    result = {'all_complete_video_audio_outputs_equal': True, 'complete_output_checks': len(outputs),
              'inputs_immutable_and_identical_on_all_ranks': True, 'reference_output': outputs[0], 'cases': {},
              'scope': 'Paired actual service decode on one actual denoiser output. Not a normal request wall time or native49-step quality.'}
    for mode, indices in (('baseline', (2, 4, 6)), ('candidate', (3, 5, 7))):
        maxima = [max(row['calls'][i]['wall_s'] for row in rows) for i in indices]
        result['cases'][mode] = {'warm_rank_max_wall_s': maxima, 'median_warm_rank_max_wall_s': statistics.median(maxima),
                                 'all_warm_calls_under20s': all(value < 20 for value in maxima)}
    result['restored_baseline_rank_max_s'] = max(row['calls'][8]['wall_s'] for row in rows)
    result['speedup'] = result['cases']['baseline']['median_warm_rank_max_wall_s'] / result['cases']['candidate']['median_warm_rank_max_wall_s']
    return result


def diagnose_failed_pair(rows):
    """Retain the failed whole-A/V gate; report disjoint video/audio evidence."""
    modes = [False, True, False, True, False, True, False, True, False]
    if len(rows) != 8 or any(row['mode'] != 'paired' or row.get('completed') or row.get('failure', {}).get('type') != 'AssertionError'
           or row.get('failure', {}).get('message') != 'Paired same-input complete service outputs differ'
           or len(row['calls']) != 9 for row in rows):
        raise ValueError('Not the retained known complete-output identity failure')
    if any([call['candidate_enabled'] for call in row['calls']] != modes for row in rows):
        raise ValueError('Unexpected failed-pair mode sequence')
    immutable = all(row['input_before'] == row['input_after'] == rows[0]['input_before'] for row in rows)
    if not immutable:
        raise ValueError('Mutated/nonidentical inputs need a separate diagnosis, not a native audio variance claim')
    videos = [call['output']['video'] for row in rows for call in row['calls']]
    video_exact = immutable and all(video == videos[0] for video in videos)
    video_contract = (videos[0]['shape'] == [1, 3, 124, 768, 1344] and videos[0]['dtype'] == 'torch.uint8'
                      and videos[0]['bytes'] == 383975424)
    per_rank = []
    for row in rows:
        values = [call['output']['audio'] for call in row['calls']]
        if any(value['shape'] != [1, 2, 165600] or value['dtype'] != 'torch.float32' or value['bytes'] != 1324800 for value in values):
            raise ValueError('Failed gate does not contain the expected real complete audio')
        per_rank.append({'rank': row['rank'], 'native_baseline_unique_audio_hashes': len({v['sha256'] for v in values[::2]}),
                         'candidate_unique_audio_hashes': len({v['sha256'] for v in values[1::2]}),
                         'all_call_unique_audio_hashes': len({v['sha256'] for v in values})})
    if any(row['native_baseline_unique_audio_hashes'] != 5 or row['candidate_unique_audio_hashes'] != 4 for row in per_rank):
        raise ValueError('Observed output variance differs from the classified native-audio failure')
    result = {'failed_whole_audio_video_identity_gate': True, 'all_complete_video_audio_outputs_equal': False,
              'video_only_identity_accepted': video_exact and video_contract, 'complete_video_checks': len(videos),
              'video_reference': videos[0], 'inputs_immutable_and_identical_on_all_ranks': immutable,
              'audio_repeat_evidence': per_rank,
              'scope': '72 complete same-input service video outputs; whole-A/V identity FAILED because native audio itself is not bit-repeatable. No output error magnitude can be inferred from hashes alone.', 'cases': {}}
    for mode, indices in (('baseline', (2, 4, 6)), ('candidate', (3, 5, 7))):
        maxima = [max(row['calls'][index]['wall_s'] for row in rows) for index in indices]
        result['cases'][mode] = {'warm_rank_max_wall_s': maxima, 'median_warm_rank_max_wall_s': statistics.median(maxima),
                                 'all_warm_calls_under20s': all(value < 20 for value in maxima)}
    result['restored_baseline_rank_max_s'] = max(row['calls'][8]['wall_s'] for row in rows)
    result['speedup'] = result['cases']['baseline']['median_warm_rank_max_wall_s'] / result['cases']['candidate']['median_warm_rank_max_wall_s']
    return result


def summarize(root=ROOT):
    http = json.loads((root / 'http' / 'result.json').read_text())
    failed_pair = http.get('failure', {}).get('type') == 'HTTPError'
    if not http.get('completed') and not failed_pair:
        raise ValueError('HTTP benchmark incomplete for an unclassified reason')
    requests = {row['label']: row for row in http['requests']}
    expected_labels = LABELS[:-1] if failed_pair else LABELS
    if set(requests) != set(expected_labels) or len(http['requests']) != len(expected_labels):
        raise ValueError('Missing/duplicate completed real HTTP requests')
    if hashlib.sha256((root / 'service_http64.py').read_bytes()).hexdigest() != http['source_sha256']:
        raise ValueError('Actual HTTP workload/manager source mismatch')
    result = {'scope': 'Actual 8x910A FL2VA HTTP one-interval request; measured complete service decode includes video+audio+residency, excludes subsequent video D2H and MP4. Not49-step quality, T2VA or official benchmark.',
              'normal_requests': {}, 'stage_accepted_under20s': False, 'official_lossless_goal_complete': False,
              'http_run_completed': bool(http.get('completed')), 'http_failure_retained': http.get('failure')}
    source_sha = hashlib.sha256((root / 'h3_service_audit64.py').read_bytes()).hexdigest()
    for label in LABELS:
        if label == 'paired' and failed_pair:
            rows = [json.loads((root / 'timing' / ('paired_rank%d.json' % rank)).read_text()) for rank in range(8)]
            if any(row['rank'] != rank or row['label'] != 'paired' or row['timing_source_sha256'] != source_sha for rank, row in enumerate(rows)):
                raise ValueError('Invalid source/rank/label in failed pair')
            result['same_input_service_decode'] = diagnose_failed_pair(rows)
            continue
        rows = rank_rows(root / 'timing', label)
        if any(row['timing_source_sha256'] != source_sha for row in rows):
            raise ValueError('Timing source changed from actual execution')
        expected_mode = 'paired' if label == 'paired' else ('candidate' if label.startswith('c') else 'baseline')
        if requests[label]['mode'] != expected_mode or any(row['mode'] != expected_mode for row in rows):
            raise ValueError('Label/HTTP/worker path mismatch: ' + label)
        if label == 'paired':
            result['same_input_service_decode'] = paired(rows)
        else:
            summary = normal(rows)
            item = requests[label]
            summary.update(warmup_excluded=item['warmup_excluded'], request_wall_s=item['request_wall_s'],
                           server_inference_time_s=item['response']['inference_time_s'], mp4_sha256=item['mp4_sha256'])
            result['normal_requests'][label] = summary
    result['normal_warm_cases'] = {}
    for mode, labels in (('baseline', ('b1', 'b2', 'b3')), ('candidate', ('c1', 'c2', 'c3'))):
        stage = [result['normal_requests'][label]['rank_max_wall_s'] for label in labels]
        wall = [result['normal_requests'][label]['request_wall_s'] for label in labels]
        result['normal_warm_cases'][mode] = {'decode_rank_max_wall_s': stage,
                                             'median_decode_s': statistics.median(stage),
                                             'all_decode_calls_under20s': all(value < 20 for value in stage),
                                             'median_request_wall_s': statistics.median(wall)}
    result['normal_decode_speedup'] = result['normal_warm_cases']['baseline']['median_decode_s'] / result['normal_warm_cases']['candidate']['median_decode_s']
    result['normal_downloaded_mp4s_all_identical'] = len({requests[label]['mp4_sha256'] for label in LABELS if label not in ('bWarm', 'cWarm', 'paired')}) == 1
    result['timing_only_target_met'] = result['normal_warm_cases']['candidate']['all_decode_calls_under20s']
    result['stage_accepted_under20s'] = (result['normal_warm_cases']['candidate']['all_decode_calls_under20s']
                                           and result['same_input_service_decode']['all_complete_video_audio_outputs_equal']
                                           and result['same_input_service_decode']['cases']['candidate']['all_warm_calls_under20s'])
    audio = json.loads((root / 'audio_repeat01.json').read_text())
    if not audio.get('completed') or audio.get('failure') or audio['source_sha256'] != hashlib.sha256((root / 'audio_repeat64.py').read_bytes()).hexdigest():
        raise ValueError('Native audio control incomplete/source mismatch')
    if audio['input_sha_before'] != audio['input_sha_after'] or audio['cpu_master_sha_before'] != audio['cpu_master_sha_after']:
        raise ValueError('Audio control input or immutable weight state changed')
    result['native_audio_control'] = {'scope': audio['scope'], 'input_and_cpu_weight_masters_immutable': True,
                                      'production_audio_modified': audio['production_audio_modified'], 'cases': {}}
    for name, case in audio['cases'].items():
        if not case.get('completed') or len(case['calls']) != 5:
            raise ValueError('Incomplete native audio repeat group')
        result['native_audio_control']['cases'][name] = {
            'distinct_hashes': len({call['sha256'] for call in case['calls']}),
            'rng_state_unchanged': case['rng_state_unchanged'],
            'median_after_first_s': case['median_after_first_s'],
            'max_abs_vs_first_in_case': max(call['vs_first_in_case']['max_abs_error'] for call in case['calls']),
            'max_relative_rms_vs_first_in_case': max(call['vs_first_in_case']['relative_rms'] for call in case['calls']),
            'all_finite': all(call['vs_first_in_case']['finite'] for call in case['calls'])}
    return result


if __name__ == '__main__':
    result = summarize()
    (ROOT / 'summary.json').write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result, indent=2))
