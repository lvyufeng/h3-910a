#!/usr/bin/env python3
"""Summarize actual original component repetitions without changing numeric policy."""
import argparse
import hashlib
import json
from pathlib import Path
import statistics
from fingerprints129 import normalized_fingerprint
from original_state137 import same_original_state

SOURCE=Path(__file__).resolve().parent


def summarize(root):
    process=json.loads((root/'engine_components137c_process.json').read_text());life=json.loads((root/'engine_components137c/component_lifecycle137.json').read_text())
    reports=sorted([json.loads(p.read_text()) for p in root.glob('component_rank*137.json')],key=lambda r:r['rank'])
    result={'scope':'Original native fixed encoder and fixed audio latent repetitions only','process_exit_code':process.get('exit_code'),
        'completion_unknown':process.get('completion_unknown',False),'lifecycle_completed':life['completed'],
        'engine_initialization_s':life.get('engine_initialization_s'),'reports':len(reports),'components':[],
        'new_math_flag_policy_or_tolerance':False,'sampler_or_DiT_evaluations':0,'native_false_throughout':True}
    for path,digest in process['source_sha256'].items():assert hashlib.sha256((SOURCE/'executed_sources137b'/Path(path).name).read_bytes()).hexdigest()==digest
    result['executed_source_SHA_verified']=True
    if life.get('failure'):result['failure']=life['failure']
    if (root/'cohort_failure128.json').exists():result['cohort_failure']=json.loads((root/'cohort_failure128.json').read_text())
    if not reports:return result
    result['worker_pids']=sorted({r['pid'] for r in reports})
    for component in ('encoder','audio_decoder'):
        sections=[next((c for c in r['components'] if c['component']==component),None) for r in reports]
        if not all(sections):continue
        values={'component':component,'trace_first_changes_by_rank':[],'phases':[],
            'execution_scope':'Actual TP8 encoder' if component=='encoder' else 'Original full FP32 audio decoder independently replicated on each rank, not TP8 decoder'}
        for rank,section in zip(reports,sections):
            values['trace_first_changes_by_rank'].append({'rank':rank['rank'],'first_change':section.get('trace_first_change')})
        for phase,count in (('native',8),('trace',2),('observer_restored',4)):
            groups=[[p for p in s['passes'] if p['phase']==phase] for s in sections]
            if not all(groups):continue
            n=min(len(g) for g in groups);passes=[p for group in groups for p in group]
            row={'phase':phase,'recorded_repeats_each_rank':n,'complete_expected_repeats':n==count,
                'all_words_same_phase_first':all(p['output_vs_phase_first']['all_words_equal'] for p in passes),
                'max_changed_words_self':max(p['output_vs_phase_first']['different_words'] for p in passes),
                'max_abs_difference_self':max(p['output_vs_phase_first']['max_abs_difference'] for p in passes),
                'max_relative_RMS_percent_self':max(p['output_vs_phase_first']['relative_RMS_percent'] for p in passes),
                'max_changed_words_vs_native_first':max(p['output_vs_native_first']['different_words'] for p in passes),
                'max_abs_difference_vs_native_first':max(p['output_vs_native_first']['max_abs_difference'] for p in passes),
                'max_relative_RMS_percent_vs_native_first':max(p['output_vs_native_first']['relative_RMS_percent'] for p in passes),
                'word_bits':passes[0]['output_vs_phase_first']['word_bits'],'total_output_words':passes[0]['output_vs_phase_first']['total_words'],
                'per_repeat_coordinated_rankmax_s':[max(g[i]['elapsed_full_component_sync_CPU_output_s'] for g in groups) for i in range(n)],
                'input_fingerprint_and_available_format_unchanged':all(p['input_before']==p['input_after'] and p['input_values_layout_and_available_format_unchanged'] for p in passes),
                'CPU_NPU_RNG_snapshots_unchanged_within_each_call':all(p['rng_before']==p['rng_after'] for p in passes),
                'native_false_HF32off_parametrize_cache_disabled':all(not p['runtime']['deterministic'] and not p['runtime']['warn_only'] and not p['runtime']['HF32_matmul'] and not p['runtime']['HF32_conv'] and not p['runtime']['parametrization_cache_enabled'] for p in passes),
                'max_PyTorch_peak_allocated_GiB':max(p['PyTorch_peak_allocated_GiB'] for p in passes),
                'nonclean_observer_audits_and_sync_not_pure_kernel_or_quality_acceptance':True}
            if n>1:row['median_coordinated_rankmax_excluding_phase_first_s']=statistics.median(row['per_repeat_coordinated_rankmax_s'][1:])
            values['phases'].append(row)
        if life['completed']:
            assert all(len(s['passes'])==14 and s['original_parameters_buffers_and_CPU_masters_unchanged'] and s['own_observers_removed'] for s in sections)
            assert all(same_original_state(s['state_before'],s['state_after']) for s in sections)
        result['components'].append(values)
    if life['completed']:
        assert process['exit_code']==0 and process['all_private_children_exited'] and not process.get('cohort_monitor_errors')
        assert len(reports)==8 and [r['rank'] for r in reports]==list(range(8)) and len(result['worker_pids'])==8
        assert all(r['completed'] and r['original_encoder_method_restored'] and r['initial_runtime']==r['final_runtime'] for r in reports)
        result.update(original_methods_flags_and_CPU_master_states_restored=True,actual_encoder_preparation_calls=8,
            actual_fixed_encoder_replay_calls=112,actual_fixed_audio_decoder_replay_calls=112)
    return result


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--root',type=Path,required=True);parser.add_argument('--output',type=Path,required=True);args=parser.parse_args()
    result=summarize(args.root)
    if args.output.exists():assert json.loads(args.output.read_text())==result
    else:args.output.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2))


if __name__=='__main__':main()
