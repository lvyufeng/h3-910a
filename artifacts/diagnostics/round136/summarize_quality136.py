#!/usr/bin/env python3
"""Summarize actual full49 scalar evidence, not a quality acceptance threshold."""
import argparse
import hashlib
import json
from pathlib import Path
import statistics
from fingerprints129 import normalized_fingerprint

SOURCE=Path(__file__).resolve().parent


def summarize(root):
    process=json.loads((root/'engine_quality136a_process.json').read_text());session=root/'engine_quality136a'
    life=json.loads((session/'lifecycle_result136.json').read_text())
    events=[json.loads(line) for p in session.glob('worker_runtime_pid*127.jsonl') for line in p.read_text().splitlines()]
    reports=sorted([json.loads(p.read_text()) for p in root.glob('quality_worker_rank*136.json')],key=lambda r:r['rank'])
    result={'process_exit_code':process.get('exit_code'),'completion_unknown':process.get('completion_unknown',False),
        'lifecycle_completed':life['completed'],'engine_initialization_s':life.get('engine_initialization_s'),
        'scope':'OriginalFalse / temporaryglobalTrue+finalAdaLN / restoredFalse, three49 evaluation quality requests',
        'quality_or_numeric_adoption':False,'new_tolerance':False,'copy_candidate':False,
        'full49_candidate_repeatability_tested':False,'reports':len(reports),'requests':[]}
    for original,digest in process['source_sha256'].items():
        path=SOURCE/'executed_sources136a'/Path(original).name
        assert hashlib.sha256(path.read_bytes()).hexdigest()==digest,original
    result['exact_executed_source_SHA_verified']=True
    if life.get('failure'):result['failure']=life['failure']
    for request in life['requests']:
        index=request['index'];rows=[r for r in events if r['request_index']==index]
        steps=[r for r in rows if r['event']=='dit_step_end'];pids=sorted({r['pid'] for r in steps})
        assert len(pids)==8 and len(steps)==49*8
        by_step=[[r for r in steps if r['step_index']==step] for step in range(49)]
        assert all(len(r)==8 and {x['pid'] for x in r}==set(pids) for r in by_step)
        rankmax=[max(r['elapsed_s_includes_offload_and_D2H_audit'] for r in group) for group in by_step]
        assert all(r['actual_Half_matrix_modules_this_step']==155 and r['actual_FA_launches']==50 and r['actual_FA_errors']==0 for r in steps)
        assert all(r['actual_deterministic_flag']==(index==2) and r['actual_candidate_calls_this_step']==int(index==2) for r in steps)
        item={'index':index,'label':request['label'],'generation_wall_s':request['generation_wall_s'],'worker_pids':pids,
            'actual_evaluations_per_rank':49,'sigma_boundaries':50,'strict_Half_checks':sum(r['actual_Half_matrix_modules_this_step'] for r in steps),
            'actual_DiT_FA_launches':sum(r['actual_FA_launches'] for r in steps),'actual_FA_errors':0,
            'actual_candidate_Python_calls':sum(r['actual_candidate_calls_this_step'] for r in steps),
            'first_step_coordinated_rankmax_s':rankmax[0],'median_remaining48_coordinated_rankmax_s':statistics.median(rankmax[1:]),
            'all49_step_coordinated_rankmax_s':rankmax,'pipeline_stage_rankmax_host_s':{},'prefetch_observed_method_calls':0,
            'step_timing_contains_original_range_and_prediction_D2H_audits':True}
        for name in ('pipeline._prepare_request_inputs','pipeline.encode_prompt','pipeline.diffuse','pipeline.decode','video_vae.decode_latent','audio_vae.decode_latent'):
            stage=[r for r in rows if r['event']=='stage_end' and r['stage']==name]
            if stage:item['pipeline_stage_rankmax_host_s'][name]=max(r['host_elapsed_s'] for r in stage)
        ends=[r for r in rows if r['event']=='native_request_end'];assert len(ends)==8
        item['prefetch_observed_method_calls']=sum(p['calls'] for r in ends for p in r['prefetch'])
        item['prefetch_method_calls_not_copy_engine_kernel_or_actual_transferred_byte_proof']=True
        states=[r['state']['memory'] for r in ends if 'peak_allocated_bytes' in r['state']['memory']]
        if states:item['rankmax_PyTorch_peak_allocated_GiB']=max(r['peak_allocated_bytes'] for r in states)/(1024**3)
        encoders=[r for r in rows if r['event']=='live_NPU_encoder_hidden'];assert len(encoders)==8 and {r['pid'] for r in encoders}==set(pids)
        assert all(r['actual_encoder_TP_size']==8 and r['encoder_group_ranks']==list(range(8)) for r in encoders)
        assert sorted(r['encoder_vocab_partition_rank'] for r in encoders)==list(range(8))
        item['actual_encoder_TP8_shards']=[{k:r[k] for k in ('pid','encoder_vocab_partition_rank','encoder_embedding_shard_shape','encoder_embedding_shard_dtype','encoder_embedding_shard_device')} for r in sorted(encoders,key=lambda r:r['encoder_vocab_partition_rank'])]
        result['requests'].append(item)
    if life['completed']:
        assert process['exit_code']==0 and process['all_private_children_exited'] and not process.get('cohort_monitor_errors')
        assert len(reports)==8 and [r['rank'] for r in reports]==list(range(8))
        assert all(r['completed'] and r['candidate_method_restored'] and not r['final_runtime']['deterministic'] and not r['final_runtime']['warn_only'] for r in reports)
        for r in reports:
            assert r['actual_original_evaluations']==147 and r['strict_Half_checks']==22785 and r['actual_FA_launches']==7350 and r['actual_FA_errors']==0
            assert r['actual_candidate_calls']==49 and r['candidate_row_count_calls']=={'2':1,'3':48}
            assert r['model_before_state']['native_masters']==r['model_after_state']['native_masters']
            assert normalized_fingerprint(r['model_before_state']['materialized_parameters_and_buffers'])==normalized_fingerprint(r['model_after_state']['materialized_parameters_and_buffers'])
        assert [r['worker_pids'] for r in result['requests']]==[result['requests'][0]['worker_pids']]*3
        result['strict_Half_checks_total']=sum(r['strict_Half_checks'] for r in result['requests']);assert result['strict_Half_checks_total']==182280
        result['actual_DiT_FA_launches_total']=sum(r['actual_DiT_FA_launches'] for r in result['requests']);assert result['actual_DiT_FA_launches_total']==58800
        result['actual_candidate_Python_calls_total']=sum(r['actual_candidate_Python_calls'] for r in result['requests']);assert result['actual_candidate_Python_calls_total']==392
        result['original_masters_resident_state_methods_and_false_flags_restored']=True
        result['all_original_R129_master_reference_checks_verified']=all(r['original_master_reference']['all50_original_R129_master_reference_SHA_verified'] for r in reports)
    return result


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--root',type=Path,required=True);parser.add_argument('--output',type=Path,required=True);args=parser.parse_args()
    result=summarize(args.root)
    if args.output.exists():assert json.loads(args.output.read_text())==result
    else:args.output.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2))


if __name__=='__main__':main()
