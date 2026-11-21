#!/usr/bin/env python3
"""R126 topology-aware live FP16 encoder TP4 within ONE confirmed HCCS mesh island."""
from __future__ import annotations
import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import statistics
import time

ROOT=Path('/tmp/h3_perf12601')
CHECKPOINT=Path('/home/mseco/models/MiniMax-H3/FL2VA')


def digest(tensor):
    return hashlib.sha256(tensor.detach().cpu().contiguous().view(__import__('torch').uint8).numpy().tobytes()).hexdigest()


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--label',required=True)
    p.add_argument('--warmup',type=int,default=1)
    p.add_argument('--repeats',type=int,default=3)
    p.add_argument('--reference-hidden',required=True)
    args=p.parse_args();assert '/' not in args.label
    out=ROOT/args.label;assert out.is_dir()
    rank,local=int(os.environ['RANK']),int(os.environ['LOCAL_RANK'])
    assert int(os.environ['WORLD_SIZE'])==4
    physical_group=[int(x) for x in os.environ['ASCEND_RT_VISIBLE_DEVICES'].split(',')]
    assert physical_group in ([0,1,2,3],[4,5,6,7])
    receipt=out/f'rank{rank}.json';assert not receipt.exists()
    r={'scope':__doc__,'completed':False,'rank':rank,'live_vision_and50_text_layers':True,
       'hidden_conditioning_cache':False,'CPU_model_inference':False,'cases':[],
       'actual_world_size':4,'physical_HCCS_island':physical_group,'encoder_TP_size':4,
       'cross_CPU_collectives_inside_encoder_group':False,'new_precision_policy_accepted':False}
    def publish():receipt.write_text(json.dumps(r,indent=2)+'\n')
    import torch
    import torch_npu
    from PIL import Image,ImageDraw
    from safetensors import safe_open
    from transformers import Qwen2TokenizerFast,Qwen3VLProcessor
    from vllm_omni.diffusion.distributed.parallel_state import init_distributed_environment,initialize_model_parallel,destroy_distributed_env,init_world_group
    from vllm_omni.model_executor.models.minimax_h3.preprocessing import build_minimax_h3_presentation
    from native_geometry_memo126 import NativeFP16GeometryMemo126
    torch.npu.set_device(local);torch.npu.matmul.allow_hf32=False;torch.npu.conv.allow_hf32=False
    torch.set_num_threads(4);device=torch.device(f'npu:{local}')
    try:
        init_distributed_environment(world_size=4,rank=rank,local_rank=local,distributed_init_method='env://',backend='hccl')
        initialize_model_parallel(tensor_parallel_size=2,sequence_parallel_size=2,ulysses_degree=2,ring_degree=1,backend='hccl')
        group=init_world_group(ranks=list(range(4)),local_rank=local,backend='hccl')
        assert group.world_size==4
        r['encoder_group_rank']=group.rank_in_group
        r['actual_visible_device_count']=torch.npu.device_count()
        assert r['actual_visible_device_count']==4
        ref_path=Path(args.reference_hidden).resolve()
        assert ref_path.parent==Path('/tmp/h3_perf12601/NPU_full_generation126') and ref_path.suffix=='.pt'
        eight_rank_reference=torch.load(ref_path,map_location='cpu',weights_only=True)['hidden_states'].contiguous()
        assert eight_rank_reference.shape==(2043,5120) and eight_rank_reference.dtype==torch.float16
        r['comparison_reference_path']=str(ref_path)
        r['eight_rank_hidden_reference_used_only_for_metrics_not_input']=True
        source=ROOT/'NPU_encoder_FP16_124.py';raw=source.read_bytes()
        assert hashlib.sha256(raw).hexdigest()=='0959895b778fc0900d785250b71279055d3940d975f3523909856072960f5026'
        spec=importlib.util.spec_from_file_location('own_NPU_encoder126',source)
        module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
        started=time.perf_counter()
        # Keep parameter version counters available for exact geometry invalidation.
        encoder=module.MiniMaxH3Qwen3VLEncoder(str(CHECKPOINT/'text_encoder'),device=device,load_model=True,encoder_group=group,quant_config=None)
        index=json.loads((CHECKPOINT/'text_encoder/model.safetensors.index.json').read_text())['weight_map']
        needed=[]
        for name in index:
            mapped=encoder._map_weight_name(name)
            if mapped is not None and not ('.layers.' in mapped[0] and int(mapped[0].split('.layers.')[1].split('.')[0])>=50):needed.append(name)
        def weights():
            for filename in sorted({index[name] for name in needed}):
                with safe_open(str(CHECKPOINT/'text_encoder'/filename),framework='pt',device='cpu') as f:
                    for name in sorted(needed):
                        if index[name]==filename:yield name,f.get_tensor(name)
        encoder.load_weights(weights());encoder.eval().to(device)
        assert all(p.dtype==torch.float16 for p in encoder.parameters() if p.is_floating_point())
        r['setup_s']=time.perf_counter()-started
        r['parameter_bytes']=sum(p.numel()*p.element_size() for p in encoder.parameters())
        tokenizer=Qwen2TokenizerFast.from_pretrained(str(CHECKPOINT),subfolder='tokenizer',local_files_only=True)
        processor=Qwen3VLProcessor.from_pretrained(str(CHECKPOINT),subfolder='processor',local_files_only=True)
        images=[]
        for x in [320,1024]:
            image=Image.new('RGB',(1344,768),(200,205,212));draw=ImageDraw.Draw(image)
            draw.rectangle((0,640,1344,768),fill=(98,102,108));draw.ellipse((x-45,550,x+45,640),fill=(215,30,30));images.append(image)
        started=time.perf_counter();vision=processor.image_processor(images=images,return_tensors='pt')
        ids,tags=build_minimax_h3_presentation(tokenizer,prompt='a red ball bouncing from left to right on the floor',task='fl2va',
            condition_labels=[('image',1),('image',2)],image_grid_thw=vision['image_grid_thw'],video_grid_thw=None,
            video_timestamps=None,merge_size=int(processor.image_processor.merge_size))
        r['processor_and_presentation_s']=time.perf_counter()-started
        r['input_ids_sha256']=digest(ids);r['pixels_sha256']=digest(vision['pixel_values'].half())
        r['grid_values']=vision['image_grid_thw'].tolist()
        def runtime():
            try:auto=torch.is_autocast_enabled('npu')
            except TypeError:auto=torch.is_autocast_enabled()
            try:dt=str(torch.get_autocast_dtype('npu'))
            except (AttributeError,RuntimeError):dt='torch.float16'
            return {'default_dtype':str(torch.get_default_dtype()),'default_device':str(torch.get_default_device()),
                    'autocast_enabled':bool(auto),'autocast_dtype':dt,
                    'matmul_allow_hf32':bool(torch.npu.matmul.allow_hf32),
                    'conv_allow_hf32':bool(torch.npu.conv.allow_hf32),
                    'stream_id':str(torch.npu.current_stream(device).npu_stream)}
        memo=NativeFP16GeometryMemo126(encoder,source,runtime_state=runtime)
        def run():
            return encoder.encode_ids(ids,pixel_values=vision['pixel_values'].half(),image_grid_thw=vision['image_grid_thw'])
        reference=None
        with torch.inference_mode():
            for case_index,enabled in enumerate([False]):
                memo.set_enabled(enabled)
                for _ in range(args.warmup):
                    h=run();torch.npu.synchronize();del h
                rows=[]
                for repeat in range(args.repeats):
                    torch.npu.synchronize();torch.distributed.barrier()
                    start=time.perf_counter();hidden=run();torch.npu.synchronize();local_s=time.perf_counter()-start
                    ts=torch.tensor([local_s],dtype=torch.float32,device=device)
                    torch.distributed.all_reduce(ts,op=torch.distributed.ReduceOp.MAX);max_s=float(ts.cpu())
                    cpu=hidden.detach().cpu().contiguous();assert cpu.shape==(2043,5120) and cpu.dtype==torch.float16
                    assert bool(torch.isfinite(cpu).all())
                    if reference is None:reference=cpu.clone()
                    changed=int((cpu.view(torch.int16)!=reference.view(torch.int16)).sum())
                    diff=cpu.double()-reference.double()
                    rel=float(diff.square().sum().sqrt()/reference.double().square().sum().sqrt().clamp_min(1e-30))
                    ref_diff=cpu.double()-eight_rank_reference.double()
                    reference_rel=float(ref_diff.square().sum().sqrt()/eight_rank_reference.double().square().sum().sqrt().clamp_min(1e-30))
                    del ref_diff
                    row={'relative_RMS_vs_live_eight_rank_reference':reference_rel,'raw_words_equal_vs_live_eight_rank_reference':bool(torch.equal(cpu.view(torch.uint8),eight_rank_reference.view(torch.uint8))),
                         'comparison_is_not_an_automatic_tolerance':True,'repeat':repeat,'local_s':local_s,'rank_max_s':max_s,'hidden_sha256':digest(cpu),
                         'different_words_vs_first_baseline':changed,'relative_RMS_vs_first_baseline':rel}
                    rows.append(row)
                    r['cases']=[x for x in r['cases'] if x['case_index']!=case_index]+[{'case_index':case_index,'memo_enabled':enabled,'checks':rows,
                         'rank_max_median_s':statistics.median(x['rank_max_s'] for x in rows),'memo_stats':dict(memo.stats)}]
                    publish()
                if enabled:
                    # Proof mode is deliberately outside clean timing. Use a
                    # temporary dispatch wrapper, not a replacement of owned
                    # geometry methods. Cached and original see the SAME grid;
                    # return the original output so proof cannot affect hidden.
                    proofs=[]
                    old_dispatch=memo._dispatch
                    def proof_dispatch(name,owner,grid):
                        candidate=old_dispatch(name,owner,grid)
                        reference=memo._originals[name](grid)
                        ac=reference.cpu().contiguous();bc=candidate.cpu().contiguous()
                        proofs.append({'method':name,'shape':list(ac.shape),'dtype':str(ac.dtype),
                                       'elements':ac.numel(),
                                       'bit_identical':bool(torch.equal(ac.view(torch.uint8),bc.view(torch.uint8))),
                                       'candidate_hits_at_proof':memo.stats['hits']})
                        return reference
                    memo._dispatch=proof_dispatch
                    try:
                        proof_hidden=run();torch.npu.synchronize()
                    finally:
                        memo._dispatch=old_dispatch
                    r['same_input_geometry_leaf_proof']=proofs
                    local_fail=int(len(proofs)!=2 or not all(x['bit_identical'] for x in proofs))
                    bad=torch.tensor([local_fail],dtype=torch.int32,device=device)
                    torch.distributed.all_reduce(bad,op=torch.distributed.ReduceOp.MAX)
                    if int(bad.cpu()):
                        memo.set_enabled(False)
                        r['memo_disabled_for_user_review']=True
                        publish()
                        raise FloatingPointError('Geometry cache changed a same-input native leaf')
                    del proof_hidden
        r['completed']=True;r['memo_hits']=memo.stats.get('hits',0);publish()
        if rank==0:print(json.dumps({'encoder_completed':True,'cases':r['cases'],'memo_stats':memo.stats},indent=2),flush=True)
        memo.uninstall()
    except BaseException as exc:
        r['failure']={'type':type(exc).__name__,'message':str(exc)};publish();raise
    finally:
        if torch.distributed.is_initialized():destroy_distributed_env()

if __name__=='__main__':main()
