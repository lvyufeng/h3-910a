#!/usr/bin/env python3
"""R126 same-latent VAE operand-cache trial with unchanged FP32 parameters.

Native decoder/installer source is pinned. Baseline and candidate warm timing
have no profiler, activation-capture or proof hooks. Every actual local tile is
then replayed before quantization; any raw-word difference prevents deployment.
No model weights or activations are downloaded or published by this script.
"""
from __future__ import annotations
import argparse
import datetime
import hashlib
import inspect
import json
import os
from pathlib import Path
import statistics
import sys
import time

from contracts126 import check_local_files,validate_manifest,sha_file,atomic_json


def parse():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--manifest',required=True)
    p.add_argument('--root',required=True)
    p.add_argument('--label',required=True)
    p.add_argument('--execute',action='store_true')
    p.add_argument('--repeats',type=int,default=3)
    p.add_argument('--tile-input-budget',type=int,default=32*1024**2)
    return p.parse_args()


def main():
    args=parse();manifest=validate_manifest(json.loads(Path(args.manifest).read_text()))
    root=Path(args.root).resolve();assert root==Path('/tmp/h3_perf12601')
    assert '/' not in args.label and args.repeats>=1
    check_local_files(manifest,manifest['checkpoint'],manifest['latents'])
    if not args.execute:
        print(json.dumps({'validated':True,'model_executed':False,'action':'require --execute'}));return
    os.environ.update(manifest['environment'])
    sys.path.insert(0,manifest['nan_compat_dir'])
    import torch
    import torch_npu
    from vllm_omni.diffusion.models.minimax_h3 import vae as adapter
    from vllm_omni.platforms import current_omni_platform
    import npu_half_nan_compat124 as compat
    from decode_only126 import observed_installer,parameters,all_float_fp32,verify_native_sources
    from autocast_operand_cache126 import AutocastOperandCache126
    assert torch.__version__==manifest['torch_version'] and torch_npu.__version__==manifest['torch_npu_version']
    rank=int(os.environ['RANK']);local=int(os.environ['LOCAL_RANK'])
    torch.npu.set_device(local);device=torch.device('npu',local)
    torch.npu.matmul.allow_hf32=False;torch.npu.conv.allow_hf32=False;torch.set_num_threads(4)
    torch.distributed.init_process_group('hccl',timeout=datetime.timedelta(seconds=1800))
    out=root/args.label
    if rank==0:out.mkdir(mode=0o700,exist_ok=False)
    torch.distributed.barrier()
    r={'scope':__doc__,'completed':False,'rank':rank,'installer':{'calls':0,'native_return':None},'cases':[],
       'driver_sha256':sha_file(__file__),'operand_cache_sha256':sha_file(Path(__file__).with_name('autocast_operand_cache126.py')),
       'manifest_sha256':sha_file(args.manifest),'latents_sha256':manifest['latents_sha256'],
       'original_parameters_rewritten':False,'numerical_changes_not_auto_accepted':True}
    def publish():atomic_json(out/f'rank{rank}.json',r)
    cache=None;video=None;tile_inputs=[]
    def gate(local_fail,message):
        bad=torch.tensor([int(local_fail)],dtype=torch.int32,device=device)
        torch.distributed.all_reduce(bad,op=torch.distributed.ReduceOp.MAX)
        if int(bad.cpu()):
            r['requires_user_numerical_decision']=True;r['deployment_stopped']=True;publish()
            raise FloatingPointError(message)
    def words_equal(a,b):
        aa=a.detach().cpu().contiguous();bb=b.detach().cpu().contiguous()
        return bool(torch.equal(aa.view(torch.uint8),bb.view(torch.uint8))),aa,bb
    def digest(t):return hashlib.sha256(t.contiguous().view(torch.uint8).numpy().tobytes()).hexdigest()
    try:
        old_dtype=torch.get_default_dtype();torch.set_default_dtype(torch.float32)
        begun=time.perf_counter()
        try:
            with observed_installer(torch,adapter,'observe-native',r['installer']):
                video=adapter.MiniMaxH3VideoVAE(str(Path(manifest['checkpoint'])/'video_vae'),device=device,load_device=torch.device('cpu'),decode_only=True,trust_remote_code=True)
        finally:torch.set_default_dtype(old_dtype)
        video.set_parallel_size(8,process_group=torch.distributed.group.WORLD)
        video.decoder_component.load_to_device();torch.npu.synchronize()
        assert all_float_fp32(torch,video)
        assert r['installer']['native_return'] is False
        r['load_and_construct_s']=time.perf_counter()-begun
        r['resident_parameters_before']=parameters(video)
        native=Path(inspect.getfile(type(video.model.decoder)))
        assert sha_file(native)==manifest['native_source_sha256']['native_decoder']
        bundle=torch.load(manifest['latents'],map_location='cpu',weights_only=True)
        latent=bundle['video_latent'].to(device)
        assert latent.dtype==torch.float32 and list(latent.shape)==manifest['video_latent_shape']
        def whole_decode():
            with torch.inference_mode(),compat.half_nan_scope(enabled=True):
                with current_omni_platform.create_autocast_context(device_type='npu',dtype=torch.float16,enabled=True):
                    return video.decode_latent(latent)
        def tile_decode(x):
            with torch.inference_mode(),compat.half_nan_scope(enabled=True):
                with current_omni_platform.create_autocast_context(device_type='npu',dtype=torch.float16,enabled=True):
                    return video.model.decoder(x)
        # The cold/capture pass is NOT a clean warm performance sample.
        input_bytes=0
        def capture(module,inputs):
            nonlocal input_bytes
            assert len(inputs)==1 and isinstance(inputs[0],torch.Tensor)
            x=inputs[0];input_bytes+=x.numel()*x.element_size()
            assert input_bytes<=args.tile_input_budget
            tile_inputs.append(x.detach().clone())
        handle=video.model.decoder.register_forward_pre_hook(capture)
        cold=time.perf_counter()
        try:
            cold_result=whole_decode();torch.npu.synchronize()
        finally:handle.remove()
        r['cold_capture_pass_s']=time.perf_counter()-cold
        baseline=cold_result.detach().cpu().contiguous();del cold_result
        assert list(baseline.shape)==manifest['video_output_shape'] and baseline.dtype==torch.uint8
        r['actual_local_tile_inputs']=[{'shape':list(x.shape),'dtype':str(x.dtype)} for x in tile_inputs]
        r['captured_tile_input_bytes']=input_bytes
        assert tile_inputs and len(tile_inputs)<=128
        cache=AutocastOperandCache126(video.model.decoder)
        start=time.perf_counter();cache.prepare();r['cache_prime_s']=time.perf_counter()-start
        r['cache_before_trial']=cache.snapshot()
        def clean_case(name,enabled):
            cache.set_enabled(enabled)
            # Explicit warmup for each selected graph/operand variant.
            z=whole_decode();torch.npu.synchronize();del z
            rows=[]
            for i in range(args.repeats):
                torch.npu.synchronize();torch.distributed.barrier()
                t=time.perf_counter();value=whole_decode();torch.npu.synchronize();elapsed=time.perf_counter()-t
                ts=torch.tensor([elapsed],dtype=torch.float32,device=device)
                torch.distributed.all_reduce(ts,op=torch.distributed.ReduceOp.MAX)
                cpu=value.detach().cpu().contiguous();del value
                eq=bool(torch.equal(cpu,baseline))
                rows.append({'repeat':i,'rank_max_s':float(ts.cpu()),'local_s':elapsed,
                             'complete_Uint8_words_equal':eq,'sha256':digest(cpu),
                             'Uint8_equality_alone_is_not_precision_proof':True,
                             'peak_allocated_bytes':torch.npu.max_memory_allocated(device)})
                gate(not eq,'Complete VAE output changed; operand cache remains undeployed')
                del cpu
            r['cases'].append({'name':name,'cached_operands_enabled':enabled,'checks':rows,
                               'median_rank_max_s':statistics.median(v['rank_max_s'] for v in rows),
                               'no_profile_or_capture_hooks':True});publish()
        clean_case('native_before',False)
        # Every eligible Linear sees native A, native B and cached C on the same
        # actual FIRST tile input. The proof returns A to all downstream ops.
        cache.begin_proof();local_error=None
        try:
            proof_result=tile_decode(tile_inputs[0]);torch.npu.synchronize();del proof_result
        except BaseException as exc:
            local_error={'type':type(exc).__name__,'message':str(exc)}
        finally:r['first_tile_all_Linear_leaf_proof']=cache.end_proof()
        r['leaf_proof_error']=local_error;publish()
        leaf=r['first_tile_all_Linear_leaf_proof']
        gate(local_error is not None or not leaf['all_same_input_leaf_words_equal'] or not leaf['all_eligible_modules_executed'],
             'Same-input cached Linear operands failed raw-word proof; stop and ask user')
        # Complete 36-layer prequantized decoder output pairs for EVERY actual
        # locally owned tile, not only final quantized frames or selected values.
        tile_rows=[]
        from tile_rounds126 import aligned_local_tile_rounds
        local_count=len(tile_inputs)
        max_count=torch.tensor([local_count],dtype=torch.int32,device=device)
        torch.distributed.all_reduce(max_count,op=torch.distributed.ReduceOp.MAX)
        global_count=int(max_count.cpu())
        r['local_tile_count']=local_count;r['globally_aligned_tile_proof_rounds']=global_count
        for i,has_local_tile in aligned_local_tile_rounds(local_count,global_count):
            if not has_local_tile:
                # Ranks own different numbers of spatial tiles (28 vs 21 on
                # this request). Still enter EACH gate, never a later decode
                # barrier while peers are in an all-reduce proof gate.
                gate(False,'A peer prequantized tile failed its word proof')
                continue
            x=tile_inputs[i]
            cache.set_enabled(False);a=tile_decode(x);b=tile_decode(x)
            cache.set_enabled(True);c=tile_decode(x);torch.npu.synchronize()
            self_same,aa,bb=words_equal(a,b);candidate_same,_,cc=words_equal(a,c)
            delta=cc.double()-aa.double()
            tile_rows.append({'tile_index':i,'shape':list(aa.shape),'dtype':str(aa.dtype),'elements':aa.numel(),
                'native_self_repeat_words_equal':self_same,'cached_words_equal':candidate_same,
                'native_sha256':digest(aa),'cached_sha256':digest(cc),
                'finite_all':bool(torch.isfinite(aa).all() and torch.isfinite(bb).all() and torch.isfinite(cc).all()),
                'max_abs_error':float(delta.abs().max()),
                'relative_RMS':float(delta.square().sum().sqrt()/aa.double().square().sum().sqrt().clamp_min(1e-30))})
            r['all_local_prequantized_tile_proofs']=tile_rows;publish()
            gate(not self_same or not candidate_same,'Full prequantized tile changed or native self-repeat unresolved; stop and ask user')
            del a,b,c,aa,bb,cc,delta
        clean_case('cached',True)
        clean_case('native_restored',False)
        r['cache_after_trial']=cache.snapshot();assert r['cache_after_trial']['original_parameters_unchanged']
        r['resident_parameters_after']=parameters(video)
        r['all_prequantized_tile_pairs_equal']=all(x['cached_words_equal'] and x['native_self_repeat_words_equal'] for x in tile_rows)
        r['completed']=True;r['accepted_lossless']=False;publish()
        if rank==0:print(json.dumps({'VAE_operand_cache_completed':True,'cases':r['cases'],'cache':r['cache_after_trial']},indent=2),flush=True)
    except BaseException as exc:
        r['failure']={'type':type(exc).__name__,'message':str(exc)};publish();raise
    finally:
        if cache is not None:cache.close()
        tile_inputs.clear()
        if video is not None:video.decoder_component.offload_to_cpu()
        if torch.distributed.is_initialized():torch.distributed.destroy_process_group()

if __name__=='__main__':main()
