#!/usr/bin/env python3
"""R126 diagnostic full live FP16 graph captures; never use capture timing as clean speed."""
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
    args=p.parse_args();assert '/' not in args.label
    out=ROOT/args.label;assert out.is_dir()
    rank,local=int(os.environ['RANK']),int(os.environ['LOCAL_RANK'])
    receipt=out/f'rank{rank}.json';assert not receipt.exists()
    r={'scope':__doc__,'completed':False,'rank':rank,'live_vision_and50_text_layers':True,
       'hidden_conditioning_cache':False,'CPU_model_inference':False,'cases':[]}
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
        init_distributed_environment(world_size=8,rank=rank,local_rank=local,distributed_init_method='env://',backend='hccl')
        initialize_model_parallel(tensor_parallel_size=2,sequence_parallel_size=4,ulysses_degree=4,ring_degree=1,backend='hccl')
        group=init_world_group(ranks=list(range(8)),local_rank=local,backend='hccl')
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
        targets=[('embedding',encoder.text_model.embed_tokens),('vision.root',encoder.vision),('vision.merger',encoder.vision.merger),('text.root',encoder.text_model)]
        targets.extend((f'vision.block.{i:02d}',b) for i,b in enumerate(encoder.vision.blocks))
        targets.extend((f'vision.deepstack.{i}',b) for i,b in enumerate(encoder.vision.deepstack_merger_list))
        targets.extend((f'text.layer.{i:02d}',b) for i,b in enumerate(encoder.text_model.layers))
        assert len(encoder.vision.blocks)==27 and len(encoder.text_model.layers)==50 and len(encoder.vision.deepstack_merger_list)==3
        r['capture_GPU_budget_each_pass_bytes']=2*1024**3
        r['capture_has_no_per_module_device_sync']=True
        r['capture_is_diagnostic_not_clean_performance']=True
        def leaves(value):
            if isinstance(value,torch.Tensor):return [value]
            if isinstance(value,(list,tuple)):return [t for v in value for t in leaves(v)]
            raise TypeError(type(value).__name__)
        reference=None
        with torch.inference_mode():
            for phase,enabled in [('native_A',False),('native_B',False)]:
                memo.set_enabled(enabled)
                if enabled:
                    # Fill geometry only through one ORIGINAL live vision call;
                    # cached hidden/DeepStack/text states are never substituted.
                    warm=run();torch.npu.synchronize();del warm
                capture={};captured_bytes=[0];handles=[];counts={}
                def collector(name):
                    def hook(module,inputs,output):
                        assert name not in capture,'A supposedly one-call graph node repeated'
                        values=leaves(output)
                        captured_bytes[0]+=sum(t.numel()*t.element_size() for t in values)
                        assert captured_bytes[0]<=r['capture_GPU_budget_each_pass_bytes']
                        capture[name]=[t.detach().clone() for t in values]
                        counts[name]=counts.get(name,0)+1
                    return hook
                for name,module in targets:handles.append(module.register_forward_hook(collector(name)))
                try:
                    final=run();torch.npu.synchronize()
                finally:
                    for handle in handles:handle.remove()
                assert set(capture)=={name for name,_ in targets}
                # All raw copies/metrics are AFTER the complete graph. Only one
                # <=2GiB device capture exists at a time; the reference stays on
                # this NPU machine's CPU, never on the client and never as input.
                cpu_values={name:[t.cpu().contiguous() for t in tensors] for name,tensors in capture.items()}
                capture.clear()
                torch.npu.synchronize()
                components=[]
                for name,_ in targets:
                    for leaf_index,tensor in enumerate(cpu_values[name]):
                        assert bool(torch.isfinite(tensor).all())
                        entry={'name':name,'leaf':leaf_index,'shape':list(tensor.shape),'dtype':str(tensor.dtype),
                               'elements':tensor.numel(),'sha256':digest(tensor)}
                        if reference is not None:
                            gold=reference[name][leaf_index]
                            assert gold.shape==tensor.shape and gold.dtype==tensor.dtype
                            entry['raw_words_equal_to_native_A']=bool(torch.equal(gold.view(torch.uint8),tensor.view(torch.uint8)))
                            word_dtype={2:torch.int16,4:torch.int32,8:torch.int64}.get(tensor.element_size(),torch.uint8)
                            entry['different_words']=int((gold.view(word_dtype)!=tensor.view(word_dtype)).sum())
                            diff=tensor.double()-gold.double()
                            entry['max_abs_error']=float(diff.abs().max())
                            entry['relative_RMS']=float(diff.square().sum().sqrt()/gold.double().square().sum().sqrt().clamp_min(1e-30))
                            del diff
                        components.append(entry)
                final_cpu=final.detach().cpu().contiguous()
                r['cases'].append({'phase':phase,'geometry_memo_enabled':enabled,'forward_call_counts':counts,
                                   'captured_GPU_bytes':captured_bytes[0],'components':components,
                                   'final_hidden_sha256':digest(final_cpu),'memo_stats':memo.snapshot()})
                if reference is None:reference=cpu_values
                else:cpu_values.clear()
                del final,final_cpu
                publish()
        r['all_27_vision_and_50_text_modules_executed_each_pass']=True
        r['numerical_acceptance']=False
        r['baseline_self_repeat_is_measured_not_assumed_zero']=True
        r['completed']=True;publish()
        memo.uninstall()
        if rank==0:
            print(json.dumps({'encoder_diagnostic_completed':True,'cases':[(c['phase'],c['captured_GPU_bytes']) for c in r['cases']]},indent=2),flush=True)
    except BaseException as exc:
        r['failure']={'type':type(exc).__name__,'message':str(exc)};publish();raise
    finally:
        if torch.distributed.is_initialized():destroy_distributed_env()

if __name__=='__main__':main()
