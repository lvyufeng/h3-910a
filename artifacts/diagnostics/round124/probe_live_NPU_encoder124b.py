#!/usr/bin/env python3
"""Live FP16 own-910A Qwen3-VL conditioning for the original H3 first/last-image request.

No GPU encoder activation substitution, no CPU inference, no BF16 roundtrip, and
no model downloads. Original local NPU checkpoint weights, TP8 live encoder, same
prompt/images/tokenizer/processor, own vision plus retained50 decoder layers.
The output is saved only for diagnostics, not a hidden full-generation cache.
"""
from __future__ import annotations
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import time

ROOT=Path('/tmp/h3_align12401')
CHECKPOINT=Path('/home/mseco/models/MiniMax-H3/FL2VA')


def main():
    import torch
    import torch_npu
    from PIL import Image,ImageDraw
    from safetensors import safe_open
    from transformers import Qwen2TokenizerFast,Qwen3VLProcessor
    from vllm_omni.diffusion.distributed.parallel_state import init_distributed_environment,initialize_model_parallel,destroy_distributed_env,init_world_group
    from vllm_omni.model_executor.models.minimax_h3.preprocessing import build_minimax_h3_presentation
    rank=int(os.environ['RANK']);local_rank=int(os.environ['LOCAL_RANK'])
    torch.npu.set_device(local_rank);torch.npu.matmul.allow_hf32=False;torch.npu.conv.allow_hf32=False
    torch.set_num_threads(4);device=torch.device(f'npu:{local_rank}')
    out=ROOT/'own_NPU_encoder_probe124b';out.mkdir(exist_ok=True)
    receipt=out/f'encoder_rank{rank}.json';assert not receipt.exists()
    r={'scope':__doc__,'completed':False,'rank':rank,'teacher_forcing_used':False,'CPU_model_inference_executed':False,
       'model_weight_downloaded':False,'encoder_TP':8,'encoder_matrix_weights_dtype':'torch.float16','BF16_output_roundtrip':False}
    def publish():receipt.write_text(json.dumps(r,indent=2)+'\n')
    publish()
    try:
        init_distributed_environment(world_size=8,rank=rank,local_rank=local_rank,distributed_init_method='env://',backend='hccl')
        initialize_model_parallel(tensor_parallel_size=2,sequence_parallel_size=4,ulysses_degree=4,ring_degree=1,backend='hccl')
        group=init_world_group(ranks=list(range(8)),local_rank=local_rank,backend='hccl')
        path=ROOT/'NPU_encoder_FP16_124.py'
        spec=importlib.util.spec_from_file_location('own_NPU_FP16_encoder124',path)
        module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
        start=time.perf_counter()
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
        loaded=encoder.load_weights(weights());r['loaded_parameter_names_count']=len(loaded)
        encoder.eval().to(device)
        assert all(p.dtype==torch.float16 for p in encoder.parameters() if p.is_floating_point())
        r['setup_s']=time.perf_counter()-start
        tokenizer=Qwen2TokenizerFast.from_pretrained(str(CHECKPOINT),subfolder='tokenizer',local_files_only=True)
        processor=Qwen3VLProcessor.from_pretrained(str(CHECKPOINT),subfolder='processor',local_files_only=True)
        images=[]
        for x in [320,1024]:
            image=Image.new('RGB',(1344,768),(200,205,212));draw=ImageDraw.Draw(image)
            draw.rectangle((0,640,1344,768),fill=(98,102,108));draw.ellipse((x-45,550,x+45,640),fill=(215,30,30));images.append(image)
        vision=processor.image_processor(images=images,return_tensors='pt')
        ids,tags=build_minimax_h3_presentation(tokenizer,prompt='a red ball bouncing from left to right on the floor',task='fl2va',
                                             condition_labels=[('image',1),('image',2)],image_grid_thw=vision['image_grid_thw'],video_grid_thw=None,video_timestamps=None,merge_size=int(processor.image_processor.merge_size))
        r['input_token_ids_sha256']=hashlib.sha256(ids.contiguous().view(torch.uint8).numpy().tobytes()).hexdigest()
        r['token_rows']=int(ids.numel())
        with torch.inference_mode():
            torch.npu.synchronize();start=time.perf_counter()
            hidden=encoder.encode_ids(ids,pixel_values=vision['pixel_values'].to(torch.float16),image_grid_thw=vision['image_grid_thw'])
            torch.npu.synchronize();r['live_encoder_elapsed_s']=time.perf_counter()-start
        assert hidden.dtype==torch.float16 and hidden.shape==(2043,5120)
        r['hidden_finite']=bool(torch.isfinite(hidden).all());assert r['hidden_finite']
        cpu=hidden.cpu().contiguous();r['hidden_shape']=list(cpu.shape)
        r['hidden_sha256_raw']=hashlib.sha256(cpu.view(torch.uint8).numpy().tobytes()).hexdigest();r['hidden_max_abs']=float(cpu.float().abs().max())
        if rank==0:
            path=out/'own_NPU_encoder_hidden124.pt'
            with path.open('xb') as f:torch.save({'hidden_states':cpu,'token_tags':tags.cpu(),'input_ids':ids.cpu()},f)
            r['diagnostic_capture_path']=str(path);r['capture_sha256']=hashlib.sha256(path.read_bytes()).hexdigest()
        r['completed']=True;publish();print(json.dumps({'own_NPU_encoder_completed':True,'rank':rank,'hidden_finite':True,'shape':list(hidden.shape),'elapsed_s':r['live_encoder_elapsed_s']}),flush=True)
    except BaseException as exc:r['failure']={'type':type(exc).__name__,'message':str(exc)};publish();raise
    finally:
        if torch.distributed.is_initialized():destroy_distributed_env()


if __name__=='__main__':main()
