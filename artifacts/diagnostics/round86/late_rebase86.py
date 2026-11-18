#!/usr/bin/env python3
"""Rebase decoder25..49 from unchanged CPUFP16 rawlayer24;not a full encode timer."""
from pathlib import Path
import os
import hashlib
import json
import time
ROOT=Path('/tmp/h3_round86_late01')


def main():
    import torch
    import torch_npu
    import full_row_fp32_fixed84 as previous
    from evidence_io86 import EvidenceWriter
    from safetensors import safe_open
    from transformers import Qwen3VLConfig
    rank=int(os.environ['RANK']);size=int(os.environ['WORLD_SIZE']);assert size==8
    writer=EvidenceWriter(ROOT/('tp8_rank%d.json'%rank))
    r={'scope':__doc__,'rank':rank,'world_size':size,'completed':False,
       'source_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
       'not_full_encoder_timing':True,'full_encoder_precision_accepted':False,
       'whole_goal_complete':False,'numerical_tolerance':None,'cases':{}}
    ready=False;kv=None;row=None;engine=None
    try:
        torch.set_num_threads(2);torch.npu.set_device(rank)
        previous.ROOT=ROOT;reference=previous.load_cpu_gate(r);candidate=previous.load_candidate(r)
        previous.install_fp16_f32_math_attention(candidate,r)
        from kv_padding81 import install as install_kv
        kv=install_kv(candidate)
        from vllm_omni.diffusion.distributed.group_coordinator import GroupCoordinator
        from vllm_omni.diffusion.distributed.parallel_state import init_distributed_environment,initialize_model_parallel
        init_distributed_environment(world_size=size,rank=rank,local_rank=rank,backend='hccl');ready=True
        initialize_model_parallel(sequence_parallel_size=size,ulysses_degree=size,tensor_parallel_size=1,backend='hccl')
        group=GroupCoordinator([list(range(size))],rank,'hccl')
        assert previous.file_sha(previous.MODEL/'config.json')==previous.CPU_CONFIG_SHA256
        assert previous.file_sha(previous.MODEL/'model.safetensors.index.json')==previous.CHECKPOINT_INDEX_SHA256
        cfg=Qwen3VLConfig.from_pretrained(str(previous.MODEL),trust_remote_code=False)
        assert cfg.text_config.intermediate_size==25600
        encoder=candidate.MiniMaxH3Qwen3VLEncoder(str(previous.MODEL),device=torch.device('npu',rank),load_model=False,encoder_group=group)
        encoder.compute_dtype=torch.float16;encoder._r79_fp16=True;encoder._tp_size=size
        encoder.text_model=torch.nn.Module()
        encoder.text_model.layers=torch.nn.ModuleDict({str(i):candidate.MiniMaxH3Qwen3VLTextDecoderLayer(group,cfg.text_config,torch.float16,prefix='text_encoder.text_model.layers.%d'%i)for i in range(25,50)})
        encoder.text_model.rotary_emb=candidate.MiniMaxH3Qwen3VLTextRotaryEmbedding(cfg.text_config)
        idx=json.loads((previous.MODEL/'model.safetensors.index.json').read_text())['weight_map']
        names=[]
        for name in idx:
            if not name.startswith('model.language_model.layers.'):continue
            if 25<=int(name.split('.')[3])<50:names.append(name)
        def tensors():
            for shard in sorted({idx[name]for name in names}):
                with safe_open(str(previous.MODEL/shard),framework='pt',device='cpu')as handle:
                    for name in sorted(names):
                        if idx[name]==shard:yield name,handle.get_tensor(name)
        with torch.no_grad():loaded=encoder.load_weights(tensors())
        assert loaded==set(dict(encoder.named_parameters()))
        encoder.to(device=torch.device('npu',rank));encoder.eval()
        assert all(p.dtype==torch.float16 for p in encoder.parameters())
        r['loaded_parameter_tensors']=len(loaded)
        from runtime5gm83 import Runtime5GM
        from row_fp32_adapter_fixed84 import install as install_row
        from attention_engine85 import AccurateAttention85
        row=install_row(candidate,Runtime5GM(rank),ROOT/'row_kernels',expected_layer_count=50)
        row.prime(encoder.text_model);row.set_enabled(False)
        engine=AccurateAttention85(candidate,Runtime5GM(rank),ROOT/'attention_kernels');engine.prepare();engine.set_enabled(False)
        input_file=previous.CPU_ROOT/'FP16_layer24.pt';gold_file=previous.CPU_ROOT/'FP16_layer49.pt'
        assert previous.file_sha(input_file)==previous.EXPECTED_CPU_LAYERS[24]
        assert previous.file_sha(gold_file)==previous.EXPECTED_CPU_LAYERS[49]
        cpu_input=torch.load(input_file,map_location='cpu',weights_only=True)
        gold=torch.load(gold_file,map_location='cpu',weights_only=True)
        payload=torch.load(previous.CPU_ROOT/'FP16_multimodal_inputs.pt',map_location='cpu',weights_only=True)
        dev=torch.device('npu',rank);inp=cpu_input.to(dev);positions=payload['position_ids'].to(dev)
        assert inp.shape==(1,2043,5120)and inp.dtype==torch.float16
        r['CPU_layer24_transfer']=previous.compare(inp,cpu_input);assert r['CPU_layer24_transfer']['raw_bits_equal']
        before=previous.parameters_sha(encoder);input_sha=previous.tensor_info(inp)['sha256']
        cpu_hashes={item['layer']:item['output']['sha256']for item in reference['language_layers']}
        capture={}
        def capture_mlp(module,args):capture['mlp_input']=args[0].detach().clone()
        def suffix(record):
            state=inp.clone();pos=encoder.text_model.rotary_emb(state,positions);layers=[]
            for i in range(25,50):
                state=encoder.text_model.layers[str(i)](state,position_embeddings=pos)
                if record:
                    info=previous.tensor_info(state)
                    layers.append({'layer':i,'npu':info,'CPU_reference_sha256':cpu_hashes[i]})
                    assert info['finite']
            return state.cpu(),layers
        native=None
        for name,attn,rows in(('native',False,False),('accurate_attention_native_rows',True,False),('accurate_attention_FP32_rows',True,True)):
            engine.set_enabled(attn);row.set_enabled(rows)
            handle=encoder.text_model.layers['25'].mlp.register_forward_pre_hook(capture_mlp)if name=='native'else None
            try:
                with torch.inference_mode():out,layers=suffix(True)
            finally:
                if handle is not None:handle.remove()
            with torch.inference_mode():repeated,_=suffix(False)
            repeat=previous.compare(repeated,out);assert repeat['raw_bits_equal']
            r['cases'][name]={'raw25to49':layers,'final49_vs_CPU':previous.compare(out,gold),
                              'repeat_bits':repeat,'engine':engine.snapshot(),'row_adapter':row.snapshot(),
                              'no_DeepStack_or_final_norm_added':True}
            if name=='native':native=out
        engine.set_enabled(False);row.set_enabled(False)
        with torch.inference_mode():restored,_=suffix(False)
        r['native_recovered']=previous.compare(restored,native);assert r['native_recovered']['raw_bits_equal']
        from late_leaf86 import run_controls
        with torch.inference_mode():r['block25_leaf_controls']=run_controls(encoder.text_model.layers['25'],inp,capture['mlp_input'],previous.compare,previous.tensor_info)
        r['original_parameters_unchanged']=before==previous.parameters_sha(encoder);assert r['original_parameters_unchanged']
        r['original_CPU_layer24_input_unchanged']=input_sha==previous.tensor_info(inp)['sha256'];assert r['original_CPU_layer24_input_unchanged']
        r['CPU_reference_receipt_unchanged']=previous.file_sha(previous.CPU_ROOT/'reference.json')=='44acac61f9b4d9bc8afc91632392502ccaa29dee7bfab951e2b15902a0be74c9'
        assert r['CPU_reference_receipt_unchanged'];r['completed']=True
    except BaseException as error:
        r['failure']={'type':type(error).__name__,'message':str(error)};raise
    finally:
        if engine is not None:r['engine_final']=engine.snapshot();engine.uninstall()
        if row is not None:r['row_final']=row.snapshot();row.uninstall()
        if kv is not None:r['kv_final']=kv.snapshot();kv.uninstall()
        if ready:
            from vllm_omni.diffusion.distributed.parallel_state import destroy_model_parallel
            destroy_model_parallel()
            if torch.distributed.is_initialized():torch.distributed.destroy_process_group()
        writer.write(r)


if __name__=='__main__':main()
