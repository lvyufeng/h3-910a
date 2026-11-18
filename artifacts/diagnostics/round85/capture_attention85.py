#!/usr/bin/env python3
"""Capture actual TP8 decoder0 Half QKV and compare the leaf to CPU Half MATH SDPA.

Only the original first decoder block is constructed/loaded; original CPUFP16
multimodal input is reused to isolate attention. This is not a full encode timer.
"""
from pathlib import Path
import os
import json
import hashlib
import time

ROOT=Path('/tmp/h3_round85_capture01')


def main():
    import torch
    import torch_npu
    import full_row_fp32_fixed84 as previous
    from evidence_io85 import EvidenceWriter
    from safetensors import safe_open
    from transformers import Qwen3VLConfig
    from torch.nn.attention import SDPBackend,sdpa_kernel
    rank=int(os.environ['RANK']);size=int(os.environ['WORLD_SIZE'])
    writer=EvidenceWriter(ROOT/('tp8_rank%d.json'%rank))
    r={'scope':__doc__,'rank':rank,'world_size':size,'completed':False,
       'source_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
       'not_full_encoder_timing':True,'full_encoder_precision_accepted':False,
       'whole_goal_complete':False,'numerical_tolerance':None}
    ready=False;kv=None
    try:
        assert size==8
        torch.set_num_threads(2);torch.npu.set_device(rank)
        previous.ROOT=ROOT
        reference=previous.load_cpu_gate(r)
        candidate=previous.load_candidate(r)
        previous.install_fp16_f32_math_attention(candidate,r)
        from kv_padding81 import install
        kv=install(candidate)
        from vllm_omni.diffusion.distributed.group_coordinator import GroupCoordinator
        from vllm_omni.diffusion.distributed.parallel_state import init_distributed_environment,initialize_model_parallel,destroy_model_parallel
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
        encoder.text_model.layers=torch.nn.ModuleList([candidate.MiniMaxH3Qwen3VLTextDecoderLayer(group,cfg.text_config,torch.float16,prefix='text_encoder.text_model.layers.0')])
        encoder.text_model.rotary_emb=candidate.MiniMaxH3Qwen3VLTextRotaryEmbedding(cfg.text_config)
        index=json.loads((previous.MODEL/'model.safetensors.index.json').read_text())['weight_map']
        names=sorted(name for name in index if name.startswith('model.language_model.layers.0.'))
        def tensors():
            for shard in sorted({index[name]for name in names}):
                with safe_open(str(previous.MODEL/shard),framework='pt',device='cpu')as handle:
                    for name in names:
                        if index[name]==shard:yield name,handle.get_tensor(name)
        with torch.no_grad():loaded=encoder.load_weights(tensors())
        assert loaded==set(dict(encoder.named_parameters()))
        encoder.to(device=torch.device('npu',rank));encoder.eval()
        assert all(p.dtype==torch.float16 for p in encoder.parameters())
        r['loaded_parameter_tensors']=len(loaded)
        before=previous.parameters_sha(encoder)
        cpu_input=torch.load(previous.CPU_ROOT/'FP16_multimodal_inputs.pt',map_location='cpu',weights_only=True)
        input_=cpu_input['inputs_embeds'].to(torch.device('npu',rank))
        positions=cpu_input['position_ids'].to(torch.device('npu',rank))
        assert input_.shape==(1,2043,5120)and input_.dtype==torch.float16
        old_sdpa=candidate._scaled_dot_product_attention;calls=[]
        def captured_sdpa(q,k,v):
            assert q.dtype==k.dtype==v.dtype==torch.float16
            out=old_sdpa(q,k,v)
            calls.append({'q':q.detach().cpu().contiguous(),'k':k.detach().cpu().contiguous(),
                          'v':v.detach().cpu().contiguous(),'native_out':out.detach().cpu().contiguous()})
            return out
        candidate._scaled_dot_product_attention=captured_sdpa
        with torch.inference_mode():
            pos=encoder.text_model.rotary_emb(input_,positions)
            first=encoder.text_model.layers[0](input_.clone(),position_embeddings=pos).cpu()
            second=encoder.text_model.layers[0](input_.clone(),position_embeddings=pos).cpu()
        assert len(calls)==2
        r['raw_decoder0_vs_CPU']=previous.compare(first,torch.load(previous.CPU_ROOT/'FP16_layer00.pt',map_location='cpu',weights_only=True))
        r['raw_decoder0_repeat']=previous.compare(second,first)
        assert r['raw_decoder0_repeat']['raw_bits_equal']
        r['raw_decoder0_reproduces_R84_native_CPU_input']=previous.tensor_info(first)['sha256']=='29a39654df6143e0280d53fa0088e8454fdd6bf46a3b9ff5f4f45ebe0f0ad5ef'
        assert r['raw_decoder0_reproduces_R84_native_CPU_input']
        r['operand_repeat']={name:previous.compare(calls[1][name],calls[0][name])for name in('q','k','v','native_out')}
        assert all(item['raw_bits_equal']for item in r['operand_repeat'].values())
        capture=calls[0];q=capture['q'];k=capture['k'];v=capture['v']
        groups=q.shape[1]//k.shape[1];ke=k.repeat_interleave(groups,1);ve=v.repeat_interleave(groups,1)
        assert q.shape==(1,8,2043,128)and k.shape==(1,1,2043,128)
        with torch.inference_mode(),sdpa_kernel(SDPBackend.MATH):
            cpu_gold=torch.nn.functional.scaled_dot_product_attention(q,ke,ve,is_causal=True,dropout_p=0.0)
            cpu_repeat=torch.nn.functional.scaled_dot_product_attention(q,ke,ve,is_causal=True,dropout_p=0.0)
        assert torch.equal(cpu_gold.view(torch.uint8),cpu_repeat.view(torch.uint8))
        r['attention_native_vs_CPU_Half_MATH']=previous.compare(capture['native_out'],cpu_gold)
        r['CPU_Half_MATH_repeat']=previous.compare(cpu_repeat,cpu_gold)
        capture['cpu_half_math_out']=cpu_gold
        capture['CPU_multimodal_source_sha256']=previous.file_sha(previous.CPU_ROOT/'FP16_multimodal_inputs.pt')
        capture['source_parameters_sha256']=before
        target=ROOT/('QKV_rank%d.pt'%rank);assert not target.exists();torch.save(capture,target)
        r['capture']={'path':str(target),'sha256':previous.file_sha(target),
                      'q':previous.tensor_info(q),'k':previous.tensor_info(k),'v':previous.tensor_info(v)}
        # Isolate score/softmax/PV for one real 512-query window. Float leaf
        # intermediates are diagnostic only; actual teacher is Half MATH SDPA above.
        begin=0;end=512;scale=q.shape[-1]**-0.5
        scores_cpu=torch.matmul(q.float()[...,begin:end,:],ke.float().transpose(-2,-1))*scale
        mask=torch.arange(2043)[None,:]>torch.arange(begin,end)[:,None]
        prob_cpu=torch.softmax(scores_cpu.masked_fill(mask,float('-inf')),-1)
        pv_cpu=torch.matmul(prob_cpu,ve.float())
        dev=torch.device('npu',rank);qn=q.to(dev);kn=ke.to(dev);vn=ve.to(dev)
        scores_npu=torch.matmul(qn.float()[...,begin:end,:],kn.float().transpose(-2,-1))*scale
        prob_npu=torch.softmax(scores_npu.masked_fill(mask.to(dev),float('-inf')),-1)
        pv_native_probs=torch.matmul(prob_npu,vn.float())
        pv_exact_probs=torch.matmul(prob_cpu.to(dev),vn.float())
        r['first_512_query_leaf_diagnostics']={
            'scores_unmasked':previous.compare(scores_npu,scores_cpu),
            'probabilities':previous.compare(prob_npu,prob_cpu),
            'PV_native_probabilities':previous.compare(pv_native_probs,pv_cpu),
            'PV_exact_CPU_probabilities':previous.compare(pv_exact_probs,pv_cpu),
            'scope':'Float intermediate diagnostics on actual frozen HalfQKV;not a new whole-FP32teacher',
        }
        r['parameters_unchanged']=before==previous.parameters_sha(encoder);assert r['parameters_unchanged']
        r['CPU_reference_receipt_unchanged']=previous.file_sha(previous.CPU_ROOT/'reference.json')=='44acac61f9b4d9bc8afc91632392502ccaa29dee7bfab951e2b15902a0be74c9'
        assert r['CPU_reference_receipt_unchanged']
        r['completed']=True
    except BaseException as error:
        r['failure']={'type':type(error).__name__,'message':str(error)};raise
    finally:
        if kv is not None:r['kv_padding_final']=kv.snapshot();kv.uninstall()
        if ready:
            from vllm_omni.diffusion.distributed.parallel_state import destroy_model_parallel
            destroy_model_parallel()
            if torch.distributed.is_initialized():torch.distributed.destroy_process_group()
        writer.write(r)


if __name__=='__main__':main()
