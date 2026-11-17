#!/usr/bin/env python3
"""Genuine eight-rank raw BF16/INT32 gather on the CURRENT EXPERIMENTAL encoder group.

This uses the same vllm_omni GroupCoordinator constructor as the resident
experimental encoder,not a claim about vLLM's different device communicator.
GroupCoordinator.all_gather has NO FP16 cast in this source;its all_reduce does.
BF16 direct transmission and INT32 pair transport are separate run roots. Each
payload covers raw signedzero/subnormal/extreme/Inf/NaN words;proofs use integer
channels,never float()/BF16 CPU conversion that can normalize NaN payloads.
Only fixed equal-size collectives are exercised. No whole-encoder performance.
"""
import hashlib
import inspect
import json
import os
from pathlib import Path
import time
import torch
import torch_npu
from vllm_omni.diffusion.distributed.group_coordinator import GroupCoordinator
from vllm_omni.diffusion.distributed.parallel_state import init_distributed_environment
from raw_wire75 import pack,unpack
from evidence_io75 import EvidenceWriter

MODE=os.environ.get('H3_WIRE_MODE','int32')
assert MODE in('int32','bf16')
ROOT=Path('/tmp/h3_round75_wire_'+MODE+'01')
rank=int(os.environ['LOCAL_RANK']);world=int(os.environ['WORLD_SIZE']);assert world==8
writer=EvidenceWriter(ROOT/f'wire_rank{rank}.json');torch.set_num_threads(4)
torch.npu.set_device(rank)
r={'rank':rank,'mode':MODE,'world_size':world,'device_index':torch.npu.current_device(),'source_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
   'scope':__doc__,'cases':[],'production_communicator_proven':False,'whole_encoder_acceleration_proven':False,'official_goal_complete':False}


def seed(owner,n,c):
    assert c%2==0
    words=((torch.arange(n*c,dtype=torch.int64)*257+31+owner*7919)&65535).to(torch.int32)
    special=[0x1230+owner,0x0000,0x8000,0x0001,0x8001,0x007f,0x0080,0x7f7f,0xff7f,0x7f80,0xff80,0x7fc0,0x7fc1,0x7f95,0xffc0,0xffed]
    words[:len(special)]=torch.tensor(special,dtype=torch.int32)
    return words.to(torch.uint16).view(torch.int32).reshape(n,c//2)


def raw(value):return hashlib.sha256(memoryview(value.detach().cpu().contiguous().view(torch.uint8).numpy())).hexdigest()


def proof(a,b):
    assert a.shape==b.shape and a.dtype==b.dtype==torch.int32
    aa,bb=a.detach().cpu().contiguous(),b.detach().cpu().contiguous()
    return {'shape_int32':list(aa.shape),'int32_elements':aa.numel(),'bf16_words':aa.numel()*2,
            'different_int32_pairs':int((aa!=bb).sum()),'different_bf16_words':int((aa.view(torch.int16)!=bb.view(torch.int16)).sum()),
            'raw_pairs_equal':bool(torch.equal(aa,bb)),'observed_sha256':raw(aa),'expected_sha256':raw(bb)}


try:
    init_distributed_environment(world_size=world,rank=rank,local_rank=rank,backend='hccl')
    group=GroupCoordinator([list(range(world))],rank,'hccl')
    r['actual_group_class']=type(group).__module__+'.'+type(group).__name__
    r['actual_group_source_path']=inspect.getsourcefile(type(group));r['actual_group_source_sha256']=hashlib.sha256(Path(r['actual_group_source_path']).read_bytes()).hexdigest()
    r['device_group_backend']=torch.distributed.get_backend(group.device_group);assert r['device_group_backend']=='hccl'and group.world_size==8
    r['integer_transport_only']=MODE=='int32'
    with torch.inference_mode():
        for n,c in((128,544),(2016,432),(8064,432),(8064,144),(8064,538),(8064,544)):
            expected_local=seed(rank,n,c);base=expected_local.npu();value=unpack(base)
            before=raw(base);loop=pack(value)
            view_proof=proof(loop,expected_local);assert view_proof['raw_pairs_equal']
            row={'rows':n,'bf16_columns':c,'source_physical_format':int(torch_npu.get_npu_format(value)),
                 'source_stride':list(value.stride()),'source_storage_offset':value.storage_offset(),'source_ptr_mod32':value.data_ptr()%32,
                 'raw_device_view_roundtrip':view_proof,'directions':{}}
            prefix=expected_local.view(torch.int16).reshape(-1)[:65568].to(torch.int32)&65535
            row['distinct_bf16_words_in_first65568_source_words']=int(prefix.unique().numel())
            assert row['distinct_bf16_words_in_first65568_source_words']==65536
            for dim in(0,-1):
                expected=torch.cat([seed(owner,n,c)for owner in range(world)],dim=0 if dim==0 else 1)
                def gather():
                    source=pack(value)if MODE=='int32'else value
                    out=group.all_gather(source,dim=dim)
                    return pack(unpack(out)) if MODE=='int32'else pack(out)
                torch.distributed.barrier();torch.npu.synchronize();start=time.perf_counter();first=gather();torch.npu.synchronize()
                excluded=time.perf_counter()-start
                p=proof(first,expected);assert p['raw_pairs_equal']
                times=[];repeats=[]
                for sample in range(3):
                    torch.distributed.barrier();torch.npu.synchronize();start=time.perf_counter();out=gather();torch.npu.synchronize()
                    elapsed=time.perf_counter()-start;times.append(elapsed);repeats.append(proof(out,expected))
                    assert repeats[-1]['raw_pairs_equal']
                row['directions'][str(dim)]={'first_compile_control_s_excluded':excluded,'proof':p,'warm_same_call_s':times,'repeat_proofs':repeats}
            row['source_readonly']=before==raw(base);assert row['source_readonly']
            if n==128:
                negatives={}
                for label,bad in(('odd_columns',value.reshape(-1)[:1].reshape(1,1)),('odd_offset',value.reshape(-1)[1:549].reshape(1,548)),('noncontiguous',value[:,::2])):
                    try:pack(bad)
                    except ValueError as e:negatives[label]={'rejected':True,'reason':str(e)}
                    else:raise AssertionError('Raw-view guard accepted bad '+label)
                row['guard_rejections']=negatives
            r['cases'].append(row);writer.write(r)
            print(json.dumps({'rank':rank,'mode':MODE,'N':n,'C':c,'all_raw_word_proofs_equal':True}),flush=True)
    r['completed']=True;r['all_tested_raw_word_transfers_equal']=True
except BaseException as e:
    r['failure']={'type':type(e).__name__,'message':str(e)}
    r['partial_case']=row if 'row'in globals()else None
    raise
finally:
    writer.write(r)
    if torch.distributed.is_initialized():torch.distributed.destroy_process_group()
