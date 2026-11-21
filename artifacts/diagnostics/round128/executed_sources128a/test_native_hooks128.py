#!/usr/bin/env python3
"""50 native hook rehearsal with small synthetic tensors, not H3 speed/precision acceptance."""
from __future__ import annotations
import hashlib
import json
from pathlib import Path
import threading

ROOT=Path('/tmp/h3_overlap12801')


def main():
    import torch
    import torch_npu
    from vllm_omni.platforms import current_omni_platform as platform
    from vllm_omni.diffusion.offloader.layerwise_backend import _install_layerwise_hook_group
    from vllm_omni.diffusion.offloader.tensor_utils import clear_block_storage
    from native_copy_adapter128 import NativeCopyAdapter128
    path=ROOT/'native_hook_unit128b.json';assert not path.exists()
    torch.npu.set_device(0);torch.npu.matmul.allow_hf32=False;torch.npu.conv.allow_hf32=False
    torch.set_num_threads(4);factory=platform.Stream
    class Block(torch.nn.Module):
        def __init__(self,index):
            super().__init__();self.index=index;self.fail=False
            self.weight=torch.nn.Parameter((torch.arange(256,dtype=torch.float16).reshape(16,16)/256).t(),requires_grad=False)
            self.bias=torch.nn.Parameter(torch.full((16,),index/1000,dtype=torch.float32),requires_grad=False)
            self.register_buffer('scale',torch.tensor(0.01,dtype=torch.float32))
        def forward(self,x):
            if self.fail:raise RuntimeError('Injected block failure for rollback rehearsal')
            return x.float().mm(self.weight.float())*self.scale+self.bias
    blocks=[Block(i) for i in range(50)]
    hooks=_install_layerwise_hook_group(blocks,torch.device('npu:0'),platform.current_stream(),True)
    hooks[0].prefetch_layer(non_blocking=False);torch.npu.synchronize()
    def hashes():
        return [{str(dtype):hashlib.sha256(memoryview(master.view(torch.uint8).numpy())).hexdigest()
                 for dtype,master in h.dtype_cpu_flattened_weights.items()} for h in hooks]
    before=hashes();old_streams=[h.copy_stream for h in hooks]
    x=torch.arange(128,dtype=torch.float32,device='npu:0').reshape(8,16)/128
    def run(capture=True):
        value=x;leaves=[]
        for block in blocks:
            value=block(value)
            if capture:leaves.append(value.detach().cpu().contiguous())
        torch.npu.synchronize()
        return value,leaves
    baseline,reference=run();baseline=baseline.cpu().contiguous();records=[]
    def check(leaves):
        assert len(leaves)==50
        assert all(torch.equal(a.view(torch.uint8),b.view(torch.uint8)) for a,b in zip(leaves,reference))
    for asynchronous in (False,True):
        adapter=NativeCopyAdapter128(torch,blocks,hooks,asynchronous=asynchronous).activate()
        adapter.audit_each_copy=True
        _,leaves=run();check(leaves)
        assert len(adapter.byte_proofs)==100 and {r['target_block'] for r in adapter.byte_proofs}==set(range(50))
        adapter.audit_each_copy=False
        adapter.begin_profile();run(capture=False);profile=adapter.end_profile()
        if asynchronous:
            clear_block_storage(dict(blocks[0].named_parameters()),dict(blocks[0].named_buffers()),None)
            _,leaves=run();check(leaves);assert adapter.recovered_missing==1
            # An incomplete forward must not silently reset to a supported execution epoch.
            blocks[20].fail=True
            try:run()
            except RuntimeError as exc:assert 'Injected block failure' in str(exc)
            else:raise AssertionError('Failure injection did not execute')
            assert adapter.active_owner==20
            blocks[20].fail=False
        facts=adapter.facts()
        adapter.rollback();assert adapter.state=='closed' and adapter.capability.closed
        assert all(h.copy_stream==old for h,old in zip(hooks,old_streams))
        _,restored=run();check(restored)
        assert hashes()==before
        records.append({'mode':'async' if asynchronous else 'serial','all50_consumer_outputs_raw_words_equal':True,
            'all50_native_flat_masters_byteproof':adapter.byte_proofs,'profile':profile,'facts':facts,
            'native_restored_forward_bit_identical':True,'all_cpu_masters_SHA_unchanged':True,
            'missing_initial_block_and_midforward_failure_recovered':asynchronous})
    assert platform.Stream is factory
    result={'scope':__doc__,'completed':True,'native_thread_id':threading.get_native_id(),'original_CPU_master_layouts_unchanged':True,
        'only_synthetic_small_block_data_used':True,'model_weights_loaded_or_transferred':False,
        'not_original_155check_or_real_H3_generation_gate':True,'global_factory_unchanged':True,
        'source_sha256':{p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in ROOT.glob('*.py')},'records':records}
    path.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2),flush=True)


if __name__=='__main__':main()
