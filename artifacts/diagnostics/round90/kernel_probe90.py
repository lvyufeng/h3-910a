#!/usr/bin/env python3
"""Build and execute new Float-vector GELU on actualHalf and signed/extreme controls."""
from pathlib import Path
import json
ROOT=Path('/tmp/h3_round90_vision01')


def main():
    import torch
    import torch_npu
    import full_row_fp32_fixed84 as previous
    import erf_gelu90 as kernel
    from runtime5gm83 import Runtime5GM
    from evidence_io90 import EvidenceWriter
    torch.set_num_threads(2);torch.npu.set_device(0);runtime=Runtime5GM(0)
    writer=EvidenceWriter(ROOT/'kernel_probe90.json')
    r={'scope':__doc__,'completed':False,'source_SHA256':previous.file_sha(Path(__file__)),
       'full_encoder_precision_accepted':False,'whole_goal_complete':False,'numerical_tolerance':None,'cases':{}}
    try:
        frozen=Path('/tmp/h3_round89_vision02/CPUEntryAndActualLeaves89.pt')
        assert previous.file_sha(frozen)=='97f80014b35abdae2af45fc0fcabc3577d5260b025e299192439012c81a697ef'
        inputs=torch.load(frozen,map_location='cpu',weights_only=True)['actual_leaf_inputs']
        preflight=json.loads((ROOT/'gelu_preflight90.json').read_text());assert preflight['completed']
        aux=torch.zeros(16,dtype=torch.float16,device='npu:0')
        special=[0.,-0.,2**-24,-2**-24,.0001,-.0001,.1,-.1,1.,-1.,3.,-3.,5.,-5.,8.,-8.,65504.,-65504.]
        smoke=torch.linspace(-8,8,8192,dtype=torch.float16);smoke[:len(special)]=torch.tensor(special,dtype=torch.float16)
        cases=[('signed_extreme_smoke',smoke)]+[(name,inputs[name])for name in('deepstack_merger_list.0.act_fn','merger.act_fn')]
        with torch.inference_mode():
            for name,cpu_input in cases:
                cpu_gold=torch.nn.functional.gelu(cpu_input,approximate='none')
                cpu_repeat=torch.nn.functional.gelu(cpu_input,approximate='none');assert previous.compare(cpu_repeat,cpu_gold)['raw_bits_equal']
                if name in preflight['leafs']:
                    assert previous.tensor_info(cpu_gold)['sha256']==preflight['leafs'][name]['native_none_vs_CPU_Half_erf']['cpu']['sha256']
                meta=kernel.build_kernel(numel=cpu_input.numel(),out_dir=ROOT/'erf_kernels')
                x=cpu_input.to('npu:0').reshape(-1).contiguous();before=previous.tensor_info(x)['sha256']
                out=torch.empty_like(x)
                runtime.launch(meta,[x,aux,aux,aux,out],synchronize=True)
                first=out.reshape(cpu_input.shape).cpu()
                runtime.launch(meta,[x,aux,aux,aux,out],synchronize=True)
                repeat=previous.compare(out.reshape(cpu_input.shape),first);assert repeat['raw_bits_equal']
                compare=previous.compare(first,cpu_gold);assert compare['finite']
                assert before==previous.tensor_info(x)['sha256']
                row={'CPU_Half_erf':compare,'repeat_bits':repeat,'input_SHA_unchanged':True,'metadata':meta}
                if name=='signed_extreme_smoke':
                    row['special_values']=[{'x':float(cpu_input[i]),'CPU_Half':float(cpu_gold[i]),'kernel':float(first[i]),
                                             'CPU_bits':int(cpu_gold[i].view(torch.int16)),'kernel_bits':int(first[i].view(torch.int16))}for i in range(len(special))]
                r['cases'][name]=row;writer.write(r)
        r['improves_both_real_CPU_erf_leafs']=all(r['cases'][name]['CPU_Half_erf']['relative_rms']<preflight['leafs'][name]['native_none_vs_CPU_Half_erf']['relative_rms']for name in('deepstack_merger_list.0.act_fn','merger.act_fn'))
        assert r['improves_both_real_CPU_erf_leafs'],'kernel doesnot improveoriginalnativeCPUHalferfleaves;full-modelintegrationrefused'
        r['runtime']=runtime.snapshot();r['completed']=True
    except BaseException as error:r['failure']={'type':type(error).__name__,'message':str(error)};raise
    finally:writer.write(r)


if __name__=='__main__':main()
