#!/usr/bin/env python3
"""Real frozenHalf GELU:none/tanh/explicitFloat erf,without changing any model."""
from pathlib import Path
import json
import hashlib
ROOT=Path('/tmp/h3_round90_vision01')
FROZEN=Path('/tmp/h3_round89_vision02/CPUEntryAndActualLeaves89.pt')
FROZEN_SHA256='97f80014b35abdae2af45fc0fcabc3577d5260b025e299192439012c81a697ef'
LEAVES=('blocks.0.mlp.act_fn','deepstack_merger_list.0.act_fn','merger.act_fn')


def main():
    import torch
    import torch_npu
    import full_row_fp32_fixed84 as previous
    from evidence_io90 import EvidenceWriter
    torch.set_num_threads(2);torch.npu.set_device(0)
    writer=EvidenceWriter(ROOT/'gelu_preflight90.json')
    r={'scope':__doc__,'completed':False,'source_SHA256':previous.file_sha(Path(__file__)),
       'no_model_or_parameter_changed':True,'no_Float_weight_teacher':True,
       'full_encoder_precision_accepted':False,'whole_goal_complete':False,'numerical_tolerance':None,'leafs':{}}
    try:
        assert previous.file_sha(FROZEN)==FROZEN_SHA256
        assert previous.file_sha(previous.CPU_ROOT/'reference.json')=='44acac61f9b4d9bc8afc91632392502ccaa29dee7bfab951e2b15902a0be74c9'
        payload=torch.load(FROZEN,map_location='cpu',weights_only=True)
        old=json.loads(Path('/tmp/h3_round89_vision02/vision89.json').read_text())
        with torch.inference_mode():
            for name in LEAVES:
                x=payload['actual_leaf_inputs'][name];assert x.dtype==torch.float16
                before=previous.tensor_info(x)['sha256']
                cpu_none=torch.nn.functional.gelu(x,approximate='none')
                cpu_none_repeat=torch.nn.functional.gelu(x,approximate='none')
                assert previous.compare(cpu_none_repeat,cpu_none)['raw_bits_equal']
                original_mode='tanh'if name.startswith('blocks.')else'none'
                original_cpu=torch.nn.functional.gelu(x,approximate=original_mode)
                expected=old['actual_nonlinear_leaf']['controls'][name]['native_vs_cpu_half']['cpu']['sha256']
                assert previous.tensor_info(original_cpu)['sha256']==expected
                cpu_tanh=torch.nn.functional.gelu(x,approximate='tanh')
                xf=x.float();cpu_erf=(.5*xf*(1+torch.erf(xf*(2**-.5)))).to(torch.float16)
                xn=x.to('npu:0');npu_input_sha=previous.tensor_info(xn)['sha256']
                nn=torch.nn.functional.gelu(xn,approximate='none')
                nt=torch.nn.functional.gelu(xn,approximate='tanh')
                fn=torch.nn.functional.gelu(xn.float(),approximate='none').to(torch.float16)
                nr=torch.nn.functional.gelu(xn,approximate='none');assert previous.compare(nr,nn)['raw_bits_equal']
                row={'input':previous.tensor_info(x),'original_CPU_Half_output_matches_R89_SHA':True,
                     'CPU_Half_none_repeat_bits':True,'NPU_Half_none_repeat_bits':True,
                     'native_none_vs_CPU_Half_erf':previous.compare(nn,cpu_none),
                     'native_tanh_vs_CPU_Half_erf':previous.compare(nt,cpu_none),
                     'native_none_vs_CPU_Half_tanh':previous.compare(nn,cpu_tanh),
                     'native_tanh_vs_CPU_Half_tanh':previous.compare(nt,cpu_tanh),
                     'native_none_vs_native_tanh':previous.compare(nn,nt),
                     'CPU_tanh_vs_CPU_Half_erf':previous.compare(cpu_tanh,cpu_none),
                     'CPU_Float_explicit_erf_then_Half_vs_CPU_Half_erf':previous.compare(cpu_erf,cpu_none),
                     'NPU_Float_none_then_Half_vs_CPU_Half_erf':previous.compare(fn,cpu_none),
                     'NPU_explicit_erf_supported':False}
                try:
                    yf=.5*xn.float()*(1+torch.erf(xn.float()*(2**-.5)))
                    ef=yf.to(torch.float16)
                    er=(.5*xn.float()*(1+torch.erf(xn.float()*(2**-.5)))).to(torch.float16)
                    assert previous.compare(er,ef)['raw_bits_equal']
                    row['NPU_explicit_erf_supported']=True
                    row['NPU_Float_explicit_erf_then_Half_vs_CPU_Half_erf']=previous.compare(ef,cpu_none)
                    row['NPU_explicit_erf_repeat_bits']=True
                except Exception as error:row['NPU_explicit_erf_error']={'type':type(error).__name__,'message':str(error)}
                row['original_CPU_and_NPU_inputs_SHA_unchanged']=before==previous.tensor_info(x)['sha256']and npu_input_sha==previous.tensor_info(xn)['sha256']
                assert row['original_CPU_and_NPU_inputs_SHA_unchanged'];r['leafs'][name]=row;writer.write(r)
        r['frozen_input_SHA256']=FROZEN_SHA256;r['completed']=True
    except BaseException as error:r['failure']={'type':type(error).__name__,'message':str(error)};raise
    finally:writer.write(r)


if __name__=='__main__':main()
