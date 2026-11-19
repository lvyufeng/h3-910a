#!/usr/bin/env python3
"""Two-head TP attention: all eight head shards vs original CPU Half MATH and BC128 bits."""
from pathlib import Path
import copy
import json
import statistics
import time
from types import SimpleNamespace
ROOT=Path('/tmp/h3_round107_head2_01')


def main():
    import torch
    import torch_npu
    import full_row_fp32_fixed84 as previous
    from runtime5gm83 import Runtime5GM
    from flash_engine101 import FlashVisionAttention101
    from head2_attention107 import Head2VisionAttention107
    torch.set_num_threads(2);torch.npu.set_device(0);previous.ROOT=ROOT
    r={'scope':__doc__,'completed':False,'source_SHA256':previous.file_sha(Path(__file__)),
       'no_model_weight_or_FloatLinear_teacher':True,'not_full_encoder_timing':True,
       'full_encoder_precision_accepted':False,'whole_goal_complete':False,'numerical_tolerance':None,'cases':[]}
    old=None;new=None
    try:
        candidate=previous.load_candidate(r);original_helper=candidate._vision_sdpa_bf16_compatible
        path=Path('/tmp/h3_round88_vision01/FrozenVisionQKV88.pt')
        assert previous.file_sha(path)=='3fb5d524373c96de48a4be583cee87f171f5e20fdaa8ac57fd4a31cb709ce552'
        capture=torch.load(path,map_location='cpu',weights_only=True)
        ns=SimpleNamespace(__file__=candidate.__file__,_vision_sdpa_bf16_compatible=original_helper)
        old=FlashVisionAttention101(ns,Runtime5GM(0),ROOT/'old_BC128_kernels');new=Head2VisionAttention107(candidate,Runtime5GM(0),ROOT/'head2_kernels')
        r['old_prepare']=copy.deepcopy(old.prepare());r['head2_prepare']=new.prepare();old.set_enabled(True);new.set_enabled(True)
        assert candidate._vision_sdpa_bf16_compatible is original_helper
        for image,item in enumerate(capture['images']):
            with torch.inference_mode():full=old.forward(*(item[name].to('npu:0')for name in('q','k','v')))
            assert full.shape==(1,16,4032,72)
            for rank in range(8):
                start=rank*2;args=[item[name][:,start:start+2].to('npu:0')for name in('q','k','v')]
                before=[previous.tensor_info(x)['sha256']for x in args]
                with torch.inference_mode():first=new.forward(*args);repeat=new.forward(*args)
                gold=item['CPU_Half_MATH_output'][:,start:start+2].contiguous();old_slice=full[:,start:start+2].contiguous()
                assert first.shape==gold.shape==old_slice.shape==(1,2,4032,72)
                same=previous.compare(first,old_slice);assert same['raw_bits_equal']
                assert previous.compare(repeat,first)['raw_bits_equal']
                times=[]
                for _ in range(5):
                    torch.npu.synchronize();t=time.perf_counter()
                    with torch.inference_mode():out=new.forward(*args)
                    torch.npu.synchronize();times.append(time.perf_counter()-t)
                    assert previous.compare(out,first)['raw_bits_equal']
                assert before==[previous.tensor_info(x)['sha256']for x in args]
                r['cases'].append({'image':image,'TP_rank':rank,'heads':[start,start+1],
                                   'vs_original_CPU_Half_MATH':previous.compare(first,gold),'vs_original_BC128_head_slice':same,
                                   'all_repeats_and_inputs_unchanged':True,'clock_s':times,'median_s':statistics.median(times)})
        assert candidate._vision_sdpa_bf16_compatible is original_helper
        r['global_candidate_helper_unchanged']=True;r['head2_final']=new.snapshot();r['old_final']=old.snapshot();r['completed']=True
    except BaseException as error:r['failure']={'type':type(error).__name__,'message':str(error)};raise
    finally:
        if new is not None:new.uninstall()
        if old is not None:old.uninstall()
        with(ROOT/'head2_probe107.json').open('x')as f:f.write(json.dumps(r,indent=2)+'\n')
        print(json.dumps({'completed':r['completed'],'cases':r['cases'],'failure':r.get('failure')},indent=2),flush=True)


if __name__=='__main__':main()
