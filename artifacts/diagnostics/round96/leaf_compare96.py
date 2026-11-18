#!/usr/bin/env python3
"""Actual frozen originalvision QKV A/B old vs fused PV,independent scoped helpers."""
from pathlib import Path
import json
import os
import time
import statistics
from types import SimpleNamespace
ROOT=Path(os.environ.get('R96_ROOT','/tmp/h3_round96_fused01'))


def main():
    import torch
    import torch_npu
    import full_row_fp32_fixed84 as previous
    from runtime5gm83 import Runtime5GM
    from accurate_vision_attention88 import AccurateVisionAttention88
    from accurate_vision_attention_pv96 import AccurateVisionAttentionPV96
    torch.set_num_threads(2);torch.npu.set_device(0);previous.ROOT=ROOT
    r={'scope':__doc__,'completed':False,'source_SHA256':previous.file_sha(Path(__file__)),
       'not_full_encoder_timing':True,'no_model_weight_or_FloatLinear_teacher':True,
       'full_encoder_precision_accepted':False,'whole_goal_complete':False,'numerical_tolerance':None,'images':[]}
    old=None;new=None
    try:
        expected=json.loads((ROOT/'expected_inputs96.json').read_text());candidate=previous.load_candidate(r)
        path=Path('/tmp/h3_round88_vision01/FrozenVisionQKV88.pt');assert previous.file_sha(path)==expected['frozenQKVfileSHA256']
        capture=torch.load(path,map_location='cpu',weights_only=True)
        # Never nest the patches: each engine owns a different private helper namespace.
        ns_old=SimpleNamespace(__file__=candidate.__file__,_vision_sdpa_bf16_compatible=candidate._vision_sdpa_bf16_compatible)
        ns_new=SimpleNamespace(__file__=candidate.__file__,_vision_sdpa_bf16_compatible=candidate._vision_sdpa_bf16_compatible)
        old=AccurateVisionAttention88(ns_old,Runtime5GM(0),ROOT/'old_attention_kernels')
        new=AccurateVisionAttentionPV96(ns_new,Runtime5GM(0),ROOT/'fused_attention_kernels')
        r['old_prepare']=old.prepare();r['fused_prepare']=new.prepare();old.set_enabled(True);new.set_enabled(True)
        for image,item in enumerate(capture['images']):
            args=[item[name].to('npu:0')for name in('q','k','v')];before=[previous.tensor_info(x)['sha256']for x in args]
            with torch.inference_mode():
                baseline=old.forward(*args);candidate_out=new.forward(*args)
                old_repeat=old.forward(*args);new_repeat=new.forward(*args)
            assert previous.tensor_info(baseline)['sha256']==expected['R88_precise_leaf_output_SHAs'][image]
            assert previous.compare(old_repeat,baseline)['raw_bits_equal']and previous.compare(new_repeat,candidate_out)['raw_bits_equal']
            case={'image':image,'old_vs_CPU_Half_MATH':previous.compare(baseline,item['CPU_Half_MATH_output']),
                  'fused_vs_CPU_Half_MATH':previous.compare(candidate_out,item['CPU_Half_MATH_output']),
                  'fused_vs_old':previous.compare(candidate_out,baseline),'old_warm_s':[],'fused_warm_s':[]}
            for _ in range(7):
                for name,engine,gold in(('old',old,baseline),('fused',new,candidate_out)):
                    torch.npu.synchronize();start=time.perf_counter()
                    with torch.inference_mode():out=engine.forward(*args)
                    torch.npu.synchronize();case[name+'_warm_s'].append(time.perf_counter()-start)
                    assert previous.compare(out,gold)['raw_bits_equal']
            case['old_median_s']=statistics.median(case['old_warm_s']);case['fused_median_s']=statistics.median(case['fused_warm_s'])
            case['old_over_fused_median_speedup']=case['old_median_s']/case['fused_median_s']
            assert before==[previous.tensor_info(x)['sha256']for x in args]
            case['all_warm_repeats_and_QKV_unchanged']=True;r['images'].append(case)
        r['old_final']=old.snapshot();r['fused_final']=new.snapshot();r['completed']=True
    except BaseException as error:r['failure']={'type':type(error).__name__,'message':str(error)};raise
    finally:
        if new is not None:new.uninstall()
        if old is not None:old.uninstall()
        with(ROOT/'leaf_compare96.json').open('x')as f:f.write(json.dumps(r,indent=2)+'\n')
        print(json.dumps({'completed':r['completed'],'images':r['images'],'failure':r.get('failure')},indent=2),flush=True)


if __name__=='__main__':main()
