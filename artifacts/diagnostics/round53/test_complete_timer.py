#!/usr/bin/env python3
"""Run extracted old/new denoise loops with exact synthetic clocks and CPU tensors."""
import ast
from contextlib import nullcontext, redirect_stderr
import io
import json
from pathlib import Path
import re
import types
from typing import Any, Callable
from unittest.mock import patch
import torch

ROOT=Path(__file__).resolve().parent
class Clock:
    def __init__(self): self.t=0.
    def advance(self,x): self.t+=x
    def read(self): return self.t

def run(path,perf):
    clock=Clock(); events=[]
    node=next(x for x in ast.parse(path.read_text()).body if isinstance(x,ast.FunctionDef) and x.name=='minimax_h3_denoise_loop')
    def prepare(**kw):
        return kw['initial_video_rows'].clone(),kw['initial_audio_rows'].clone(),None,None
    def edit(rows,mask,edit,t,ct,sigma): return rows,torch.full((rows.shape[0],),t)
    def cancel(synchronize=False):
        # Actual helper: successful, uncancelled execution does not drain.
        events.append(('cancel',synchronize))
    def measurement_sync():
        clock.advance(5.); events.append(('measurement_sync',))
    def model(**kw):
        clock.advance(2.); events.append(('model',))
        return kw['video_rows']*.25,kw['audio_rows']*.5
    def callback(step,v,a): clock.advance(3.); events.append(('callback',step))
    ns={'torch':torch,'Any':Any,'Callable':Callable,'AbstractContextManager':object,'nullcontext':nullcontext,
        'MiniMaxH3DenoiseBranch':object,'MiniMaxH3LatentEdit':object,
        'MINIMAX_H3_IMGVID_COND_TIMESTEP':0.,'MINIMAX_H3_AUDIO_REF_COND_TIMESTEP':0.,
        'current_omni_platform':types.SimpleNamespace(is_npu=lambda:False,synchronize=measurement_sync),
        'minimax_h3_prepare_denoise_rows':prepare,'minimax_h3_prepare_edit_rows':edit,
        'minimax_h3_publish_denoise_progress':lambda *args:None,'check_request_cancellation':cancel,
        'minimax_h3_rf_v_to_x0':lambda x,v,t:x-v*t,
        'minimax_h3_euler_eta0_step':lambda x,x0,sigma_curr,sigma_next:x*.5+x0*.5}
    exec(compile(ast.fix_missing_locations(ast.Module(body=[node],type_ignores=[])),str(path),'exec'),ns)
    branch=types.SimpleNamespace(update_mask_dev=torch.ones(4,dtype=torch.bool),audio_update_mask_dev=torch.ones(3,dtype=torch.bool),locked_audio_rows=None)
    branch.forward_kwargs=lambda **kw:{'video_rows':kw['video_rows'],'audio_rows':kw['audio_rows']}
    logs=io.StringIO()
    with patch.dict('os.environ',{'VLLM_OMNI_910A_PERF':str(perf)}),patch('time.perf_counter',clock.read),redirect_stderr(logs):
        out=ns['minimax_h3_denoise_loop'](model=model,positive=branch,initial_video_rows=torch.ones(4,8),initial_audio_rows=torch.ones(3,8),keyframe_cond_rows=None,sigmas_video=[1.,.5,0.],sigmas_audio=[1.,.5,0.],device=torch.device('cpu'),on_step=callback)
    return out,events,logs.getvalue()

old,eold,lold=run(ROOT/'source/denoise_loop.py',1)
new,enew,lnew=run(ROOT/'candidate_denoise_loop.py',1)
quiet,equiet,lquiet=run(ROOT/'candidate_denoise_loop.py',0)
assert eold==equiet
assert eold==[e for e in enew if e[0]!='measurement_sync']
assert sum(e[0]=='measurement_sync' for e in enew)==2
assert all(torch.equal(a,b) and torch.equal(a,c) for a,b,c in zip(old,new,quiet))
rows=[list(map(float,x)) for x in re.findall(r'dstep_complete (\d+) total=([0-9.]+)s callback_and_sync=([0-9.]+)s',lnew)]
assert rows==[[1.,10.,8.],[2.,10.,8.]],rows
assert 'observed_total=20.000000s' in lnew and 'after_first_avg=10.000000s' in lnew
assert 'dstep_complete' not in lold and 'dstep_complete' not in lquiet
report={'passed':True,'exact_output_old_new_perf_off':True,
        'same_model_and_callback_sequence':True,'perf_disabled_original_cancellation_behavior':True,
        'perf_mode_explicit_measurement_sync':True,
        'synthetic_model_s':2,'synthetic_callback_s':3,'synthetic_measurement_sync_s':5,
        'new_complete_steps':rows,'new_logs':lnew,'old_logs':lold,
        'scope':'Extracted actual loop with synthetic math/clock and actual successful-cancellation semantics: explicit PERF-only sync includes queued work; PERF-off outputs/execution unchanged, not a real model quality test'}
(ROOT/'timer_test.json').write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps(report,indent=2))
