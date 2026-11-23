"""R148 guard fast-path identity check, with a native repeat control.

For every FP32 range-guard linear call this records:
  * the native result, computed twice, so the module's own repeat noise is known;
  * the candidate fast-path result for the scale==1 regime;
and compares bits. The value RETURNED to the model is always the native first
result, so the model's math and the default path are completely unchanged.
"""
from __future__ import annotations
import math
import statistics

_STATE={'installed':False,'calls':[],'limits':None}


def _stats(a,b,torch):
    au=a.detach().cpu().contiguous();bu=b.detach().cpu().contiguous()
    same=bool(torch.equal(au.view(torch.uint8),bu.view(torch.uint8)))
    af=au.float().double();bf=bu.float().double()
    diff=af-bf
    denom=float(bf.square().sum().sqrt().clamp_min(1e-30))
    return {'bit_identical':same,'differing_elements':int((au!=bu).sum()),
            'total_elements':int(au.numel()),
            'relative_RMS':float(diff.square().sum().sqrt())/denom,
            'max_abs_diff':float(diff.abs().max())}


def install(torch,limit=100):
    import h3_replay123.transformer as replay
    cls=replay._MiniMaxH3HalfInputFP32OutputLinearMethod
    original=cls.apply
    _STATE['limits']=limit

    def apply(self,layer,x,bias=None):
        native=original(self,layer,x,bias)
        if len(_STATE['calls'])>=limit:
            return native
        product=x.float()
        peak=float(product.abs().amax())
        exponent=max(0,math.ceil(math.log2(peak/32752.0))) if peak>32752.0 else 0
        record={'peak':peak,'exponent':exponent,'scale':float(2**exponent),
                'in_fast_path_regime':exponent==0,'shape':list(x.shape)}
        control=original(self,layer,x,bias)          # native again: the repeat control
        if exponent==0:
            half_input=product                       # scale==1 so the narrow is the identity
            fast=torch.mm(half_input.reshape(-1,half_input.shape[-1]),
                          layer.weight.t(),out_dtype=torch.float32)
            fast=fast.reshape(*x.shape[:-1],layer.weight.shape[0])
            if bias is not None:fast=fast+bias.float()
            record['native_vs_fast']=_stats(native,fast,torch)
        else:
            record['native_vs_fast']={'skipped':'peak exceeds 32752 so the fast path does not apply'}
        record['native_vs_native_control']=_stats(native,control,torch)
        _STATE['calls'].append(record)
        return native

    cls.apply=apply
    _STATE['installed']=True
    return True


def collect():
    calls=_STATE['calls']
    out={'installed':_STATE['installed'],'calls_recorded':len(calls),
         'measurement_only_no_math_change':True,
         'model_always_returns_the_native_result':True,
         'note':'the candidate result is computed but never returned; the control re-runs the native path'}
    if not calls:
        return out
    peaks=[c['peak'] for c in calls]
    out['peak']={'min':min(peaks),'median':statistics.median(peaks),'max':max(peaks),
                 'all_within_32752':all(c['in_fast_path_regime'] for c in calls),
                 'exceeding_32752':sum(1 for c in calls if not c['in_fast_path_regime'])}
    cmp=[c['native_vs_fast'] for c in calls if 'bit_identical' in c.get('native_vs_fast',{})]
    ctl=[c['native_vs_native_control'] for c in calls]
    def summ(rows):
        if not rows:return None
        return {'count':len(rows),'all_bit_identical':all(r['bit_identical'] for r in rows),
                'max_differing_elements':max(r['differing_elements'] for r in rows),
                'max_relative_RMS':max(r['relative_RMS'] for r in rows),
                'max_abs_diff':max(r['max_abs_diff'] for r in rows)}
    out['native_vs_fast']=summ(cmp)
    out['native_vs_native_control']=summ(ctl)
    if cmp and ctl:
        worst_fast=max(r['differing_elements'] for r in cmp)
        worst_ctl=max(r['differing_elements'] for r in ctl)
        out['verdict']={'fast_path_no_worse_than_native_repeat':worst_fast<=worst_ctl,
                        'worst_fast_differing':worst_fast,'worst_control_differing':worst_ctl}
    out['sample']=calls[:3]
    return out
