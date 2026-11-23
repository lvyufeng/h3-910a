"""R163: verify an equivalent-but-cheap implementation of the 155 safety checks.

The shipped check calls ieee_nonfinite_masks, which format-casts and copies the whole
tensor, widens the bit view from int16 to int32 (doubling traffic), and materialises
three boolean masks of full tensor size before summing. For an FP16 tensor the same
verdict can be obtained from reductions alone:

  non-finite count  = isnan(x).sum() + isinf(x).sum()
  saturation count  = (x.abs() == 65504).sum()
                    = presence test, because 65504 is the largest FP16 magnitude:
                      max(x) == 65504  or  min(x) == -65504

This module installs a hook that computes BOTH implementations for every audited call,
compares the integer counts exactly, and records the time of each. The audit verdict
used by the harness stays the native one, so this run is a pure verification: no model
arithmetic, no check is removed, nothing is adopted.
"""
from __future__ import annotations
import statistics

_STATE = {'installed': False, 'rows': []}


def _native_counts(torch, value):
    from npu_half_nan_compat124 import ieee_nonfinite_masks
    nan_mask, posinf, neginf = ieee_nonfinite_masks(value)
    bad = (nan_mask | posinf | neginf).sum(dtype=torch.int32)
    boundary = (value.abs() == 65504).sum(dtype=torch.int32)
    return bad, boundary


def _fast_counts(torch, value):
    bad = torch.isnan(value).sum(dtype=torch.int32) + torch.isinf(value).sum(dtype=torch.int32)
    # 65504 is the maximum finite FP16 magnitude, so saturation exists exactly when the
    # extreme value reaches it. NaN cannot hide this because it is counted above.
    top = value.max()
    bottom = value.min()
    hits = int(bool(top == 65504)) + int(bool(bottom == -65504))
    return bad, torch.tensor(hits, dtype=torch.int32, device=value.device)


def install(torch):
    if _STATE['installed']:
        return True
    from matrix_safety126 import MatrixSafetyAudit126

    def _hook(self, name):
        def hook(module, inputs, result):
            value = result[0] if isinstance(result, tuple) else result
            if not isinstance(value, torch.Tensor) or value.dtype != torch.float16:
                return
            start = torch.npu.Event(enable_timing=True); end = torch.npu.Event(enable_timing=True)
            start.record()
            bad_native, boundary_native = _native_counts(torch, value)
            end.record()
            start2 = torch.npu.Event(enable_timing=True); end2 = torch.npu.Event(enable_timing=True)
            start2.record()
            bad_fast, boundary_fast = _fast_counts(torch, value)
            end2.record()
            _STATE['rows'].append({
                'name': name, 'shape': tuple(value.shape), 'numel': value.numel(),
                'native_events': (start, end), 'fast_events': (start2, end2),
                'counts': (int(bad_native), int(boundary_native), int(bad_fast), int(boundary_fast)),
            })
            # keep the native verdict authoritative for this verification run
            if int(bad_native) or int(boundary_native):
                raise FloatingPointError(f'Half matrix range failure at {name}')
        return hook

    MatrixSafetyAudit126._hook = _hook
    _STATE['installed'] = True
    return True


def collect(torch):
    if not _STATE['installed']:
        return {'installed': False}
    torch.npu.synchronize()
    rows = _STATE['rows']
    out = {'installed': True, 'verification_only': True,
           'native_verdict_still_authoritative': True,
           'no_check_removed': True, 'calls_compared': len(rows)}
    if not rows:
        return out
    mismatches = [r for r in rows if r['counts'][0] != r['counts'][2] or r['counts'][1] != r['counts'][3]]
    native_ms, fast_ms = [], []
    for r in rows:
        try:
            native_ms.append(r['native_events'][0].elapsed_time(r['native_events'][1]))
            fast_ms.append(r['fast_events'][0].elapsed_time(r['fast_events'][1]))
        except BaseException:
            pass
    out['equivalent_on_every_call'] = not mismatches
    out['mismatch_count'] = len(mismatches)
    out['mismatch_examples'] = [{'name': m['name'], 'counts': m['counts']} for m in mismatches[:3]]
    if native_ms:
        out['native'] = {'calls': len(native_ms), 'median_ms': round(statistics.median(native_ms), 3),
                         'sum_ms': round(sum(native_ms), 1)}
    if fast_ms:
        out['fast'] = {'calls': len(fast_ms), 'median_ms': round(statistics.median(fast_ms), 3),
                       'sum_ms': round(sum(fast_ms), 1)}
    if native_ms and fast_ms:
        steps = len(native_ms) / 155.0
        out['per_step'] = {
            'native_s': round(sum(native_ms) / steps / 1000, 3),
            'fast_s': round(sum(fast_ms) / steps / 1000, 3),
            'saving_s': round((sum(native_ms) - sum(fast_ms)) / steps / 1000, 3)}
    # biggest tensors dominate, so record them
    rows_sorted = sorted(rows, key=lambda r: -r['numel'])[:3]
    out['largest'] = [{'name': r['name'], 'shape': list(r['shape'])} for r in rows_sorted]
    _STATE['rows'] = []
    return out
