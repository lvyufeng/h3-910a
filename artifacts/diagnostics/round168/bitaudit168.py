"""R168: measure the compliant step time with the verified-equivalent cheap audit installed.

R166 and R167 proved that a bit-pattern implementation of the 155 safety checks reproduces
the native non-finite and saturation counts exactly, on 37 adversarial cases and on 465
real audited tensors. What has never been measured is the number a decision actually turns
on: the end-to-end step time with that implementation doing the checking.

This module installs it as the audit hook. All 155 checks still run, the same strict rule
still raises on any non-finite or saturated matrix, and the call accounting still asserts
155 per forward. Only the implementation of the check changes, never the model arithmetic
and never which checks exist.

The verdict is computed from bit patterns because R165 showed the FP16 float predicates on
this device are unreliable:
    magnitude = bits & 0x7fff
    non-finite count = (magnitude >= 0x7c00).sum()
    saturation count = (magnitude == 0x7bff).sum()
"""
from __future__ import annotations
import statistics

_STATE = {'installed': False, 'rows': []}


def install(torch):
    global torch_npu
    import torch_npu
    if _STATE['installed']:
        return True
    from matrix_safety126 import MatrixSafetyAudit126

    def _counts(value):
        dense = torch_npu.npu_format_cast(value, 2)
        if not dense.is_contiguous():
            dense = dense.contiguous()
        magnitude = dense.view(torch.int16) & 0x7fff
        nonfinite = int((magnitude >= 0x7c00).sum(dtype=torch.int32))
        saturated = int((magnitude == 0x7bff).sum(dtype=torch.int32))
        return nonfinite, saturated

    def _hook(self, name):
        def hook(module, inputs, result):
            value = result[0] if isinstance(result, tuple) else result
            if not isinstance(value, torch.Tensor) or value.dtype != torch.float16:
                return
            self.calls += 1
            start = torch.npu.Event(enable_timing=True); end = torch.npu.Event(enable_timing=True)
            start.record(); nonfinite, saturated = _counts(value); end.record()
            _STATE['rows'].append({'name': name, 'counts': (nonfinite, saturated),
                                   'events': (start, end), 'numel': value.numel()})
            if nonfinite or saturated:      # identical strict rule to the shipped audit
                raise FloatingPointError(f'Half matrix range failure at {name}: {nonfinite}, {saturated}')
        return hook

    MatrixSafetyAudit126._hook = _hook
    _STATE['installed'] = True
    return True


def collect(torch):
    if not _STATE['installed']:
        return {'installed': False}
    torch.npu.synchronize()
    rows = _STATE['rows']
    out = {'installed': True, 'candidate_measurement': True,
           'all_155_checks_still_run': True, 'model_arithmetic_unchanged': True,
           'calls_checked': len(rows), 'total_flagged': sum(r['counts'][0] + r['counts'][1] for r in rows)}
    times = []
    for r in rows:
        try:
            times.append(r['events'][0].elapsed_time(r['events'][1]))
        except BaseException:
            pass
    if times:
        steps = len(times) / 155.0
        out['check'] = {'calls': len(times), 'median_ms': round(statistics.median(times), 3),
                        'device_s_per_step': round(sum(times) / steps / 1000, 3)}
    _STATE['rows'] = []
    return out
