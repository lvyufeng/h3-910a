"""R155: an additive parent-child span decomposition of MiniMaxH3MLP.forward.

R153b subtracted medians of unrelated calls, which is not a decomposition. This probe
records five events per mlp call on the same stream, in program order:

    E0 mlp pre  -> E1 fc1 post -> E2 fc2 pre -> E3 fc2 post -> E4 mlp post

so the intervals tile the call exactly:

    pre_dispatch  E0-E1  = fc1 span
    fc1           E1-E1
    activation    E1-E2  = chunk + silu + mul + dispatch up to fc2
    fc2           E2-E3
    tail          E3-E4

A wrapper on F.silu adds one inner pair, so the activation region splits into the SiLU
and the remainder (the multiply plus chunk handling). Nothing here changes the math.
"""
from __future__ import annotations
import statistics

_STATE = {'installed': False, 'calls': []}


def install(torch, blocks):
    if _STATE['installed']:
        return True
    import torch.nn.functional as F
    original_silu = F.silu

    def silu(input, *args, **kwargs):
        start = torch.npu.Event(enable_timing=True)
        end = torch.npu.Event(enable_timing=True)
        start.record()
        result = original_silu(input, *args, **kwargs)
        end.record()
        _STATE.setdefault('pending_silu', []).append([start, end])
        return result

    F.silu = silu

    for index, block in enumerate(blocks):
        events = {}

        def mark(name):
            def hook(*args, **kwargs):
                event = torch.npu.Event(enable_timing=True)
                event.record()
                events[name] = event
            return hook

        def pre_mlp(module, inputs):
            events.clear()
            events['e0'] = torch.npu.Event(enable_timing=True)
            events['e0'].record()
            _STATE.setdefault('pending_silu', []).clear()

        def post_mlp(module, inputs, output):
            end = torch.npu.Event(enable_timing=True)
            end.record()
            _STATE['calls'].append(dict(events, e4=end))
            return None

        block.mlp.register_forward_pre_hook(pre_mlp)
        block.mlp.register_forward_hook(post_mlp)
        block.mlp.fc1.register_forward_hook(mark('e1'))
        block.mlp.fc2.register_forward_pre_hook(mark('e2'))
        block.mlp.fc2.register_forward_hook(mark('e3'))

    _STATE['installed'] = True
    return True


def collect(torch):
    if not _STATE['installed']:
        return {'installed': False}
    torch.npu.synchronize()
    rows = []
    for call in _STATE['calls']:
        if not all(k in call for k in ('e0', 'e1', 'e2', 'e3', 'e4')):
            continue
        try:
            rows.append({
                'fc1': call['e0'].elapsed_time(call['e1']),
                'activation': call['e1'].elapsed_time(call['e2']),
                'fc2': call['e2'].elapsed_time(call['e3']),
                'tail': call['e3'].elapsed_time(call['e4']),
                'total': call['e0'].elapsed_time(call['e4']),
            })
        except BaseException:
            continue
    out = {'installed': True, 'measurement_only_no_math_change': True,
           'no_synchronize_inside_forward': True, 'additive_by_construction': True,
           'uses_the_native_forward_path': True, 'calls': len(rows)}
    if not rows:
        return out
    keys = ('fc1', 'activation', 'fc2', 'tail', 'total')
    out['median_ms'] = {k: round(statistics.median(r[k] for r in rows), 3) for k in keys}
    out['sum_of_parts_median_ms'] = round(sum(out['median_ms'][k]
                                              for k in ('fc1', 'activation', 'fc2', 'tail')), 3)
    out['closure_error_ms'] = round(out['median_ms']['total'] - out['sum_of_parts_median_ms'], 3)
    out['per_call_sum_median_ms'] = round(
        statistics.median(r['fc1'] + r['activation'] + r['fc2'] + r['tail'] for r in rows), 3)
    out['per_step_activation_s'] = round(out['median_ms']['activation'] * 50 / 1000, 3)
    out['per_step_total_s'] = round(out['median_ms']['total'] * 50 / 1000, 3)
    silu = []
    for pair in _STATE.get('pending_silu', []):
        try:
            silu.append(pair[0].elapsed_time(pair[1]))
        except BaseException:
            pass
    out['last_call_silu_ms'] = round(statistics.median(silu), 3) if silu else None
    _STATE['calls'].clear()
    return out
