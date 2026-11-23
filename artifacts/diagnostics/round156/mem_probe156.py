"""R156: is the MLP activation region waiting on the allocator?

R125 already showed that fusing the three pointwise kernels into one buys no time at
all, and R155 showed the region is a real additive 34.7 ms against 3.2 ms standalone.
Both point at waiting rather than computing. The cheapest decisive check is the NPU
allocator's own counters: if it is retrying allocations or forcing stream syncs while
the activation allocates ~900 MB of FP32 temporaries per layer, those counters rise
every step and correlate with the region.

This probe reads torch.npu memory statistics around the whole step and around each
block, and keeps the same additive mlp spans so the two can be lined up. It changes
no math and installs no candidate.
"""
from __future__ import annotations
import statistics

_STATE = {'installed': False, 'calls': [], 'steps': []}


def _stats(torch):
    try:
        raw = torch.npu.memory_stats()
    except BaseException:
        return {}
    keys = ('num_alloc_retries', 'num_sync_all_streams', 'num_ooms', 'num_device_alloc',
            'num_device_free', 'allocated_bytes.all.allocated', 'allocated_bytes.all.freed',
            'reserved_bytes.all.current', 'reserved_bytes.all.peak',
            'active_bytes.all.current', 'active_bytes.all.peak')
    out = {}
    for key in keys:
        if key in raw:
            out[key] = int(raw[key])
    return out


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

    for block in blocks:
        events = {}

        def pre_mlp(module, inputs):
            events.clear()
            events['mem_before'] = _stats(torch)
            events['e0'] = torch.npu.Event(enable_timing=True)
            events['e0'].record()
            _STATE.setdefault('pending_silu', []).clear()

        def post_mlp(module, inputs, output):
            end = torch.npu.Event(enable_timing=True)
            end.record()
            events['e4'] = end
            events['mem_after'] = _stats(torch)
            _STATE['calls'].append((dict(events), list(_STATE.get('pending_silu', []))))
            return None

        block.mlp.register_forward_pre_hook(pre_mlp)
        block.mlp.register_forward_hook(post_mlp)

        def post_block(module, inputs, output):
            _STATE['steps'].append(_stats(torch))
            return None

        block.register_forward_hook(post_block)

    _STATE['installed'] = True
    return True


def collect(torch):
    if not _STATE['installed']:
        return {'installed': False}
    torch.npu.synchronize()
    call_rows = []
    for pair, silu_pairs in _STATE['calls']:
        if 'e0' not in pair or 'e4' not in pair:
            continue
        silu_ms = []
        for start, end in silu_pairs:
            try:
                silu_ms.append(start.elapsed_time(end))
            except BaseException:
                pass
        call_rows.append({'region': pair['e0'].elapsed_time(pair['e4']),
                          'silu': sum(silu_ms),
                          'mem_before': pair.get('mem_before', {}),
                          'mem_after': pair.get('mem_after', {})})
    out = {'installed': True, 'measurement_only_no_math_change': True,
           'no_synchronize_inside_forward': True, 'calls': len(call_rows)}
    if call_rows:
        out['mlp_total_median_ms'] = round(statistics.median(r['region'] for r in call_rows), 3)
        out['silu_median_ms'] = round(statistics.median(r['silu'] for r in call_rows), 3)
        # allocator counters: does any of them move during a call, and how often
        moved = {}
        for key in call_rows[0]['mem_after']:
            deltas = []
            for r in call_rows:
                if key in r['mem_before'] and key in r['mem_after']:
                    deltas.append(r['mem_after'][key] - r['mem_before'][key])
            if deltas and any(d != 0 for d in deltas):
                moved[key] = {'calls_with_change': sum(1 for d in deltas if d != 0),
                              'total_change': sum(deltas),
                              'median_change': statistics.median(deltas)}
        out['counters_that_move_during_a_call'] = moved
        out['first_call_mem'] = {k: v for k, v in list(call_rows[0]['mem_before'].items())[:6]}
        out['last_call_mem'] = {k: v for k, v in list(call_rows[-1]['mem_after'].items())[:6]}
    _STATE['calls'].clear()
    _STATE['steps'].clear()
    return out
