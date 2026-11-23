"""R170: the two measurements that settle whether the safety checks can be deferred.

R169 priced deferring the 155 checks at about 19.4 GiB of live tensors against about 20.4 GiB
of headroom, but two of its three terms were assumptions, and the headroom figure came from
inside an mlp call rather than from a whole step. This run measures exactly what was missing:

  1. the real output shape of every audited tensor, per module kind;
  2. the peak allocated and reserved bytes across a whole block stack, not just one call.

It installs ADDITIONAL forward hooks and leaves the native strict audit untouched: all 155
checks still run and still raise on failure, the model arithmetic is unchanged, and nothing
is adopted. This is a read-only diagnostic.
"""
from __future__ import annotations
import statistics

_STATE = {'installed': False, 'shapes': {}, 'peaks': [], 'reset_done': False}

_AUDITED = ('.qkv_proj', '.mlp.fc1', '.adaln_proj.linear')


def install(torch, blocks):
    """Hook the audited modules of every block. `blocks` is the layerwise block list, which
    is exactly what the offloader hook group receives, so no model internals are needed."""
    if _STATE['installed']:
        return True
    installed = 0
    first = last = None
    for block in blocks:
        for name, module in block.named_modules():
            kind = None
            for tag in ('qkv_proj', 'fc1', 'linear'):
                if name.endswith(tag):
                    kind = tag
                    break
            if kind is None:
                continue

            def make(kind=kind):
                def hook(module, inputs, result):
                    value = result[0] if isinstance(result, tuple) else result
                    if not isinstance(value, torch.Tensor) or value.dtype != torch.float16:
                        return
                    row = _STATE['shapes'].setdefault(kind, {'calls': 0, 'shapes': {}, 'bytes': 0})
                    row['calls'] += 1
                    key = 'x'.join(str(d) for d in value.shape)
                    row['shapes'][key] = row['shapes'].get(key, 0) + 1
                    row['bytes'] = max(row['bytes'], value.numel() * value.element_size())
                return hook
            module.register_forward_hook(make())
            installed += 1
        if first is None:
            first = block
        last = block
    if first is not None:
        def reset_hook(module, args):
            if not _STATE['reset_done']:
                try:
                    torch.npu.reset_peak_memory_stats()
                except BaseException:
                    pass

        def read_hook(module, args, result):
            try:
                _STATE['peaks'].append({
                    'peak_allocated_bytes': int(torch.npu.max_memory_allocated()),
                    'peak_reserved_bytes': int(torch.npu.max_memory_reserved()),
                    'allocated_now': int(torch.npu.memory_allocated()),
                    'reserved_now': int(torch.npu.memory_reserved())})
                _STATE['reset_done'] = True
            except BaseException:
                pass

        first.register_forward_pre_hook(reset_hook)
        last.register_forward_hook(read_hook)
    _STATE['installed'] = installed > 0
    _STATE['installed_hooks'] = installed
    return _STATE['installed']


def collect(torch):
    if not _STATE['installed']:
        return {'installed': False}
    torch.npu.synchronize()
    GiB = 1024 ** 3
    out = {'installed': True, 'read_only_diagnostic': True, 'native_audit_untouched': True,
           'model_arithmetic_unchanged': True, 'nothing_adopted': True}
    kinds = {}
    for kind, row in _STATE['shapes'].items():
        kinds[kind] = {'calls': row['calls'], 'shapes': row['shapes'],
                       'max_bytes_per_tensor': row['bytes'],
                       'max_mib_per_tensor': round(row['bytes'] / (1024 ** 2), 1)}
    out['audited_tensor_shapes'] = kinds
    # price the deferred set from the measured shapes alone
    per_step = {'qkv_proj': 50, 'fc1': 50, 'adaln_proj.linear': 55}
    deferred = 0
    for kind, row in kinds.items():
        count = per_step.get(kind, 0)
        if kind == 'linear':
            count = per_step['adaln_proj.linear']
        deferred += count * row['max_bytes_per_tensor']
        row['per_step_count'] = count
        row['per_step_gib'] = round(count * row['max_bytes_per_tensor'] / GiB, 2)
    out['deferred_total_gib_from_measured_shapes'] = round(deferred / GiB, 2)
    if _STATE['peaks']:
        peaks = _STATE['peaks']
        out['whole_block_stack_peak'] = {
            'samples': len(peaks),
            'peak_allocated_gib_max': round(max(p['peak_allocated_bytes'] for p in peaks) / GiB, 2),
            'peak_allocated_gib_median': round(statistics.median([p['peak_allocated_bytes'] for p in peaks]) / GiB, 2),
            'peak_reserved_gib_max': round(max(p['peak_reserved_bytes'] for p in peaks) / GiB, 2),
            'peak_reserved_gib_median': round(statistics.median([p['peak_reserved_bytes'] for p in peaks]) / GiB, 2)}
    _STATE['shapes'] = {}
    _STATE['peaks'] = []
    return out
