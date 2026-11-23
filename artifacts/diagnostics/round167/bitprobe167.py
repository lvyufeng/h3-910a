
_STATE = {'installed': False, 'rows': []}


def _native_counts(torch, value):
    from npu_half_nan_compat124 import ieee_nonfinite_masks
    nan_mask, posinf, neginf = ieee_nonfinite_masks(value)
    return int((nan_mask | posinf | neginf).sum(dtype=torch.int32)), int((value.abs() == 65504).sum(dtype=torch.int32))


def _bit_counts(torch, value):
    dense = torch_npu.npu_format_cast(value, 2)
    if not dense.is_contiguous():
        dense = dense.contiguous()
    bits = dense.view(torch.int16)
    magnitude = bits & 0x7fff
    return int((magnitude >= 0x7c00).sum(dtype=torch.int32)), int((magnitude == 0x7bff).sum(dtype=torch.int32))


def install(torch):
    global torch_npu
    import torch_npu
    if _STATE['installed']:
        return True
    from matrix_safety126 import MatrixSafetyAudit126

    def _hook(self, name):
        def hook(module, inputs, result):
            value = result[0] if isinstance(result, tuple) else result
            if not isinstance(value, torch.Tensor) or value.dtype != torch.float16:
                return
            self.calls += 1
            s1 = torch.npu.Event(enable_timing=True); e1 = torch.npu.Event(enable_timing=True)
            s1.record(); nb, ns = _native_counts(torch, value); e1.record()
            s2 = torch.npu.Event(enable_timing=True); e2 = torch.npu.Event(enable_timing=True)
            s2.record(); fb, fs = _bit_counts(torch, value); e2.record()
            _STATE['rows'].append({'name': name, 'numel': value.numel(),
                                   'counts': (nb, ns, fb, fs), 'native_events': (s1, e1), 'bit_events': (s2, e2)})
            if nb or ns:      # native verdict remains authoritative for this verification run
                raise FloatingPointError(f'Half matrix range failure at {name}: {nb}, {ns}')
        return hook

    MatrixSafetyAudit126._hook = _hook
    _STATE['installed'] = True
    return True


def collect(torch):
    import statistics
    if not _STATE['installed']:
        return {'installed': False}
    torch.npu.synchronize()
    rows = _STATE['rows']
    out = {'installed': True, 'verification_only': True, 'native_verdict_authoritative': True,
           'no_check_removed': True, 'calls_compared': len(rows)}
    if not rows:
        return out
    mismatches = [r for r in rows if r['counts'][0] != r['counts'][2] or r['counts'][1] != r['counts'][3]]
    native_ms, bit_ms = [], []
    for r in rows:
        try:
            native_ms.append(r['native_events'][0].elapsed_time(r['native_events'][1]))
            bit_ms.append(r['bit_events'][0].elapsed_time(r['bit_events'][1]))
        except BaseException:
            pass
    out['exactly_count_equivalent_on_every_call'] = not mismatches
    out['mismatch_count'] = len(mismatches)
    if native_ms and bit_ms:
        steps = len(native_ms) / 155.0
        out['native'] = {'median_ms': round(statistics.median(native_ms), 3), 'device_s_per_step': round(sum(native_ms)/steps/1000, 3)}
        out['bit'] = {'median_ms': round(statistics.median(bit_ms), 3), 'device_s_per_step': round(sum(bit_ms)/steps/1000, 3)}
        out['speedup'] = round(statistics.median(native_ms)/statistics.median(bit_ms), 2)
        out['device_saving_s_per_step'] = round((sum(native_ms)-sum(bit_ms))/steps/1000, 3)
    _STATE['rows'] = []
    return out
