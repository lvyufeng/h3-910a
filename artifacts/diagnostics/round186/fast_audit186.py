"""R186: verdict-equivalence verification for the stage A fast path (contract B).

H3_FAST_AUDIT=verify makes the audit compute BOTH verdicts for every audited tensor, compare
them, and record every comparison. This is a verification run: it is slower than either single
path by construction, and its only product is the evidence that the fast verdict equals the
native one on real tensors.

H3_FAST_AUDIT=1 keeps the R185 measurement behaviour (fast only).
Unset keeps the untouched native path.
"""
from __future__ import annotations
import atexit
import ctypes
import json
import os

MODE = os.environ.get('H3_FAST_AUDIT', '0')
_LIB = None
_RECORD = []          # (module_name, native_nonfinite, native_saturated, fast_nonfinite, fast_saturated, max_mag)
_OUT = os.environ.get('H3_FAST_VERIFY_OUT', '')


def enabled():
    return MODE in ('1', 'verify')


def verify_mode():
    return MODE == 'verify'


def _lib():
    global _LIB
    if _LIB is None:
        path = os.environ.get('H3_FAST_LIB', '/tmp/h3_lib184/libh3fast184.so')
        lib = ctypes.CDLL(path)
        lib.h3_max_mag_bits.restype = ctypes.c_int
        lib.h3_max_mag_bits.argtypes = [ctypes.c_void_p, ctypes.c_int64, ctypes.c_int64,
                                        ctypes.c_void_p]
        _LIB = lib
    return _LIB


def fast_matrix_verdict(torch_npu, value):
    """(nonfinite_present, saturated_present, max_mag_bits) from one custom-op call."""
    if not value.is_contiguous():
        value = value.contiguous()
    max_mag = _lib().h3_max_mag_bits(ctypes.c_void_p(value.data_ptr()),
                                     int(value.shape[0]), int(value.shape[1]),
                                     ctypes.c_void_p(torch_npu.npu.current_stream().npu_stream))
    if max_mag < 0:
        raise RuntimeError('h3_max_mag_bits failed')
    return (max_mag >= 0x7c00), (max_mag == 0x7bff), max_mag


def record(name, native_pair, fast_pair, max_mag):
    _RECORD.append((name, int(native_pair[0]), int(native_pair[1]),
                    int(fast_pair[0]), int(fast_pair[1]), int(max_mag)))


def flush(rank):
    """Write this rank's comparisons and a summary. Called once after the audit ends."""
    if not verify_mode():
        return None
    total = len(_RECORD)
    mismatches = [r for r in _RECORD if (r[1] != r[3]) or (r[2] != r[4])]
    summary = {
        'rank': rank, 'mode': MODE, 'comparisons': total,
        'mismatches': len(mismatches),
        'agree': total > 0 and not mismatches,
        'nonfinite_hits': sum(1 for r in _RECORD if r[1] or r[3]),
        'saturated_hits': sum(1 for r in _RECORD if r[2] or r[4]),
        'module_name_examples': sorted({r[0] for r in _RECORD})[:6],
        'mismatch_examples': mismatches[:5],
    }
    if _OUT:
        with open(f'{_OUT}.rank{rank}.json', 'w') as fh:
            json.dump({'summary': summary, 'rows': _RECORD}, fh)
    return summary


def _atexit_flush():
    """Safety net: the benchmark may end the run without closing the audit."""
    if not verify_mode() or not _OUT:
        return
    try:
        import torch.distributed as dist
        rank = dist.get_rank() if dist.is_initialized() else 0
    except Exception:
        rank = 0
    try:
        flush(rank)
    except Exception:
        pass


if MODE == 'verify':
    atexit.register(_atexit_flush)
