"""R185: drop-in fast path for the matrix safety audit, behind a default-off env flag.

Same verdicts, same semantics, but one custom-op call per audited tensor instead of the
PyTorch mask cascade. Used only to MEASURE the end-to-end step time of the stage A candidate;
the default path is untouched and nothing is adopted by running this.

Enable with H3_FAST_AUDIT=1. When unset, matrix_safety126's original code runs unchanged.
"""
from __future__ import annotations
import ctypes
import os

# visible evidence that the flag reached the worker and the branch actually ran
_MARK = os.environ.get('H3_FAST_MARK', '')
_count = 0


def _mark():
    global _count
    _count += 1
    if _MARK:
        with open(_MARK, 'a') as fh:
            fh.write(f'{os.getpid()} {_count}\n')

_LIB = None
_ENABLED = os.environ.get('H3_FAST_AUDIT', '0') == '1'


def enabled():
    return _ENABLED


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


def _stream_ptr(torch_npu):
    return ctypes.c_void_p(torch_npu.npu.current_stream().npu_stream)


def fast_matrix_verdict(torch_npu, value):
    """Return (nonfinite_present, saturated_present, max_mag_bits) for a 2D fp16 NPU tensor.

    The kernel receives the tensor's own device buffer, so no copy or format cast is needed;
    the audit tensors are already contiguous fp16 results.
    """
    assert value.dtype is not None and str(value.dtype) == 'torch.float16', value.dtype
    assert value.dim() == 2, tuple(value.shape)
    if not value.is_contiguous():
        value = value.contiguous()
    _mark()
    max_mag = _lib().h3_max_mag_bits(ctypes.c_void_p(value.data_ptr()),
                                     int(value.shape[0]), int(value.shape[1]),
                                     _stream_ptr(torch_npu))
    if max_mag < 0:
        raise RuntimeError('h3_max_mag_bits failed')
    return (max_mag >= 0x7c00), (max_mag == 0x7bff), max_mag
