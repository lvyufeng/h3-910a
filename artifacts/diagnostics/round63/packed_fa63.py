"""Experimental VAE dense FA with exact fused FP16 layout conversion on 910A.

Packing/unpacking targets bitwise equivalence to the existing FA wrapper, NOT
bitwise equivalence between FA and the legacy eager attention. This module is
standalone/current-stream only; no production import or general concurrent ABI.
"""
import ctypes
import torch
import torch_npu
from vllm_omni.diffusion.attention.backends import fa910a as fa
from pack_vector63 import build


class PackedFA:
    def __init__(self, n=1797, heads=32, padded=2048, v_row_stride=6144, v_head_stride=192, br=64, bc=1024):
        assert 0 < n <= padded and padded % 16 == 0
        self.n, self.heads, self.padded, self.d = n, heads, padded, 64
        self.v_row_stride, self.v_head_stride = v_row_stride, v_head_stride
        self.device = torch.device('npu', torch.npu.current_device())
        self.lib = fa._lib()
        self.cores = fa._runtime_core_budget()
        self.nz_shape = (1, heads, 4, padded // 16, 16, 16)
        self.objects = {}
        self.handles = {}
        for kind in ('pack', 'unpack'):
            result = build(kind, n, padded, heads, self.cores, v_row_stride, v_head_stride)
            self._register(kind, result)
        self._register('attention', fa._build(1, heads, padded, self.d, br_cap=br, bc_cap=bc, key_seq_limit=n))

    def _register(self, kind, result):
        obj, name, blocks = result
        handle = self.lib.fa910a_register(obj.encode(), name.encode())
        if not handle:
            raise RuntimeError('Kernel registration failed: ' + kind)
        self.objects[kind] = {'object': obj, 'symbol': name, 'block_dim': blocks}
        self.handles[kind] = handle

    def _launch(self, kind, tensors):
        assert len(tensors) == 5
        info = self.objects[kind]
        rc = self.lib.fa910a_launch(self.handles[kind], info['symbol'].encode(), info['block_dim'],
                                    *[tensor.data_ptr() for tensor in tensors],
                                    ctypes.c_void_p(torch.npu.current_stream().npu_stream))
        if rc:
            raise RuntimeError('%s launch failed: %d' % (kind, rc))

    def _validate(self, q, k, v):
        shape = (1, self.n, self.heads, self.d)
        if any(tuple(x.shape) != shape or x.dtype != torch.float16 or x.device != self.device for x in (q, k, v)):
            raise ValueError('Expected matched FP16 B1/BSND QKV on the prepared device')
        if not q.is_contiguous() or not k.is_contiguous():
            raise ValueError('Prepared Q/K must be dense BSND')
        if tuple(v.stride()[1:]) != (self.v_row_stride, self.v_head_stride, 1):
            raise ValueError('V stride differs from compiled GM descriptor')
        if any(torch_npu.get_npu_format(x) not in (0, 2) for x in (q, k, v)):
            raise ValueError('Only base physical ND/NCHW input formats are supported')
        for x in (q, k, v):
            last = x.storage_offset() + sum((s - 1) * stride for s, stride in zip(x.shape, x.stride())) + 1
            if last * x.element_size() > x.untyped_storage().nbytes():
                raise ValueError('Input view exceeds physical storage')

    def pack(self, q, k, v):
        self._validate(q, k, v)
        combined = torch.empty((3, *self.nz_shape), dtype=torch.float16, device=self.device)
        self._launch('pack', (q, k, v, q, combined))
        return combined

    def unpack(self, nz):
        if tuple(nz.shape) != self.nz_shape or nz.dtype != torch.float16 or nz.device != self.device or not nz.is_contiguous():
            raise ValueError('Expected the prepared dense NZ output buffer')
        out = torch.empty((1, self.n, self.heads, self.d), dtype=torch.float16, device=self.device)
        self._launch('unpack', (nz, nz, nz, nz, out))
        return out

    def __call__(self, q, k, v):
        combined = self.pack(q, k, v)
        onz = torch.empty(self.nz_shape, dtype=torch.float16, device=self.device)
        lse = torch.empty((1, self.heads, self.padded), dtype=torch.float32, device=self.device)
        self._launch('attention', (combined[0], combined[1], combined[2], onz, lse))
        return self.unpack(onz)
