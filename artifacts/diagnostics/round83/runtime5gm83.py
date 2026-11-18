"""Verified five-GM-pointer TIK runtime bridge;no custom C compilation.

The library is the same immutable legacy launcher used by R82 actual execution.
Torch supplies the current NPU stream;arguments are retained by the caller.
"""
import ctypes
import hashlib
from pathlib import Path

LIBRARY=Path('/tmp/libfa910a.so')
LIBRARY_SHA256='bd50d7005bf48e9a9a853c01750eaea4eefbb3e0f69abb2e7153aa7139e8b043'


class Runtime5GM:
    def __init__(self,device=0):
        import torch
        import torch_npu
        if hashlib.sha256(LIBRARY.read_bytes()).hexdigest()!=LIBRARY_SHA256:
            raise RuntimeError('Proven launcher binary changed')
        torch.npu.set_device(device)
        probe=torch.empty((1,),dtype=torch.float16,device=f'npu:{device}')
        torch.npu.synchronize();del probe
        self.lib=ctypes.CDLL(str(LIBRARY));self.device=device;self.cache={}
        self.lib.fa910a_init.argtypes=[ctypes.c_int];self.lib.fa910a_init.restype=ctypes.c_int
        self.lib.fa910a_register.argtypes=[ctypes.c_char_p,ctypes.c_char_p]
        self.lib.fa910a_register.restype=ctypes.c_void_p
        self.lib.fa910a_launch.argtypes=[ctypes.c_void_p,ctypes.c_char_p,ctypes.c_uint32]+[ctypes.c_void_p]*6
        self.lib.fa910a_launch.restype=ctypes.c_int
        self.lib.fa910a_sync.argtypes=[ctypes.c_void_p];self.lib.fa910a_sync.restype=ctypes.c_int
        if self.lib.fa910a_init(0)!=0:raise RuntimeError('Legacy runtime init failed')
    def register(self,meta):
        obj=Path(meta['obj_path']);symbol=meta['kernel_symbol']
        key=(meta['obj_sha256'],symbol)
        # The registered device binary is immutable even if its source file
        # later changes. Verify once;do not put disk hashing in warm timings.
        if key in self.cache:return self.cache[key]
        digest=hashlib.sha256(obj.read_bytes()).hexdigest()
        if digest!=meta['obj_sha256']:raise RuntimeError('Kernel object changed after build')
        if key not in self.cache:
            handle=self.lib.fa910a_register(str(obj).encode(),symbol.encode())
            if not handle:raise RuntimeError('Kernel registration failed')
            self.cache[key]=handle
        return self.cache[key]
    def launch(self,meta,buffers,synchronize=True):
        import torch
        import torch_npu
        if len(buffers)!=5:raise RuntimeError('Five GM arguments required')
        for value in buffers:
            if (value.device.type not in('npu','privateuseone')or value.device.index!=self.device
                    or not value.is_contiguous()or value.storage_offset()!=0):
                raise RuntimeError('Unvalidated GM operand device/layout/offset')
            if torch_npu.get_npu_format(value)not in(0,2):raise RuntimeError('GM operand is not dense')
        handle=self.register(meta)
        stream=ctypes.c_void_p(torch.npu.current_stream().npu_stream)
        rc=self.lib.fa910a_launch(handle,meta['kernel_symbol'].encode(),int(meta['block_dim']),
                                 *[value.data_ptr()for value in buffers],stream)
        if rc!=0:raise RuntimeError(f'Kernel launch returned{rc}')
        if synchronize:
            rc=self.lib.fa910a_sync(stream)
            if rc!=0:raise RuntimeError(f'Kernel synchronize returned{rc}')
    def snapshot(self):
        return {'library':str(LIBRARY),'library_sha256':LIBRARY_SHA256,'registered_kernels':len(self.cache),
                'ABI':'fiveGMpointers','current_stream':'torch.npu.current_stream().npu_stream',
                'no_production_launcher_modified':True}
