"""Self-developed FP16 indexed AdaLN/gated-residual TBE kernels on910A.

Match the CURRENT eager FP16 sequence contract, including intermediate casts;
this is distinct from the official Triton FP32-intermediate/BF16 formula.
Norm-containing variants may change FP32 reduction order and are not bitwise
accepted unless their actual comparisons pass. No production dispatch changes.
"""
import ctypes
import fcntl
import hashlib
import json
import os
from pathlib import Path
import shutil
import tempfile
import torch

ROOT = Path('/tmp/h3_round61_adaln01')
KERNELS = ROOT / 'kernels'
_HASH = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()[:12]
_LIB = None
_BUILDS = {}
_HANDLES = {}
_CORE = {}


def library():
    global _LIB
    if _LIB is None:
        lib = ctypes.CDLL(str(ROOT/'libvec61.so'))
        lib.fa910a_init.argtypes=[ctypes.c_int];lib.fa910a_init.restype=ctypes.c_int
        lib.fa910a_register.argtypes=[ctypes.c_char_p,ctypes.c_char_p];lib.fa910a_register.restype=ctypes.c_void_p
        lib.vec61_launch_n.argtypes=[ctypes.c_void_p,ctypes.c_uint32,ctypes.c_uint32,ctypes.POINTER(ctypes.c_void_p),ctypes.c_void_p]
        lib.vec61_launch_n.restype=ctypes.c_int
        assert lib.fa910a_init(0)==0
        _LIB=lib
    return _LIB


def core_budget():
    device=torch.npu.current_device()
    if device not in _CORE:
        lib=library();lib.aclrtGetDeviceInfo.argtypes=[ctypes.c_uint32,ctypes.c_int,ctypes.POINTER(ctypes.c_int64)]
        lib.aclrtGetDeviceInfo.restype=ctypes.c_int
        value=ctypes.c_int64();assert lib.aclrtGetDeviceInfo(device,101,ctypes.byref(value))==0
        assert 0<value.value<=64
        _CORE[device]=int(value.value)
    return _CORE[device]


def round_half(t, src, half, width):
    t.vconv(64,'none',half,src,width//64,1,1,4,8)
    t.vconv(64,'none',src,half,width//64,1,1,8,4)


def _build(kind,rows,width,table_rows,eps,cores):
    if kind in ('mod','gate') and width%128==0 and table_rows<=6:
        from point_vector61 import build_point
        return build_point(kind,rows,width,table_rows,cores)
    from tbe import tik
    import tbe.common.platform as platform
    platform.set_current_compile_soc_info('Ascend910A')
    name=f'adaln61_{kind}_r{rows}_d{width}_m{table_rows}_c{cores}_{_HASH}'
    name+='_e'+hashlib.sha256(str(eps).encode()).hexdigest()[:8]
    obj=KERNELS/name/(name+'.o')
    if obj.exists():
        meta=json.loads(obj.with_suffix('.json').read_text());assert meta['blockDim']==cores
        return str(obj),name+'__kernel0',cores
    norm=kind in ('rms_mod','gate_rms_mod')
    gate=kind in ('gate','gate_rms_mod')
    t=tik.Tik(disable_debug=True)
    xgm=t.Tensor('float16',(rows,width),name='adaln_xgm',scope=tik.scope_gm)
    idxgm=t.Tensor('int32',(((rows+7)//8)*8,),name='adaln_idxgm',scope=tik.scope_gm)
    # Only tensors participating in this static kind enter its ABI.
    gm={}
    if gate:
        gm['gate']=t.Tensor('float16',(table_rows,width),name='adaln_gategm',scope=tik.scope_gm)
        gm['branch']=t.Tensor('float16',(rows,width),name='adaln_branchgm',scope=tik.scope_gm)
    if norm:
        gm['weight']=t.Tensor('float16',(width,),name='adaln_weightgm',scope=tik.scope_gm)
    if kind!='gate':
        gm['shift']=t.Tensor('float16',(table_rows,width),name='adaln_shiftgm',scope=tik.scope_gm)
        gm['scale']=t.Tensor('float16',(table_rows,width),name='adaln_scalegm',scope=tik.scope_gm)
    ygm=t.Tensor('float16',(rows,width),name='adaln_ygm',scope=tik.scope_gm)
    ugm=t.Tensor('float16',(rows,width),name='adaln_ugm',scope=tik.scope_gm) if kind=='gate_rms_mod' else None
    with t.for_range(0,cores,block_num=cores) as core:
        x=t.Tensor('float32',(width,),name='adaln_x',scope=tik.scope_ubuf)
        y=t.Tensor('float32',(width,),name='adaln_y',scope=tik.scope_ubuf)
        scale=t.Tensor('float32',(width,),name='adaln_scale',scope=tik.scope_ubuf)
        shift=t.Tensor('float32',(width,),name='adaln_shift',scope=tik.scope_ubuf)
        half=t.Tensor('float16',(width,),name='adaln_half_ub',scope=tik.scope_ubuf)
        idxub=t.Tensor('int32',(8,),name='adaln_index_ub',scope=tik.scope_ubuf)
        if norm:
            weight=t.Tensor('float32',(width,),name='adaln_weight',scope=tik.scope_ubuf)
            tmp=t.Tensor('float32',(width,),name='adaln_tmp',scope=tik.scope_ubuf)
            total=t.Tensor('float32',(8,),name='adaln_total',scope=tik.scope_ubuf)
            work=t.Tensor('float32',(((width//64+7)//8)*8,),name='adaln_reduce_work',scope=tik.scope_ubuf)
            variance=t.Tensor('float32',(64,),name='adaln_variance',scope=tik.scope_ubuf)
            inv=t.Tensor('float32',(64,),name='adaln_inv',scope=tik.scope_ubuf)
            correction=t.Tensor('float32',(64,),name='adaln_correction',scope=tik.scope_ubuf)
            t.data_move(half,gm['weight'],0,1,width//16,0,0)
            t.vconv(64,'none',weight,half,width//64,1,1,8,4)
        with t.for_range(0,(rows+cores-1)//cores) as step:
            row=core+step*cores
            with t.if_scope(row<rows):
                t.data_move(idxub,idxgm[(row//8)*8],0,1,1,0,0)
                index=t.Scalar('int32',name='adaln_index');index.set_as(idxub[row%8])
                t.data_move(half,xgm[row,0],0,1,width//16,0,0)
                t.vconv(64,'none',x,half,width//64,1,1,8,4)
                if gate:
                    t.data_move(half,gm['gate'][index,0],0,1,width//16,0,0)
                    t.vconv(64,'none',scale,half,width//64,1,1,8,4)
                    t.data_move(half,gm['branch'][row,0],0,1,width//16,0,0)
                    t.vconv(64,'none',y,half,width//64,1,1,8,4)
                    t.h_mul(y,y,scale)
                    round_half(t,y,half,width)
                    t.h_add(x,x,y)
                    round_half(t,x,half,width)
                    if ugm is not None:
                        t.data_move(ugm[row,0],half,0,1,width//16,0,0)
                if norm:
                    t.h_mul(tmp,x,x)
                    t.vec_reduce_add(64,total,tmp,work,width//64,8)
                    t.vmuls(8,total,total,1.0/width,1,1,1,1,1)
                    t.vadds(8,total,total,eps,1,1,1,1,1)
                    value=t.Scalar('float32',name='adaln_var');value.set_as(total[0])
                    t.vector_dup(64,variance,value,1,1,8)
                    t.vec_rsqrt(64,inv,variance,1,8,8)
                    for _ in range(2):
                        t.h_mul(correction,inv,inv);t.h_mul(correction,correction,variance)
                        t.vmuls(64,correction,correction,-.5,1,1,1,8,8)
                        t.vadds(64,correction,correction,1.5,1,1,1,8,8)
                        t.h_mul(inv,inv,correction)
                    scalar=t.Scalar('float32',name='adaln_inv_scalar');scalar.set_as(inv[0])
                    t.vmuls(64,x,x,scalar,width//64,1,1,8,8)
                    t.h_mul(x,x,weight)
                    round_half(t,x,half,width)
                if kind!='gate':
                    t.data_move(half,gm['scale'][index,0],0,1,width//16,0,0)
                    t.vconv(64,'none',scale,half,width//64,1,1,8,4)
                    t.data_move(half,gm['shift'][index,0],0,1,width//16,0,0)
                    t.vconv(64,'none',shift,half,width//64,1,1,8,4)
                    t.vadds(64,scale,scale,1.0,width//64,1,1,8,8)
                    round_half(t,scale,half,width)
                    t.h_mul(x,x,scale)
                    round_half(t,x,half,width)
                    t.h_add(x,x,shift)
                t.vconv(64,'none',half,x,width//64,1,1,4,8)
                t.data_move(ygm[row,0],half,0,1,width//16,0,0)
    if kind=='mod':names=('shift','scale')
    elif kind=='gate':names=('gate','branch')
    elif kind=='rms_mod':names=('weight','shift','scale')
    else:names=('gate','branch','weight','shift','scale')
    workdir=os.path.realpath(tempfile.mkdtemp(prefix='adaln61_build_',dir=ROOT))
    if Path(workdir).parent!=ROOT.resolve() or not Path(workdir).name.startswith('adaln61_build_'):
        raise RuntimeError('Unexpected AdaLN private build directory')
    old=os.getcwd()
    try:
        os.chdir(workdir)
        t.BuildCCE(kernel_name=name,inputs=[xgm,*[gm[key] for key in names],idxgm],outputs=[ygm] if ugm is None else [ugm,ygm])
        source=Path(workdir)/'kernel_meta'/(name+'.o');assert source.exists() and source.resolve()==source.absolute()
        obj.parent.mkdir(exist_ok=False)
        shutil.copyfile(source.with_suffix('.json'),obj.with_suffix('.json'));shutil.copyfile(source,obj)
    finally:
        os.chdir(old)
        if Path(workdir).resolve()==Path(workdir).absolute() and Path(workdir).parent==ROOT.resolve():shutil.rmtree(workdir)
    meta=json.loads(obj.with_suffix('.json').read_text());assert meta['blockDim']==cores
    return str(obj),name+'__kernel0',cores


class PreparedIndices:
    """Explicit immutable metadata snapshot with checked bounds and padded GM.

    Not a cache of a mutable original tensor: callers deliberately use this
    snapshot's values. Its own normal tensor version detects illicit mutation.
    Stream changes are rejected until cross-stream ordering is validated.
    """
    def __init__(self,indices,table_rows):
        assert indices.ndim==1 and indices.dtype in (torch.int32,torch.int64)
        cpu=indices.detach().cpu()
        assert cpu.numel()>0 and int(cpu.min())>=0 and int(cpu.max())<table_rows
        self.rows,self.table_rows=indices.numel(),table_rows
        with torch.inference_mode(False):
            self.tensor=indices.to(torch.int32).contiguous().clone()
            if (-self.rows)%8:
                self.tensor=torch.cat((self.tensor,torch.zeros((-self.rows)%8,dtype=self.tensor.dtype,device=self.tensor.device)))
        self.version=self.tensor._version
        self.stream=torch.npu.current_stream().npu_stream

    def checked(self,rows,table_rows,device):
        assert self.rows==rows and self.table_rows==table_rows and self.tensor.device==device
        assert self.tensor._version==self.version, 'Prepared metadata was mutated'
        assert torch.npu.current_stream().npu_stream==self.stream, 'Unvalidated cross-stream metadata use'
        return self.tensor


def launch(kind,x,indices,*,shift=None,scale=None,gate=None,branch=None,weight=None,eps=1e-5):
    assert kind in ('mod','gate','rms_mod','gate_rms_mod')
    assert x.ndim==2 and x.dtype==torch.float16 and x.device.type in ('npu','privateuseone')
    assert x.is_contiguous() and 64<=x.shape[1]<=8192 and x.shape[1]%64==0
    rows,width=x.shape;assert rows>0
    tensors=[]
    if kind in ('gate','gate_rms_mod'):tensors.extend((gate,branch))
    if kind in ('rms_mod','gate_rms_mod'):tensors.append(weight)
    if kind!='gate':tensors.extend((shift,scale))
    import torch_npu
    assert torch_npu.get_npu_format(x) in (0,2)
    assert not (torch.is_grad_enabled() and x.requires_grad), 'Inference-only AdaLN prototype'
    assert all(t.dtype==x.dtype and t.device==x.device and t.is_contiguous() and torch_npu.get_npu_format(t) in (0,2) for t in tensors)
    table_rows=gate.shape[0] if gate is not None else shift.shape[0]
    assert table_rows>0
    if gate is not None:assert gate.shape==(table_rows,width) and branch.shape==x.shape
    if weight is not None:assert weight.shape==(width,)
    if kind!='gate':assert shift.shape==scale.shape==(table_rows,width)
    if isinstance(indices,PreparedIndices):
        idx=indices.checked(rows,table_rows,x.device)
    else:
        assert indices.shape==(rows,) and indices.device==x.device and indices.dtype in (torch.int32,torch.int64)
        # Ordinary tensor API revalidates its mutable values on every call;
        # faster prepared-snapshot use is explicit, not an unchecked cache.
        cpu_indices=indices.detach().cpu()
        assert int(cpu_indices.min())>=0 and int(cpu_indices.max())<table_rows
        idx=indices.to(torch.int32).contiguous()
        padding=(-rows)%8
        if padding:idx=torch.cat((idx,torch.zeros(padding,dtype=idx.dtype,device=idx.device)))
    cores=min(core_budget(),rows);key=(kind,rows,width,table_rows,float(eps),cores)
    info=_BUILDS.get(key)
    if info is None:
        with (KERNELS/'.compile.lock').open('a') as lock:
            fcntl.flock(lock,fcntl.LOCK_EX);info=_build(kind,rows,width,table_rows,eps,cores)
        _BUILDS[key]=info
    obj,symbol,cores=info
    if obj not in _HANDLES:
        handle=library().fa910a_register(obj.encode(),symbol.encode())
        if not handle:raise RuntimeError('AdaLN registration failed')
        _HANDLES[obj]=handle
    y=torch.empty_like(x);updated=torch.empty_like(x) if kind=='gate_rms_mod' else None
    buffers=[x,*tensors,idx]+([y] if updated is None else [updated,y])
    pointers=(ctypes.c_void_p*len(buffers))(*[t.data_ptr() for t in buffers])
    rc=library().vec61_launch_n(_HANDLES[obj],cores,len(buffers),pointers,ctypes.c_void_p(torch.npu.current_stream().npu_stream))
    if rc:raise RuntimeError('AdaLN launch failed rc=%s'%rc)
    return y if updated is None else (updated,y)
