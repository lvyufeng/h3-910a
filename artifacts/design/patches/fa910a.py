"""Flash attention on Ascend 910A: build, cache, and launch the TBE kernel.

Design notes that matter:

* The TBE kernel bakes a per-block core task map in at compile time, so it is
  built per (B, H, N, D) and cached. Building the real shape takes ~19 s.
* Tile sizes MUST divide N. The reference's `last_Bc` path (used when
  N % Bc != 0) is numerically wrong -- 20% relative error on every row at
  N=1280/Bc=512. So we pick the largest 16-aligned tile <= cap that divides N,
  which also keeps the cube efficient.
* The kernel computes softmax(QK^T)V WITHOUT 1/sqrt(d), so the caller scales Q.
"""
import ctypes
import hashlib
import os
import sys

import numpy as np

_LIB = None
_LIB_PATH = "/tmp/libfa910a.so"
_KERNEL_DIR = "/tmp/fa910a_kernels"


class FlashAttn910AError(RuntimeError):
    pass


def _lib():
    global _LIB
    if _LIB is None:
        if not os.path.exists(_LIB_PATH):
            raise FlashAttn910AError("launcher library missing: %s" % _LIB_PATH)
        lib = ctypes.CDLL(_LIB_PATH)
        lib.fa910a_init.restype = ctypes.c_int
        lib.fa910a_init.argtypes = [ctypes.c_int]
        lib.fa910a_register.restype = ctypes.c_void_p
        lib.fa910a_register.argtypes = [ctypes.c_char_p, ctypes.c_char_p]
        lib.fa910a_launch.restype = ctypes.c_int
        lib.fa910a_launch.argtypes = [ctypes.c_void_p, ctypes.c_char_p,
                                      ctypes.c_uint32] + [ctypes.c_void_p] * 6
        lib.fa910a_sync.restype = ctypes.c_int
        lib.fa910a_sync.argtypes = [ctypes.c_void_p]
        # torch_npu already brought ACL up; do not force a device bind.
        rc = lib.fa910a_init(0)
        if rc != 0:
            raise FlashAttn910AError("fa910a_init failed rc=%d" % rc)
        _LIB = lib
    return _LIB


def pick_tile(n, cap, align=16):
    """Largest multiple of `align` that is <= cap and divides n exactly."""
    t = (cap // align) * align
    while t >= align:
        if n % t == 0:
            return t
        t -= align
    raise FlashAttn910AError("no %d-aligned tile <= %d divides %d" % (align, cap, n))


def _build(b, h, n, d, br_cap=64, bc_cap=512):
    """Build (or reuse) the kernel for this shape. Returns (.o path, kernel name, blockdim)."""
    br = pick_tile(n, br_cap)
    bc = pick_tile(n, bc_cap)
    key = "b%d_h%d_n%d_d%d_br%d_bc%d" % (b, h, n, d, br, bc)
    dpath = os.path.join(_KERNEL_DIR, key)
    obj = os.path.join(dpath, "%s.o" % key)
    if not os.path.exists(obj):
        os.makedirs(_KERNEL_DIR, exist_ok=True)
        sys.path.insert(0, "/tmp/fa_pkg")
        import tbe.common.platform as plat
        plat.set_current_compile_soc_info("Ascend910A")
        from flash_attention.ops.flash_attention.flash_attention_fwd import flash_attention
        os.environ["FA_BR"] = str(br)
        os.environ["FA_BC"] = str(bc)
        nz = [b, h, d // 16, n // 16, 16, 16]
        def desc(ori, dt, fmt="NZ"):
            return {"shape": list(nz), "ori_shape": list(ori), "dtype": dt, "format": fmt}
        flash_attention(
            query=desc([b, h, n, d], "float16"), key=desc([b, h, n, d], "float16"),
            value=desc([b, h, n, d], "float16"), attn_mask=None, dropout_mask=None,
            alibi_mask=None, output=desc([b, h, n, d], "float16"),
            rowsum=desc([b, h, n], "float32", "ND"),
            tiling_stgy_name="sparse", kernel_name=key, disable_debug=True)
        src = "/tmp/kernel_meta/%s.o" % key
        if not os.path.exists(src):
            raise FlashAttn910AError("kernel build produced nothing at %s" % src)
        os.makedirs(dpath, exist_ok=True)
        import shutil
        shutil.move(src, obj)
        shutil.move(src.replace(".o", ".json"), obj.replace(".o", ".json"))
    import json
    with open(obj.replace(".o", ".json")) as f:
        bd = json.load(f)["blockDim"]
    return obj, key + "__kernel0", bd


_CACHE = {}


def flash_attention_910a(q, k, v, scale=None, stream=None):
    """q,k,v: torch_npu fp16 (B,H,N,D) NDS. Returns (B,H,N,D) fp16.

    Applies `scale` to q (default 1/sqrt(D)), because the kernel does not.
    """
    b, h, n, d = q.shape
    if scale is None:
        scale = 1.0 / (d ** 0.5)
    lib = _lib()
    obj, kern, bd = _CACHE.get((b, h, n, d)) or _build(b, h, n, d)
    _CACHE[(b, h, n, d)] = (obj, kern, bd)

    hdl = _CACHE.get(("hdl", obj))
    if hdl is None:
        hdl = lib.fa910a_register(obj.encode(), kern.encode())
        if not hdl:
            raise FlashAttn910AError("register failed for %s" % obj)
        _CACHE[("hdl", obj)] = hdl

    qs = (q.float() * scale).half()
    # NDS (B,H,N,D) -> NZ (B,H,D/16,N/16,16,16)
    def to_nz(t):
        return t.reshape(b, h, n // 16, 16, d // 16, 16).permute(0, 1, 4, 2, 3, 5).contiguous()
    q_nz, k_nz, v_nz = to_nz(qs), to_nz(k), to_nz(v)
    o_nz = to_nz.__self__ if False else q_nz.new_empty(q_nz.shape)
    import torch
    l = torch.empty((b, h, n), dtype=torch.float32, device=q.device)

    st = stream if stream is not None else torch.npu.current_stream().npu_stream
    rc = lib.fa910a_launch(hdl, kern.encode(), bd,
                           q_nz.data_ptr(), k_nz.data_ptr(), v_nz.data_ptr(),
                           o_nz.data_ptr(), l.data_ptr(), ctypes.c_void_p(st))
    if rc != 0:
        raise FlashAttn910AError("launch failed rc=%d" % rc)
    lib.fa910a_sync(ctypes.c_void_p(st))
    out = o_nz.permute(0, 1, 3, 4, 2, 5).reshape(b, h, n, d).contiguous()
    return out
