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
# Opt-in only; see the note at the launch site for why this is off.
_SYNC_EACH_CALL = os.environ.get("VLLM_OMNI_910A_FLASH_SYNC", "0") not in ("0", "", "false")


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


def _build(b, h, n, d, br_cap=128, bc_cap=304):
    """Build (or reuse) the kernel for this shape. Returns (.o path, kernel name, blockdim)."""
    br = pick_tile(n, br_cap)
    bc = pick_tile(n, bc_cap)
    key = "b%d_h%d_n%d_d%d_br%d_bc%d" % (b, h, n, d, br, bc)
    dpath = os.path.join(_KERNEL_DIR, key)
    obj = os.path.join(dpath, "%s.o" % key)
    if not os.path.exists(obj):
        os.makedirs(_KERNEL_DIR, exist_ok=True)
        # TIK writes ./kernel_meta relative to the *process cwd*, which the
        # service shares across all 8 USP workers -- and a plain path lookup
        # would also miss it entirely. Build inside an exclusive lock in a
        # private directory, then publish the artefact into the cache.
        import fcntl
        import shutil
        import tempfile
        lock_path = os.path.join(_KERNEL_DIR, ".build.lock")
        with open(lock_path, "w") as lockf:
            fcntl.flock(lockf, fcntl.LOCK_EX)
            if not os.path.exists(obj):          # another worker may have won
                sys.path.insert(0, "/tmp/fa_pkg")
                import tbe.common.platform as plat
                plat.set_current_compile_soc_info("Ascend910A")
                from flash_attention.ops.flash_attention.flash_attention_fwd import flash_attention
                os.environ["FA_BR"] = str(br)
                os.environ["FA_BC"] = str(bc)
                nz = [b, h, d // 16, n // 16, 16, 16]
                def desc(ori, dt, fmt="NZ"):
                    return {"shape": list(nz), "ori_shape": list(ori), "dtype": dt, "format": fmt}
                work = tempfile.mkdtemp(prefix="fa910a_build_")
                prev = os.getcwd()
                try:
                    os.chdir(work)
                    flash_attention(
                        query=desc([b, h, n, d], "float16"), key=desc([b, h, n, d], "float16"),
                        value=desc([b, h, n, d], "float16"), attn_mask=None, dropout_mask=None,
                        alibi_mask=None, output=desc([b, h, n, d], "float16"),
                        rowsum=desc([b, h, n], "float32", "ND"),
                        tiling_stgy_name="sparse", kernel_name=key, disable_debug=True)
                    src = os.path.join(work, "kernel_meta", "%s.o" % key)
                    if not os.path.exists(src):
                        raise FlashAttn910AError("kernel build produced nothing at %s" % src)
                    os.makedirs(dpath, exist_ok=True)
                    shutil.move(src, obj)
                    shutil.move(src.replace(".o", ".json"), obj.replace(".o", ".json"))
                finally:
                    os.chdir(prev)
                    shutil.rmtree(work, ignore_errors=True)
    import json
    with open(obj.replace(".o", ".json")) as f:
        bd = json.load(f)["blockDim"]
    return obj, key + "__kernel0", bd


_CACHE = {}
# Per-shape device scratch, reused across calls; see the note at the allocation site.
_SCRATCH = {}


def flash_attention_910a(q, k, v, scale=None, stream=None, n_pad=0):
    """q,k,v: torch_npu fp16 (B,H,N,D) NDS. Returns (B,H,N,D) fp16.

    Applies `scale` to q (default 1/sqrt(D)), because the kernel does not.

    `n_pad` is the number of trailing padded keys. The packed batch is allocated
    with torch.zeros, so a padded key row contributes k=0 and v=0: its score is
    q.0 = 0, so exp(0)=1 to the softmax denominator, and nothing to the
    numerator. The kernel therefore returns

        O_kernel = num / (den_valid + n_pad)

    and the exact result is recovered by dividing by (1 - n_pad*exp(-l)), using
    the log-sum-exp `l` the kernel already produces. No mask, no extra matmul.
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

    import torch
    # Reuse the NZ scratch buffers per shape instead of allocating four ~113 MB
    # tensors on every call. At 200 calls per run that allocation churn was
    # enough to be a suspect for the VAE decode slowdown, and reuse removes it
    # as a variable either way.
    bufs = _SCRATCH.get((b, h, n, d, q.device))
    if bufs is None:
        nz_shape = (b, h, d // 16, n // 16, 16, 16)
        bufs = {
            "q": torch.empty(nz_shape, dtype=torch.float16, device=q.device),
            "k": torch.empty(nz_shape, dtype=torch.float16, device=q.device),
            "v": torch.empty(nz_shape, dtype=torch.float16, device=q.device),
            "o": torch.empty(nz_shape, dtype=torch.float16, device=q.device),
            "l": torch.empty((b, h, n), dtype=torch.float32, device=q.device),
            "isrc": torch.empty((b, h, n, d), dtype=torch.float16, device=q.device)
            if scale != 1.0 else None,
        }
        _SCRATCH[(b, h, n, d, q.device)] = bufs
    q_nz, k_nz, v_nz, o_nz, l = bufs["q"], bufs["k"], bufs["v"], bufs["o"], bufs["l"]

    # NDS (B,H,N,D) -> NZ (B,H,D/16,N/16,16,16), written straight into scratch.
    def to_nz(t, dst):
        dst.copy_(t.reshape(b, h, n // 16, 16, d // 16, 16).permute(0, 1, 4, 2, 3, 5))
        return dst

    # Scale inside fp16 (1.16 ms -> 0.24 ms versus a float32 round trip), and
    # only when needed, reusing its own scratch.
    if scale == 1.0:
        qs = q
    else:
        qs = bufs["isrc"]
        torch.mul(q, scale, out=qs)
    to_nz(qs, q_nz); to_nz(k, k_nz); to_nz(v, v_nz)

    st = stream if stream is not None else torch.npu.current_stream().npu_stream
    rc = lib.fa910a_launch(hdl, kern.encode(), bd,
                           q_nz.data_ptr(), k_nz.data_ptr(), v_nz.data_ptr(),
                           o_nz.data_ptr(), l.data_ptr(), ctypes.c_void_p(st))
    if rc != 0:
        raise FlashAttn910AError("launch failed rc=%d" % rc)
    # Do NOT synchronise here. The kernel is enqueued on the torch stream, so
    # torch's own ordering (and its eventual sync) already covers it. A full
    # aclrtSynchronizeStream per attention call serialises the whole diffusion
    # pipeline -- measured as one chip at 83% AICore while the other seven sat
    # at 7%. Sync is only needed by callers that read the result immediately.
    if _SYNC_EACH_CALL:
        lib.fa910a_sync(ctypes.c_void_p(st))
    out = o_nz.permute(0, 1, 3, 4, 2, 5).reshape(b, h, n, d)
    if n_pad:
        # out was num/(den_valid+n_pad); rescale to num/den_valid. Only the
        # padded case pays the float32 round trip.
        corr = 1.0 - float(n_pad) * torch.exp(-l.float())
        out = (out.float() / corr.unsqueeze(-1)).to(q.dtype)
    return out.contiguous()
