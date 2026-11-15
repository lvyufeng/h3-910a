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


_CORE_BUDGETS = {}


def _query_runtime_aicore_count(device_id):
    """Query physical scheduling capacity, not a compile-time SoC default.

    This 910A toolchain reports CORE_NUM=32, while ACL and DCMI report 30
    available cores. A 32-task static equal-work grid on 30 cores leaves two
    tasks for a second wave (558 ms); repartitioning across 30 gives 317 ms.
    ACL_DEV_ATTR_AICORE_CORE_NUM=101 is from the installed CANN acl_rt.h.
    The query uses ACL's device ID, so visible-device remapping stays in ACL.
    """
    try:
        query = _lib().aclrtGetDeviceInfo
    except AttributeError as exc:
        raise FlashAttn910AError("ACL runtime lacks aclrtGetDeviceInfo; cannot safely choose a core budget") from exc
    query.argtypes = [ctypes.c_uint32, ctypes.c_int, ctypes.POINTER(ctypes.c_int64)]
    query.restype = ctypes.c_int
    value = ctypes.c_int64(0)
    rc = query(device_id, 101, ctypes.byref(value))
    if rc != 0 or not 1 <= value.value <= 64:
        raise FlashAttn910AError("AI Core capacity query failed for device %d: rc=%d count=%d"
                               % (device_id, rc, value.value))
    return int(value.value)


def _runtime_core_budget():
    import torch
    if not hasattr(torch, "npu"):
        import torch_npu  # noqa: F401 -- installs torch.npu for standalone builders
    device_id = torch.npu.current_device()
    if device_id not in _CORE_BUDGETS:
        _CORE_BUDGETS[device_id] = _query_runtime_aicore_count(device_id)
    return _CORE_BUDGETS[device_id]


def _kernel_cache_key(b, h, n, d, br, bc, core_budget):
    # The partition map is baked into the binary. Never reuse a 32-core binary
    # for a 30-core device merely because its Q/K/V shapes match.
    return "b%d_h%d_n%d_d%d_br%d_bc%d_c%d" % (b, h, n, d, br, bc, core_budget)


def _build(b, h, n, d, br_cap=128, bc_cap=304):
    """Build/reuse by shape, tiles AND physical core budget. Return (.o, symbol, blockDim)."""
    br = pick_tile(n, br_cap)
    bc = pick_tile(n, bc_cap)
    core_budget = _runtime_core_budget()
    key = _kernel_cache_key(b, h, n, d, br, bc, core_budget)
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
                from flash_attention.ops.flash_attention.flash_attention_fwd import FlashAttentionFwd
                from flash_attention.ops.flash_attention.tiling_strategy.strategy import TilingStrategy
                os.environ["FA_BR"] = str(br)
                os.environ["FA_BC"] = str(bc)
                nz = [b, h, d // 16, n // 16, 16, 16]
                def desc(ori, dt, fmt="NZ"):
                    return {"shape": list(nz), "ori_shape": list(ori), "dtype": dt, "format": fmt}
                work = tempfile.mkdtemp(prefix="fa910a_build_")
                prev = os.getcwd()
                try:
                    os.chdir(work)
                    instance = FlashAttentionFwd(
                        q=desc([b, h, n, d], "float16"), k=desc([b, h, n, d], "float16"),
                        v=desc([b, h, n, d], "float16"), attn_mask=None, dropout_mask=None,
                        alibi_mask=None, tiling_stgy=TilingStrategy.from_strategy_name("sparse"),
                        kernel_name=key, disable_debug=True)
                    instance.core_num = min(instance.core_num, core_budget)
                    instance.compute_process()
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
    if not isinstance(bd, int) or not 1 <= bd <= core_budget:
        raise FlashAttn910AError("cached blockDim=%r exceeds runtime core budget=%d" % (bd, core_budget))
    if os.environ.get("VLLM_OMNI_910A_PERF", "0") != "0":
        print("[FA910A] core_budget=%d blockDim=%d B=%d H=%d N=%d D=%d Br=%d Bc=%d"
              % (core_budget, bd, b, h, n, d, br, bc), file=sys.stderr, flush=True)
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
    cache_key = (b, h, n, d, _runtime_core_budget())
    obj, kern, bd = _CACHE.get(cache_key) or _build(b, h, n, d)
    _CACHE[cache_key] = (obj, kern, bd)

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
