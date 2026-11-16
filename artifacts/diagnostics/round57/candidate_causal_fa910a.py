"""Flash attention on Ascend 910A: build, cache, and launch the TBE kernel.

Design notes that matter:

* The TBE kernel bakes a per-block core task map in at compile time, so it is
  built per (B, H, N, D) and cached. Building the real shape takes ~19 s.
* Tiles divide the compiled sequence length, avoiding the earlier failed
  remainder path. If the input has inefficient divisors, a bounded expansion
  permits larger tiles and masks added keys before rowmax/softmax. Cache identity
  includes both the runtime core budget and this compiled key boundary.
* The kernel computes softmax(QK^T)V WITHOUT 1/sqrt(d), so the caller scales Q.
"""
import ctypes
import hashlib
import json
import math
import operator
import os
import sys

import numpy as np

_LIB = None
_LIB_PATH = "/tmp/libfa910a.so"
_KERNEL_DIR = "/tmp/fa910a_kernels"
# Opt-in only; see the note at the launch site for why this is off.
_SYNC_EACH_CALL = os.environ.get("VLLM_OMNI_910A_FLASH_SYNC", "0") not in ("0", "", "false")
_USE_TILE_PADDING = os.environ.get("VLLM_OMNI_910A_FA_PAD_TILES", "1").lower() not in ("0", "", "false", "off")
_PLAN_LOGGED = set()


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


def _kernel_sequence_length(n, d, enabled=True):
    """Expand only inefficient, measured-width shapes with <=10% extra work rows."""
    if not isinstance(n, int) or n <= 0 or n % 16:
        raise FlashAttn910AError("sequence length must be a positive multiple of 16")
    if not enabled or d != 128 or pick_tile(n, 128) * pick_tile(n, 304) >= 128 * 128:
        return n
    quantum = math.lcm(128, 304)
    padded = ((n + quantum - 1) // quantum) * quantum
    return padded if (padded - n) * 10 <= n else n


def _partial_column_mask(first_invalid):
    """128-bit vector mask for an invalid suffix of each FP16 NZ row of 16."""
    if not isinstance(first_invalid, int) or not 1 <= first_invalid < 16:
        raise FlashAttn910AError("invalid partial NZ column mask")
    lane = ((1 << 16) - 1) ^ ((1 << first_invalid) - 1)
    word = sum(lane << (16 * i) for i in range(4))
    return [word, word]


def _install_key_sequence_limit(instance, key_limit, causal=False):
    """Exclude every invalid prefix-tail key, including sub-16-column tails.

    Modulation makes original padded K/V nonzero in actual H3 layers. Mask all
    columns beyond the metadata's true valid-key length, not a zero-key LSE
    subtraction. Use finite-min scores before rowmax and exact zero probabilities
    before both rowsum and P@V. The packed score tile remains FP16.
    """
    import types
    from flash_attention.ops.flash_attention.constants import FP16, L1
    original = instance.rowmax_compute
    if causal:
        original_qblock = instance.compute_in_each_q_block
        def current_qblock(self, batch_start, batch_idx, q_index, q_height):
            self._causal_q_index = q_index
            return original_qblock(batch_start, batch_idx, q_index, q_height)
        instance.compute_in_each_q_block = types.MethodType(current_qblock, instance)

    def fill_tail(self, scores, kv_index, value):
        if causal:
            # Equal square Br/Bc tiles: future tiles are fully invalid; only
            # diagonal tiles need a compile-time triangular per-row mask.
            assert self.Br == self.Bc
            with self.tik_instance.if_scope(kv_index > self._causal_q_index):
                self.tik_instance.h_duplicate(scores, value)
            with self.tik_instance.if_scope(kv_index == self._causal_q_index):
                for row in range(scores.shape[1]):
                    block, valid = divmod(row + 1, 16)
                    if valid:
                        mask = ((1 << 16) - 1) ^ ((1 << valid) - 1)
                        self.tik_instance.vector_dup([0, mask], scores[block, row, 0], value, 1, 1, 8)
                        block += 1
                    if block < scores.shape[0]:
                        self.tik_instance.vector_dup(16, scores[block, row, 0], value,
                                                     scores.shape[0] - block, 1, scores.shape[1])
        # Bc is finalized by compute_process, not by the constructor.
        cutoff, remainder = divmod(key_limit, self.Bc)
        if remainder:
            block, column = divmod(remainder, 16)
            with self.tik_instance.if_scope(kv_index == cutoff):
                if column:
                    # Each vector repeat is eight 16-column FP16 NZ rows. A
                    # repeated bitmask fills only invalid columns in each row.
                    self.tik_instance.vector_dup(_partial_column_mask(column), scores[block, 0, 0],
                                                 value, scores.shape[1] // 8, 1, 8)
                    block += 1
                if block < scores.shape[0]:
                    self.tik_instance.h_duplicate(scores[block:, :, :], value)
            with self.tik_instance.if_scope(kv_index > cutoff):
                self.tik_instance.h_duplicate(scores, value)
        else:
            with self.tik_instance.if_scope(kv_index >= cutoff):
                self.tik_instance.h_duplicate(scores, value)

    def limited_rowmax(self, scores, mi, kv_index):
        fill_tail(self, scores, kv_index, -65504.0)
        return original(scores, mi, kv_index)

    def limited_softmax(self, scores, li, mi, kv_index, drop_offset):
        mi, update_scale = self.rowmax_compute(scores, mi, kv_index)
        n1, m, n0 = scores.shape
        with self.tik_instance.new_stmt_scope(disable_sync=False):
            broadcast = self.tik_ops_utils.broadcast_vec_from_M_to_MN0(mi).reshape((1, m, n0))
            for idx in range(n1):
                self.tik_instance.h_sub(scores[idx, :, :], scores[idx, :, :], broadcast)
        self.tik_instance.h_exp(scores, scores)
        # Mask exactly before both rowsum and P@V, not by denominator subtraction.
        fill_tail(self, scores, kv_index, 0.0)
        probabilities = self.tik_instance.Tensor(FP16, (n1, m, n0), scope=L1, name="Pij_l1_zN")
        self.cont_data_mv_1_bust(dst=probabilities, src=scores, burst=n1 * m * n0 // 16)
        li = self.rowsum_compute(probabilities, li, update_scale, kv_index)
        if self.has_drop_mask:
            self.do_dropout_mask(scores, drop_offset, n1 * n0, n1 * n0, m, m)
            self.cont_data_mv_1_bust(dst=probabilities, src=scores, burst=n1 * m * n0 // 16)
        return probabilities, li, mi, update_scale

    instance.rowmax_compute = types.MethodType(limited_rowmax, instance)
    instance.softmax_compute = types.MethodType(limited_softmax, instance)


def _kernel_cache_key(b, h, n, d, br, bc, core_budget, key_seq_limit=None, causal=False):
    # Partition AND clipped-key geometry are baked into the binary. Never reuse
    # either a 32-core binary or the unmasked zero-pad prototype for this kernel.
    key = "b%d_h%d_n%d_d%d_br%d_bc%d_c%d" % (b, h, n, d, br, bc, core_budget)
    key = key if key_seq_limit is None else key + "_k%d_v3" % key_seq_limit
    return key + "_causal_v4" if causal else key


def _build(b, h, n, d, br_cap=128, bc_cap=304, key_seq_limit=None, causal=False):
    """Build/reuse by shape, tiles AND physical core budget. Return (.o, symbol, blockDim)."""
    if key_seq_limit is not None:
        key_seq_limit = operator.index(key_seq_limit)
        if key_seq_limit <= 0 or key_seq_limit > n:
            raise FlashAttn910AError("invalid key sequence limit")
        if key_seq_limit == n:
            key_seq_limit = None
    br = pick_tile(n, br_cap)
    bc = br if causal else pick_tile(n, bc_cap)
    core_budget = _runtime_core_budget()
    key = _kernel_cache_key(b, h, n, d, br, bc, core_budget, key_seq_limit, causal)
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
                work = os.path.realpath(tempfile.mkdtemp(prefix="fa910a_build_"))
                if (os.path.dirname(work) != os.path.realpath(tempfile.gettempdir())
                        or not os.path.basename(work).startswith("fa910a_build_")):
                    raise FlashAttn910AError("unexpected private build directory")
                prev = os.getcwd()
                try:
                    os.chdir(work)
                    instance = FlashAttentionFwd(
                        q=desc([b, h, n, d], "float16"), k=desc([b, h, n, d], "float16"),
                        v=desc([b, h, n, d], "float16"), attn_mask=None, dropout_mask=None,
                        alibi_mask=None, tiling_stgy=TilingStrategy.from_strategy_name("sparse"),
                        kernel_name=key, disable_debug=True)
                    instance.core_num = min(instance.core_num, core_budget)
                    if key_seq_limit is not None or causal:
                        _install_key_sequence_limit(instance, key_seq_limit or n, causal=causal)
                    instance.compute_process()
                    src = os.path.join(work, "kernel_meta", "%s.o" % key)
                    if not os.path.exists(src):
                        raise FlashAttn910AError("kernel build produced nothing at %s" % src)
                    if key_seq_limit is not None or causal:
                        meta_path = src.replace(".o", ".json")
                        with open(meta_path) as f:
                            meta = json.load(f)
                        meta["fa910a_key_seq_limit"] = key_seq_limit
                        meta["fa910a_causal"] = causal
                        meta["fa910a_kernel_version"] = 4 if causal else 3
                        with open(meta_path, "w") as f:
                            json.dump(meta, f)
                    os.makedirs(dpath, exist_ok=True)
                    if (os.path.realpath(src) != os.path.abspath(src)
                            or os.path.realpath(dpath) != os.path.abspath(dpath)):
                        raise FlashAttn910AError("unexpected resolved kernel publication path")
                    # Publish metadata before the object readiness marker so a
                    # second worker cannot see an object without its mask tag.
                    shutil.move(src.replace(".o", ".json"), obj.replace(".o", ".json"))
                    shutil.move(src, obj)
                finally:
                    os.chdir(prev)
                    if (os.path.realpath(work) == os.path.abspath(work)
                            and os.path.basename(work).startswith("fa910a_build_")):
                        shutil.rmtree(work, ignore_errors=True)
    with open(obj.replace(".o", ".json")) as f:
        meta = json.load(f)
    bd = meta["blockDim"]
    if (key_seq_limit is not None or causal) and (meta.get("fa910a_key_seq_limit") != key_seq_limit
                                       or meta.get("fa910a_kernel_version") != (4 if causal else 3)
                                       or bool(meta.get("fa910a_causal", False)) != causal):
        raise FlashAttn910AError("cached kernel lacks its requested key-tail mask")
    if not isinstance(bd, int) or not 1 <= bd <= core_budget:
        raise FlashAttn910AError("cached blockDim=%r exceeds runtime core budget=%d" % (bd, core_budget))
    if os.environ.get("VLLM_OMNI_910A_PERF", "0") != "0":
        print("[FA910A] core_budget=%d blockDim=%d B=%d H=%d N=%d D=%d Br=%d Bc=%d"
              % (core_budget, bd, b, h, n, d, br, bc), file=sys.stderr, flush=True)
    return obj, key + "__kernel0", bd


_CACHE = {}
# Per-shape device scratch, reused across calls; see the note at the allocation site.
_SCRATCH = {}
_PAD_BUFFERS = {}


def _pad_qkv(q, k, v, kernel_n):
    """Reuse aligned buffers and clear every tail on every call/length change."""
    import torch
    b, h, source_n, d = q.shape
    key = (b, h, kernel_n, d, q.device)
    padded = _PAD_BUFFERS.get(key)
    if padded is None:
        padded = tuple(torch.empty((b, h, kernel_n, d), dtype=q.dtype, device=q.device) for _ in range(3))
        _PAD_BUFFERS[key] = padded
    for src, dst in zip((q, k, v), padded):
        # The installed zero_ kernel warns on nonzero storage-offset views.
        # Clear the full contiguous allocation, then overwrite the valid prefix;
        # this also safely handles a later source length shrinking in-place.
        dst.zero_()
        dst[:, :, :source_n, :].copy_(src)
    return padded


def flash_attention_910a(q, k, v, scale=None, stream=None, n_pad=0):
    """Keep caller shape/dtype; auto-align inefficient shapes with masked keys."""
    import torch
    if q.ndim != 4 or tuple(q.shape) != tuple(k.shape) or tuple(q.shape) != tuple(v.shape):
        raise FlashAttn910AError("Q/K/V must have the same rank-4 self-attention shape")
    if any(t.dtype != torch.float16 or t.device != q.device for t in (q, k, v)):
        raise FlashAttn910AError("Q/K/V must be FP16 on the same device")
    if q.device.type not in ("npu", "privateuseone"):
        raise FlashAttn910AError("FlashAttention910A requires NPU tensors")
    if isinstance(n_pad, (bool, torch.Tensor)):
        raise FlashAttn910AError("padding count must be a plain integer")
    try:
        n_pad = operator.index(n_pad)
    except TypeError as exc:
        raise FlashAttn910AError("padding count must be an integer") from exc
    b, h, source_n, d = q.shape
    if b <= 0 or h <= 0 or d <= 0 or d % 16 or n_pad < 0 or n_pad >= source_n:
        raise FlashAttn910AError("invalid shape or original padding count")
    valid_keys = source_n - n_pad
    kernel_n = _kernel_sequence_length(source_n, d, enabled=_USE_TILE_PADDING)
    if kernel_n == source_n:
        return _flash_attention_core(q, k, v, scale=scale, stream=stream,
                                      key_seq_limit=valid_keys if n_pad else None)
    pq, pk, pv = _pad_qkv(q, k, v, kernel_n)
    tag = (b, h, source_n, kernel_n, d, n_pad, q.device)
    if os.environ.get("VLLM_OMNI_910A_PERF", "0") != "0" and tag not in _PLAN_LOGGED:
        _PLAN_LOGGED.add(tag)
        print("[FA910A] plan input_N=%d kernel_N=%d key_limit=%d original_pad=%d"
              % (source_n, kernel_n, valid_keys, n_pad), file=sys.stderr, flush=True)
    out = _flash_attention_core(pq, pk, pv, scale=scale, stream=stream,
                                key_seq_limit=valid_keys)
    return out[:, :, :source_n, :].contiguous()


def _flash_attention_core(q, k, v, scale=None, stream=None, key_seq_limit=None, causal=False):
    """FP16 NZ launch with exact compiled valid-key prefix masking.

    Scale Q because the raw kernel does not. Original and internally added
    invalid keys are excluded in the score/probability tiles; no assumption
    about zero-valued modulated padding and no LSE subtraction is needed.
    """
    b, h, n, d = q.shape
    if scale is None:
        scale = 1.0 / (d ** 0.5)
    lib = _lib()
    cache_key = (b, h, n, d, _runtime_core_budget(), key_seq_limit, causal)
    obj, kern, bd = _CACHE.get(cache_key) or _build(b, h, n, d, key_seq_limit=key_seq_limit, causal=causal)
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
        if bufs["isrc"] is None:
            bufs["isrc"] = torch.empty((b, h, n, d), dtype=q.dtype, device=q.device)
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
    return out.contiguous()
