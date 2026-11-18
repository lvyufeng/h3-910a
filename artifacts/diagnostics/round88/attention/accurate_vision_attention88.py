#!/usr/bin/env python3
"""Round88 gated ACCURATE vision SDPA engine (SOURCE ONLY).

WHAT THIS IS
------------
The Round79 private candidate's ``_vision_sdpa_bf16_compatible`` keeps Half Q/K/V
but promotes them to Float32 and runs three Float32 torch matmuls. Round85 already
proved, on the *text* tower, that this stack's Float32 matmul does not preserve a
true FP32 accumulation: injecting exact CPU probabilities still left ~1.2e-4 PV
error. The vision tower uses the very same helper, so its P@V carries the same
kind of unverified Float32 matmul precision.

This module replaces only that one module-level helper with a gated alternative
that keeps the arithmetic exact by construction:

* original Half q/k/v (head_dim 72) are copied unchanged into zero-padded width
  128 operands and a zero-padded key axis 4032 -> 4096;
* the executed Round85 batched Half x Half -> FP32 L0C -> FP32 GM kernel
  (``batch_gemm85``, SHA-pinned, per-head distinct B, one FP32 accumulator live
  across every K tile, no Half partial and no FP32 weight shadow) produces the
  QK scores and both P@V products;
* scores are cropped to the true 4032 keys *before* the unmasked Float softmax,
  and the ORIGINAL scale ``72 ** -0.5`` is applied - never ``128 ** -0.5`` for
  the padded width;
* P (Float) is split into ``P_hi = Half(P)`` and
  ``P_lo = Half((P - Float(P_hi)) * 1024)``; the two P@V products are summed in
  Float32 and only then narrowed once to the activation dtype;
* the 72 real output columns are cropped from the padded 128.

WHY THE PADDING IS SAFE
-----------------------
Zero-padding the head dimension contributes exactly zero to every dot product
(the extra 56 terms are exact products of zeros). Zero-padding the key axis to
4096 would, however, corrupt the softmax denominator if it were left in, because
a padded key scores exactly 0 and ``exp(0) == 1``. The score tensor is therefore
cropped back to ``[:, :, :4032]`` before the softmax. On the P@V side the padded
key rows of P and the padded key columns of V are both exactly zero, so the extra
contraction terms are exactly zero as well.

WHAT THIS DELIBERATELY DOES NOT DO
----------------------------------
* No new TIK kernel and no new C launcher: it reuses the immutable Round85
  ``batch_gemm85`` builder and the proven five-GM ``runtime5gm83`` bridge.
* No global ``torch.nn.functional`` monkeypatch and no production edit; only the
  candidate module's own ``_vision_sdpa_bf16_compatible`` attribute is scoped.
* No Float32 weight shadow, no BF16 flag, no model weight is read or converted:
  this helper only ever sees the Half q/k/v activations of one attention leaf.
* The biased Half linears and the vision MLP/merger are untouched; the Round87
  bias adapter stays a separate switch.
* Activated K/V packing is cached only inside one call (one image), never across
  layers, blocks or inputs.

SCOPE / NON-CLAIMS
------------------
Source only. Nothing here has been compiled, launched, imported on an NPU,
executed on a model, installed into production or committed by its author: the
author performed no SSH, no NPU/GPU work and no CANN build. There is no numerical
result, no tolerance, no speed claim and no whole-encoder acceptance. The parent
driver captures real vision q/k/v, builds the CPU Half MATH reference and runs
this engine; every number belongs to that driver's receipts.
"""

from __future__ import annotations

import ast
import hashlib
from pathlib import Path
import textwrap

# --- pinned immutable inputs -------------------------------------------------
#: The executed Round85 batched Half->FP32 GEMM builder used unmodified.
PINNED_KERNEL_MODULE = "batch_gemm85"
PINNED_KERNEL_SHA256 = "546cd211ddb3938c3d8dac9d5207a79646c1458a3887df7e8c8562426069854a"
#: The proven five-GM-pointer runtime bridge and its immutable launcher library.
PINNED_RUNTIME_MODULE = "runtime5gm83"
PINNED_RUNTIME_MODULE_SHA256 = "c8890dbcd825fe9177c7ee8d5c6b46e4d00231d4d61e6421f83491fec85c52a6"
PINNED_RUNTIME_LIBRARY_SHA256 = (
    "bd50d7005bf48e9a9a853c01750eaea4eefbb3e0f69abb2e7153aa7139e8b043")
#: R79 private candidate and the exact original helper this engine replaces.
PINNED_CANDIDATE_SHA256 = "2141b29adc9ecc601d7fa2599e6ba56351adfbdaa5f16acb62c00d95cee3cf78"
PINNED_HELPER_SOURCE_SHA256 = (
    "6393bee7067d4854a7f4d47f192d0b3b6673acfc7cf951e6c86eb52b5a39d2e5")
HELPER_NAME = "_vision_sdpa_bf16_compatible"

# --- real H3 vision attention geometry (validated against the checkpoint) -----
NUM_HEADS = 16
ORIGINAL_HEAD_DIM = 72           # 1152 / 16
PADDED_HEAD_DIM = 128            # cube tile multiple
ORIGINAL_KEY_LEN = 4032          # 48 * 84 patches of one image
PADDED_KEY_LEN = 4096            # 4032 rounded up to a 128 multiple
QUERY_CHUNK = 512
#: The original helper scales by ``q.shape[-1] ** -0.5`` = 72 ** -0.5 and NOT by
#: the padded width: the padded columns contribute exactly zero, so rescaling by
#: 128 ** -0.5 would rescale the genuine scores by a factor of 72/128.
ORIGINAL_SCALE = ORIGINAL_HEAD_DIM ** -0.5
RESIDUAL_SCALE = 1024.0

HALF_DTYPE_NAME = "torch.float16"
NPU_DEVICE_PREFIXES = ("npu", "privateuseone")
EXPECTED_ACTIVATION_SHAPE = (1, NUM_HEADS, ORIGINAL_KEY_LEN, ORIGINAL_HEAD_DIM)


class VisionAttentionError(RuntimeError):
    """Raised instead of guessing, silently falling back or editing the graph."""


# ---------------------------------------------------------------------------
# Pure geometry helpers (stdlib only: unit-testable without torch or TIK)
# ---------------------------------------------------------------------------


def _ceil_to(value: int, multiple: int) -> int:
    if multiple <= 0:
        raise VisionAttentionError("multiple must be positive")
    return ((value + multiple - 1) // multiple) * multiple


def query_chunks(total: int = ORIGINAL_KEY_LEN, chunk: int = QUERY_CHUNK) -> tuple:
    """``((start, end, m), ...)`` for the query axis; 7x512 + one 448 tail."""
    if not isinstance(total, int) or isinstance(total, bool) or total <= 0:
        raise VisionAttentionError(f"total must be a positive int, got {total!r}")
    if not isinstance(chunk, int) or isinstance(chunk, bool) or chunk <= 0:
        raise VisionAttentionError(f"chunk must be a positive int, got {chunk!r}")
    out = []
    for start in range(0, total, chunk):
        end = min(start + chunk, total)
        out.append((start, end, end - start))
    return tuple(out)


def planned_kernel_shapes() -> tuple:
    """The four ``(batch, m, k, n)`` binaries ``prepare()`` builds and pins.

    Two QK binaries (full 512-query chunk and the 448-query tail; both pad to
    M=512 inside the kernel) and two P@V binaries for the same query chunks.
    ``k``/``n`` here are already padded: 128 for the head dimension, 4096 for the
    padded key axis.
    """
    chunks = query_chunks()
    query_ms = sorted({m for _start, _end, m in chunks}, reverse=True)
    out = []
    for m in query_ms:
        out.append((NUM_HEADS, m, PADDED_HEAD_DIM, PADDED_KEY_LEN))
    for m in query_ms:
        out.append((NUM_HEADS, m, PADDED_KEY_LEN, PADDED_HEAD_DIM))
    return tuple(out)


def check_activation_spec(shape, dtype_name, device_type, device_index, runtime_device) -> dict:
    """Fail-closed structural check of one leaf's q/k/v, torch-free for testing.

    ``shape`` may be a tuple/list of ints; ``dtype_name`` a ``str(dtype)`` style
    string; ``device_type`` ``"npu"``/``"privateuseone"``; ``device_index`` int
    or ``None``.
    """
    dims = tuple(int(v) for v in shape)
    if dims != EXPECTED_ACTIVATION_SHAPE:
        raise VisionAttentionError(
            f"vision SDPA operands are {dims}, expected {EXPECTED_ACTIVATION_SHAPE}; this "
            "engine is fixed to the 16-head / 4032-key / 72-dim real vision geometry and "
            "refuses to guess another one")
    if dtype_name != HALF_DTYPE_NAME:
        raise VisionAttentionError(
            f"vision SDPA operands are {dtype_name}, expected {HALF_DTYPE_NAME}; the original "
            "helper's Float32 matmul is only being replaced on the FP16 path")
    if device_type not in NPU_DEVICE_PREFIXES:
        raise VisionAttentionError(
            f"vision SDPA operands are on {device_type!r}, expected an Ascend NPU device")
    if int(runtime_device) < 0:
        raise VisionAttentionError("runtime object reports no device index")
    if device_index is not None and int(device_index) != int(runtime_device):
        raise VisionAttentionError(
            f"vision SDPA operands are on device {device_index} but the runtime owns "
            f"device {runtime_device}")
    return {"shape": list(dims), "dtype": dtype_name, "device_type": device_type,
            "device_index": None if device_index is None else int(device_index),
            "runtime_device": int(runtime_device)}


def padded_buffer_bytes() -> dict:
    """Deterministic extra device-buffer accounting for one enabled call.

    These are the buffers this engine allocates per (image) attention call on top
    of whatever the caller already holds. They are freed when the call returns;
    nothing here is cached across calls, so this is a transient cost, not a
    resident weight shadow.
    """
    h = NUM_HEADS
    q = h * ORIGINAL_KEY_LEN * PADDED_HEAD_DIM * 2
    kv = h * PADDED_KEY_LEN * PADDED_HEAD_DIM * 2
    vt = h * PADDED_HEAD_DIM * PADDED_KEY_LEN * 2
    k1 = PADDED_HEAD_DIM // 16
    packed_k = h * k1 * PADDED_KEY_LEN * 16 * 2
    k1v = PADDED_KEY_LEN // 16
    packed_v = h * k1v * PADDED_HEAD_DIM * 16 * 2
    m_pad = _ceil_to(max(c[2] for c in query_chunks()), 128)
    score_c = h * (PADDED_KEY_LEN // 16) * m_pad * 16 * 4
    pv_c = h * (PADDED_HEAD_DIM // 16) * m_pad * 16 * 4
    pack_a_score = h * (PADDED_HEAD_DIM // 16) * m_pad * 16 * 2
    pack_a_pv = h * (PADDED_KEY_LEN // 16) * m_pad * 16 * 2
    chunk_p = h * m_pad * PADDED_KEY_LEN * 2
    return {
        "padded_q_half": q,
        "padded_k_half": kv,
        "padded_v_half": kv,
        "transposed_v_half": vt,
        "packed_k_half": packed_k,
        "packed_v_half": packed_v,
        "score_fp32_gm": score_c,
        "pv_fp32_gm": pv_c,
        "pack_a_score_half": pack_a_score,
        "pack_a_pv_half": pack_a_pv,
        "p_hi_or_lo_padded_half": chunk_p,
        "call_persistent_total": q + kv + kv + vt + packed_k + packed_v,
        "per_chunk_transient_total": score_c + pv_c + pack_a_score + pack_a_pv + 2 * chunk_p,
    }


def planned_launch_counts() -> dict:
    """Counts a single enabled call must produce (a structural contract)."""
    chunks = len(query_chunks())
    return {"query_chunks": chunks, "pack_w_calls": 2,
            "kernel_launches": chunks * 3,          # 1 QK + 2 P@V per query chunk
            "float_before_half_chunks": chunks}


# ---------------------------------------------------------------------------
# Source pinning helpers (stdlib only)
# ---------------------------------------------------------------------------


def _sha256_file(path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _normalize_source(segment: str) -> str:
    lines = [line.rstrip() for line in textwrap.dedent(segment).splitlines()]
    while lines and not lines[-1]:
        lines.pop()
    return "\n".join(lines) + "\n"


def helper_source_digest(candidate_file) -> str:
    """SHA-256 of the normalized original ``_vision_sdpa_bf16_compatible`` source."""
    text = Path(candidate_file).read_text()
    for node in ast.parse(text).body:
        if isinstance(node, ast.FunctionDef) and node.name == HELPER_NAME:
            segment = ast.get_source_segment(text, node)
            if segment is None:
                break
            return hashlib.sha256(_normalize_source(segment).encode()).hexdigest()
    raise VisionAttentionError(f"{HELPER_NAME} not found in {candidate_file}")


# ---------------------------------------------------------------------------
# The gated engine
# ---------------------------------------------------------------------------


class AccurateVisionAttention88:
    """Gated accurate vision SDPA: Half in, FP32 accumulation, one Half cast out."""

    def __init__(self, candidate, runtime, kernel_dir):
        self._candidate = candidate
        self._runtime = runtime
        self._kernel_dir = Path(kernel_dir)
        original = getattr(candidate, HELPER_NAME, None)
        if not callable(original):
            raise VisionAttentionError(
                f"candidate module has no callable {HELPER_NAME}; refusing to patch an "
                "unknown graph")
        self._original = original
        self._enabled = False
        self._installed = True
        self._metas: dict = {}
        self._aux = None
        self._stats = {
            "calls_enabled": 0,
            "calls_disabled": 0,
            "calls_rejected": 0,
            "kernel_launches": 0,
            "pack_w_calls": 0,
            "query_chunks_processed": 0,
            "float_before_half_chunks": 0,
            "half_output_chunks": 0,
            "kernel_shapes": {},
            "device": int(getattr(runtime, "device", -1)),
        }
        # Scoped assignment on the isolated diagnostic module only. The text
        # SDPA symbol and torch.nn.functional are never touched.
        setattr(candidate, HELPER_NAME, self.forward)

    # -- preparation -------------------------------------------------------

    def _verify_sources(self) -> dict:
        import batch_gemm85 as kernel
        import sys

        kernel_sha = _sha256_file(kernel.__file__)
        if kernel_sha != PINNED_KERNEL_SHA256:
            raise VisionAttentionError(
                f"{PINNED_KERNEL_MODULE} changed: {kernel_sha} != {PINNED_KERNEL_SHA256}; the "
                "executed FP32-accumulator arithmetic this engine relies on no longer matches")
        module_name = type(self._runtime).__module__
        runtime_module = sys.modules.get(module_name)
        runtime_file = getattr(runtime_module, "__file__", None)
        if not runtime_file:
            raise VisionAttentionError(
                f"cannot locate the runtime module {module_name!r} to verify its pin")
        runtime_sha = _sha256_file(runtime_file)
        if runtime_sha != PINNED_RUNTIME_MODULE_SHA256:
            raise VisionAttentionError(
                f"{PINNED_RUNTIME_MODULE} changed: {runtime_sha} != "
                f"{PINNED_RUNTIME_MODULE_SHA256}")
        reported = getattr(self._runtime, "library_sha256", None)
        if reported is None:
            snapshot = getattr(self._runtime, "snapshot", None)
            if callable(snapshot):
                reported = snapshot().get("library_sha256")
        if reported != PINNED_RUNTIME_LIBRARY_SHA256:
            raise VisionAttentionError(
                f"runtime bridge reports launcher {reported!r}, expected "
                f"{PINNED_RUNTIME_LIBRARY_SHA256}")
        candidate_file = getattr(self._candidate, "__file__", None)
        if not candidate_file:
            raise VisionAttentionError("candidate module has no __file__; cannot verify its pin")
        candidate_sha = _sha256_file(candidate_file)
        if candidate_sha != PINNED_CANDIDATE_SHA256:
            raise VisionAttentionError(
                f"candidate changed: {candidate_sha} != {PINNED_CANDIDATE_SHA256}")
        helper_sha = helper_source_digest(candidate_file)
        if helper_sha != PINNED_HELPER_SOURCE_SHA256:
            raise VisionAttentionError(
                f"{HELPER_NAME} source changed: {helper_sha} != {PINNED_HELPER_SOURCE_SHA256}; "
                "the operator this engine replaces is no longer the reviewed one")
        return {"kernel_module_sha256": kernel_sha,
                "runtime_module_sha256": runtime_sha,
                "runtime_library_sha256": reported,
                "candidate_sha256": candidate_sha,
                "helper_source_sha256": helper_sha,
                "kernel_module": PINNED_KERNEL_MODULE,
                "runtime_module": PINNED_RUNTIME_MODULE}

    def _validate_kernel_meta(self, meta: dict, batch: int, m: int, k: int, n: int) -> dict:
        import batch_gemm85 as kernel

        m_pad = _ceil_to(m, 128)
        m_tiles, n_tiles = m_pad // 128, n // 128
        tasks = batch * m_tiles * n_tiles
        expected_cores = min(30, tasks)
        checks = {
            "dst_dtype": ("float32", meta.get("dst_dtype")),
            "input_dtype": ("float16", meta.get("input_dtype")),
            "accumulator_dtype": ("float32", meta.get("accumulator_dtype")),
            "GM_argument_count": (5, meta.get("GM_argument_count")),
            "block_dim": (expected_cores, int(meta.get("block_dim", -1))),
            "grid_cores": (expected_cores, int((meta.get("grid") or {}).get("cores", -1))),
        }
        for name, (expected, actual) in checks.items():
            if actual != expected:
                raise VisionAttentionError(
                    f"kernel metadata {name} is {actual!r}, expected {expected!r}")
        logical = meta.get("logical_shape") or {}
        if (int(logical.get("batch", -1)), int(logical.get("m", -1)),
                int(logical.get("k", -1)), int(logical.get("n", -1))) != (batch, m, k, n):
            raise VisionAttentionError(f"unexpected kernel logical shape {logical}")
        padded = meta.get("padded_shape") or {}
        if int(padded.get("m_pad", -1)) != m_pad:
            raise VisionAttentionError(
                f"kernel M pad {padded.get('m_pad')} != expected {m_pad}")
        shapes = meta.get("gm_shapes") or {}
        want_a = [batch, k // 16, m_pad, 16]
        want_b = [batch, k // 16, n, 16]
        want_c = [batch, n // 16, m_pad, 16]
        for key, want in (("a", want_a), ("b", want_b), ("c", want_c)):
            got = [int(v) for v in shapes.get(key, ())]
            if got != want:
                raise VisionAttentionError(f"kernel GM {key} shape {got} != {want}")
        if (meta.get("accumulation") or {}).get("half_partial") is not False:
            raise VisionAttentionError("kernel metadata does not guarantee no Half partial")
        if (meta.get("accumulation") or {}).get("fp32_ub_partial_sum") is not False:
            raise VisionAttentionError("kernel metadata does not guarantee a single L0C sum")
        obj = Path(meta["obj_path"])
        if not obj.is_file():
            raise VisionAttentionError(f"kernel object missing: {obj}")
        if _sha256_file(obj) != meta["obj_sha256"]:
            raise VisionAttentionError("kernel object changed after build")
        if kernel.PINNED_CORE_NUM != 30:
            raise VisionAttentionError("kernel core pin changed")
        return meta

    def prepare(self) -> dict:
        """Build (or reuse) the four pinned binaries under one shared lock."""
        import batch_gemm85 as kernel
        import fcntl

        sources = self._verify_sources()
        self._kernel_dir.mkdir(parents=True, exist_ok=True)
        built = {}
        with (self._kernel_dir / ".vision_engine88.lock").open("a") as lock:
            fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
            try:
                for batch, m, k, n in planned_kernel_shapes():
                    meta = kernel.build_kernel(batch=batch, m=m, k=k, n=n,
                                               out_dir=self._kernel_dir)
                    self._validate_kernel_meta(meta, batch, m, k, n)
                    self._metas[(m, k, n)] = meta
                    built[(m, k, n)] = {
                        "kernel_name": meta["kernel_name"],
                        "kernel_symbol": meta["kernel_symbol"],
                        "obj_sha256": meta["obj_sha256"],
                        "block_dim": int(meta["block_dim"]),
                        "compiled_now": bool(meta.get("compiled_now")),
                    }
            finally:
                fcntl.flock(lock.fileno(), fcntl.LOCK_UN)
        if len(self._metas) != len(planned_kernel_shapes()):
            raise VisionAttentionError("not all planned vision kernels were prepared")
        self._sources = sources
        return {"sources": sources, "kernels": {str(key): value for key, value in built.items()}}

    def set_enabled(self, enabled: bool) -> None:
        enabled = bool(enabled)
        if enabled and len(self._metas) != len(planned_kernel_shapes()):
            raise VisionAttentionError(
                "accurate vision attention enabled before all four kernels were prepared")
        self._enabled = enabled

    # -- runtime path ------------------------------------------------------

    def _aux_buffer(self, device):
        import torch

        if self._aux is None or self._aux.device != device:
            self._aux = torch.zeros(16, dtype=torch.float16, device=device)
        return self._aux

    def _gemm(self, a, packed_weight, n: int):
        """One batched kernel launch; returns the FP32 GM output ``(H, m, n)``."""
        import torch
        import batch_gemm85 as kernel

        h, m, k = (int(v) for v in a.shape)
        meta = self._metas[(m, k, n)]
        packed_a = kernel.pack_a(a)
        c = torch.empty(meta["gm_shapes"]["c"], dtype=torch.float32, device=a.device)
        aux = self._aux_buffer(a.device)
        self._runtime.launch(meta, [packed_a, packed_weight, aux, aux, c], synchronize=True)
        self._stats["kernel_launches"] += 1
        shape_key = f"h{h}_m{m}_k{k}_n{n}"
        self._stats["kernel_shapes"][shape_key] = int(
            self._stats["kernel_shapes"].get(shape_key, 0)) + 1
        out = kernel.unpack_c(c, m=m)
        if out.dtype != torch.float32:
            raise VisionAttentionError(
                f"kernel output is {out.dtype}, expected float32 before the single Half cast")
        return out

    def forward(self, q, k, v):
        if not self._enabled:
            self._stats["calls_disabled"] += 1
            # Disabled means exactly the captured original operator, unchanged.
            return self._original(q, k, v)
        import torch
        import batch_gemm85 as kernel

        if len(self._metas) != len(planned_kernel_shapes()):
            self._stats["calls_rejected"] += 1
            raise VisionAttentionError("accurate vision attention used before prepare()")
        spec = check_activation_spec(tuple(q.shape), str(q.dtype), q.device.type,
                                     q.device.index, int(getattr(self._runtime, "device", -1)))
        for name, tensor in (("k", k), ("v", v)):
            if (tuple(int(x) for x in tensor.shape) != tuple(spec["shape"])
                    or str(tensor.dtype) != HALF_DTYPE_NAME
                    or tensor.device.type != q.device.type
                    or tensor.device.index != q.device.index):
                self._stats["calls_rejected"] += 1
                raise VisionAttentionError(f"vision SDPA operand {name} does not match q")
        self._stats["calls_enabled"] += 1
        counts = planned_launch_counts()
        launches_before = self._stats["kernel_launches"]
        device = q.device

        # 1. Copy the original Half values into zero-padded operands. The indexed
        #    copy is exact for any stride pattern the caller's split produced.
        qp = torch.zeros((NUM_HEADS, ORIGINAL_KEY_LEN, PADDED_HEAD_DIM),
                         dtype=torch.float16, device=device)
        qp[..., :ORIGINAL_HEAD_DIM] = q[0]
        kp = torch.zeros((NUM_HEADS, PADDED_KEY_LEN, PADDED_HEAD_DIM),
                         dtype=torch.float16, device=device)
        kp[:, :ORIGINAL_KEY_LEN, :ORIGINAL_HEAD_DIM] = k[0]
        vp = torch.zeros((NUM_HEADS, PADDED_KEY_LEN, PADDED_HEAD_DIM),
                         dtype=torch.float16, device=device)
        vp[:, :ORIGINAL_KEY_LEN, :ORIGINAL_HEAD_DIM] = v[0]

        # 2. Pack B once per call (never across layers/inputs): the transposed
        #    value matrix is the P@V right operand. Distinct B per head.
        packed_k = kernel.pack_w(kp)
        packed_v = kernel.pack_w(vp.transpose(1, 2).contiguous())
        self._stats["pack_w_calls"] += 2

        result = torch.empty((NUM_HEADS, ORIGINAL_KEY_LEN, ORIGINAL_HEAD_DIM),
                             dtype=q.dtype, device=device)
        for start, end, m in query_chunks():
            chunk = qp[:, start:end, :]                       # (H, m, 128)
            scores = self._gemm(chunk, packed_k, PADDED_KEY_LEN)
            # 3. Drop the padded keys BEFORE the softmax: a padded key scores
            #    exactly 0, and exp(0)=1 would corrupt the denominator.
            scores = scores[:, :, :ORIGINAL_KEY_LEN] * ORIGINAL_SCALE
            p = torch.softmax(scores, dim=-1)                 # Float, UNMASKED
            if p.dtype != torch.float32:
                self._stats["calls_rejected"] += 1
                raise VisionAttentionError(f"probabilities are {p.dtype}, expected float32")

            # 4. Float P -> (Half high, 1024-scaled Half residual), zero key padding.
            high = p.to(torch.float16)
            low = ((p - high.float()) * RESIDUAL_SCALE).to(torch.float16)
            p_high = torch.zeros((NUM_HEADS, m, PADDED_KEY_LEN),
                                 dtype=torch.float16, device=device)
            p_high[:, :, :ORIGINAL_KEY_LEN] = high
            p_low = torch.zeros_like(p_high)
            p_low[:, :, :ORIGINAL_KEY_LEN] = low

            # 5. Both products stay Float32; they are summed in Float32 and only
            #    then narrowed exactly once to the activation dtype.
            out = (self._gemm(p_high, packed_v, PADDED_HEAD_DIM)
                   + self._gemm(p_low, packed_v, PADDED_HEAD_DIM)
                   * (1.0 / RESIDUAL_SCALE))
            if out.dtype != torch.float32:
                self._stats["calls_rejected"] += 1
                raise VisionAttentionError("PV sum left float32 before the Half cast")
            self._stats["float_before_half_chunks"] += 1
            result[:, start:end, :] = out.to(q.dtype)[:, :, :ORIGINAL_HEAD_DIM]
            self._stats["half_output_chunks"] += 1
            self._stats["query_chunks_processed"] += 1
        if self._stats["kernel_launches"] - launches_before != counts["kernel_launches"]:
            self._stats["calls_rejected"] += 1
            raise VisionAttentionError(
                "unexpected kernel launch count for one enabled call")
        return result.unsqueeze(0)

    # -- checks / reporting / teardown --------------------------------------

    def padding_selfcheck(self, q, k, v) -> dict:
        """Verify the zero-padding contract on real operands (call outside warm).

        Returns plain booleans only; it never changes the graph and never chooses
        a numerical tolerance.
        """
        import torch
        import batch_gemm85 as kernel

        qp = torch.zeros((NUM_HEADS, ORIGINAL_KEY_LEN, PADDED_HEAD_DIM),
                         dtype=torch.float16, device=q.device)
        qp[..., :ORIGINAL_HEAD_DIM] = q[0]
        kp = torch.zeros((NUM_HEADS, PADDED_KEY_LEN, PADDED_HEAD_DIM),
                         dtype=torch.float16, device=q.device)
        kp[:, :ORIGINAL_KEY_LEN, :ORIGINAL_HEAD_DIM] = k[0]
        vp = torch.zeros((NUM_HEADS, PADDED_KEY_LEN, PADDED_HEAD_DIM),
                         dtype=torch.float16, device=q.device)
        vp[:, :ORIGINAL_KEY_LEN, :ORIGINAL_HEAD_DIM] = v[0]
        packed_k = kernel.pack_w(kp)
        return {
            "original_scale": float(ORIGINAL_SCALE),
            "not_padded_width_scale": float(ORIGINAL_SCALE) != float(PADDED_HEAD_DIM ** -0.5),
            "q_prefix_exact": bool(torch.equal(qp[..., :ORIGINAL_HEAD_DIM], q[0].contiguous())),
            "q_width_tail_zero": int(torch.count_nonzero(qp[..., ORIGINAL_HEAD_DIM:]).item()) == 0,
            "k_key_tail_zero": int(torch.count_nonzero(kp[:, ORIGINAL_KEY_LEN:]).item()) == 0,
            "k_prefix_exact": bool(torch.equal(
                kp[:, :ORIGINAL_KEY_LEN, :ORIGINAL_HEAD_DIM], k[0].contiguous())),
            "v_prefix_exact": bool(torch.equal(
                vp[:, :ORIGINAL_KEY_LEN, :ORIGINAL_HEAD_DIM], v[0].contiguous())),
            "v_width_tail_zero": int(torch.count_nonzero(vp[..., ORIGINAL_HEAD_DIM:]).item()) == 0,
            "packed_k_head_distinct": list(packed_k.shape) == [
                NUM_HEADS, PADDED_HEAD_DIM // 16, PADDED_KEY_LEN, 16],
            "query_chunks": [list(c) for c in query_chunks()],
            "extra_bytes": padded_buffer_bytes(),
            "launch_counts": planned_launch_counts(),
            "no_tolerance_chosen": True,
        }

    def snapshot(self) -> dict:
        return {
            "round": 88,
            "installed": bool(self._installed),
            "enabled": bool(self._enabled),
            "prepared": len(self._metas) == len(planned_kernel_shapes()),
            "helper_name": HELPER_NAME,
            "kernels_prepared": sorted(str(key) for key in self._metas),
            "stats": {key: (dict(value) if isinstance(value, dict) else value)
                      for key, value in self._stats.items()},
            "geometry": {"num_heads": NUM_HEADS,
                         "original_head_dim": ORIGINAL_HEAD_DIM,
                         "padded_head_dim": PADDED_HEAD_DIM,
                         "original_key_len": ORIGINAL_KEY_LEN,
                         "padded_key_len": PADDED_KEY_LEN,
                         "query_chunk": QUERY_CHUNK,
                         "original_scale": float(ORIGINAL_SCALE),
                         "residual_scale": RESIDUAL_SCALE,
                         "unmasked": True, "causal": False},
            "extra_buffer_bytes": padded_buffer_bytes(),
            "planned_launch_counts": planned_launch_counts(),
            "one_half_cast_per_chunk": True,
            "float_through_both_pv_products_before_cast": True,
            "no_global_monkeypatch": True,
            "text_sdpa_symbol_untouched": True,
            "no_model_weight_shadow": True,
            "kernel_not_pv_fused": True,
            "pinned": {"kernel_module": PINNED_KERNEL_MODULE,
                       "kernel_module_sha256": PINNED_KERNEL_SHA256,
                       "runtime_module": PINNED_RUNTIME_MODULE,
                       "runtime_module_sha256": PINNED_RUNTIME_MODULE_SHA256,
                       "runtime_library_sha256": PINNED_RUNTIME_LIBRARY_SHA256,
                       "candidate_sha256": PINNED_CANDIDATE_SHA256,
                       "helper_source_sha256": PINNED_HELPER_SOURCE_SHA256},
            "source_sha256": _sha256_file(__file__),
            "full_encoder_precision_accepted": False,
            "whole_goal_complete": False,
        }

    def uninstall(self) -> dict:
        """Restore the captured original helper; idempotent, never blind."""
        if self._installed:
            current = getattr(self._candidate, HELPER_NAME, None)
            # Bound-method identity, not ``is``: a fresh attribute access would
            # produce a new bound-method object for the same function.
            is_ours = (getattr(current, "__self__", None) is self
                       and getattr(current, "__func__", None) is type(self).forward)
            if not is_ours:
                raise VisionAttentionError(
                    f"{HELPER_NAME} was replaced by a third party after install; the original "
                    "was not restored blindly")
            setattr(self._candidate, HELPER_NAME, self._original)
            self._installed = False
        self._enabled = False
        self._metas.clear()
        self._aux = None
        summary = self.snapshot()
        summary["uninstalled"] = True
        return summary


def install(candidate, runtime, kernel_dir) -> AccurateVisionAttention88:
    """Factory: install the gated engine on the candidate's vision SDPA helper.

    Returns an engine with ``enabled == False``; the caller must ``prepare()`` and
    then ``set_enabled(True)`` for the accurate path.
    """
    return AccurateVisionAttention88(candidate, runtime, kernel_dir)


def planned_geometries() -> list:
    """Plain description of the four planned binaries (for manifests/tests)."""
    return [{"batch": batch, "m": m, "k": k, "n": n,
             "m_pad": _ceil_to(m, 128),
             "role": "score" if k == PADDED_HEAD_DIM else "pv"}
            for batch, m, k, n in planned_kernel_shapes()]