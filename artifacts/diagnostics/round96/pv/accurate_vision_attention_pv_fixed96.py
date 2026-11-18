#!/usr/bin/env python3
"""Round96 gated accurate vision SDPA with fused two-Half-probability P@V.

SOURCE ONLY. This file has not been imported on an NPU, compiled, executed against
a model, installed into production or committed by its author. There is no
numerical result, no tolerance, no speed claim and no whole-encoder acceptance
here; the root driver owns every receipt.

WHAT CHANGES COMPARED WITH ROUND88
----------------------------------
The executed Round88 engine already made the vision attention leaf exact by
construction (padded Half operands, one FP32 L0C accumulator per product, Float32
softmax, two-Half probability split). Its remaining cost is structural: per query
chunk it issues **three** kernel launches (one QK plus two separate P@V) and brings
**two** FP32 P@V products back to the host, where torch multiplies the low product
by ``1/1024``, adds, and only then narrows once to Half.

This engine keeps the identical arithmetic contract but fuses the two P@V products
into ONE kernel launch (``pv_high_low96``): two independent FP32 L0C accumulators
over the full K axis, a Float32 UB scale+add, one ``h_cast('none')`` and one Half
store. Per call that is **two** launches and **one** Half-size product buffer per
chunk instead of three launches, two FP32 buffers, a host multiply, a host add and
a host cast.

Unchanged on purpose
--------------------
* The original Half q/k/v are only ever copied unchanged into zero-padded operands:
  no Float32 promotion, no Float32 weight shadow, no BF16 flag, no teacher.
* Head dim 72 -> 128 zero padding, key axis 4032 -> 4096 zero padding, scores
  cropped to the true 4032 keys **before** the unmasked Float32 softmax, and the
  original ``72 ** -0.5`` scale (never ``128 ** -0.5``).
* ``P_hi = Half(P)`` and ``P_lo = Half((P - Float(P_hi)) * 1024)``, combined as
  ``Float(hi) + Float(lo) / 1024`` then narrowed exactly once.
* Only the candidate module's own ``_vision_sdpa_bf16_compatible`` attribute is
  scoped. ``torch.nn.functional``, the text SDPA symbol, the biased linears, the
  MLP and the merger are never touched.

SCOPE / NON-CLAIMS
------------------
Source only. The parent driver captures real vision q/k/v, builds the CPU Half
MATH reference and compares leaves and the whole vision tower. Fewer launches and
less host traffic are a *structural* expectation, not a measured speedup, and the
fused kernel's Float32 combine is expected to match the executed Round88 host
combine bit-for-bit only if the vector unit rounds the same way - which the root
must prove, not assume.
"""

from __future__ import annotations

import ast
import hashlib
from pathlib import Path
import textwrap

# --- pinned immutable inputs -------------------------------------------------
PINNED_KERNEL_MODULE = "batch_gemm85"
PINNED_KERNEL_SHA256 = "546cd211ddb3938c3d8dac9d5207a79646c1458a3887df7e8c8562426069854a"
PINNED_RUNTIME_MODULE = "runtime5gm83"
PINNED_RUNTIME_MODULE_SHA256 = (
    "c8890dbcd825fe9177c7ee8d5c6b46e4d00231d4d61e6421f83491fec85c52a6")
PINNED_RUNTIME_LIBRARY_SHA256 = (
    "bd50d7005bf48e9a9a853c01750eaea4eefbb3e0f69abb2e7153aa7139e8b043")
PINNED_CANDIDATE_SHA256 = (
    "2141b29adc9ecc601d7fa2599e6ba56351adfbdaa5f16acb62c00d95cee3cf78")
PINNED_HELPER_SOURCE_SHA256 = (
    "6393bee7067d4854a7f4d47f192d0b3b6673acfc7cf951e6c86eb52b5a39d2e5")
PINNED_PV_MODULE = "pv_high_low96"
HELPER_NAME = "_vision_sdpa_bf16_compatible"
#: The fused kernel's five GM pointers, in the exact order the bridge passes them.
EXPECTED_PV_ARG_ORDER = ["hi_gm_float16", "lo_gm_float16", "v_gm_float16",
                         "aux_gm_float32", "out_gm_float16"]

# --- real H3 vision attention geometry (validated against the checkpoint) -----
NUM_HEADS = 16
ORIGINAL_HEAD_DIM = 72
PADDED_HEAD_DIM = 128
ORIGINAL_KEY_LEN = 4032
PADDED_KEY_LEN = 4096
QUERY_CHUNK = 512
ORIGINAL_SCALE = ORIGINAL_HEAD_DIM ** -0.5
RESIDUAL_SCALE = 1024.0
HALF_DTYPE_NAME = "torch.float16"
NPU_DEVICE_PREFIXES = ("npu", "privateuseone")
EXPECTED_ACTIVATION_SHAPE = (1, NUM_HEADS, ORIGINAL_KEY_LEN, ORIGINAL_HEAD_DIM)


class VisionAttentionPvError(RuntimeError):
    """Raised instead of guessing, silently falling back or editing the graph."""


def _ceil_to(value: int, multiple: int) -> int:
    return ((value + multiple - 1) // multiple) * multiple


def query_chunks(total: int = ORIGINAL_KEY_LEN, chunk: int = QUERY_CHUNK) -> tuple:
    """``((start, end, m), ...)``: seven full 512-query chunks plus one 448 tail."""
    if not isinstance(total, int) or isinstance(total, bool) or total <= 0:
        raise VisionAttentionPvError(f"total must be a positive int, got {total!r}")
    if not isinstance(chunk, int) or isinstance(chunk, bool) or chunk <= 0:
        raise VisionAttentionPvError(f"chunk must be a positive int, got {chunk!r}")
    return tuple((start, min(start + chunk, total), min(start + chunk, total) - start)
                 for start in range(0, total, chunk))


def planned_launch_counts() -> dict:
    """Launches one enabled call must issue: QK + ONE fused PV per query chunk."""
    chunks = len(query_chunks())
    return {"query_chunks": chunks, "pack_w_calls": 2, "pack_a_pv_calls": 2,
            "kernel_launches": chunks * 2, "qk_launches": chunks,
            "fused_pv_launches": chunks, "half_output_chunks": chunks,
            "float_before_half_chunks": chunks,
            "round88_launches_without_fusion": chunks * 3}


def planned_qk_shapes() -> tuple:
    """``(batch, m, k, n)`` for the two QK binaries (head dim x padded keys)."""
    return tuple((NUM_HEADS, m, PADDED_HEAD_DIM, PADDED_KEY_LEN)
                 for m in sorted({m for _s, _e, m in query_chunks()}, reverse=True))


def planned_pv_shapes() -> tuple:
    """``(batch, m, k, n)`` for the two fused-PV binaries (padded keys x head dim)."""
    return tuple((NUM_HEADS, m, PADDED_KEY_LEN, PADDED_HEAD_DIM)
                 for m in sorted({m for _s, _e, m in query_chunks()}, reverse=True))


def check_activation_spec(shape, dtype_name, device_type, device_index, runtime_device) -> dict:
    """Fail-closed structural check of one leaf's q/k/v (torch-free, testable)."""
    dims = tuple(int(v) for v in shape)
    if dims != EXPECTED_ACTIVATION_SHAPE:
        raise VisionAttentionPvError(
            f"vision SDPA operands are {dims}, expected {EXPECTED_ACTIVATION_SHAPE}; this "
            "engine is fixed to the real 16-head/4032-key/72-dim vision geometry")
    if dtype_name != HALF_DTYPE_NAME:
        raise VisionAttentionPvError(
            f"vision SDPA operands are {dtype_name}, expected {HALF_DTYPE_NAME}")
    if device_type not in NPU_DEVICE_PREFIXES:
        raise VisionAttentionPvError(
            f"vision SDPA operands are on {device_type!r}, expected an Ascend NPU device")
    if int(runtime_device) < 0:
        raise VisionAttentionPvError("runtime object reports no device index")
    if device_index is not None and int(device_index) != int(runtime_device):
        raise VisionAttentionPvError(
            f"vision SDPA operands are on device {device_index} but the runtime owns "
            f"device {runtime_device}")
    return {"shape": list(dims), "dtype": dtype_name, "device_type": device_type,
            "device_index": None if device_index is None else int(device_index),
            "runtime_device": int(runtime_device)}


def padded_buffer_bytes() -> dict:
    """Deterministic operand/transient byte accounting for one enabled call."""
    h, m_pad = NUM_HEADS, QUERY_CHUNK
    q = h * ORIGINAL_KEY_LEN * PADDED_HEAD_DIM * 2
    kv = h * PADDED_KEY_LEN * PADDED_HEAD_DIM * 2
    vt = h * PADDED_HEAD_DIM * PADDED_KEY_LEN * 2
    packed_k = h * (PADDED_KEY_LEN // 16) * PADDED_HEAD_DIM * 16 * 2
    packed_v = h * (PADDED_KEY_LEN // 16) * PADDED_HEAD_DIM * 16 * 2
    score_c = h * (PADDED_KEY_LEN // 16) * m_pad * 16 * 4
    out_half_gm = h * (PADDED_HEAD_DIM // 16) * m_pad * 16 * 2
    pack_a_score = h * (PADDED_HEAD_DIM // 16) * m_pad * 16 * 2
    pack_a_pv = h * (PADDED_KEY_LEN // 16) * m_pad * 16 * 2
    chunk_p = h * m_pad * PADDED_KEY_LEN * 2
    return {
        "padded_q_half": q, "padded_k_half": kv, "padded_v_half": kv,
        "transposed_v_half": vt, "packed_k_half": packed_k, "packed_v_half": packed_v,
        "score_fp32_gm": score_c, "fused_pv_half_gm": out_half_gm,
        "pack_a_score_half": pack_a_score, "pack_a_pv_half": pack_a_pv,
        "p_hi_or_lo_padded_half": chunk_p,
        "call_persistent_total": q + kv + kv + vt + packed_k + packed_v,
        "per_chunk_transient_total": (score_c + out_half_gm + pack_a_score + pack_a_pv
                                      + 2 * chunk_p),
        "round88_per_chunk_transient_total": (score_c + h * (PADDED_HEAD_DIM // 16) * m_pad
                                             * 16 * 4 + pack_a_score + pack_a_pv
                                             + 2 * chunk_p),
        "note": "the fused PV product returns one Half buffer instead of two FP32 buffers; "
                "this is byte accounting, not a measured speedup",
    }


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
    raise VisionAttentionPvError(f"{HELPER_NAME} not found in {candidate_file}")


class AccurateVisionAttentionPv96:
    """Gated accurate vision SDPA with one fused two-Half-probability P@V launch."""

    def __init__(self, candidate, runtime, kernel_dir):
        self._candidate = candidate
        self._runtime = runtime
        self._kernel_dir = Path(kernel_dir)
        original = getattr(candidate, HELPER_NAME, None)
        if not callable(original):
            raise VisionAttentionPvError(
                f"candidate module has no callable {HELPER_NAME}; refusing to patch an "
                "unknown graph")
        self._original = original
        self._enabled = False
        self._installed = True
        self._qk_metas: dict = {}
        self._pv_metas: dict = {}
        self._aux = None
        self._stats = {
            "calls_enabled": 0, "calls_disabled": 0, "calls_rejected": 0,
            "kernel_launches": 0, "qk_launches": 0, "fused_pv_launches": 0,
            "pack_w_calls": 0, "pack_a_pv_calls": 0, "query_chunks_processed": 0,
            "half_output_chunks": 0, "float_before_half_chunks": 0,
            "kernel_shapes_qk": {}, "kernel_shapes_pv": {},
            "device": int(getattr(runtime, "device", -1)),
        }
        # Scoped assignment on the isolated diagnostic module only.
        setattr(candidate, HELPER_NAME, self.forward)

    # -- preparation -------------------------------------------------------

    def _verify_sources(self) -> dict:
        import sys

        import batch_gemm85 as kernel

        receipt = {}
        kernel_sha = _sha256_file(kernel.__file__)
        if kernel_sha != PINNED_KERNEL_SHA256:
            raise VisionAttentionPvError(
                f"{PINNED_KERNEL_MODULE} changed: {kernel_sha} != {PINNED_KERNEL_SHA256}; the "
                "executed QK FP32-accumulator arithmetic this engine relies on no longer matches")
        receipt["kernel_module"] = {"path": str(kernel.__file__), "sha256": kernel_sha}

        runtime_module = sys.modules.get(type(self._runtime).__module__)
        runtime_file = getattr(runtime_module, "__file__", None)
        if not runtime_file:
            raise VisionAttentionPvError(
                f"cannot locate the {PINNED_RUNTIME_MODULE} module to pin its bridge ABI")
        runtime_sha = _sha256_file(runtime_file)
        if runtime_sha != PINNED_RUNTIME_MODULE_SHA256:
            raise VisionAttentionPvError(
                f"{PINNED_RUNTIME_MODULE} changed: {runtime_sha} != "
                f"{PINNED_RUNTIME_MODULE_SHA256}")
        if int(getattr(self._runtime, "device", -1)) < 0:
            raise VisionAttentionPvError("runtime object reports no device index")
        if self._runtime.snapshot().get('library_sha256') != PINNED_RUNTIME_LIBRARY_SHA256:
            raise VisionAttentionPvError('runtime launcher library no longer matches its pin')
        receipt["runtime_module"] = {"path": str(runtime_file), "sha256": runtime_sha,
                                     "library_sha256": PINNED_RUNTIME_LIBRARY_SHA256,
                                     "ABI": "fiveGMpointers"}

        candidate_file = getattr(self._candidate, "__file__", None)
        if not candidate_file:
            raise VisionAttentionPvError("candidate module exposes no __file__ to pin")
        candidate_sha = _sha256_file(candidate_file)
        if candidate_sha != PINNED_CANDIDATE_SHA256:
            raise VisionAttentionPvError(
                f"candidate source changed: {candidate_sha} != {PINNED_CANDIDATE_SHA256}")
        digest = helper_source_digest(candidate_file)
        if digest != PINNED_HELPER_SOURCE_SHA256:
            raise VisionAttentionPvError(
                f"{HELPER_NAME} source changed: {digest} != {PINNED_HELPER_SOURCE_SHA256}")
        receipt["candidate"] = {"path": str(candidate_file), "sha256": candidate_sha,
                                "helper_source_sha256": digest,
                                "matches_pin": True}
        return receipt

    def _verify_pv_module(self):
        import pv_high_low96 as pv

        receipt = {"module": PINNED_PV_MODULE,
                   "path": str(Path(pv.__file__).resolve()),
                   "sha256": _sha256_file(pv.__file__),
                   "fused_single_launch": True,
                   "kernel_arg_order": list(pv.KERNEL_ARG_ORDER),
                   "GM_argument_count": 5}
        if receipt["kernel_arg_order"] != EXPECTED_PV_ARG_ORDER:
            raise VisionAttentionPvError(
                "pv module ABI order changed; the fused launch would pass the wrong operand")
        return pv, receipt

    def prepare(self) -> dict:
        """Build and pin the two QK and two fused-PV binaries; disabled afterwards."""
        import torch

        receipt = {"sources": self._verify_sources()}
        pv, pv_receipt = self._verify_pv_module()
        receipt["pv_module"] = pv_receipt
        receipt["shapes_qk"] = [list(s) for s in planned_qk_shapes()]
        receipt["shapes_pv"] = [list(s) for s in planned_pv_shapes()]
        receipt["built"] = {"qk": [], "pv": []}
        self._qk_metas.clear()
        self._pv_metas.clear()

        for batch, m, k, n in planned_qk_shapes():
            meta = self._build_qk(batch, m, k, n)
            self._qk_metas[(m, k, n)] = meta
            receipt["built"]["qk"].append({key: meta[key] for key in (
                "kernel_name", "kernel_symbol", "block_dim", "obj_sha256",
                "logical_shape", "dst_dtype")})
        for batch, m, k, n in planned_pv_shapes():
            meta = pv.build_kernel(batch, m, k, n, out_dir=self._kernel_dir / "pv")
            self._check_pv_meta(meta, batch, m, k, n)
            self._pv_metas[(m, k, n)] = meta
            receipt["built"]["pv"].append({key: meta[key] for key in (
                "kernel_name", "kernel_symbol", "block_dim", "obj_sha256",
                "logical_shape", "dst_dtype", "single_final_half_cast",
                "no_mid_product_fp32_gm", "no_host_combine", "kernel_arg_order")})
        receipt["launch_plan"] = planned_launch_counts()
        receipt["bytes"] = padded_buffer_bytes()
        receipt["default_disabled"] = not self._enabled
        receipt["not_a_speed_claim"] = True
        receipt["not_a_precision_result"] = True
        return receipt

    def _build_qk(self, batch, m, k, n):
        import batch_gemm85 as kernel

        meta = kernel.build_kernel(batch, m, k, n, out_dir=self._kernel_dir / "qk")
        expected = {"batch": batch, "m": m, "k": k, "n": n}
        if meta.get("logical_shape") != expected:
            raise VisionAttentionPvError(
                f"QK kernel geometry {meta.get('logical_shape')} != {expected}")
        if str(meta.get("dst_dtype")) != "float32":
            raise VisionAttentionPvError("QK kernel output is not the expected float32 scores")
        return meta

    @staticmethod
    def _check_pv_meta(meta, batch, m, k, n) -> None:
        if meta.get("logical_shape") != {"batch": batch, "m": m, "k": k, "n": n}:
            raise VisionAttentionPvError(
                f"fused PV kernel geometry {meta.get('logical_shape')} != "
                f"{(batch, m, k, n)}")
        required = {"dst_dtype": "float16", "input_dtype": "float16",
                    "accumulator_dtype": "float32", "GM_argument_count": 5,
                    "single_final_half_cast": True, "no_mid_product_fp32_gm": True,
                    "no_host_combine": True, "fused_products": ["high", "low"]}
        for key, want in required.items():
            if meta.get(key) != want:
                raise VisionAttentionPvError(
                    f"fused PV kernel {key}={meta.get(key)!r}, expected {want!r}")
        if list(meta["kernel_arg_order"]) != EXPECTED_PV_ARG_ORDER:
            raise VisionAttentionPvError(
                f"fused PV kernel argument order {meta['kernel_arg_order']} != "
                f"{EXPECTED_PV_ARG_ORDER}; the five GM pointers would be mismatched")
        if not meta.get("one_kernel_launch_per_tile"):
            raise VisionAttentionPvError("fused PV kernel is not a single-launch contract")
        if not meta.get("no_half_partial"):
            raise VisionAttentionPvError("fused PV kernel reports a Half partial product")
        if int(meta.get("block_dim", -1)) != int(meta.get("grid", {}).get("cores", -2)):
            raise VisionAttentionPvError("fused PV kernel block_dim disagrees with its grid")

    # -- enable / disable ---------------------------------------------------

    def set_enabled(self, enabled: bool) -> bool:
        if not isinstance(enabled, bool):
            raise VisionAttentionPvError("set_enabled() takes a bool")
        if enabled and len(self._qk_metas) != len(planned_qk_shapes()):
            raise VisionAttentionPvError("set_enabled(True) requires a successful prepare()")
        if enabled and len(self._pv_metas) != len(planned_pv_shapes()):
            raise VisionAttentionPvError("set_enabled(True) requires a successful prepare()")
        self._enabled = enabled
        return self._enabled

    # -- device work -------------------------------------------------------

    def _aux_buffer(self, device):
        import torch

        if self._aux is None or self._aux.device != device:
            # Unused five-GM placeholder (the fused kernel declares it float32).
            self._aux = torch.zeros((16,), dtype=torch.float32, device=device)
        return self._aux

    def _run_qk(self, a, packed_weight, n):
        """Unchanged Round85 QK product: Half in, one FP32 L0C over K, FP32 GM out."""
        import torch

        import batch_gemm85 as kernel

        h, m, k = (int(v) for v in a.shape)
        meta = self._qk_metas[(m, k, n)]
        packed_a = kernel.pack_a(a)
        c = torch.empty(meta["gm_shapes"]["c"], dtype=torch.float32, device=a.device)
        aux = self._aux_buffer(a.device)
        self._runtime.launch(meta, [packed_a, packed_weight, aux, aux, c], synchronize=True)
        self._stats["kernel_launches"] += 1
        self._stats["qk_launches"] += 1
        key = f"h{h}_m{m}_k{k}_n{n}"
        self._stats["kernel_shapes_qk"][key] = int(
            self._stats["kernel_shapes_qk"].get(key, 0)) + 1
        out = kernel.unpack_c(c, m=m)
        if out.dtype != torch.float32:
            raise VisionAttentionPvError(
                f"QK output is {out.dtype}, expected float32 before the Float softmax")
        return out

    def _run_fused_pv(self, p_high, p_low, packed_v, m):
        """ONE launch: two FP32 L0C products over all K, Float combine, one Half cast."""
        import torch

        import pv_high_low96 as pv

        h, mm, k = (int(v) for v in p_high.shape)
        if mm != m:
            raise VisionAttentionPvError(f"probability rows {mm} != chunk m={m}")
        meta = self._pv_metas[(m, k, PADDED_HEAD_DIM)]
        packed_hi = pv.pack_a(p_high)
        packed_lo = pv.pack_a(p_low)
        self._stats["pack_a_pv_calls"] += 2
        out_gm = torch.empty(meta["gm_shapes"]["out"], dtype=torch.float16,
                            device=p_high.device)
        self._runtime.launch(meta, [packed_hi, packed_lo, packed_v,
                                    self._aux_buffer(p_high.device), out_gm],
                             synchronize=True)
        self._stats["kernel_launches"] += 1
        self._stats["fused_pv_launches"] += 1
        key = f"h{h}_m{m}_k{k}_n{PADDED_HEAD_DIM}"
        self._stats["kernel_shapes_pv"][key] = int(
            self._stats["kernel_shapes_pv"].get(key, 0)) + 1
        out = pv.unpack_out_half(out_gm, m=m, n=PADDED_HEAD_DIM)
        if out.dtype != torch.float16:
            raise VisionAttentionPvError(
                f"fused PV output is {out.dtype}, expected the single Half store")
        return out

    def forward(self, q, k, v):
        if not self._enabled:
            self._stats["calls_disabled"] += 1
            return self._original(q, k, v)
        import torch

        import batch_gemm85 as kernel

        if len(self._qk_metas) != len(planned_qk_shapes()) or \
                len(self._pv_metas) != len(planned_pv_shapes()):
            self._stats["calls_rejected"] += 1
            raise VisionAttentionPvError("fused accurate vision attention used before prepare()")
        spec = check_activation_spec(tuple(q.shape), str(q.dtype), q.device.type,
                                     q.device.index, int(getattr(self._runtime, "device", -1)))
        for name, tensor in (("k", k), ("v", v)):
            if (tuple(int(x) for x in tensor.shape) != tuple(spec["shape"])
                    or str(tensor.dtype) != HALF_DTYPE_NAME
                    or tensor.device.type != q.device.type
                    or tensor.device.index != q.device.index):
                self._stats["calls_rejected"] += 1
                raise VisionAttentionPvError(
                    f"vision SDPA operand {name} does not match q")
        self._stats["calls_enabled"] += 1
        launches_before = self._stats["kernel_launches"]
        device = q.device

        # 1. Copy the original Half values into the zero-padded operands (exact).
        qp = torch.zeros((NUM_HEADS, ORIGINAL_KEY_LEN, PADDED_HEAD_DIM),
                         dtype=torch.float16, device=device)
        qp[..., :ORIGINAL_HEAD_DIM] = q[0]
        kp = torch.zeros((NUM_HEADS, PADDED_KEY_LEN, PADDED_HEAD_DIM),
                         dtype=torch.float16, device=device)
        kp[:, :ORIGINAL_KEY_LEN, :ORIGINAL_HEAD_DIM] = k[0]
        vp = torch.zeros((NUM_HEADS, PADDED_KEY_LEN, PADDED_HEAD_DIM),
                         dtype=torch.float16, device=device)
        vp[:, :ORIGINAL_KEY_LEN, :ORIGINAL_HEAD_DIM] = v[0]

        # 2. Pack B operands once per call (never across layers/inputs).
        packed_k = kernel.pack_w(kp)
        packed_v = kernel.pack_w(vp.transpose(1, 2).contiguous())
        self._stats["pack_w_calls"] += 2

        result = torch.empty((NUM_HEADS, ORIGINAL_KEY_LEN, ORIGINAL_HEAD_DIM),
                             dtype=q.dtype, device=device)
        for start, end, m in query_chunks():
            chunk = qp[:, start:end, :]
            scores = self._run_qk(chunk, packed_k, PADDED_KEY_LEN)
            # 3. Crop padded keys BEFORE the softmax: a padded key scores exactly 0
            #    and exp(0)=1 would corrupt the denominator. UNMASKED softmax.
            scores = scores[:, :, :ORIGINAL_KEY_LEN] * ORIGINAL_SCALE
            p = torch.softmax(scores, dim=-1)
            if p.dtype != torch.float32:
                self._stats["calls_rejected"] += 1
                raise VisionAttentionPvError(
                    f"probabilities are {p.dtype}, expected float32")

            # 4. Float P -> (Half high, 1024-scaled Half residual), zero key padding.
            high = p.to(torch.float16)
            low = ((p - high.float()) * RESIDUAL_SCALE).to(torch.float16)
            p_high = torch.zeros((NUM_HEADS, m, PADDED_KEY_LEN),
                                 dtype=torch.float16, device=device)
            p_high[:, :, :ORIGINAL_KEY_LEN] = high
            p_low = torch.zeros_like(p_high)
            p_low[:, :, :ORIGINAL_KEY_LEN] = low

            # 5. ONE fused launch: hi@V and lo@V in two FP32 L0C, Float scale+add,
            #    one Half cast, one Half GM store. Padded key rows of P and padded
            #    key columns of V are both exactly zero, so those terms vanish.
            fused = self._run_fused_pv(p_high, p_low, packed_v, m)
            self._stats["half_output_chunks"] += 1
            self._stats["float_before_half_chunks"] += 1
            self._stats["query_chunks_processed"] += 1
            result[:, start:end, :] = fused[:, :, :ORIGINAL_HEAD_DIM]
            del p_high, p_low, high, low, p, scores

        issued = self._stats["kernel_launches"] - launches_before
        expected = planned_launch_counts()["kernel_launches"]
        if issued != expected:
            self._stats["calls_rejected"] += 1
            raise VisionAttentionPvError(
                f"enabled call issued {issued} kernel launches, expected exactly {expected}")
        return result.unsqueeze(0)

    # -- reporting / restore ------------------------------------------------

    def padding_selfcheck(self) -> dict:
        zero = None
        return {
            "head_dim_padded": [ORIGINAL_HEAD_DIM, PADDED_HEAD_DIM],
            "key_len_padded": [ORIGINAL_KEY_LEN, PADDED_KEY_LEN],
            "m_padded_by_kernel": [QUERY_CHUNK, _ceil_to(QUERY_CHUNK, 128)],
            "query_chunks": [list(chunk) for chunk in query_chunks()],
            "score_cropped_before_softmax": True,
            "softmax_masked": False,
            "original_scale": ORIGINAL_SCALE,
            "padded_width_scale_used": False,
            "probability_high_low_residual_scale": RESIDUAL_SCALE,
            "operand_padding_is_zero": True,
            "score_pad_magnitude": zero,
            "reference_softmax_denominator_corrupted": False,
        }

    def snapshot(self) -> dict:
        import sys

        pv_module = sys.modules.get(PINNED_PV_MODULE)
        return {
            "round": 96,
            "installed": bool(self._installed), "enabled": bool(self._enabled),
            "default_disabled": True, "per_instance_patch_only": True,
            "patches_torch_global": False, "patches_torch_functional": False,
            "patches_candidate_class": False, "patched_helper": HELPER_NAME,
            "fused_pv_single_launch": True,
            "pv_module": None if pv_module is None else {
                "path": str(Path(pv_module.__file__).resolve()),
                "sha256": _sha256_file(pv_module.__file__)},
            "kernel_arg_order": None if pv_module is None else list(pv_module.KERNEL_ARG_ORDER),
            "padding": self.padding_selfcheck(),
            "launch_plan": planned_launch_counts(),
            "qk_kernels": len(self._qk_metas), "pv_kernels": len(self._pv_metas),
            "stats": dict(self._stats),
            "stats_kernel_shapes_qk": dict(self._stats["kernel_shapes_qk"]),
            "stats_kernel_shapes_pv": dict(self._stats["kernel_shapes_pv"]),
            "bytes": padded_buffer_bytes(),
            "original_helper_preserved": callable(self._original),
            "no_fp32_weight_shadow": True, "no_bf16_teacher": True,
            "float32_combine_in_vector_unit": True,
            "host_combine_removed": True,
            "full_encoder_precision_accepted": False, "whole_goal_complete": False,
            "numerical_tolerance": None, "not_a_speed_claim": True,
        }

    def uninstall(self) -> dict:
        if self._installed:
            current = getattr(self._candidate, HELPER_NAME, None)
            ours = getattr(current, "__self__", None) is self
            if not (current == self.forward or ours):
                raise VisionAttentionPvError(
                    f"candidate.{HELPER_NAME} was replaced by a third party; the original "
                    "was not restored blindly")
            setattr(self._candidate, HELPER_NAME, self._original)
        self._installed = False
        self._enabled = False
        self._qk_metas.clear()
        self._pv_metas.clear()
        self._aux = None
        summary = self.snapshot()
        summary["uninstalled"] = True
        return summary


# Public acronym spelling used by the root integration driver.
AccurateVisionAttentionPV96 = AccurateVisionAttentionPv96


def install(candidate, runtime, kernel_dir) -> AccurateVisionAttentionPv96:
    """Factory: install the fused engine disabled; caller must prepare() then enable."""
    return AccurateVisionAttentionPv96(candidate, runtime, kernel_dir)