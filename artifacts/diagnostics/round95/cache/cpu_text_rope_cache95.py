#!/usr/bin/env python3
"""Round95 per-instance original-HF CPU Half text M-RoPE table cache.

SOURCE ONLY. Nothing here has been executed by its author against torch,
torch_npu, an NPU, the HF model or CANN: no SSH, no model run, no kernel build,
no installation, no production edit, no commit. Every real number belongs to the
root agent that primes the cache and measures the complete encoder.

WHAT THIS CACHES, AND WHY IT IS NOT A TEACHER
---------------------------------------------
``prime(x, position_ids)`` computes the **original HuggingFace CPU**
``Qwen3VLTextRotaryEmbedding`` cos/sin table for the pinned real M-RoPE positions
``[[3,1,2043]]`` (R78 tensor SHA pinned) with an *empty metadata-only* CPU Half
tensor whose content is never read (the original HF forward only consults
``x.dtype`` and ``x.device``). The table is computed twice and must be raw-byte
identical, then copied bit-exactly to the NPU. No hidden state, embedding,
pixel, patch, DeepStack or image conditioning value is copied, cached or used as
a teacher. No Linear weight is changed or shadowed in Float, and no BF16 gold
output is cast anywhere.

Honest labelling
----------------
The native R79 NPU Half text cos/sin and this original CPU Half table are **not
bit-identical**: Round94 measured cos relative RMS 7.1403e-6 (max exactly one
Half ULP 0.00048828125) and sin 2.80e-8. This cache is therefore a *precision
alignment* to the original HF CPU table, **not** a lossless native-bit cache.
``snapshot()`` reports ``native_rounding_changed: True``, measured
``native_equals_CPU_table_bits`` and no tolerance waiver. The absolute frequency
math (R79 Float direct multiply) is already correct and is not duplicated here.

FAIL-CLOSED STRUCTURE
---------------------
* Candidate provenance: ``candidate_encoder79.py`` file SHA, the reviewed rotary
  class name/module, and line-based source digests of ``__init__``,
  ``_get_inv_freq`` and ``forward`` are pinned.
* Reference provenance: the original HF modeling source file SHA is pinned.
* Config: ``dim``, ``mrope_section``, pinned CPU ``inv_freq`` SHA, default
  ``rope_type``, ``attention_scaling == 1.0`` and zero learnable parameters are
  required; the candidate ``_get_inv_freq(cpu)`` must be raw-byte equal to the
  original HF CPU inverse frequencies.
* Real fixture: NPU positions must be int64 ``(3,1,2043)`` whose values hash to
  the pinned R78 tensor; the CPU table must reproduce the R94-measured original
  CPU cos/sin SHAs.
* Per enabled call: config/source/positions/dtype/device/shape drift and cached
  tensor mutation raise. An inference tensor without a version counter is always
  value-validated; a pointer reuse or a new object is never served from a fast
  path.
* Restore: the instance ``forward`` is put back only when it is still ours; a
  third-party patch is never clobbered. Class level, ``torch.nn.functional`` and
  private candidate functions are untouched; default is disabled/native.
"""
from __future__ import annotations

import ast
import hashlib
import json
import math
import sys
import time
from pathlib import Path
from types import MethodType

ROUND = 95
CANDIDATE_MODULE_NAME = "candidate_encoder79"
CANDIDATE_BASENAME = "candidate_encoder79.py"
ROTARY_CLASS_NAME = "MiniMaxH3Qwen3VLTextRotaryEmbedding"
FORWARD = "forward"

PINNED_CANDIDATE_SHA256 = (
    "2141b29adc9ecc601d7fa2599e6ba56351adfbdaa5f16acb62c00d95cee3cf78")
PINNED_CANDIDATE_METHOD_SHA256 = {
    "__init__": "2f00425ab44bfcf62d11af46e9a1fa83c4830d0705f76e9a8d0ace36044192dd",
    "_get_inv_freq": "0a11e01d19e63812988d48d668c193115cf14256ef23b925ec33a5597c75b4a2",
    "forward": "bcb2afebb7f6f3402bb2916f2322da491df4f082cb3d5d13c171908213cf7a35",
}
PINNED_HF_SOURCE_SHA256 = (
    "fcb5571ef95e4a6679af866ef9ef4dfa5cee017f9f91453daf28c7804c7a50fe")

PINNED_CPU_INV_FREQ_SHA256 = (
    "29fc897bbb1fda6b314c1bcffa10e39b067913f538e4f07ed7cbd223b84c4cbf")
PINNED_POSITION_IDS_SHA256 = (
    "05a92359cd6a171559fadc0185ea2c7619a1969f7d771bb89c6a7d4df31a814c")
PINNED_CPU_COS_SHA256 = (
    "f0480c8783ff7ec9663f4f79035019a00c17219152f9bab4210097317ab7f996")
PINNED_CPU_SIN_SHA256 = (
    "0c147aa3d84e46345c8ead59721a04ccd1fec602c01ec54e570006bd2cda0b08")
# Round94 actually measured native NPU cos/sin for this fixture; recorded as a
# cross-check only, never used as the cached value and never asserted equal.
R94_NATIVE_OBSERVED_SHA256 = {
    "cos": "7044e477f0f49b80ed78428430671e009cd688695582b6796187edd43522ae1c",
    "sin": "74c703b256c5e3838cbd8a641007e6b8c4d0089cb3f981c73a5a16da8430b64a",
}

EXPECTED_HIDDEN_SHAPE = (1, 2043, 5120)
EXPECTED_POSITION_SHAPE = (3, 1, 2043)
EXPECTED_TABLE_SHAPE = (1, 2043, 128)
EXPECTED_STATE_DTYPE = "torch.float16"
EXPECTED_POSITION_DTYPE = "torch.int64"
EXPECTED_DIM = 128
EXPECTED_MROPE_SECTION = (24, 20, 20)
EXPECTED_ROPE_TYPE = "default"
EXPECTED_ATTENTION_SCALING = 1.0
EXPECTED_TABLE_BYTES = 2 * 1 * 2043 * 128 * 2  # cos + sin, 1,046,016 B
NPU_DEVICE_TYPES = ("npu", "privateuseone")
INFERENCE_VERSION_ERROR = "Inference tensors do not track version counter"


class TextRopeCacheError(RuntimeError):
    """Raised when the reviewed table cache contract cannot be honoured."""


# ---------------------------------------------------------------------------
# small stdlib helpers (duck-typed tensor access; no torch imported here)
# ---------------------------------------------------------------------------
def _sha256_bytes(data):
    return hashlib.sha256(bytes(data)).hexdigest()


def _file_sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _default_candidate_path():
    here = Path(__file__).resolve().parent
    for candidate in (here / CANDIDATE_BASENAME, here.parent / CANDIDATE_BASENAME,
                      here.parent.parent / "round79" / "candidate" / CANDIDATE_BASENAME,
                      Path.cwd() / CANDIDATE_BASENAME):
        if candidate.is_file():
            return candidate
    raise TextRopeCacheError(
        "cannot locate the reviewed %s; stage it beside this module or pass "
        "candidate_path" % CANDIDATE_BASENAME)


def method_source_digests(path, class_name):
    """Line-based digests of a class's methods; stable across Python versions."""
    text = Path(path).read_text()
    tree = ast.parse(text)
    lines = text.splitlines()
    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef) and node.name == class_name:
            digests = {}
            for item in node.body:
                if isinstance(item, ast.FunctionDef):
                    segment = "\n".join(lines[item.lineno - 1:item.end_lineno]) + "\n"
                    digests[item.name] = _sha256_bytes(segment.encode())
            return digests
    raise TextRopeCacheError("class %s not found in %s" % (class_name, path))


def tensor_signature(value):
    """Cheap identity/version fingerprint; no data is read or copied."""
    device = getattr(value, "device", None)
    try:
        version = int(value._version)
    except RuntimeError as error:
        if INFERENCE_VERSION_ERROR not in str(error):
            raise
        version = None
    except AttributeError:
        version = None
    return {
        "data_ptr": int(value.data_ptr()),
        "storage_offset": int(value.storage_offset()),
        "shape": [int(size) for size in value.shape],
        "stride": [int(size) for size in value.stride()],
        "dtype": str(value.dtype),
        "device_type": str(getattr(device, "type", "")),
        "device_index": None if device is None or device.index is None else int(device.index),
        "version": version,
    }


def raw_bytes(value):
    """Exact raw little-endian bytes of a dense copy (outside the warm clock)."""
    dense = value.detach().cpu().contiguous()
    return bytes(dense.numpy().tobytes())


def tensor_raw_sha(value):
    return _sha256_bytes(raw_bytes(value))


def _device_type(value):
    return str(getattr(getattr(value, "device", None), "type", ""))


def _is_npu_device(value):
    return _device_type(value) in NPU_DEVICE_TYPES


def _device_identity(value):
    device = getattr(value, "device", None)
    return (str(getattr(device, "type", "")),
            None if device is None or device.index is None else int(device.index))


def _opt_int(config, name):
    value = getattr(config, name, None)
    return None if value is None else int(value)


def config_fingerprint(rotary):
    """Torch-free structural fingerprint of every RoPE-relevant field."""
    config = getattr(rotary, "config", None)
    if config is None:
        raise TextRopeCacheError("rotary module has no config")
    rope = dict(getattr(config, "rope_parameters", None) or {})
    base = float(getattr(rotary, "base"))
    mrope = [int(value) for value in getattr(rotary, "mrope_section")]
    return {
        "dim": int(getattr(rotary, "dim")),
        "base": base,
        "mrope_section": mrope,
        "rope_type": str(rope.get("rope_type", EXPECTED_ROPE_TYPE)),
        "rope_theta": float(rope.get("rope_theta", base)),
        "config_mrope_section": [int(value) for value in
                                 rope.get("mrope_section", mrope)],
        "max_position_embeddings": _opt_int(config, "max_position_embeddings"),
        "hidden_size": _opt_int(config, "hidden_size"),
        "num_attention_heads": _opt_int(config, "num_attention_heads"),
        "head_dim": _opt_int(config, "head_dim"),
    }


def pinned_config_gate(rotary):
    """Refuse anything but the reviewed default-rope H3 text geometry."""
    fingerprint = config_fingerprint(rotary)
    if fingerprint["dim"] != EXPECTED_DIM:
        raise TextRopeCacheError(
            "rotary dim is %r, expected %d" % (fingerprint["dim"], EXPECTED_DIM))
    if tuple(fingerprint["mrope_section"]) != EXPECTED_MROPE_SECTION:
        raise TextRopeCacheError("mrope_section is %r, expected %r"
                                 % (fingerprint["mrope_section"], EXPECTED_MROPE_SECTION))
    if tuple(fingerprint["config_mrope_section"]) != EXPECTED_MROPE_SECTION:
        raise TextRopeCacheError("config mrope_section is %r, expected %r"
                                 % (fingerprint["config_mrope_section"],
                                    EXPECTED_MROPE_SECTION))
    base = fingerprint["base"]
    if not math.isfinite(base) or base <= 1.0:
        raise TextRopeCacheError("rotary base %r is not a finite default rope base" % (base,))
    if fingerprint["rope_type"] != EXPECTED_ROPE_TYPE:
        raise TextRopeCacheError(
            "rope_type is %r; only the reviewed default rope is supported"
            % (fingerprint["rope_type"],))
    if fingerprint["rope_theta"] != base:
        raise TextRopeCacheError(
            "candidate base %r and config rope_theta %r disagree"
            % (base, fingerprint["rope_theta"]))
    if callable(getattr(rotary, "_get_inv_freq", None)) is False:
        raise TextRopeCacheError("candidate rotary has no _get_inv_freq")
    return fingerprint


def _torch():
    import torch
    return torch


def _hf_module():
    from transformers.models.qwen3_vl import modeling_qwen3_vl as hf
    path = Path(hf.__file__)
    actual = _file_sha(path)
    if actual != PINNED_HF_SOURCE_SHA256:
        raise TextRopeCacheError(
            "imported HF modeling source is not the reviewed original: %s" % actual)
    return hf, path


def _internal_tensor_info(value):
    import torch
    cpu = value.detach().cpu().contiguous()
    info = {
        "shape": [int(size) for size in cpu.shape],
        "dtype": str(cpu.dtype),
        "bytes": int(cpu.numel() * cpu.element_size()),
        "sha256": tensor_raw_sha(cpu),
        "finite": bool(torch.isfinite(cpu.float()).all()) if cpu.is_floating_point() else True,
    }
    if cpu.is_floating_point() and cpu.numel():
        info["max_abs"] = float(cpu.float().abs().max())
    return info


def _internal_compare(value, reference):
    """Byte-for-byte the staged full_row_fp32_fixed84.compare semantics."""
    import torch
    a = value.detach().cpu().contiguous()
    b = reference.detach().cpu().contiguous()
    if a.shape != b.shape and a.numel() == b.numel():
        b = b.reshape(a.shape)
    result = {"npu": _internal_tensor_info(a), "cpu": _internal_tensor_info(b)}
    if a.shape != b.shape:
        result.update({"shape_mismatch": True, "comparable": False, "raw_bits_equal": False})
        return result
    finite = bool(torch.isfinite(a.float()).all() and torch.isfinite(b.float()).all())
    result.update({"shape_mismatch": False, "comparable": True, "finite": finite,
                   "dtype_matches": a.dtype == b.dtype})
    if not finite or a.dtype != b.dtype:
        result["raw_bits_equal"] = False
        return result
    delta = a.double() - b.double()
    denominator = b.double().square().mean().sqrt().clamp_min(1e-30)
    result.update({
        "raw_bits_equal": bool(torch.equal(a.view(torch.uint8), b.view(torch.uint8))),
        "elements": int(a.numel()),
        "max_abs_error": float(delta.abs().max()),
        "RMSE": float(delta.square().mean().sqrt()),
        "relative_rms": float(delta.square().mean().sqrt() / denominator),
    })
    return result


# ---------------------------------------------------------------------------
# the cache
# ---------------------------------------------------------------------------
class CPUTextRopeCache95:
    """Bit-exact original-HF CPU Half text M-RoPE table for one rotary instance.

    The constructor verifies provenance/config and installs a per-instance
    ``forward`` wrapper that is **disabled** by default: while disabled every
    call delegates to the captured original. ``prime(x, position_ids)`` builds
    the table outside the warm clock; ``set_enabled(True)`` then serves it.
    """

    def __init__(self, rotary, candidate_path=None):
        self._rotary = rotary
        self._candidate_path = (Path(candidate_path) if candidate_path is not None
                                else _default_candidate_path())
        self._provenance = self._verify_provenance()
        self._structure = config_fingerprint(rotary)
        pinned_config_gate(rotary)
        if int(sum(parameter.numel() for parameter in rotary.parameters())) != 0:
            raise TextRopeCacheError("reviewed rotary is expected to have zero parameters")
        self._hf_source_path = None
        self._originals = {}
        self._wrappers = {}
        self._installed = False
        self._enabled = False
        self._primed = False
        self._cache = {}
        self._positions_object = None
        self._positions_signature = None
        self._positions_bytes = None
        self._x_metadata = None
        self._cpu_table = None
        self._native_probe = None
        self._prime_seconds = None
        self._prime_count = 0
        self._stats = {"calls": 0, "disabled_calls": 0, "hits": 0,
                       "position_fast_hits": 0, "position_value_rereads": 0,
                       "validation_failures": 0, "returned_cached_object": 0}
        self._install()

    # -- provenance -------------------------------------------------------
    def _verify_provenance(self):
        rotary_type = type(self._rotary)
        if getattr(rotary_type, "__module__", None) != CANDIDATE_MODULE_NAME:
            raise TextRopeCacheError(
                "rotary class module is %r, expected the reviewed %s"
                % (getattr(rotary_type, "__module__", None), CANDIDATE_MODULE_NAME))
        if getattr(rotary_type, "__name__", None) != ROTARY_CLASS_NAME:
            raise TextRopeCacheError(
                "rotary class is %r, expected %s"
                % (getattr(rotary_type, "__name__", None), ROTARY_CLASS_NAME))
        actual = _file_sha(self._candidate_path)
        if actual != PINNED_CANDIDATE_SHA256:
            raise TextRopeCacheError(
                "candidate source is not the reviewed file: %s != %s"
                % (actual, PINNED_CANDIDATE_SHA256))
        module = sys.modules.get(CANDIDATE_MODULE_NAME)
        module_file = getattr(module, "__file__", None)
        if module_file is not None and _file_sha(module_file) != PINNED_CANDIDATE_SHA256:
            raise TextRopeCacheError(
                "imported candidate module file differs from the pinned source")
        digests = method_source_digests(self._candidate_path, ROTARY_CLASS_NAME)
        for name, pinned in PINNED_CANDIDATE_METHOD_SHA256.items():
            if digests.get(name) != pinned:
                raise TextRopeCacheError(
                    "reviewed %s.%s source changed: %r" % (ROTARY_CLASS_NAME, name,
                                                          digests.get(name)))
        return {"candidate_path": str(self._candidate_path),
                "candidate_sha256": actual, "method_sha256": digests,
                "matches_pins": True, "no_reference_source_edited_by_this_round": True}

    def _revalidate_structure(self):
        """Fail closed if the reviewed config or source files drifted since prime."""
        if config_fingerprint(self._rotary) != self._structure:
            raise TextRopeCacheError(
                "rotary config changed after prime(); uninstall and build a new cache")
        if _file_sha(self._candidate_path) != PINNED_CANDIDATE_SHA256:
            raise TextRopeCacheError("candidate source changed after prime()")
        if self._hf_source_path is None:
            raise TextRopeCacheError("HF reference source path was not recorded at prime()")
        if _file_sha(self._hf_source_path) != PINNED_HF_SOURCE_SHA256:
            raise TextRopeCacheError("original HF reference source changed after prime()")

    # -- install / restore ------------------------------------------------
    def _install(self):
        if FORWARD in getattr(self._rotary, "__dict__", {}):
            raise TextRopeCacheError(
                "rotary already carries an instance-level forward; refusing an ambiguous patch")
        original = getattr(self._rotary, FORWARD, None)
        if not callable(original):
            raise TextRopeCacheError("rotary has no callable forward")
        self._originals[FORWARD] = original

        def wrapper(module, x, position_ids=None, *args, **kwargs):
            return self._dispatch(x, position_ids, args, kwargs)

        self._wrappers[FORWARD] = wrapper
        setattr(self._rotary, FORWARD, MethodType(wrapper, self._rotary))
        self._installed = True
        return self

    def uninstall(self):
        if self._installed:
            current = getattr(self._rotary, FORWARD, None)
            wrapper = self._wrappers[FORWARD]
            if not (getattr(current, "__self__", None) is self._rotary
                    and getattr(current, "__func__", None) is wrapper):
                raise TextRopeCacheError(
                    "rotary.forward was replaced by a third party after install; "
                    "the original was not restored blindly")
            delattr(self._rotary, FORWARD)
            self._installed = False
        self._enabled = False
        summary = self.snapshot(verify_cache=False)
        summary["uninstalled"] = True
        return summary

    # -- prime ------------------------------------------------------------
    def prime(self, x, position_ids, compare=None, tensor_info=None):
        """Build and verify the table outside the warm clock; disabled only."""
        if not self._installed:
            raise TextRopeCacheError("prime() called after uninstall()")
        if self._enabled:
            raise TextRopeCacheError(
                "prime() is refused while enabled; set_enabled(False) first (no silent "
                "re-prime of a table already in use)")
        torch = _torch()
        hf, hf_path = _hf_module()
        started = time.perf_counter()
        self._verify_provenance()
        pinned_config_gate(self._rotary)
        x_metadata = self._x_metadata_from(x)
        position_sha = self._position_fixture_gate(position_ids)
        cpu_positions = position_ids.detach().to("cpu").contiguous()

        cpu_rot = hf.Qwen3VLTextRotaryEmbedding(self._rotary.config, device="cpu")
        if str(getattr(cpu_rot, "rope_type", None)) != EXPECTED_ROPE_TYPE:
            raise TextRopeCacheError("original CPU rotary is not the reviewed default rope")
        if float(cpu_rot.attention_scaling) != EXPECTED_ATTENTION_SCALING:
            raise TextRopeCacheError(
                "original CPU rotary attention_scaling is %r, expected 1.0"
                % (cpu_rot.attention_scaling,))
        if list(int(v) for v in cpu_rot.mrope_section) != list(EXPECTED_MROPE_SECTION):
            raise TextRopeCacheError("original CPU rotary mrope_section differs")
        for label, module in (("original HF CPU rotary", cpu_rot),
                              ("candidate rotary", self._rotary)):
            if int(sum(p.numel() for p in module.parameters())) != 0:
                raise TextRopeCacheError("%s unexpectedly has learnable parameters" % label)
        inv_cpu = cpu_rot.inv_freq.detach().cpu().contiguous()
        inv_candidate = self._rotary._get_inv_freq(torch.device("cpu")).detach().cpu().contiguous()
        if tensor_raw_sha(inv_cpu) != PINNED_CPU_INV_FREQ_SHA256:
            raise TextRopeCacheError("original CPU inverse frequencies are not the pinned table")
        if raw_bytes(inv_cpu) != raw_bytes(inv_candidate):
            raise TextRopeCacheError(
                "candidate _get_inv_freq(cpu) is not raw-byte equal to the original HF CPU "
                "inverse frequencies")

        dummy = torch.empty(EXPECTED_HIDDEN_SHAPE, dtype=torch.float16, device="cpu")
        first = cpu_rot(dummy, cpu_positions)
        second = cpu_rot(dummy, cpu_positions)
        cpu_cos, cpu_sin = first
        for tensor, expected_sha, label in ((cpu_cos, PINNED_CPU_COS_SHA256, "cos"),
                                            (cpu_sin, PINNED_CPU_SIN_SHA256, "sin")):
            if tuple(int(v) for v in tensor.shape) != EXPECTED_TABLE_SHAPE:
                raise TextRopeCacheError("CPU table %s shape changed" % label)
            if str(tensor.dtype) != EXPECTED_STATE_DTYPE:
                raise TextRopeCacheError("CPU table %s dtype is %s" % (label, tensor.dtype))
            if not bool(torch.isfinite(tensor.float()).all()):
                raise TextRopeCacheError("CPU table %s is not finite" % label)
            if tensor_raw_sha(tensor) != expected_sha:
                raise TextRopeCacheError(
                    "CPU table %s does not reproduce the R94-measured original HF CPU table"
                    % label)
        if raw_bytes(first[0]) != raw_bytes(second[0]) or raw_bytes(first[1]) != raw_bytes(second[1]):
            raise TextRopeCacheError("original CPU table is not bit-repeatable")

        with torch.inference_mode(False), torch.no_grad():
            cache_cos = cpu_cos.to(x.device).clone()
            cache_sin = cpu_sin.to(x.device).clone()
            cache_cos.requires_grad_(False)
            cache_sin.requires_grad_(False)
        cached = {}
        for name, tensor, source in (("cos", cache_cos, cpu_cos), ("sin", cache_sin, cpu_sin)):
            if tuple(int(v) for v in tensor.shape) != EXPECTED_TABLE_SHAPE:
                raise TextRopeCacheError("cached %s shape changed during the NPU copy" % name)
            if str(tensor.dtype) != EXPECTED_STATE_DTYPE or _device_identity(tensor) != _device_identity(x):
                raise TextRopeCacheError("cached %s dtype/device changed during the NPU copy" % name)
            if raw_bytes(tensor) != raw_bytes(source):
                raise TextRopeCacheError("cached %s is not bit-identical to the CPU table" % name)
            signature = tensor_signature(tensor)
            if signature["version"] is None:
                raise TextRopeCacheError(
                    "cached %s is not a version-tracked tensor; build it outside inference_mode"
                    % name)
            cached[name] = {"tensor": tensor, "signature": signature,
                            "sha256": tensor_raw_sha(tensor),
                            "shape": list(EXPECTED_TABLE_SHAPE),
                            "dtype": EXPECTED_STATE_DTYPE,
                            "bytes": int(EXPECTED_TABLE_SHAPE[0] * EXPECTED_TABLE_SHAPE[1]
                                         * EXPECTED_TABLE_SHAPE[2] * 2)}

        chosen_compare = compare if compare is not None else _internal_compare
        chosen_info = tensor_info if tensor_info is not None else _internal_tensor_info
        with torch.no_grad():
            native = self._originals[FORWARD](x, position_ids)
        if not isinstance(native, (tuple, list)) or len(native) != 2:
            raise TextRopeCacheError("original rotary forward did not return a (cos, sin) pair")
        native_cos, native_sin = native[0], native[1]
        for tensor, label in ((native_cos, "cos"), (native_sin, "sin")):
            if tuple(int(v) for v in tensor.shape) != EXPECTED_TABLE_SHAPE:
                raise TextRopeCacheError("native probe %s shape changed" % label)
            if not bool(torch.isfinite(tensor.float()).all()):
                raise TextRopeCacheError("native probe %s is not finite" % label)
        native_probe = {
            "cos": chosen_info(native_cos), "sin": chosen_info(native_sin),
            "cos_vs_CPU_table": chosen_compare(native_cos, cpu_cos),
            "sin_vs_CPU_table": chosen_compare(native_sin, cpu_sin),
            "probe_scope": "one original native forward at prime; never cached",
            "R94_observed_native_SHA256": dict(R94_NATIVE_OBSERVED_SHA256),
            "matches_R94_observed_native_SHA":
                (tensor_raw_sha(native_cos) == R94_NATIVE_OBSERVED_SHA256["cos"]
                 and tensor_raw_sha(native_sin) == R94_NATIVE_OBSERVED_SHA256["sin"]),
        }
        self._hf_source_path = hf_path
        self._structure = config_fingerprint(self._rotary)
        self._x_metadata = x_metadata
        self._positions_object = position_ids
        self._positions_signature = tensor_signature(position_ids)
        self._positions_bytes = raw_bytes(position_ids)
        self._cache = cached
        self._cpu_table = {
            "cos": chosen_info(cpu_cos), "sin": chosen_info(cpu_sin),
            "repeat_bits_equal": True,
            "matches_R94_measured_CPU_table":
                (tensor_raw_sha(cpu_cos) == PINNED_CPU_COS_SHA256
                 and tensor_raw_sha(cpu_sin) == PINNED_CPU_SIN_SHA256),
            "inv_freq_sha256": tensor_raw_sha(inv_cpu),
            "inv_freq_matches_candidate_cpu_bits": True,
            "attention_scaling": float(cpu_rot.attention_scaling),
            "rope_type": str(cpu_rot.rope_type),
        }
        self._native_probe = native_probe
        self._prime_seconds = time.perf_counter() - started
        self._prime_count += 1
        self._primed = True
        return self.snapshot()

    def _x_metadata_from(self, x):
        metadata = {"shape": [int(v) for v in x.shape], "dtype": str(x.dtype),
                    "device_type": _device_type(x),
                    "device_index": _device_identity(x)[1]}
        if tuple(metadata["shape"]) != EXPECTED_HIDDEN_SHAPE:
            raise TextRopeCacheError("x shape is %r, expected %r"
                                     % (metadata["shape"], EXPECTED_HIDDEN_SHAPE))
        if metadata["dtype"] != EXPECTED_STATE_DTYPE:
            raise TextRopeCacheError("x dtype is %s, expected %s"
                                     % (metadata["dtype"], EXPECTED_STATE_DTYPE))
        if metadata["device_type"] not in NPU_DEVICE_TYPES:
            raise TextRopeCacheError("x must live on an NPU device, got %r"
                                     % (metadata["device_type"],))
        return metadata

    def _position_fixture_gate(self, position_ids):
        shape = tuple(int(v) for v in position_ids.shape)
        if shape != EXPECTED_POSITION_SHAPE:
            raise TextRopeCacheError("position_ids shape is %r, expected %r"
                                     % (list(shape), EXPECTED_POSITION_SHAPE))
        if str(position_ids.dtype) != EXPECTED_POSITION_DTYPE:
            raise TextRopeCacheError("position_ids dtype is %s" % position_ids.dtype)
        if not _is_npu_device(position_ids):
            raise TextRopeCacheError("position_ids must live on an NPU device")
        sha = tensor_raw_sha(position_ids)
        if sha != PINNED_POSITION_IDS_SHA256:
            raise TextRopeCacheError(
                "position_ids values are not the pinned real R78 M-RoPE fixture: %s" % sha)
        return sha

    # -- gating -----------------------------------------------------------
    def set_enabled(self, value):
        if not isinstance(value, bool):
            raise TextRopeCacheError("set_enabled() takes a bool")
        if value and not self._primed:
            raise TextRopeCacheError("set_enabled(True) requires a successful prime()")
        self._enabled = bool(value)
        return self.snapshot(verify_cache=False)

    def _validate_x(self, x):
        if self._x_metadata is None:
            raise TextRopeCacheError("x metadata was not primed")
        current = {"shape": [int(v) for v in x.shape], "dtype": str(x.dtype),
                   "device_type": _device_type(x),
                   "device_index": _device_identity(x)[1]}
        if current != self._x_metadata:
            raise TextRopeCacheError(
                "x metadata changed after prime(): %r != %r" % (current, self._x_metadata))

    def _validate_positions(self, position_ids):
        signature = tensor_signature(position_ids)
        if signature["shape"] != list(EXPECTED_POSITION_SHAPE) or \
                signature["dtype"] != EXPECTED_POSITION_DTYPE or \
                signature["device_type"] != self._positions_signature["device_type"] or \
                signature["device_index"] != self._positions_signature["device_index"]:
            raise TextRopeCacheError("position_ids metadata changed after prime()")
        fast = (position_ids is self._positions_object
                and signature["version"] is not None
                and signature == self._positions_signature)
        if fast:
            self._stats["position_fast_hits"] += 1
            return
        values = raw_bytes(position_ids)
        self._stats["position_value_rereads"] += 1
        if values != self._positions_bytes:
            raise TextRopeCacheError(
                "position_ids values differ from the primed real fixture; the table cache "
                "refuses a different sequence")
        # Track the new object, but only a versioned one can ever take a fast path.
        self._positions_object = position_ids
        self._positions_signature = signature

    def _validate_cache_tensors(self):
        for name, entry in self._cache.items():
            if tensor_signature(entry["tensor"]) != entry["signature"]:
                raise TextRopeCacheError(
                    "cached %s table tensor was modified after prime(); refusing to serve it"
                    % name)

    def _dispatch(self, x, position_ids, extra_args, kwargs):
        if not self._enabled:
            self._stats["disabled_calls"] += 1
            if position_ids is None:
                return self._originals[FORWARD](x, *extra_args, **kwargs)
            return self._originals[FORWARD](x, position_ids, *extra_args, **kwargs)
        if position_ids is None or extra_args or kwargs:
            raise TextRopeCacheError(
                "enabled table cache only supports the two positional arguments "
                "(x, position_ids) of the reviewed rotary")
        self._stats["calls"] += 1
        try:
            self._revalidate_structure()
            self._validate_x(x)
            self._validate_positions(position_ids)
            self._validate_cache_tensors()
        except BaseException:
            self._stats["validation_failures"] += 1
            raise
        self._stats["hits"] += 1
        self._stats["returned_cached_object"] += 1
        return self._cache["cos"]["tensor"], self._cache["sin"]["tensor"]

    # -- reporting --------------------------------------------------------
    def snapshot(self, verify_cache=True):
        cached = {}
        if verify_cache and self._cache:
            self._validate_cache_tensors()
        for name, entry in self._cache.items():
            row = {"shape": entry["shape"], "dtype": entry["dtype"],
                   "bytes": entry["bytes"], "sha256": entry["sha256"],
                   "version": entry["signature"]["version"]}
            if verify_cache:
                row["sha256_unchanged_since_prime"] = (
                    tensor_raw_sha(entry["tensor"]) == entry["sha256"])
            cached[name] = row
        return {
            "round": ROUND,
            "installed": bool(self._installed),
            "enabled": bool(self._enabled),
            "primed": bool(self._primed),
            "patched_methods": [FORWARD],
            "per_instance_patch_only": True,
            "patches_torch_global": False,
            "patches_torch_functional": False,
            "patches_candidate_class": False,
            "patches_private_candidate_functions": False,
            "cache_scope": "original-HF CPU Half text M-RoPE cos/sin table only",
            "cached_rope_only": True,
            "caches_hidden_embeddings": False,
            "caches_pixels_patch_or_image_conditioning": False,
            "caches_deepstack_or_text_conditions": False,
            "teacher_weights": 0,
            "parameter_dtype_changed": False,
            "cached": cached,
            "extra_bytes": sum(entry["bytes"] for entry in self._cache.values()),
            "expected_extra_bytes": EXPECTED_TABLE_BYTES,
            "cpu_table": self._cpu_table,
            "native_probe": self._native_probe,
            "native_rounding_changed": True,
            "native_equals_CPU_table_bits": (
                None if self._native_probe is None else
                (self._native_probe["cos_vs_CPU_table"]["raw_bits_equal"]
                 and self._native_probe["sin_vs_CPU_table"]["raw_bits_equal"])),
            "precision_alignment_not_lossless_native_bits": True,
            "config": self._structure,
            "positions": {"fixture_sha256": PINNED_POSITION_IDS_SHA256,
                          "shape": list(EXPECTED_POSITION_SHAPE),
                          "dtype": EXPECTED_POSITION_DTYPE},
            "stats": json.loads(json.dumps(self._stats)),
            "prime_count": int(self._prime_count),
            "prime_seconds": self._prime_seconds,
            "provenance": self._provenance,
            "hf_source": {"path": None if self._hf_source_path is None
                          else str(self._hf_source_path),
                          "sha256": PINNED_HF_SOURCE_SHA256},
            "source_sha256": _file_sha(Path(__file__)),
            "full_encoder_precision_accepted": False,
            "whole_goal_complete": False,
            "numerical_tolerance": None,
        }


def install(rotary, candidate_path=None):
    """Factory: an installed, disabled reviewed table cache for one rotary."""
    return CPUTextRopeCache95(rotary, candidate_path)