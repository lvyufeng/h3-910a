#!/usr/bin/env python3
"""Round84 private FP32 row-parallel partial adapter (source only).

WHY THIS EXISTS
---------------
The Round79 diagnostic candidate's ``MiniMaxH3Qwen3VLRowParallelLinear.forward``
computes each rank's partial projection with ``quant_method.apply`` in the stored
compute dtype, so the custom Half GEMM rounds its FP32 accumulator to **Half**
*before* the collective::

    partial_half = self.quant_method.apply(self, input_parallel)   # premature Half
    ...
    output_parallel = output_parallel.float()                      # too late
    output_parallel = self.group.all_reduce(output_parallel)       # FP32 all-reduce
    output_parallel = output_parallel.to(self.output_dtype)        # final Half

The single-device reference accumulates the whole ``K`` in one FP32 accumulator
and rounds once. The native TP path therefore pays one extra Half rounding on
every rank's partial, and a FP32 collective cannot restore bits that were
already rounded away.

Round83 built and **actually executed** a blocked TIK kernel that computes
Half x Half -> FP32 accumulator -> **FP32 GM output** for the H3 projection
shapes (M=2043; K=1024/3456; N=5120), with bit-stable repeated outputs and
1.08e-6 relative RMS against an FP32-accumulate diagnostic. This module wires
that verified primitive into the encoder as a private, gated adapter so the
partial stays FP32 through the collective and is narrowed exactly once.

SCOPE AND NON-CLAIMS
--------------------
* Source only. Nothing in this file is compiled, launched or executed here; this
  module performs no SSH, no NPU/GPU work, no model run, no remote compile and no
  installation. It is **not** a precision result, **not** a speed result and
  **not** a full-encoder acceptance.
* The Round83 kernel and runtime modules are used **unmodified** and are pinned
  by SHA-256. The pinned Round79 candidate encoder file is never edited.
* Stored parameters and activations stay the original Half values. The only
  cached copies are Half packed permutations of the existing parameters: no
  FP32 weight shadows, no quantization, no BF16 compute, no clamping.
* Only ``MiniMaxH3Qwen3VLRowParallelLinear.forward`` is replaced, and only while
  the adapter is enabled. When disabled it calls the captured original forward
  unchanged.

PUBLIC API (used by the root driver)
------------------------------------
::

    adapter = install(candidate, runtime, kernel_dir, sequence_length=2043)
    #   candidate: the loaded candidate encoder *module*
    #   runtime:   runtime5gm83.Runtime5GM instance for this rank
    #   kernel_dir: shared write-once kernel directory for all ranks
    # returns an adapter with enabled == False; the class forward is replaced by
    # a gated wrapper that calls the original while disabled.

    adapter.prime(encoder.text_model)   # after parameters are loaded and moved
    adapter.controls(sample_input=..., k_local=...)   # optional pre-warm proof
    adapter.set_enabled(True)           # enable the FP32-partial path
    adapter.snapshot()                  # counters/dtypes/bytes (no tensors)
    adapter.release_scratch()           # free activations, keep weight caches
    adapter.uninstall()                 # restore forward, drop everything

Numerics of the enabled path, in order:

1. original Half input, sharded exactly like the native forward when
   ``input_is_parallel`` is False;
2. Half input packed to ``(K1, Mp, 16)`` with exact zero ``M`` padding rows;
3. verified R83 kernel: Half x Half -> FP32 accumulator -> FP32 GM output
   (single L0C accumulator across every K tile; no Half partial);
4. FP32 kernel output unpacked and reshaped to the native output shape;
5. the layer's own ``group.all_reduce`` on the **FP32** partial (dtype asserted
   before and after; actual dtypes recorded);
6. optional bias added in FP32;
7. exactly one ``.to(layer.output_dtype)`` narrowing.

The H3 row layers are constructed with ``bias=False``, so step 6 is inactive
here; it is implemented because the task requires a correct FP32 bias order, and
it is counted separately so an unexpected bias cannot pass silently.
"""

from __future__ import annotations

import ast
import hashlib
import importlib
import json
import os
from pathlib import Path
import textwrap
import time

__all__ = [
    "ROW_CLASS_NAME",
    "SUPPORTED_ROW_SHAPES",
    "PINNED",
    "RowAdapterError",
    "RowFp32Adapter",
    "install",
    "install_row",
    "candidate_source_digests",
    "expected_cached_weight_bytes",
    "forward_source_digest",
    "class_source_digest",
    "tensor_signature",
    "resolve_row_shape",
    "resolve_token_layout",
    "output_shape_for",
    "expected_m_pad",
]

ROUND = 84

ROW_CLASS_NAME = "MiniMaxH3Qwen3VLRowParallelLinear"

#: (input_size_per_partition, output_size) pairs this adapter is allowed to
#: serve. Both are the H3 text encoder's row-parallel projections at
#: text_encoder_tp_size = 8: attention ``o_proj`` (8192/8 = 1024) and MLP
#: ``down_proj`` (27648/8 = 3456), each producing the full hidden size 5120.
SUPPORTED_ROW_SHAPES = ((1024, 5120), (3456, 5120))

#: Pinned inputs. Every value was recomputed from the committed files at
#: authoring time; the adapter fails closed instead of guessing on any drift.
PINNED = {
    "round79_candidate_file_sha256":
        "2141b29adc9ecc601d7fa2599e6ba56351adfbdaa5f16acb62c00d95cee3cf78",
    "round79_row_forward_source_sha256":
        "9a10f625bef735cc937d3f02835838550ed4568e5bfd3e54d41354d4d1f1f4e3",
    "round79_row_class_source_sha256":
        "3d4a55c57df3118a4e741502062d1c8fc0301b9adafca8e9d8594ca93b76c2b3",
    "round83_kernel_module": "blocked_dynamic_steps83",
    "round83_kernel_module_sha256":
        "b5a06cfdb4c83a2d2f8c84eb56e8794f8a38d2732c4ff5e1fbef0901b520facb",
    "round83_runtime_module": "runtime5gm83",
    "round83_runtime_module_sha256":
        "c8890dbcd825fe9177c7ee8d5c6b46e4d00231d4d61e6421f83491fec85c52a6",
    "round83_runtime_library_sha256":
        "bd50d7005bf48e9a9a853c01750eaea4eefbb3e0f69abb2e7153aa7139e8b043",
}

#: The H3 text encoder stacks ``selected_layer`` decoder layers, each with one
#: attention ``o_proj`` and one MLP ``down_proj`` row-parallel projection.
EXPECTED_LAYER_COUNT = 100  # 50 layers x 2 row-parallel weights

EXPECTED_HIDDEN_SIZE = 5120
EXPECTED_SEQUENCE_LENGTH = 2043
EXPECTED_WORLD_SIZE = 8

#: Per-rank cached packed-Half weight bytes for the full 50-layer encoder:
#: 50 * (5120*1024 + 5120*3456) * 2 bytes = 2,293,760,000 B.
EXPECTED_CACHED_WEIGHT_BYTES = 2_293_760_000


class RowAdapterError(RuntimeError):
    """Raised instead of silently falling back when a precondition is unmet."""


# ---------------------------------------------------------------------------
# Source pinning helpers (stdlib only: usable on a machine without torch)
# ---------------------------------------------------------------------------


def _sha256_file(path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _normalize_source(segment: str) -> str:
    """Dedent, strip trailing whitespace and trailing blank lines."""
    lines = [line.rstrip() for line in textwrap.dedent(segment).splitlines()]
    while lines and not lines[-1]:
        lines.pop()
    return "\n".join(lines) + "\n"


def _extract_class_and_forward(source_text: str) -> tuple[str, str]:
    """Return the normalized source of the row class and of its ``forward``."""
    tree = ast.parse(source_text)
    for node in tree.body:
        if isinstance(node, ast.ClassDef) and node.name == ROW_CLASS_NAME:
            class_segment = ast.get_source_segment(source_text, node)
            for child in node.body:
                if isinstance(child, ast.FunctionDef) and child.name == "forward":
                    forward_segment = ast.get_source_segment(source_text, child)
                    if class_segment is None or forward_segment is None:
                        break
                    return _normalize_source(class_segment), _normalize_source(forward_segment)
            raise RowAdapterError(f"{ROW_CLASS_NAME} has no 'forward' definition")
    raise RowAdapterError(f"{ROW_CLASS_NAME} not found in the candidate source")


def forward_source_digest(source_text: str) -> str:
    """SHA-256 of the normalized original ``forward`` source."""
    return hashlib.sha256(_extract_class_and_forward(source_text)[1].encode()).hexdigest()


def class_source_digest(source_text: str) -> str:
    """SHA-256 of the normalized original row-parallel class source."""
    return hashlib.sha256(_extract_class_and_forward(source_text)[0].encode()).hexdigest()


def candidate_source_digests(candidate_file) -> dict:
    """Recompute the three pinned digests of a candidate encoder file."""
    text = Path(candidate_file).read_text()
    return {
        "file_sha256": hashlib.sha256(text.encode()).hexdigest(),
        "forward_source_sha256": forward_source_digest(text),
        "class_source_sha256": class_source_digest(text),
    }


def verify_candidate_source(candidate_file) -> dict:
    """Fail closed unless the candidate file matches every pinned digest."""
    digests = candidate_source_digests(candidate_file)
    for key, actual in (
        ("round79_candidate_file_sha256", digests["file_sha256"]),
        ("round79_row_forward_source_sha256", digests["forward_source_sha256"]),
        ("round79_row_class_source_sha256", digests["class_source_sha256"]),
    ):
        expected = PINNED[key]
        if actual != expected:
            raise RowAdapterError(
                f"candidate source drifted from the pinned {key}: "
                f"expected {expected}, got {actual} ({candidate_file})")
    return digests


def _import_pinned_module(module_name: str, expected_sha256: str, label: str,
                          verify: bool = True):
    try:
        module = importlib.import_module(module_name)
    except ImportError as exc:  # pragma: no cover - only on a misconfigured host
        raise RowAdapterError(
            f"cannot import the pinned {label} module {module_name!r}; the root driver must "
            f"put the Round83 diagnostics directory on sys.path") from exc
    if not verify:
        return module
    module_file = getattr(module, "__file__", None)
    if not module_file:
        raise RowAdapterError(f"pinned {label} module {module_name!r} has no __file__")
    actual = _sha256_file(module_file)
    if actual != expected_sha256:
        raise RowAdapterError(
            f"pinned {label} module drifted: expected {expected_sha256}, got {actual} "
            f"({module_file})")
    return module


# ---------------------------------------------------------------------------
# Pure-python shape / signature helpers (unit-testable without torch)
# ---------------------------------------------------------------------------


def resolve_row_shape(weight_shape, supported=SUPPORTED_ROW_SHAPES) -> tuple[int, int]:
    """Validate a row-parallel weight shape ``(N, K_local)`` -> ``(K, N)``."""
    dims = tuple(int(v) for v in weight_shape)
    if len(dims) != 2:
        raise RowAdapterError(f"row-parallel weight must be 2-D (N,K), got {dims}")
    n, k = dims
    if (k, n) not in supported:
        raise RowAdapterError(
            f"row-parallel geometry (K={k}, N={n}) is not one of the verified shapes "
            f"{sorted(supported)}")
    return k, n


def resolve_token_layout(input_shape, k_local: int, sequence_length: int) -> tuple[tuple[int, ...], int]:
    """Validate the native activation shape and return ``(batch_shape, tokens)``.

    The H3 text encoder produces exactly two shapes for these projections: the
    unbatched ``(M, K)`` and the batched ``(1, M, K)``. ``M`` must equal
    ``sequence_length`` exactly, so the zero ``M`` padding rows of the shared
    kernel buffer are never written. Anything else fails closed.
    """
    dims = tuple(int(v) for v in input_shape)
    if len(dims) == 2:
        batch_shape: tuple[int, ...] = ()
        tokens = dims[0]
    elif len(dims) == 3:
        if dims[0] != 1:
            raise RowAdapterError(
                f"activation shape {dims} is not supported; only (M,K) and (1,M,K) with "
                f"batch 1 are accepted")
        batch_shape = (1,)
        tokens = dims[1]
    else:
        raise RowAdapterError(
            f"activation shape {dims} is not supported; only 2-D (M,K) and 3-D (1,M,K) "
            f"are accepted")
    if dims[-1] != int(k_local):
        raise RowAdapterError(
            f"activation last dimension is {dims[-1]}, expected input_size_per_partition "
            f"{k_local}")
    if tokens != int(sequence_length):
        raise RowAdapterError(
            f"activation has {tokens} tokens, expected sequence_length {sequence_length}; "
            f"the shared zero-padded kernel buffers are sized for one sequence only")
    return batch_shape, tokens


def output_shape_for(batch_shape, tokens: int, n: int) -> tuple[int, ...]:
    """Native output shape for a validated activation layout.

    ``(M, K)`` -> ``(M, N)``; ``(1, M, K)`` -> ``(1, M, N)``. The row-parallel
    projection never changes the spatial shape, only the last dimension.
    """
    return tuple(int(v) for v in batch_shape) + (int(tokens), int(n))


def expected_m_pad(sequence_length: int, mb: int = 128) -> int:
    """Padded ``M`` the verified kernel uses: ``ceil(M / mb) * mb``."""
    sequence_length = int(sequence_length)
    mb = int(mb)
    if sequence_length <= 0 or mb <= 0:
        raise RowAdapterError("sequence_length and mb must be positive")
    return ((sequence_length + mb - 1) // mb) * mb


def tensor_signature(weight) -> tuple:
    """Identity signature of one logical parameter view.

    Deliberately uses the *logical* tensor ``data_ptr`` **and** the
    ``storage_offset`` instead of only ``untyped_storage().data_ptr()``: two
    different views of one storage (for example a fused QKV slice) must never be
    treated as the same cache entry. ``_version`` catches in-place mutation of a
    loaded parameter, and the shape/stride/dtype/device entries catch re-use.
    """
    return (
        int(weight.data_ptr()),
        int(weight.storage_offset()),
        tuple(int(v) for v in weight.shape),
        tuple(int(v) for v in weight.stride()),
        str(weight.dtype),
        str(weight.device),
        int(getattr(weight, "_version", -1)),
    )


def expected_cached_weight_bytes(layer_count: int = 50) -> int:
    """Packed-Half weight bytes for ``layer_count`` layers of both shapes."""
    per_layer = sum(n * k * 2 for k, n in SUPPORTED_ROW_SHAPES)
    return int(layer_count) * per_layer


# ---------------------------------------------------------------------------
# Private per-parameter packed-Half cache
# ---------------------------------------------------------------------------


class _WeightCache:
    """Packed Half permutation of each original parameter, plus fail-closed identity.

    Only Half copies are stored; there is no FP32 weight shadow. A signature
    change for an already-cached logical view is an error, never a silent
    re-cache: a mutated parameter would otherwise change results mid-run.
    """

    def __init__(self, pack_w):
        self._pack_w = pack_w
        self._entries: dict[tuple, object] = {}
        self._logical: dict[tuple[int, int], tuple] = {}
        self._stats = {"entries": 0, "bytes": 0, "hits": 0, "misses": 0,
                       "mutations_detected": 0, "roundtrip_verifications": 0}

    def _pack(self, weight):
        packed = self._pack_w(weight)
        # The kernel module already validates dtype/layout; repeat the exactness
        # proof here on-device (torch.equal) so the cache cannot hold a
        # numerically altered copy. No device-to-host copy is performed.
        import torch
        if packed.dtype != torch.float16:
            raise RowAdapterError(f"packed weight is {packed.dtype}, expected float16")
        if int(packed.numel()) != int(weight.numel()):
            raise RowAdapterError("packed weight element count changed")
        k1, n, k0 = (int(v) for v in packed.shape)
        if k0 != 16 or n != int(weight.shape[0]):
            raise RowAdapterError(f"unexpected packed weight shape {tuple(packed.shape)}")
        back = packed.transpose(0, 1).reshape(n, k1 * 16)
        if not torch.equal(back, weight.detach()):
            raise RowAdapterError(
                "packed Half weight does not round-trip to the original parameter; "
                "refusing to cache an altered copy")
        self._stats["roundtrip_verifications"] += 1
        return packed

    def get(self, weight):
        signature = tensor_signature(weight)
        entry = self._entries.get(signature)
        if entry is not None:
            self._stats["hits"] += 1
            return entry
        logical = (signature[0], signature[1])
        prior = self._logical.get(logical)
        if prior is not None and prior != signature:
            self._stats["mutations_detected"] += 1
            raise RowAdapterError(
                "cached row-parallel parameter signature changed "
                "(shape/stride/dtype/device/_version); refusing to silently re-cache a "
                "mutated parameter")
        packed = self._pack(weight)
        self._entries[signature] = packed
        self._logical[logical] = signature
        self._stats["entries"] = len(self._entries)
        self._stats["bytes"] += int(packed.numel() * packed.element_size())
        self._stats["misses"] += 1
        return packed

    def verify_unchanged(self, weight) -> bool:
        """True when the weight still matches its cached logical signature."""
        signature = tensor_signature(weight)
        if signature in self._entries:
            return True
        logical = (signature[0], signature[1])
        if self._logical.get(logical) != signature:
            raise RowAdapterError("a primed row-parallel parameter changed after prime")
        return True

    def clear(self) -> None:
        self._entries.clear()
        self._logical.clear()

    def stats(self) -> dict:
        return dict(self._stats)


# ---------------------------------------------------------------------------
# The adapter
# ---------------------------------------------------------------------------


class RowFp32Adapter:
    """Gated FP32-partial replacement for the private row-parallel linear."""

    def __init__(self, candidate, runtime, kernel_dir, *, sequence_length=EXPECTED_SEQUENCE_LENGTH,
                 expected_world_size=EXPECTED_WORLD_SIZE,
                 expected_layer_count=EXPECTED_LAYER_COUNT,
                 verify_sources=True, allow_stream_switch=False):
        self._candidate = candidate
        self._runtime = runtime
        self._kernel_dir = Path(kernel_dir)
        self._sequence_length = int(sequence_length)
        self._expected_world_size = int(expected_world_size)
        self._expected_layer_count = int(expected_layer_count)
        self._verify_sources = bool(verify_sources)
        self._allow_stream_switch = bool(allow_stream_switch)
        self._kernel_module = None
        self._kernel_metas: dict[tuple[int, int], dict] = {}
        self._scratch: dict[tuple[int, int], dict] = {}
        self._packed_weights = _WeightCache(self._pack_w)
        self._stream_ptr = None
        self._primed_metadata = None
        self._primed_layers: list[tuple[object, tuple]] = []
        self._enabled = False
        self._installed = True
        self._stats = {
            "calls_disabled": 0,
            "calls_enabled": 0,
            "calls_rejected": 0,
            "kernel_launches": 0,
            "fp32_collective_count": 0,
            "fp32_dtype_before_collective": {},
            "fp32_dtype_after_collective": {},
            "half_cast_count": 0,
            "bias_added_fp32_count": 0,
            "single_rank_no_collective_calls": 0,
            "kernel_shapes": {},
            "stream_binds": 0,
            "stream_conflicts": 0,
            "pack_controls": 0,
            "input_is_parallel_split": 0,
            "enable_transitions": [],
        }
        self._cls = None
        self._original_forward = None

    # -- installation ------------------------------------------------------

    @property
    def row_class(self):
        return self._cls

    def _install_forward(self) -> None:
        cls = getattr(self._candidate, ROW_CLASS_NAME, None)
        if cls is None:
            raise RowAdapterError(
                f"candidate module has no {ROW_CLASS_NAME}; refusing to patch an unknown class")
        current = cls.__dict__.get("forward")
        if current is None:
            raise RowAdapterError(f"{ROW_CLASS_NAME} has no forward in its own __dict__")
        if getattr(current, "_row84_gated_wrapper", False):
            raise RowAdapterError(
                f"{ROW_CLASS_NAME}.forward is already wrapped by a Round84 adapter; "
                f"uninstall the previous adapter first")
        self._cls = cls
        self._original_forward = current

        adapter = self

        def gated_forward(self, input_):
            if not adapter._enabled:
                adapter._stats["calls_disabled"] += 1
                return adapter._original_forward(self, input_)
            try:
                return adapter._forward(self, input_)
            except RowAdapterError:
                adapter._stats["calls_rejected"] += 1
                raise

        gated_forward._row84_gated_wrapper = True
        gated_forward.__name__ = "forward"
        gated_forward.__qualname__ = f"{cls.__qualname__}.forward"
        gated_forward.__doc__ = (
            "Round84 gated wrapper: original forward while disabled, verified "
            "FP32-partial row-parallel path while enabled.")
        cls.forward = gated_forward

    def uninstall(self) -> dict:
        """Restore the original forward and drop every cached tensor."""
        summary = self.snapshot()
        if self._installed and self._cls is not None:
            current = self._cls.__dict__.get("forward")
            if getattr(current, "_row84_gated_wrapper", False):
                self._cls.forward = self._original_forward
            elif current is not None:
                summary["uninstall_warning"] = (
                    "class forward was replaced by a third party after install; the original "
                    "was not restored blindly")
        self._enabled = False
        self._installed = False
        self._packed_weights.clear()
        self._scratch.clear()
        self._kernel_metas.clear()
        self._stream_ptr = None
        summary["uninstalled"] = True
        return summary

    # -- kernel preparation ------------------------------------------------

    def _resolve_kernel_module(self):
        if self._kernel_module is None:
            self._kernel_module = _import_pinned_module(
                PINNED["round83_kernel_module"],
                PINNED["round83_kernel_module_sha256"], "Round83 kernel",
                verify=self._verify_sources)
        return self._kernel_module

    def _verify_runtime(self) -> dict:
        seen = getattr(self._runtime, "library_sha256", None)
        if seen is None:
            snapshot = getattr(self._runtime, "snapshot", None)
            if callable(snapshot):
                seen = snapshot().get("library_sha256")
        if seen != PINNED["round83_runtime_library_sha256"]:
            raise RowAdapterError(
                "the Round83 runtime bridge does not report the pinned launcher library; "
                f"expected {PINNED['round83_runtime_library_sha256']}, got {seen!r}")
        if int(getattr(self._runtime, "device", -1)) < 0:
            raise RowAdapterError("runtime object has no device index")
        return {"library_sha256": seen, "device": int(self._runtime.device)}

    def _validate_kernel_meta(self, meta: dict, k_local: int) -> dict:
        expected_cores = 30
        checks = {
            "dst_dtype": ("float32", meta.get("dst_dtype")),
            "input_dtype": ("float16", meta.get("input_dtype")),
            "accumulator_dtype": ("float32", meta.get("accumulator_dtype")),
            "GM_argument_count": (5, meta.get("GM_argument_count")),
            "block_dim": (expected_cores, meta.get("block_dim")),
            "uses_fp32_bias": (False, meta.get("uses_fp32_bias")),
            "no_bias_operand": (True, meta.get("no_bias_operand")),
            "no_fp32_weight_shadow": (True, meta.get("no_fp32_weight_shadow")),
        }
        for name, (expected, actual) in checks.items():
            if actual != expected:
                raise RowAdapterError(
                    f"Round83 kernel metadata {name} is {actual!r}, expected {expected!r}")
        logical = meta.get("logical_shape") or {}
        if (int(logical.get("m", -1)), int(logical.get("k", -1)), int(logical.get("n", -1))) != (
                self._sequence_length, k_local, EXPECTED_HIDDEN_SIZE):
            raise RowAdapterError(f"unexpected kernel logical shape {logical}")
        padded = meta.get("padded_shape") or {}
        want_m_pad = expected_m_pad(self._sequence_length)
        if int(padded.get("m_pad", -1)) != want_m_pad:
            raise RowAdapterError(
                f"unexpected padded M {padded.get('m_pad')}, expected {want_m_pad}")
        shapes = meta.get("gm_shapes") or {}
        for key in ("a", "c"):
            dims = tuple(int(v) for v in shapes.get(key, ()))
            if len(dims) != 3 or dims[1] != want_m_pad:
                raise RowAdapterError(
                    f"kernel GM {key} shape {dims} does not carry the padded M={want_m_pad}")
        k_tiles = int((meta.get("accumulation") or {}).get("k_tiles", -1))
        if k_tiles != k_local // 128:
            raise RowAdapterError(f"unexpected K tile count {k_tiles} for K={k_local}")
        if (meta.get("accumulation") or {}).get("half_partial") is not False:
            raise RowAdapterError("Round83 kernel metadata does not guarantee no Half partial")
        obj = Path(meta["obj_path"])
        if not obj.is_file():
            raise RowAdapterError(f"kernel object missing: {obj}")
        if _sha256_file(obj) != meta["obj_sha256"]:
            raise RowAdapterError("kernel object changed after build")
        return meta

    def _repair_spec_file_if_needed(self, meta: dict) -> bool:
        """Rewrite a torn ``spec83.json`` atomically (the builder writes it in place).

        Eight ranks can reach the builder's cache-hit path at the same time; the
        builder's own ``.compile.lock`` covers compilation but its informational
        ``spec83.json`` write is a plain in-place write. The adapter serialises
        calls with its own lock, and repairs a file that still parses badly.
        """
        spec_path = Path(meta["obj_path"]).parent / "spec83.json"
        if spec_path.is_file():
            try:
                if isinstance(json.loads(spec_path.read_text()), dict):
                    return False
            except (json.JSONDecodeError, UnicodeDecodeError):
                pass
        payload = json.dumps(meta, indent=2, sort_keys=True) + "\n"
        root = self._kernel_dir.resolve(strict=True)
        resolved_spec = spec_path.resolve(strict=False)
        if (not resolved_spec.is_relative_to(root)
                or resolved_spec.name != 'spec83.json'
                or resolved_spec.parent != Path(meta['obj_path']).resolve(strict=True).parent):
            raise RowAdapterError('Unexpected absolute kernel-spec repair target')
        tmp = resolved_spec.with_suffix(".json.tmp84")
        if tmp.resolve(strict=False).parent != resolved_spec.parent:
            raise RowAdapterError('Unexpected temporary kernel-spec path')
        tmp.write_text(payload)
        # Both absolute source/destination paths have been checked above.
        os.replace(tmp, resolved_spec)
        return True

    def prepare_kernels(self) -> dict:
        """Build or reuse the two verified kernels under one cross-process lock."""
        kernel_module = self._resolve_kernel_module()
        runtime_info = self._verify_runtime()
        self._kernel_dir.mkdir(parents=True, exist_ok=True)
        lock_path = self._kernel_dir / ".row84_build.lock"
        deadline = time.monotonic() + 1800.0
        with lock_path.open("a") as lock:
            try:
                import fcntl
            except ImportError as exc:  # pragma: no cover - Linux only in practice
                raise RowAdapterError("fcntl is required for the shared kernel lock") from exc
            while True:
                try:
                    fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
                    break
                except BlockingIOError:
                    if time.monotonic() > deadline:
                        raise RowAdapterError(
                            "timed out waiting for another rank's kernel build lock "
                            f"({lock_path})")
                    time.sleep(0.25)
            try:
                built = {}
                for k_local, n in SUPPORTED_ROW_SHAPES:
                    meta = kernel_module.build_kernel(
                        m=self._sequence_length, k=k_local, n=n, out_dir=self._kernel_dir)
                    meta = self._validate_kernel_meta(meta, k_local)
                    repaired = self._repair_spec_file_if_needed(meta)
                    self._kernel_metas[(k_local, n)] = meta
                    built[(k_local, n)] = {
                        "kernel_name": meta["kernel_name"],
                        "kernel_symbol": meta["kernel_symbol"],
                        "obj_sha256": meta["obj_sha256"],
                        "kernel_source_sha256": meta.get("source_sha256"),
                        "block_dim": meta["block_dim"],
                        "compiled_now": bool(meta.get("compiled_now")),
                        "repaired_spec_file": repaired,
                    }
            finally:
                fcntl.flock(lock.fileno(), fcntl.LOCK_UN)
        return {"runtime": runtime_info,
                "kernel_module_sha256": _sha256_file(self._kernel_module.__file__),
                "kernel_dir": str(self._kernel_dir), "kernels": built}

    # -- scratch buffers ---------------------------------------------------

    def _ensure_scratch(self, k_local: int, n: int) -> dict:
        key = (k_local, n)
        scratch = self._scratch.get(key)
        if scratch is not None:
            return scratch
        import torch
        meta = self._kernel_metas[key]
        device = torch.device("npu", int(self._runtime.device))
        shapes = meta["gm_shapes"]
        k1, m_pad, k0 = (int(v) for v in shapes["a"])
        n1, m_pad_c, n0 = (int(v) for v in shapes["c"])
        want_m_pad = expected_m_pad(self._sequence_length)
        if (k1, m_pad, k0) != (k_local // 16, want_m_pad, 16):
            raise RowAdapterError(f"unexpected A GM shape {shapes['a']} for K={k_local}")
        if (n1, m_pad_c, n0) != (n // 16, want_m_pad, 16):
            raise RowAdapterError(f"unexpected C GM shape {shapes['c']} for N={n}")
        # A is zero initialised once: the padded M rows must stay exactly zero
        # for every launch, and the token count is pinned to sequence_length.
        a_buffer = torch.zeros((k1, m_pad, k0), dtype=torch.float16, device=device)
        out_buffer = torch.empty((n1, m_pad, n0), dtype=torch.float32, device=device)
        aux = [torch.zeros((16,), dtype=torch.float16, device=device) for _ in range(2)]
        scratch = {
            "a": a_buffer, "c": out_buffer, "aux": aux,
            "bytes": int(a_buffer.numel() * a_buffer.element_size()
                         + out_buffer.numel() * out_buffer.element_size()
                         + sum(t.numel() * t.element_size() for t in aux)),
        }
        self._scratch[key] = scratch
        return scratch

    def release_scratch(self) -> dict:
        """Free activation buffers, keeping the primed weight caches."""
        freed = sum(entry["bytes"] for entry in self._scratch.values())
        self._scratch.clear()
        return {"freed_scratch_bytes": int(freed),
                "cached_weight_bytes": int(self._packed_weights.stats()["bytes"])}

    # -- packing helpers ---------------------------------------------------

    def _pack_w(self, weight):
        kernel_module = self._resolve_kernel_module()
        return kernel_module.pack_w(weight)

    def _pack_a_into(self, buffer, flat) -> None:
        """Write the exact Half activation into the zero-padded A buffer prefix.

        Same element order as the verified ``pack_a`` helper:
        ``buffer[:, :m, :] = flat.reshape(m, K1, 16).transpose(0, 1)``. Only the
        prefix is written, so the padding rows keep their initial zeros.
        """
        import torch
        if flat.dtype != torch.float16:
            raise RowAdapterError(f"activation to pack is {flat.dtype}, expected float16")
        if flat.dim() != 2:
            raise RowAdapterError(f"activation to pack must be 2-D, got {tuple(flat.shape)}")
        m, k = int(flat.shape[0]), int(flat.shape[1])
        k1, m_pad, k0 = (int(v) for v in buffer.shape)
        if (k // 16, 16) != (k1, k0):
            raise RowAdapterError(f"activation K={k} does not match A buffer {tuple(buffer.shape)}")
        if m > m_pad:
            raise RowAdapterError(f"activation M={m} exceeds padded buffer {m_pad}")
        buffer[:, :m, :] = flat.reshape(m, k // 16, 16).transpose(0, 1)

    # -- priming -----------------------------------------------------------

    def prime(self, encoder_text_model) -> dict:
        """Cache the packed Half weights and prepare both kernels.

        Must run after the parameters have been loaded and moved to their final
        device/dtype. Nothing here writes an activation buffer outside a control
        check, and every packed weight is proven equal to the original parameter
        on-device with ``torch.equal``.
        """
        if self._cls is None:
            raise RowAdapterError("adapter is not installed")
        if self._primed_metadata is not None:
            for layer, _signature in self._primed_layers:
                self._packed_weights.verify_unchanged(layer.weight)
            return dict(self._primed_metadata, prime_reused=True)
        import torch
        if not hasattr(encoder_text_model, "named_modules"):
            raise RowAdapterError("prime expects the encoder text model nn.Module")
        layers = []
        per_shape = {}
        signatures = {}
        for name, module in encoder_text_model.named_modules():
            if type(module) is not self._cls:
                continue
            prefix = getattr(module, "prefix", name)
            is_quantized = getattr(self._candidate, "_is_quantized_linear", None)
            if not callable(is_quantized):
                raise RowAdapterError(
                    "candidate module does not expose _is_quantized_linear; the adapter "
                    "cannot prove the row-parallel layers are unquantized")
            if is_quantized(module):
                raise RowAdapterError(
                    f"{prefix}: adapter only supports unquantized row-parallel linear layers")
            dtype_name = str(getattr(module.weight, "dtype", None))
            if dtype_name != "torch.float16":
                raise RowAdapterError(
                    f"{prefix}: weight dtype is {dtype_name}; the FP16 path requires float16")
            if str(getattr(module, "output_dtype", None)) != "torch.float16":
                raise RowAdapterError(
                    f"{prefix}: output_dtype is {module.output_dtype}; expected float16")
            k_local, n = resolve_row_shape(tuple(module.weight.shape))
            if int(getattr(module, "input_size_per_partition", -1)) != k_local:
                raise RowAdapterError(f"{prefix}: input_size_per_partition is not K={k_local}")
            tp_size = int(getattr(module, "_tp_size", -1))
            world = int(getattr(module.group, "world_size", -1))
            if tp_size != self._expected_world_size or world != tp_size:
                raise RowAdapterError(
                    f"{prefix}: tp_size={tp_size} group.world_size={world}, expected "
                    f"{self._expected_world_size}")
            packed = self._packed_weights.get(module.weight)
            if tuple(int(v) for v in packed.shape) != (k_local // 16, n, 16):
                raise RowAdapterError(f"{prefix}: unexpected packed weight shape")
            layers.append((module, tensor_signature(module.weight)))
            per_shape[(k_local, n)] = per_shape.get((k_local, n), 0) + 1
            signatures[prefix] = str(getattr(module.weight, "_version", -1))
        expected_per_shape = self._expected_layer_count // len(SUPPORTED_ROW_SHAPES)
        if len(layers) != self._expected_layer_count:
            raise RowAdapterError(
                f"primed {len(layers)} row-parallel layers, expected exactly "
                f"{self._expected_layer_count}")
        for shape, count in per_shape.items():
            if count != expected_per_shape:
                raise RowAdapterError(
                    f"shape {shape} appeared {count} times, expected {expected_per_shape}")
        kernels = self.prepare_kernels()
        for k_local, n in SUPPORTED_ROW_SHAPES:
            self._ensure_scratch(k_local, n)
        cache = self._packed_weights.stats()
        expected_bytes = expected_cached_weight_bytes(
            self._expected_layer_count // len(SUPPORTED_ROW_SHAPES))
        if int(cache["bytes"]) != expected_bytes:
            raise RowAdapterError(
                f"cached packed weight bytes {cache['bytes']} != expected {expected_bytes}")
        scratch_bytes = sum(entry["bytes"] for entry in self._scratch.values())
        self._primed_layers = layers
        self._primed_metadata = {
            "round": ROUND,
            "primed": True,
            "layer_count": len(layers),
            "layers_per_shape": {f"k{k}_n{n}": c for (k, n), c in sorted(per_shape.items())},
            "primed_parameter_versions_recorded": len(signatures),
            "cached_weight_entries": int(cache["entries"]),
            "cached_weight_bytes": int(cache["bytes"]),
            "expected_cached_weight_bytes": expected_bytes,
            "cached_weight_dtype": "float16",
            "fp32_weight_shadows": 0,
            "weight_roundtrip_verifications": int(cache["roundtrip_verifications"]),
            "scratch_bytes": int(scratch_bytes),
            "kernel_dir": kernels["kernel_dir"],
            "kernels": {f"k{k}_n{n}": v for (k, n), v in kernels["kernels"].items()},
            "runtime": kernels["runtime"],
            "kernel_module_sha256": kernels["kernel_module_sha256"],
            "pinned": dict(PINNED),
        }
        return dict(self._primed_metadata)

    # -- controls ----------------------------------------------------------

    def controls(self, sample_input=None, k_local=None) -> dict:
        """Pre-warm proof: exact A packing and unchanged parameter identity.

        With a real activation sample (the same Half tensor the encoder will
        produce) this compares the adapter's zero-padded pack against the
        verified Round83 ``pack_a`` helper bit-for-bit, and checks that the
        padded M rows are exactly zero. Without a sample it re-verifies that
        every primed parameter still matches its cached signature.
        """
        if self._primed_metadata is None:
            raise RowAdapterError("controls require a primed adapter")
        for layer, _signature in self._primed_layers:
            self._packed_weights.verify_unchanged(layer.weight)
        result = {"parameters_unchanged": True,
                  "parameter_identity_checks": len(self._primed_layers),
                  "a_pack_matches_verified_helper": None,
                  "a_pack_roundtrip_exact": None,
                  "a_pack_tail_rows_zero": None}
        if sample_input is not None:
            import torch
            kernel_module = self._resolve_kernel_module()
            if k_local is None:
                k_local = int(sample_input.shape[-1])
            if (k_local, EXPECTED_HIDDEN_SIZE) not in SUPPORTED_ROW_SHAPES:
                raise RowAdapterError(f"control sample K={k_local} is not a supported shape")
            batch_shape, m = resolve_token_layout(
                tuple(sample_input.shape), k_local, self._sequence_length)
            flat = sample_input.contiguous().reshape(m, k_local)
            reference = kernel_module.pack_a(flat)
            buffer = torch.zeros_like(reference)
            self._pack_a_into(buffer, flat)
            result["a_pack_matches_verified_helper"] = bool(torch.equal(buffer, reference))
            result["a_pack_roundtrip_exact"] = bool(torch.equal(
                buffer[:, :m, :].transpose(0, 1).reshape(m, k_local), flat))
            result["a_pack_tail_rows_zero"] = bool(
                int(torch.count_nonzero(buffer[:, m:, :])) == 0) if m < buffer.shape[1] else True
            result["sample_shape"] = list(int(v) for v in sample_input.shape)
            if not (result["a_pack_matches_verified_helper"]
                    and result["a_pack_roundtrip_exact"]
                    and result["a_pack_tail_rows_zero"]):
                raise RowAdapterError("activation packing control failed")
            self._stats["pack_controls"] += 1
        return result

    # -- enable / disable --------------------------------------------------

    def set_enabled(self, enabled: bool) -> None:
        enabled = bool(enabled)
        if enabled and self._primed_metadata is None:
            raise RowAdapterError("set_enabled(True) requires prime() first")
        if enabled != self._enabled:
            self._stats["enable_transitions"].append(
                {"enabled": enabled, "at_monotonic": round(time.monotonic(), 3)})
        self._enabled = enabled

    @property
    def enabled(self) -> bool:
        return self._enabled

    # -- stream binding ----------------------------------------------------

    def _check_stream(self) -> None:
        import torch
        stream = torch.npu.current_stream()
        pointer = int(stream.npu_stream)
        if self._stream_ptr is None:
            self._stream_ptr = pointer
            self._stats["stream_binds"] += 1
            return
        if pointer != self._stream_ptr:
            self._stats["stream_conflicts"] += 1
            if not self._allow_stream_switch:
                raise RowAdapterError(
                    "row-parallel adapter activation buffers are owned by the stream they "
                    "were created on; reusing them from another stream is refused. Call "
                    "bind_stream() after synchronising, or construct the adapter with "
                    "allow_stream_switch=True")
            torch.npu.synchronize()
            self._stream_ptr = pointer
            self._stats["stream_binds"] += 1

    def bind_stream(self) -> int:
        """Explicitly re-bind after the caller synchronised and switched stream."""
        import torch
        torch.npu.synchronize()
        self._stream_ptr = int(torch.npu.current_stream().npu_stream)
        self._stats["stream_binds"] += 1
        return self._stream_ptr

    # -- the enabled forward ----------------------------------------------

    def _record_dtype(self, counter: str, value) -> None:
        key = str(value)
        bucket = self._stats[counter]
        bucket[key] = int(bucket.get(key, 0)) + 1

    def _validate_layer(self, layer) -> tuple[int, int]:
        import torch
        if type(layer) is not self._cls:
            raise RowAdapterError(
                f"adapter refuses a layer of type {type(layer).__name__}; only the pinned "
                f"{ROW_CLASS_NAME} is supported")
        if self._candidate._is_quantized_linear(layer):
            raise RowAdapterError("adapter refuses a quantized row-parallel layer")
        weight = layer.weight
        if weight is None or not torch.is_tensor(weight):
            raise RowAdapterError("row-parallel layer has no weight tensor")
        if weight.dtype != torch.float16:
            raise RowAdapterError(f"weight dtype is {weight.dtype}, expected float16")
        k_local, n = resolve_row_shape(tuple(weight.shape))
        if str(layer.output_dtype) != "torch.float16":
            raise RowAdapterError(f"output_dtype is {layer.output_dtype}, expected float16")
        if int(layer._tp_size) != self._expected_world_size:
            raise RowAdapterError(
                f"layer tp_size is {layer._tp_size}, expected {self._expected_world_size}")
        if int(layer.group.world_size) != int(layer._tp_size):
            raise RowAdapterError("layer group world size disagrees with its tp_size")
        bias = getattr(layer, "bias", None)
        if bias is not None:
            if not torch.is_tensor(bias) or bias.dtype != torch.float16:
                raise RowAdapterError("row-parallel bias must be an original float16 tensor")
            if tuple(int(v) for v in bias.shape) != (n,):
                raise RowAdapterError(f"row-parallel bias shape {tuple(bias.shape)} != ({n},)")
        return k_local, n

    def _forward(self, layer, input_):
        self._stats["calls_enabled"] += 1
        import torch
        if not self._enabled:
            raise RowAdapterError("_forward called while the adapter is disabled")
        if self._primed_metadata is None:
            raise RowAdapterError("adapter used before prime()")
        k_local, n = self._validate_layer(layer)
        if not torch.is_tensor(input_):
            raise RowAdapterError("row-parallel input is not a tensor")
        if input_.dtype != torch.float16:
            raise RowAdapterError(f"activation dtype is {input_.dtype}, expected float16")
        device = input_.device
        if device.type not in ("npu", "privateuseone"):
            raise RowAdapterError(f"activation device {device} is not an Ascend NPU device")
        if device.index is not None and int(device.index) != int(self._runtime.device):
            raise RowAdapterError(
                f"activation is on {device} but the runtime owns device {self._runtime.device}")

        # 1. Shard the activation exactly like the native forward.
        if bool(getattr(layer, "input_is_parallel", True)):
            input_parallel = input_
        else:
            split_input = input_.split(layer.input_size_per_partition, dim=-1)
            input_parallel = split_input[layer._tp_rank].contiguous()
            self._stats["input_is_parallel_split"] += 1

        batch_shape, tokens = resolve_token_layout(
            input_parallel.shape, k_local, self._sequence_length)
        flat = input_parallel.contiguous().reshape(tokens, k_local)

        self._check_stream()
        meta = self._kernel_metas[(k_local, n)]
        scratch = self._ensure_scratch(k_local, n)
        packed_weight = self._packed_weights.get(layer.weight)

        # 2. Pack the original Half activation; padding rows stay zero.
        self._pack_a_into(scratch["a"], flat)

        # 3. Verified kernel: Half x Half -> FP32 accumulator -> FP32 GM.
        buffers = [scratch["a"], packed_weight, scratch["aux"][0], scratch["aux"][1],
                   scratch["c"]]
        self._runtime.launch(meta, buffers, synchronize=True)
        self._stats["kernel_launches"] += 1
        shape_key = f"k{k_local}_n{n}"
        self._stats["kernel_shapes"][shape_key] = int(
            self._stats["kernel_shapes"].get(shape_key, 0)) + 1

        # 4. FP32 kernel output -> native output shape (fresh ND tensor).
        kernel_module = self._resolve_kernel_module()
        partial = kernel_module.unpack_c(scratch["c"], m=tokens)
        if partial.dtype != torch.float32:
            raise RowAdapterError(
                f"row-parallel partial is {partial.dtype} after the kernel; the whole point "
                f"of this path is an FP32 partial")
        partial = partial.reshape(output_shape_for(batch_shape, tokens, n))

        # 5. Keep FP32 through the layer's own collective.
        if int(layer._tp_size) > 1:
            self._record_dtype("fp32_dtype_before_collective", partial.dtype)
            reduced = layer.group.all_reduce(partial)
            self._record_dtype("fp32_dtype_after_collective", reduced.dtype)
            self._stats["fp32_collective_count"] += 1
            if reduced.dtype != torch.float32:
                raise RowAdapterError(
                    f"row-parallel collective returned {reduced.dtype}; a narrowed collective "
                    f"cannot be undone")
        else:
            reduced = partial
            self._stats["single_rank_no_collective_calls"] += 1
        if reduced.dtype != torch.float32:
            raise RowAdapterError("reduced row-parallel output is not float32")

        # 6. Optional FP32 bias (inactive for H3: the layers are bias-free).
        bias = getattr(layer, "bias", None)
        if bias is not None:
            reduced = reduced + bias.float()
            self._stats["bias_added_fp32_count"] += 1

        # 7. Exactly one narrowing to the layer's output dtype.
        output = reduced.to(layer.output_dtype)
        self._stats["half_cast_count"] += 1
        if output.dtype != torch.float16:
            raise RowAdapterError(f"final row-parallel output is {output.dtype}")
        return output

    # -- reporting ---------------------------------------------------------

    def snapshot(self) -> dict:
        output = {
            "round": ROUND,
            "installed": bool(self._installed),
            "enabled": bool(self._enabled),
            "primed": self._primed_metadata is not None,
            "sequence_length": self._sequence_length,
            "expected_world_size": self._expected_world_size,
            "expected_layer_count": self._expected_layer_count,
            "kernel_dir": str(self._kernel_dir),
            "stream_bound": self._stream_ptr is not None,
            "stats": json.loads(json.dumps(self._stats)),
            "weight_cache": self._packed_weights.stats(),
            "scratch_bytes": int(sum(entry["bytes"] for entry in self._scratch.values())),
            "kernel_shapes_prepared": sorted(f"k{k}_n{n}" for (k, n) in self._kernel_metas),
            "pinned": dict(PINNED),
            "one_cast_per_call": True,
            "fp32_partial_through_collective": True,
            "no_fp32_weight_shadow": True,
            "no_bf16_compute": True,
        }
        if self._primed_metadata is not None:
            output["prime"] = {key: value for key, value in self._primed_metadata.items()
                               if key != "pinned"}
        try:
            import torch
            if torch.npu.is_available():
                output["device_memory"] = {
                    "allocated_bytes": int(torch.npu.memory_allocated()),
                    "reserved_bytes": int(torch.npu.memory_reserved()),
                    "max_allocated_bytes": int(torch.npu.max_memory_allocated()),
                }
        except Exception:  # noqa: BLE001 - reporting must never break the run
            output["device_memory"] = None
        return output


# ---------------------------------------------------------------------------
# Module-level install()
# ---------------------------------------------------------------------------

_INSTALLED: dict[int, RowFp32Adapter] = {}


def install(candidate, runtime, kernel_dir, *, sequence_length=EXPECTED_SEQUENCE_LENGTH,
            expected_world_size=EXPECTED_WORLD_SIZE,
            expected_layer_count=EXPECTED_LAYER_COUNT,
            verify_sources=True, allow_stream_switch=False) -> RowFp32Adapter:
    """Install the gated wrapper on the candidate's row-parallel class.

    ``verify_sources`` is True by default and re-checks the pinned candidate file
    plus the Round83 kernel/runtime module hashes lazily. It exists as an escape
    hatch for offline unit tests with stand-in modules; a production run must
    leave it True.
    """
    if verify_sources:
        candidate_file = getattr(candidate, "__file__", None)
        if not candidate_file:
            raise RowAdapterError("candidate module has no __file__; cannot verify the pin")
        verify_candidate_source(candidate_file)
    cls = getattr(candidate, ROW_CLASS_NAME, None)
    if cls is None:
        raise RowAdapterError(f"candidate module has no {ROW_CLASS_NAME}")
    existing = _INSTALLED.get(id(cls))
    if existing is not None and existing._installed:
        raise RowAdapterError("a Round84 adapter is already installed on this class")
    adapter = RowFp32Adapter(
        candidate, runtime, kernel_dir, sequence_length=sequence_length,
        expected_world_size=expected_world_size,
        expected_layer_count=expected_layer_count,
        verify_sources=verify_sources, allow_stream_switch=allow_stream_switch)
    adapter._install_forward()
    _INSTALLED[id(cls)] = adapter
    return adapter

#: The root driver imports this name; ``install`` stays the documented spelling.
install_row = install
