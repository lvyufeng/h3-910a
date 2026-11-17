#!/usr/bin/env python3
"""Round81 independent FP16 K/V small-N zero-padding stability adapter.

WHY THIS EXISTS
---------------
Round80 measured, on a private FP16 candidate, that the text-attention K/V
projections (``N=128``, ``M=2043``, ``K=5120``) are NOT bit-repeatable on 910A:
0 of 8 repeated calls matched bit-for-bit. The same GEMM re-tiled to ``N=256``
by appending **all-zero output rows** became bit-repeatable 8 of 8 times, while
the retained ``[..., :128]`` columns still came from the original FP16 weight
values. Only zero rows are added and the trailing outputs are discarded, so the
mathematical function is unchanged; only the GEMM tiling changes.

This module installs that re-tiling into the Round79 private candidate at
exactly two call sites. It is a **diagnostic stability experiment**:

* it does not touch production code, the CPU FP16 gold, or the Q projection and
  MLP;
* it never substitutes BF16, never builds FP32 shadow copies of the weights, and
  never patches module-global ``torch.nn.functional``;
* it is NOT executed here and proves nothing about full-encoder precision, the
  encoder <=2 s target, or official-lossless quality.

USAGE (root driver, inside the private NPU run)
-----------------------------------------------
    from kv_padding81 import install
    kv_adapter = install(candidate)          # after candidate import
    ...                                      # move encoder to NPU
    kv_adapter.prime(encoder.text_model)     # optional warm cache, untimed
    ...
    r['kv_padding'] = kv_adapter.snapshot()
    ...
    kv_adapter.uninstall()                   # finally: restore + clear

Importing this module does NOT require torch: every torch use is a lazy import
inside a function, so stdlib-only static tests can import and inspect it.
"""

from __future__ import annotations

import ast
import hashlib
import inspect
import textwrap

__all__ = ["install", "KVPaddingError", "PAD_OUTPUT_ROWS", "ORIGINAL_KV_ROWS"]

#: Output rows of the original K/V projection on this model (128 * 1 head).
ORIGINAL_KV_ROWS = 128
#: Diagnostic re-tiling target. Measured bit-repeatable 8/8 in Round80 micro.
PAD_OUTPUT_ROWS = 256
#: Exactly two calls must be rewritten: K and V. Q and MLP stay untouched.
EXPECTED_TARGETS = ("k_weight", "v_weight")
#: 50 decoder layers x (K, V).
EXPECTED_CACHE_ENTRIES = 100
#: The Round79 generated candidate these two call sites were verified against.
PINNED_CANDIDATE_SHA256 = (
    "2141b29adc9ecc601d7fa2599e6ba56351adfbdaa5f16acb62c00d95cee3cf78"
)


class KVPaddingError(RuntimeError):
    """Raised when the adapter refuses to proceed instead of guessing."""


def _torch():
    import torch  # lazy: keeps this module importable without torch

    return torch


def _no_grad_context(torch):
    """Prefer inference_mode (no autograd graph at all), else no_grad."""
    inference_mode = getattr(torch, "inference_mode", None)
    if callable(inference_mode):
        return inference_mode()
    return torch.no_grad()


# ---------------------------------------------------------------------------
# AST transform (pure stdlib; unit-testable without torch)
# ---------------------------------------------------------------------------


def _is_f_linear(node: ast.AST) -> bool:
    return (
        isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == "linear"
        and isinstance(node.func.value, ast.Name)
        and node.func.value.id == "F"
    )


def _target_name(node: ast.Call):
    """Return the ``<name>_weight`` identifier if this is the call we rewrite."""
    if not _is_f_linear(node):
        return None
    if len(node.args) != 2 or node.keywords:
        return None
    weight = node.args[1]
    if isinstance(weight, ast.Name) and weight.id in EXPECTED_TARGETS:
        return weight.id
    return None


def transform_forward_source(source: str, helper_name: str = "_r81_kv_linear"):
    """Rewrite exactly the K/V ``F.linear`` calls in a forward-method source.

    Returns ``(new_source, matched_names)``. ``matched_names`` must be exactly
    ``("k_weight", "v_weight")`` -- two calls, no more, no fewer. Anything else
    raises, because a silent partial rewrite would produce a mixed graph that
    looks like the experiment but is not the experiment.
    """
    dedented = textwrap.dedent(source)
    tree = ast.parse(dedented)

    matched: list[str] = []
    replaced = 0

    class _Rewriter(ast.NodeTransformer):
        def visit_Call(self, node: ast.Call):  # noqa: N802 - ast API name
            nonlocal replaced
            name = _target_name(node)
            if name is None:
                return self.generic_visit(node)
            matched.append(name)
            replaced += 1
            return ast.Call(
                func=ast.Name(id=helper_name, ctx=ast.Load()),
                args=node.args,
                keywords=node.keywords,
            )

    rewritten = _Rewriter().visit(tree)
    ast.fix_missing_locations(rewritten)

    if tuple(matched) != EXPECTED_TARGETS:
        raise KVPaddingError(
            "expected exactly the K/V F.linear calls "
            f"{EXPECTED_TARGETS}, matched {tuple(matched)}; refusing a partial rewrite"
        )
    if replaced != len(EXPECTED_TARGETS):
        raise KVPaddingError(f"rewrote {replaced} calls, expected {len(EXPECTED_TARGETS)}")

    new_source = ast.unparse(rewritten)
    # Cheap structural re-check on the emitted text: the Q call must survive.
    if new_source.count("F.linear(hidden_states, q_weight)") != 1:
        raise KVPaddingError("Q projection call was not preserved; refusing to install")
    if new_source.count(f"{helper_name}(hidden_states, k_weight)") != 1:
        raise KVPaddingError("K call was not rewritten exactly once")
    if new_source.count(f"{helper_name}(hidden_states, v_weight)") != 1:
        raise KVPaddingError("V call was not rewritten exactly once")
    return new_source, tuple(matched)


# ---------------------------------------------------------------------------
# Adapter
# ---------------------------------------------------------------------------


class KVPaddingAdapter:
    """Pads only the small-N FP16 K/V GEMM output rows with zeros."""

    def __init__(self, candidate, pad_rows=PAD_OUTPUT_ROWS):
        self._candidate = candidate
        self._pad_rows = int(pad_rows)
        self._cache = {}
        self._class = None
        self._original_forward = None
        self._installed = False
        self._primed = False
        self._transform_info = {}
        self._injected_helper_name = None
        self._previous_helper = None
        self._stats = {
            "helper_calls": 0,
            "padded_calls": 0,
            "passthrough_calls": 0,
            "cache_hits": 0,
            "cache_misses": 0,
            "cache_evictions": 0,
        }

    # -- public API --------------------------------------------------------

    def prime(self, text_model):
        """Eagerly build all K/V padded caches so they are outside timed calls.

        Iterates the decoder layers and materialises each cached weight from the
        real parameters. Returns the number of entries now cached.
        """
        if not self._installed:
            raise KVPaddingError("prime() called before install() completed")
        torch = _torch()
        layers = getattr(text_model, "layers", None)
        if layers is None:
            raise KVPaddingError("prime() expects the text model (missing .layers)")
        built = 0
        with _no_grad_context(torch):
            for layer in layers:
                attn = getattr(layer, "self_attn", None)
                qkv = getattr(attn, "qkv_proj", None)
                weight = getattr(qkv, "weight", None)
                if weight is None:
                    continue
                local = int(attn.qkv_proj.local_num_kv_heads) * int(attn.head_dim)
                q_local = int(attn.qkv_proj.local_num_heads) * int(attn.head_dim)
                for start in (q_local, q_local + local):
                    self._cached_weight(weight[start : start + local])
                    built += 1
        self._primed = True
        return built

    def snapshot(self):
        """Return measured cache/statistics data (no torch tensors)."""
        padded_bytes = sum(entry["padded_bytes"] for entry in self._cache.values())
        return {
            "installed": self._installed,
            "primed": self._primed,
            "original_kv_rows": ORIGINAL_KV_ROWS,
            "padded_output_rows": self._pad_rows,
            "only_zero_output_rows_added": True,
            "trailing_output_discarded": True,
            "cache_entries": len(self._cache),
            "expected_cache_entries": EXPECTED_CACHE_ENTRIES,
            "cache_bytes": padded_bytes,
            "padding_bytes": sum(entry["padding_bytes"] for entry in self._cache.values()),
            "pad_ratio": (self._pad_rows / ORIGINAL_KV_ROWS),
            "max_abs_prefix_mismatch": 0.0,
            "all_prefix_exact": all(entry["prefix_exact"] for entry in self._cache.values()),
            "tail_nonzero_count": sum(entry["tail_nonzero"] for entry in self._cache.values()),
            "all_cached_padded_are_zero_offset": all(
                entry["padded_storage_offset"] == 0 for entry in self._cache.values()
            ),
            "no_cached_padded_requires_grad": all(
                not entry["padded_requires_grad"] for entry in self._cache.values()
            ),
            "no_cached_padded_holds_autograd_graph": all(
                entry["padded_is_leaf"] and not entry["padded_requires_grad"]
                for entry in self._cache.values()
            ),
            "cached_prefix_sha256": None,
            "param_signature_contract": "storage_data_ptr+storage_offset+shape+stride+dtype+device+_version",
            "transform": dict(self._transform_info),
            "stats": dict(self._stats),
            "not_executed_here": True,
            "proves_full_encoder_precision": False,
        }

    def clear(self):
        """Drop cached padded weights (safe before offload)."""
        self._stats["cache_evictions"] += len(self._cache)
        self._cache.clear()
        return self

    def uninstall(self):
        """Restore the original forward and release the cache."""
        if self._installed and self._class is not None:
            self._class.forward = self._original_forward
        if self._candidate is not None and self._injected_helper_name:
            module_globals = getattr(self._candidate, "__dict__", {})
            helper = module_globals.get(self._injected_helper_name)
            # Bound-method attribute access creates a new wrapper each time;
            # `is self._kv_linear` would never match the installed object.
            if (getattr(helper, '__self__', None) is self
                    and getattr(helper, '__func__', None) is type(self)._kv_linear):
                if self._previous_helper is None:
                    module_globals.pop(self._injected_helper_name, None)
                else:
                    module_globals[self._injected_helper_name] = self._previous_helper
        self._installed = False
        self._primed = False
        self.clear()
        return self

    # -- internals ---------------------------------------------------------

    def _signature(self, weight):
        """Fail-closed identity of the exact parameter view we cache."""
        try:
            version = int(weight._version)
        except (AttributeError, TypeError):  # pragma: no cover - defensive
            version = -1
        return (
            weight.untyped_storage().data_ptr(),
            int(weight.storage_offset()),
            tuple(int(v) for v in weight.shape),
            tuple(int(v) for v in weight.stride()),
            str(weight.dtype),
            str(weight.device),
            version,
        )

    def _cached_weight(self, weight):
        torch = _torch()
        if weight.shape[0] != ORIGINAL_KV_ROWS:
            raise KVPaddingError(
                f"unexpected K/V output width {weight.shape[0]}, expected {ORIGINAL_KV_ROWS}"
            )
        key = self._signature(weight)
        entry = self._cache.get(key)
        if entry is not None:
            return entry

        # Refuse to silently re-cache a *changed* parameter under a stale key.
        for known in self._cache:
            # K and V are different views of the SAME QKV parameter storage.
            # A shared storage pointer alone does not mean a mutated view.
            # Only the same logical slice (pointer + offset) changing signature
            # is stale;distinct K/V offsets must coexist in the cache.
            if known[0] == key[0] and known[1] == key[1] and known != key:
                raise KVPaddingError(
                    "cached K/V parameter signature changed (version/shape/stride); "
                    "refusing to silently re-cache a mutated weight"
                )

        rows, width = int(weight.shape[0]), int(weight.shape[1])
        padded = torch.zeros(
            (self._pad_rows, width), dtype=weight.dtype, device=weight.device
        )
        # detach() the source so the cache can never hold an autograd graph node
        # even if the caller invokes this outside no_grad/inference_mode.
        padded[:rows].copy_(weight.detach())
        if int(padded.storage_offset()) != 0:
            raise KVPaddingError("padded weight must be a zero-offset tensor")

        # Exact, device-side verification (no D2H copy, no value hashing):
        # the retained prefix must be bit-identical and the tail must be zero.
        prefix_exact = bool(torch.equal(padded[:rows], weight.detach()))
        tail_nonzero = int(torch.count_nonzero(padded[rows:]).item())
        if not prefix_exact or tail_nonzero != 0:
            raise KVPaddingError(
                "padded weight is not an exact zero-pad of the original "
                f"(prefix exact {prefix_exact}, tail nonzero {tail_nonzero})"
            )

        entry = {
            "padded": padded,
            "padded_bytes": int(padded.numel() * padded.element_size()),
            "padding_bytes": int((self._pad_rows - rows) * width * padded.element_size()),
            "padded_storage_offset": int(padded.storage_offset()),
            "padded_requires_grad": bool(padded.requires_grad),
            "padded_is_leaf": bool(padded.is_leaf),
            "prefix_exact": prefix_exact,
            "tail_nonzero": tail_nonzero,
            "torch_dtype": str(weight.dtype),
        }
        self._cache[key] = entry
        self._stats["cache_misses"] += 1
        return entry

    def _kv_linear(self, x, weight):
        """Drop-in replacement for the two K/V ``F.linear`` calls."""
        self._stats["helper_calls"] += 1
        torch = _torch()
        usable = (
            weight.shape[0] == ORIGINAL_KV_ROWS
            and weight.dtype == torch.float16
            and x.dtype == torch.float16
            and x.device.type in ("npu", "privateuseone")
        )
        if not usable:
            # Any other geometry/dtype keeps the exact original behaviour.
            self._stats["passthrough_calls"] += 1
            return torch.nn.functional.linear(x, weight)

        entry = self._cached_weight(weight)
        self._stats["cache_hits"] += 1
        self._stats["padded_calls"] += 1
        out = torch.nn.functional.linear(x, entry["padded"])
        return out[..., :ORIGINAL_KV_ROWS].contiguous()

    # -- install -----------------------------------------------------------

    def install(self):
        candidate = self._candidate
        cls = getattr(candidate, "MiniMaxH3Qwen3VLTextAttention", None)
        if cls is None:
            raise KVPaddingError("candidate has no MiniMaxH3Qwen3VLTextAttention")
        if getattr(cls.forward, "__round81_kv_padding__", False):
            raise KVPaddingError("adapter already installed on this class")

        original_forward = cls.forward
        try:
            source = inspect.getsource(original_forward)
        except OSError as error:  # pragma: no cover - source must be inspectable
            raise KVPaddingError(f"cannot read attention forward source: {error}") from error

        new_source, matched = transform_forward_source(source)
        module_globals = getattr(candidate, "__dict__", None)
        if module_globals is None:  # pragma: no cover - defensive
            raise KVPaddingError("candidate module has no __dict__ for exec namespace")

        # Exec into the LIVE module namespace, not a copy: the driver patches
        # module-level symbols (e.g. _scaled_dot_product_attention) and the
        # rewritten forward must resolve them dynamically, in any order.
        previous_helper = module_globals.get("_r81_kv_linear")
        had_forward = "forward" in module_globals
        previous_forward_name = module_globals.get("forward")
        module_globals["_r81_kv_linear"] = self._kv_linear
        try:
            code = compile(new_source, f"<round81_kv_padding {matched}>", "exec")
            exec(code, module_globals)  # noqa: S102 - explicit, reviewed, candidate-private
            new_forward = module_globals.get("forward")
        finally:
            # Leave only the helper behind; never shadow a real module attribute.
            if had_forward:
                module_globals["forward"] = previous_forward_name
            else:
                module_globals.pop("forward", None)

        if new_forward is None:  # pragma: no cover - defensive
            raise KVPaddingError("transformed source did not define forward()")
        self._injected_helper_name = "_r81_kv_linear"
        self._previous_helper = previous_helper
        new_forward.__round81_kv_padding__ = True
        new_forward.__round81_original__ = original_forward

        self._class = cls
        self._original_forward = original_forward
        self._installed = True
        self._transform_info = {
            "candidate_module": getattr(candidate, "__name__", "candidate"),
            "class_name": "MiniMaxH3Qwen3VLTextAttention",
            "original_forward_source_sha256": hashlib.sha256(
                textwrap.dedent(source).encode("utf-8")
            ).hexdigest(),
            "transformed_source_sha256": hashlib.sha256(new_source.encode("utf-8")).hexdigest(),
            "matched_calls": list(matched),
            "rewritten_call_count": len(matched),
            "helper_name": "_r81_kv_linear",
            "exec_namespace": "live_candidate_module_dict",
            "module_namespace_left_behind": ["_r81_kv_linear"],
            "global_functional_patched": False,
            "quantized_qkv_path_untouched": True,
            "q_projection_untouched": True,
            "mlp_untouched": True,
        }
        cls.forward = new_forward
        return self


def install(candidate, pad_rows=PAD_OUTPUT_ROWS):
    """Install the K/V zero-padding adapter on a loaded candidate module."""
    return KVPaddingAdapter(candidate, pad_rows=pad_rows).install()