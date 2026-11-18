#!/usr/bin/env python3
"""Round89 real-operand vision LayerNorm/GELU controls + scoped Float adapter.

SOURCE ONLY. Nothing here has been compiled, launched or executed on an NPU by
its author: no SSH, no NPU/GPU work, no model run, no CANN build, no install, no
production edit and no commit. Every number belongs to the parent driver, which
captures the real Half operands and supplies ``compare``/``tensor_info``.

WHY THIS EXISTS
---------------
R87 (one-cast biased Half linear) and R88 (accurate head_dim=72 attention) each
improved their own operator yet moved the whole 27-block vision error only from
1.4568% to 1.2887%/1.1934%, and the attention route cost more than 2x the resident
vision time. Before another full-graph change, this module measures the remaining
*non-linear* vision leaves on their real operands:

* ``_NpuBF16CompatibleLayerNorm`` (an ``nn.LayerNorm`` subclass). For Half input it
  falls through to ``super().forward``, i.e. ``F.layer_norm``. Whether an NPU Half
  LayerNorm narrows once or more than once is measured, not assumed.
* ``nn.GELU`` in two *different* original modes that must never be unified:
  ``MiniMaxH3Qwen3VLVisionMLP.act_fn`` = ``nn.GELU(approximate="tanh")`` and
  ``MiniMaxH3Qwen3VLVisionPatchMerger.act_fn`` = ``nn.GELU()`` (exact erf).

STRUCTURE (validated from the real model object, never assumed)
---------------------------------------------------------------
27 blocks x (norm1, norm2) = 54, plus merger + 3 DeepStack mergers = 4
=> 58 LayerNorm instances: 55 with normalized_shape (1152,), 3 DeepStack ones
with (4608,) because they use ``use_postshuffle_norm``.
27 MLP GELUs (tanh) + 4 merger GELUs (none) = 31 GELU instances. eps stays 1e-6.

CONTROL MATRIX PER LEAF (same frozen real Half input and same original params)
------------------------------------------------------------------------------
1. ``native``: ``type(module).forward(module, x)`` so an instance patch can never
   contaminate the reference, called twice to prove determinism.
2. ``cpu_half``: ``F.layer_norm`` / ``F.gelu`` on CPU with Half input and the
   original Half gamma/bias -- the reference, not a Float weight teacher.
3. ``npu_float_op_then_half``: Float32 op with Float32 gamma/bias derived exactly
   from the Half values inside the op only, then one cast back to Half.
4. ``cpu_float_op_then_half``: the same experiment on CPU; comparing it with
   ``cpu_half`` tests the usual "Half op == Float op then one cast" assumption.

Variants 3 and 4 are diagnostics: no Float linear/Conv, no cached Float weight,
no BF16 value, and no replacement of the independent CPU FP16 reference. No
threshold is chosen here: a small leaf error does not make the encoder pass and a
large one is not waived.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

# --- pinned immutable source this module's assumptions were read from ---------
CANDIDATE_FILENAME = "candidate_encoder79.py"
PINNED_CANDIDATE_SHA256 = (
    "2141b29adc9ecc601d7fa2599e6ba56351adfbdaa5f16acb62c00d95cee3cf78")
NORM_CLASS_NAME = "_NpuBF16CompatibleLayerNorm"

# --- real H3 vision geometry / leaf inventory ---------------------------------
HIDDEN_SIZE = 1152
MERGED_HIDDEN_SIZE = 4608          # 1152 * spatial_merge_size**2 (spatial_merge_size=2)
MLP_INTERMEDIATE = 4304
TOKENS_PRE_MERGE = 8064            # 2 images * 4032 patch tokens
TOKENS_POST_MERGE = 2016           # 8064 / 4
EXPECTED_BLOCKS = 27
EXPECTED_MERGER_MODULES = 4        # merger + len(deepstack_visual_indexes)=3
EXPECTED_NORM_MODULES = 58         # 27*2 + 4
EXPECTED_GELU_MODULES = 31         # 27 + 4
NORM_EPS = 1e-6
EXPECTED_NORMALIZED_SHAPE_COUNTS = {1152: 55, 4608: 3}
EXPECTED_GELU_APPROXIMATE_COUNTS = {"tanh": 27, "none": 4}

#: The seven real leaf inputs the parent captures with forward pre-hooks.
LEAF_MODULE_NAMES = (
    "blocks.0.norm1",
    "blocks.0.norm2",
    "blocks.0.mlp.act_fn",
    "deepstack_merger_list.0.norm",
    "deepstack_merger_list.0.act_fn",
    "merger.norm",
    "merger.act_fn",
)

LEAF_KINDS = {
    "blocks.0.norm1": "layer_norm",
    "blocks.0.norm2": "layer_norm",
    "blocks.0.mlp.act_fn": "gelu",
    "deepstack_merger_list.0.norm": "layer_norm",
    "deepstack_merger_list.0.act_fn": "gelu",
    "merger.norm": "layer_norm",
    "merger.act_fn": "gelu",
}

#: Original activation modes. The MLP is ``tanh``; the mergers are exact ``erf``.
LEAF_GELU_APPROXIMATE = {
    "blocks.0.mlp.act_fn": "tanh",
    "deepstack_merger_list.0.act_fn": "none",
    "merger.act_fn": "none",
}

#: Real fixture geometry of those seven captures.
EXPECTED_LEAF_SHAPES = {
    "blocks.0.norm1": (TOKENS_PRE_MERGE, HIDDEN_SIZE),
    "blocks.0.norm2": (TOKENS_PRE_MERGE, HIDDEN_SIZE),
    "blocks.0.mlp.act_fn": (TOKENS_PRE_MERGE, MLP_INTERMEDIATE),
    "deepstack_merger_list.0.norm": (TOKENS_POST_MERGE, MERGED_HIDDEN_SIZE),
    "deepstack_merger_list.0.act_fn": (TOKENS_POST_MERGE, MERGED_HIDDEN_SIZE),
    "merger.norm": (TOKENS_PRE_MERGE, HIDDEN_SIZE),
    "merger.act_fn": (TOKENS_POST_MERGE, MERGED_HIDDEN_SIZE),
}

HALF_DTYPE_NAME = "torch.float16"
NPU_DEVICE_PREFIXES = ("npu", "privateuseone")


class VisionLeafError(RuntimeError):
    """Raised instead of guessing, silently falling back or relaxing a contract."""


# ---------------------------------------------------------------------------
# Stdlib helpers (no torch, no TIK: unit-testable anywhere)
# ---------------------------------------------------------------------------


def _sha256_file(path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def candidate_source_candidates() -> list:
    """Every location this round may have staged the pinned R79 candidate."""
    here = Path(__file__).resolve().parent
    return [
        here / CANDIDATE_FILENAME,
        here.parent / CANDIDATE_FILENAME,
        here.parent.parent / "round79" / "candidate" / CANDIDATE_FILENAME,
    ]


def locate_candidate_source() -> Path:
    seen = []
    for path in candidate_source_candidates():
        resolved = path.resolve()
        if resolved in seen:
            continue
        seen.append(resolved)
        if resolved.is_file():
            return resolved
    raise VisionLeafError(
        "cannot locate the pinned %s; stage it next to this module or use the "
        "committed Round79 copy" % CANDIDATE_FILENAME)


def verify_candidate_source(path=None) -> dict:
    """Fail closed unless the reviewed R79 candidate bytes are still in place."""
    resolved = Path(path).resolve() if path is not None else locate_candidate_source()
    if not resolved.is_file():
        raise VisionLeafError("candidate source missing: %s" % resolved)
    actual = _sha256_file(resolved)
    if actual != PINNED_CANDIDATE_SHA256:
        raise VisionLeafError(
            "candidate source changed: %s != %s; the vision modules this round "
            "inspects are no longer the reviewed ones" % (actual, PINNED_CANDIDATE_SHA256))
    return {"path": str(resolved), "sha256": actual,
            "expected_sha256": PINNED_CANDIDATE_SHA256, "matches_pin": True}


def leaf_kind(name: str) -> str:
    if name not in LEAF_KINDS:
        raise VisionLeafError("unknown vision leaf %r" % (name,))
    return LEAF_KINDS[name]


def planned_leaf_plan() -> list:
    """The seven leaves, their kind, expected shape and original activation mode."""
    plan = []
    for name in LEAF_MODULE_NAMES:
        kind = leaf_kind(name)
        plan.append({
            "name": name,
            "kind": kind,
            "shape": list(EXPECTED_LEAF_SHAPES[name]),
            "gelu_approximate": LEAF_GELU_APPROXIMATE.get(name),
        })
    return plan


def expected_controls_report_keys() -> tuple:
    """Keys every per-leaf control entry must carry (contract for the driver)."""
    return ("kind", "module_class", "geometry", "approximate",
            "native_vs_cpu_half", "npu_float_op_then_half_vs_cpu_half",
            "cpu_float_op_then_half_vs_cpu_half", "npu_float_op_vs_cpu_float_op",
            "native_vs_npu_float_op_then_half",
            "native_repeat_bit_identical", "cpu_half_repeat_bit_identical",
            "npu_float_op_supported", "npu_float_op_error",
            "input_sha_unchanged", "parameters_sha_unchanged",
            "no_accuracy_tolerance")


def adapter_snapshot_keys() -> tuple:
    return ("round", "installed", "modes", "norm_modules_patched",
            "gelu_modules_patched", "original_bound_methods_captured",
            "patches_torch_global", "patches_torch_functional",
            "patches_biased_linears_or_conv", "cached_float_weights",
            "norm_eps_untouched", "strict_half_and_device_guard",
            "disabled_is_exact_native", "numerical_tolerance")


# ---------------------------------------------------------------------------
# Torch-side structural validation (lazy import; no NPU call in this module)
# ---------------------------------------------------------------------------


def _validate_vision_structure(vision) -> dict:
    """Count and classify the real modules; refuse anything unexpected."""
    import torch
    from torch import nn

    blocks = getattr(vision, "blocks", None)
    mergers = getattr(vision, "deepstack_merger_list", None)
    merger = getattr(vision, "merger", None)
    if blocks is None or merger is None or mergers is None:
        raise VisionLeafError("vision model lacks blocks/merger/deepstack_merger_list")
    if len(blocks) != EXPECTED_BLOCKS:
        raise VisionLeafError("expected %d blocks, found %d" % (EXPECTED_BLOCKS, len(blocks)))
    merger_count = 1 + len(mergers)
    if merger_count != EXPECTED_MERGER_MODULES:
        raise VisionLeafError("expected %d merger modules, found %d"
                              % (EXPECTED_MERGER_MODULES, merger_count))

    norm_class = type(blocks[0].norm1)
    if norm_class.__name__ != NORM_CLASS_NAME:
        raise VisionLeafError("unexpected vision norm class %r" % (norm_class.__name__,))
    if not issubclass(norm_class, nn.LayerNorm):
        raise VisionLeafError("vision norm is not an nn.LayerNorm subclass")

    norms, gelus = {}, {}
    other_norms = []
    for name, module in vision.named_modules():
        if type(module) is norm_class:
            norms[name] = module
        elif isinstance(module, nn.LayerNorm):
            other_norms.append(name)
        if isinstance(module, nn.GELU):
            gelus[name] = module
    if other_norms:
        raise VisionLeafError("unexpected non-vision LayerNorm modules: %r" % (other_norms,))
    if len(norms) != EXPECTED_NORM_MODULES:
        raise VisionLeafError("expected %d norm modules, found %d"
                              % (EXPECTED_NORM_MODULES, len(norms)))
    if len(gelus) != EXPECTED_GELU_MODULES:
        raise VisionLeafError("expected %d GELU modules, found %d"
                              % (EXPECTED_GELU_MODULES, len(gelus)))

    shape_counts, eps_values = {}, set()
    for name, module in norms.items():
        shape = tuple(int(x) for x in module.normalized_shape)
        if len(shape) != 1 or shape[0] not in EXPECTED_NORMALIZED_SHAPE_COUNTS:
            raise VisionLeafError("unexpected normalized_shape %r at %s" % (shape, name))
        shape_counts[shape[0]] = shape_counts.get(shape[0], 0) + 1
        eps_values.add(float(module.eps))
        if module.elementwise_affine and not (module.weight is not None
                                              and module.bias is not None):
            raise VisionLeafError("norm %s claims affine but lacks weight/bias" % name)
    if shape_counts != EXPECTED_NORMALIZED_SHAPE_COUNTS:
        raise VisionLeafError("normalized_shape counts %r != %r"
                              % (shape_counts, EXPECTED_NORMALIZED_SHAPE_COUNTS))
    if eps_values != {NORM_EPS}:
        raise VisionLeafError("unexpected vision norm eps values %r" % (sorted(eps_values),))

    approximate_counts = {}
    for name, module in gelus.items():
        approximate = str(getattr(module, "approximate", "none"))
        if approximate not in ("none", "tanh"):
            raise VisionLeafError("unexpected GELU approximate %r at %s" % (approximate, name))
        approximate_counts[approximate] = approximate_counts.get(approximate, 0) + 1
    if approximate_counts != EXPECTED_GELU_APPROXIMATE_COUNTS:
        raise VisionLeafError("GELU approximate counts %r != %r"
                              % (approximate_counts, EXPECTED_GELU_APPROXIMATE_COUNTS))

    # The tested leaves must be exactly the reviewed ones and keep their modes.
    for name in LEAF_MODULE_NAMES:
        kind = leaf_kind(name)
        if kind == "layer_norm":
            if name not in norms:
                raise VisionLeafError("tested norm leaf %s is missing" % name)
            expected_width = (MERGED_HIDDEN_SIZE if name.startswith("deepstack_")
                              else HIDDEN_SIZE)
            if int(norms[name].normalized_shape[0]) != expected_width:
                raise VisionLeafError("norm leaf %s width %r != %d"
                                      % (name, norms[name].normalized_shape, expected_width))
        else:
            if name not in gelus:
                raise VisionLeafError("tested GELU leaf %s is missing" % name)
            actual = str(getattr(gelus[name], "approximate", "none"))
            if actual != LEAF_GELU_APPROXIMATE[name]:
                raise VisionLeafError(
                    "GELU leaf %s approximate is %r, original is %r"
                    % (name, actual, LEAF_GELU_APPROXIMATE[name]))

    return {
        "blocks": len(blocks),
        "merger_modules": merger_count,
        "norm_modules": len(norms),
        "gelu_modules": len(gelus),
        "norm_class": norm_class.__name__,
        "norm_eps": NORM_EPS,
        "norm_normalized_shape_counts": {str(k): v for k, v in sorted(shape_counts.items())},
        "gelu_approximate_counts": dict(sorted(approximate_counts.items())),
        "other_layernorm_modules": other_norms,
        "norms": norms,
        "gelus": gelus,
    }


def _half_operand(value, name: str, expected_shape) -> dict:
    """Validate one captured operand without converting its values."""
    import torch

    if str(value.dtype) != HALF_DTYPE_NAME:
        raise VisionLeafError("leaf input %s is %s, expected original Half"
                              % (name, value.dtype))
    if value.device.type not in NPU_DEVICE_PREFIXES:
        raise VisionLeafError("leaf input %s is on %r, expected an NPU device"
                              % (name, value.device.type))
    expected = tuple(int(x) for x in expected_shape)
    shape = tuple(int(x) for x in value.shape)
    if shape != expected:
        raise VisionLeafError("leaf input %s shape %r, expected %r"
                              % (name, list(shape), list(expected)))
    return {"shape": list(shape), "dtype": str(value.dtype),
            "device": str(value.device), "requires_grad": bool(value.requires_grad)}


def _cpu_half_op(module, kind: str, value):
    """The reference: same Half input, same original Half gamma/bias, one cast."""
    import torch
    import torch.nn.functional as F

    source = value.detach().to("cpu").contiguous()
    if kind == "layer_norm":
        return F.layer_norm(
            source, tuple(module.normalized_shape),
            module.weight.detach().to("cpu").contiguous() if module.weight is not None else None,
            module.bias.detach().to("cpu").contiguous() if module.bias is not None else None,
            float(module.eps))
    return F.gelu(source, approximate=str(getattr(module, "approximate", "none")))


def _float_op_then_half(module, kind: str, value, device: str):
    """Float32 op on the original Half values, then exactly one Half cast.

    The Float gamma/bias are derived from the original Half parameters inside the
    call and are never stored, so this is not a Float weight shadow and no Float
    linear/Conv exists.
    """
    import torch
    import torch.nn.functional as F

    source = value.detach().to(device=device).contiguous().float()
    if kind == "layer_norm":
        weight = (module.weight.detach().to(device=device).contiguous().float()
                  if module.weight is not None else None)
        bias = (module.bias.detach().to(device=device).contiguous().float()
                if module.bias is not None else None)
        out = F.layer_norm(source, tuple(module.normalized_shape), weight, bias,
                           float(module.eps))
    else:
        out = F.gelu(source, approximate=str(getattr(module, "approximate", "none")))
    return out, out.to(torch.float16)


def _cpu_float_op(module, kind: str, value):
    """Same Float experiment on CPU: isolates op precision from the cast order."""
    import torch
    import torch.nn.functional as F

    source = value.detach().to("cpu").contiguous().float()
    if kind == "layer_norm":
        weight = (module.weight.detach().cpu().contiguous().float()
                  if module.weight is not None else None)
        bias = (module.bias.detach().cpu().contiguous().float()
                if module.bias is not None else None)
        return F.layer_norm(source, tuple(module.normalized_shape), weight, bias,
                            float(module.eps))
    return F.gelu(source, approximate=str(getattr(module, "approximate", "none")))


def _native_op(module, value):
    """Class-level forward: the exact native path, immune to instance patching."""
    return type(module).forward(module, value)


def _parameter_info(module, kind: str, tensor_info) -> dict:
    if kind != "layer_norm":
        return {"parameters": 0}
    return {
        "parameters": int(module.weight is not None) + int(module.bias is not None),
        "weight": tensor_info(module.weight) if module.weight is not None else None,
        "bias": tensor_info(module.bias) if module.bias is not None else None,
        "eps": float(module.eps),
        "normalized_shape": [int(x) for x in module.normalized_shape],
        "elementwise_affine": bool(module.elementwise_affine),
    }


def _parameters_sha(module, kind: str) -> str:
    import torch

    digest = hashlib.sha256()
    if kind == "layer_norm":
        for label, value in (("weight", module.weight), ("bias", module.bias)):
            if value is None:
                digest.update(label.encode() + b":none")
                continue
            array = value.detach().cpu().contiguous().view(torch.uint8).numpy()
            digest.update(label.encode())
            digest.update(memoryview(array))
    return digest.hexdigest()


def run_controls(vision, inputs, compare, tensor_info, expected_shapes=None) -> dict:
    """Measure the seven real vision Norm/GELU leaves against a CPU Half oracle.

    ``inputs`` maps the seven :data:`LEAF_MODULE_NAMES` to the frozen NPU Half
    tensors the parent captured with forward pre-hooks. ``compare(a, b)`` and
    ``tensor_info(x)`` are injected so this module never owns a numerical
    convention. No accuracy threshold is chosen here.
    """
    import torch
    from torch import nn

    structure = _validate_vision_structure(vision)
    expected = dict(expected_shapes or EXPECTED_LEAF_SHAPES)
    if set(expected) != set(LEAF_MODULE_NAMES):
        raise VisionLeafError("expected_shapes must cover exactly the seven leaves")
    if set(inputs) != set(LEAF_MODULE_NAMES):
        raise VisionLeafError(
            "inputs must cover exactly %r, got %r"
            % (list(LEAF_MODULE_NAMES), sorted(inputs)))

    norms, gelus = structure.pop("norms"), structure.pop("gelus")
    for name in LEAF_MODULE_NAMES:
        if "forward" in _module_for(norms, gelus, name).__dict__:
            raise VisionLeafError(
                "leaf %s carries an instance forward patch; run_controls needs the "
                "untouched native operator" % name)
        expected[name] = tuple(expected[name])

    before_inputs, before_params, parameters = {}, {}, {}
    for name in LEAF_MODULE_NAMES:
        before_inputs[name] = tensor_info(inputs[name])["sha256"]
        kind = leaf_kind(name)
        module = norms[name] if kind == "layer_norm" else gelus[name]
        before_params[name] = _parameters_sha(module, kind)
        parameters[name] = _parameter_info(module, kind, tensor_info)

    controls = {}
    for name in LEAF_MODULE_NAMES:
        kind = leaf_kind(name)
        module = norms[name] if kind == "layer_norm" else gelus[name]
        operand = inputs[name]
        geometry = _half_operand(operand, name, expected[name])
        approximate = str(getattr(module, "approximate", "none")) if kind == "gelu" else None

        native = _native_op(module, operand)
        native_repeat = _native_op(module, operand)
        native_repeat_equal = bool(compare(native_repeat, native)["raw_bits_equal"])

        cpu_half = _cpu_half_op(module, kind, operand)
        cpu_half_repeat = _cpu_half_op(module, kind, operand)
        cpu_half_repeat_equal = bool(
            compare(cpu_half_repeat, cpu_half)["raw_bits_equal"])

        entry = {
            "kind": kind,
            "module_class": type(module).__name__,
            "geometry": geometry,
            "approximate": approximate,
            "native_vs_cpu_half": compare(native, cpu_half),
            "native_repeat_bit_identical": native_repeat_equal,
            "cpu_half_repeat_bit_identical": cpu_half_repeat_equal,
            "npu_float_op_supported": False,
            "npu_float_op_error": None,
            "no_accuracy_tolerance": True,
        }
        try:
            npu_float, npu_float_half = _float_op_then_half(
                module, kind, operand, device=operand.device)
            entry["npu_float_op_supported"] = True
            entry["npu_float_op_then_half_vs_cpu_half"] = compare(npu_float_half, cpu_half)
            entry["native_vs_npu_float_op_then_half"] = compare(native, npu_float_half)
            entry["npu_float_op_vs_cpu_float_op"] = compare(
                npu_float, _cpu_float_op(module, kind, operand))
        except Exception as error:  # noqa: BLE001 - recorded, never guessed away
            entry["npu_float_op_error"] = "%s: %s" % (type(error).__name__, error)
            for key in ("npu_float_op_then_half_vs_cpu_half",
                        "native_vs_npu_float_op_then_half",
                        "npu_float_op_vs_cpu_float_op"):
                entry.setdefault(key, None)
        entry["cpu_float_op_then_half_vs_cpu_half"] = compare(
            _cpu_float_op(module, kind, operand).to(torch.float16), cpu_half)

        after_params = _parameters_sha(module, kind)
        after_input = tensor_info(inputs[name])["sha256"]
        entry["input_sha_unchanged"] = after_input == before_inputs[name]
        entry["parameters_sha_unchanged"] = after_params == before_params[name]
        if not entry["parameters_sha_unchanged"]:
            raise VisionLeafError("leaf %s mutated its parameters" % name)
        if not entry["input_sha_unchanged"]:
            raise VisionLeafError("leaf %s mutated its input operand" % name)
        missing = [key for key in expected_controls_report_keys() if key not in entry]
        if missing:
            raise VisionLeafError("control entry %s misses %r" % (name, missing))
        controls[name] = entry

    report = {
        "scope": __doc__,
        "round": 89,
        "source_sha256": _sha256_file(Path(__file__)),
        "candidate_source": verify_candidate_source(),
        "structure": structure,
        "leaf_plan": planned_leaf_plan(),
        "inputs": {name: tensor_info(inputs[name]) for name in LEAF_MODULE_NAMES},
        "parameters": parameters,
        "controls": controls,
        "all_native_repeats_bit_identical": all(
            entry["native_repeat_bit_identical"] for entry in controls.values()),
        "all_cpu_half_repeats_bit_identical": all(
            entry["cpu_half_repeat_bit_identical"] for entry in controls.values()),
        "all_npu_float_diagnostics_supported": all(
            entry["npu_float_op_supported"] for entry in controls.values()),
        "all_inputs_unchanged": all(
            entry["input_sha_unchanged"] for entry in controls.values()),
        "all_leaf_parameters_unchanged": all(
            entry["parameters_sha_unchanged"] for entry in controls.values()),
        "same_frozen_half_operands_for_every_variant": True,
        "cpu_half_reference_uses_original_half_gamma_and_bias": True,
        "float_gamma_and_bias_are_ephemeral_inside_float_diagnostics": True,
        "no_float_weight_teacher": True,
        "no_float_linear_or_conv": True,
        "no_bf16_teacher": True,
        "no_accuracy_tolerance": True,
        "numerical_tolerance": None,
        "adapter_required_to_be_disabled_during_controls": True,
        "not_full_encoder_timing": True,
        "full_encoder_precision_accepted": False,
        "whole_goal_complete": False,
    }
    json.dumps(report)  # the driver writes this verbatim: it must be plain JSON
    return report


def _module_for(norms, gelus, name):
    return norms[name] if leaf_kind(name) == "layer_norm" else gelus[name]


# ---------------------------------------------------------------------------
# Optional scoped adapter: per-instance forward only, disabled by default
# ---------------------------------------------------------------------------


class VisionNonlinearAdapter89:
    """Gate Float32 Norm/GELU on exactly these vision modules, one instance each.

    Nothing global is touched: only the ``forward`` attribute of the counted norm
    and GELU *instances* is wrapped, so biased linears, the Conv3d patch embed and
    every other module keep their exact native behaviour. ``set_modes()`` toggles
    behaviour at runtime; with both modes off the wrapper calls the captured
    original bound method, so the disabled path is the native operator itself.
    """

    def __init__(self, vision):
        from types import MethodType

        structure = _validate_vision_structure(vision)
        self._vision = vision
        self._norms = structure.pop("norms")
        self._gelus = structure.pop("gelus")
        self._structure = structure
        self._original = {}
        self._had_instance_forward = {}
        self._wrappers = {}
        self._installed = False
        self._modes = {"norm": False, "gelu": False}
        self._guard_calls = 0
        self._forward_calls = {"norm": 0, "gelu": 0}
        for name in self._norms:
            module = self._norms[name]
            self._had_instance_forward[name] = "forward" in module.__dict__
            self._original[name] = getattr(module, "forward")
        for name in self._gelus:
            module = self._gelus[name]
            self._had_instance_forward[name] = "forward" in module.__dict__
            self._original[name] = getattr(module, "forward")
        for kind, modules in (("layer_norm", self._norms), ("gelu", self._gelus)):
            for name, module in modules.items():
                wrapper = MethodType(self._make_forward(name, kind), module)
                self._wrappers[name] = wrapper
                setattr(module, "forward", wrapper)
        self._installed = True

    # -- forwarding -------------------------------------------------------
    def _make_forward(self, name, kind):
        original = self._original[name]

        def forward(module, input_):
            return self._forward(name, kind, module, input_, original)

        return forward

    def _forward(self, name, kind, module, input_, original):
        mode_key = "norm" if kind == "layer_norm" else "gelu"
        if not self._installed or not self._modes[mode_key]:
            return original(input_)
        self._forward_calls[mode_key] += 1
        self._check_operand(name, input_)
        import torch
        import torch.nn.functional as F

        if kind == "layer_norm":
            weight = (module.weight.float() if module.weight is not None else None)
            bias = (module.bias.float() if module.bias is not None else None)
            out = F.layer_norm(input_.float(), tuple(module.normalized_shape),
                               weight, bias, float(module.eps))
            return out.to(input_.dtype)
        return F.gelu(input_.float(),
                      approximate=str(getattr(module, "approximate", "none"))
                      ).to(input_.dtype)

    def _check_operand(self, name, input_):
        import torch

        self._guard_calls += 1
        if not torch.is_tensor(input_):
            raise VisionLeafError("adapter leaf %s got a non-tensor input" % name)
        if input_.dtype is not torch.float16:
            raise VisionLeafError(
                "adapter leaf %s input is %s, the vision diagnostic is Half only"
                % (name, input_.dtype))
        if input_.device.type not in NPU_DEVICE_PREFIXES:
            raise VisionLeafError(
                "adapter leaf %s input is on %r, expected an NPU device"
                % (name, input_.device.type))

    # -- public API -------------------------------------------------------
    def set_modes(self, norm=False, gelu=False):
        for label, value in (("norm", norm), ("gelu", gelu)):
            if not isinstance(value, bool):
                raise VisionLeafError("set_modes(%s=) must be a bool" % label)
        if not self._installed:
            raise VisionLeafError("adapter was uninstalled; modes cannot be re-armed")
        self._modes = {"norm": bool(norm), "gelu": bool(gelu)}
        return self.snapshot()

    def set_enabled(self, value):  # convenience: both non-linear modes together
        if not isinstance(value, bool):
            raise VisionLeafError("set_enabled() must be a bool")
        return self.set_modes(norm=value, gelu=value)

    def snapshot(self) -> dict:
        return {
            "round": 89,
            "installed": bool(self._installed),
            "modes": dict(self._modes),
            "norm_modules_patched": len(self._norms),
            "gelu_modules_patched": len(self._gelus),
            "original_bound_methods_captured": len(self._original),
            "patched_module_names": sorted(self._wrappers),
            "structure": dict(self._structure),
            "per_instance_forward_only": True,
            "patches_torch_global": False,
            "patches_torch_functional": False,
            "patches_biased_linears_or_conv": False,
            "cached_float_weights": False,
            "float_gamma_bias_derived_per_call": True,
            "norm_eps_untouched": True,
            "norm_normalized_shape_untouched": True,
            "gelu_approximate_preserved_per_module": True,
            "strict_half_and_device_guard": True,
            "guard_checks": self._guard_calls,
            "forward_calls": dict(self._forward_calls),
            "disabled_is_exact_native": True,
            "single_cast_to_operand_dtype_after_float_op": True,
            "numerical_tolerance": None,
            "full_encoder_precision_accepted": False,
            "whole_goal_complete": False,
        }

    def uninstall(self) -> dict:
        if self._installed:
            for name, wrapper in self._wrappers.items():
                module = self._norms[name] if name in self._norms else self._gelus[name]
                current = getattr(module, "forward", None)
                if not self._is_our_patch(current, module, wrapper):
                    raise VisionLeafError(
                        "leaf %s forward was replaced by a third party after install; "
                        "the original was not restored blindly" % name)
                # Restore to a pristine module: drop our instance attribute when the
                # module did not carry one, so the class forward applies again and a
                # later run_controls() no longer sees an instance patch.
                if self._had_instance_forward[name]:
                    setattr(module, "forward", self._original[name])
                else:
                    delattr(module, "forward")
            self._installed = False
        self._modes = {"norm": False, "gelu": False}
        self._wrappers.clear()
        self._original.clear()
        self._had_instance_forward.clear()
        summary = self.snapshot()
        summary["uninstalled"] = True
        return summary

    @staticmethod
    def _is_our_patch(current, module, wrapper):
        # Bound-method identity, not ``is``: attribute access rebinds the function.
        return (getattr(current, "__self__", None) is module
                and getattr(current, "__func__", None) is wrapper.__func__)