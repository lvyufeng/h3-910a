#!/usr/bin/env python3
"""Round86 late-layer FP16 normalisation / MLP leaf controls (SOURCE ONLY).

WHAT THIS MEASURES
------------------
Round85 showed the complete TP8 text graph diverges mostly late: with precise
attention + FP32 row partials the raw layer24 relative error was ~4.8e-4 but
raw layer49 was ~2.7%. That localises the amplification to layers 24..49 but
does not explain it. This module isolates the *leaf operators* of one real late
decoder block against explicit CPU FP16 oracles that are computed from the
**actual captured native Half values**, so a dtype label is never trusted:

* ``input_layernorm``  : HF two-round FP16 order  (Float reduction -> Half
  normalised -> Half gamma product). This is the order R79 already implements;
  this module must not "fix" it into a single Float-gamma cast.
* MLP gate/up local linear : Half x Half F.linear on the real normed MLP input
  with the original Half weight slices (``intermediate_size_per_partition`` =
  3200 because the actual checkpoint ``intermediate_size`` is 25600; the older
  27648/K3456 figure was a Round84 planning mistake and is rejected here).
* ``F.silu`` on the **actual native Half gate**, then the Half elementwise
  multiply with the actual native Half up value.
* the local ``down_proj`` shard linear (K=3200 -> 5120). The row-parallel
  collective is deliberately NOT executed: this is a precision isolation and an
  asynchronous HCCL all-reduce on one rank would hang or distort the leaf view.

Anything Float here is an explicit diagnostic (for example ``F.silu(x.float())``
rounded once, to size the pointwise rounding cost). No FP32 weight shadows, no
BF16 teacher, no cast of an old BF16 baseline, no model/parameter mutation.

WHAT THIS IS NOT
----------------
Not a full-precision pass, not a speed result, not an official-lossless claim.
No numerical tolerance is chosen before or after seeing the numbers, and there
is no accuracy-pass field. Only structural booleans are asserted: the captured
inputs/weights must be unchanged (raw bit hashes before/after) and repeated
identical native calls must return identical bits.

This file is executed by the root driver on the leased NPU; the author performed
no SSH, NPU/GPU work, model run, installation, production edit or commit.
"""

from __future__ import annotations

import ast
import hashlib
import math
from pathlib import Path
import textwrap

# --- actual H3 text-encoder geometry (validated against the real checkpoint) --
SEQUENCE_LENGTH = 2043
HIDDEN_SIZE = 5120
#: actual ``text_config.intermediate_size`` = 25600, TP8 -> 3200 per rank.
INTERMEDIATE_SIZE = 25600
WORLD_SIZE = 8
INTERMEDIATE_PER_PARTITION = INTERMEDIATE_SIZE // WORLD_SIZE          # 3200
GATE_UP_ROWS = 2 * INTERMEDIATE_PER_PARTITION                          # 6400
DOWN_WEIGHT_SHAPE = (HIDDEN_SIZE, INTERMEDIATE_PER_PARTITION)          # 5120 x 3200
GATE_UP_WEIGHT_SHAPE = (GATE_UP_ROWS, HIDDEN_SIZE)                     # 6400 x 5120

#: The Round84 planning error that must never be reintroduced silently.
REJECTED_STALE_INTERMEDIATE = 27648

HALF_DTYPE_NAME = "torch.float16"
NPU_DEVICE_PREFIXES = ("npu", "privateuseone")

#: Pinned R79 candidate file (immutable) and the exact two-round FP16 norm
#: branch inside it. ``r79_norm_branch_digest`` re-derives the latter from the
#: file so a semantic edit fails closed instead of silently changing the oracle.
PINNED_R79_CANDIDATE_SHA256 = (
    "2141b29adc9ecc601d7fa2599e6ba56351adfbdaa5f16acb62c00d95cee3cf78")
PINNED_R79_NORM_FP16_BRANCH_SHA256 = (
    "da686cd5815ff31e13371f18ee2320232acc2be6d5da2bf2f7dccc4c2f1e8f14")

#: Class names that prove the driver handed us the real R79 diagnostic modules
#: rather than an unrelated ``nn.Module`` with the same attribute names.
EXPECTED_CLASS_NAMES = {
    "decoder_layer": "MiniMaxH3Qwen3VLTextDecoderLayer",
    "norm": "MiniMaxH3Qwen3VLRMSNorm",
    "mlp": "MiniMaxH3Qwen3VLTextMLP",
    "gate_up_proj": "MiniMaxH3Qwen3VLMergedColumnParallelLinear",
    "down_proj": "MiniMaxH3Qwen3VLRowParallelLinear",
}

UNQUANTIZED_METHOD_NAME = "UnquantizedLinearMethod"


class LeafError(RuntimeError):
    """Raised instead of falling back, guessing or tolerating an unmet input."""


def expected_geometry() -> dict:
    """Plain description of the geometry this module is allowed to measure."""
    return {
        "sequence_length": SEQUENCE_LENGTH,
        "hidden_size": HIDDEN_SIZE,
        "intermediate_size_actual_checkpoint": INTERMEDIATE_SIZE,
        "world_size": WORLD_SIZE,
        "intermediate_per_partition": INTERMEDIATE_PER_PARTITION,
        "gate_up_weight_shape": list(GATE_UP_WEIGHT_SHAPE),
        "down_weight_shape": list(DOWN_WEIGHT_SHAPE),
        "gate_weight_slice": [0, INTERMEDIATE_PER_PARTITION],
        "up_weight_slice": [INTERMEDIATE_PER_PARTITION, GATE_UP_ROWS],
        "rejected_stale_intermediate_size": REJECTED_STALE_INTERMEDIATE,
        "gate_up_rows_local_validation": GATE_UP_ROWS,
    }


# ---------------------------------------------------------------------------
# Pure-python validation helpers (no torch: exercised by stdlib fake tensors)
# ---------------------------------------------------------------------------


def _shape(value, name: str) -> tuple:
    shape = getattr(value, "shape", None)
    if shape is None:
        raise LeafError(f"{name}: expected a tensor-like object with .shape, got "
                        f"{type(value).__name__}")
    try:
        return tuple(int(v) for v in shape)
    except (TypeError, ValueError) as error:
        raise LeafError(f"{name}: unreadable shape {shape!r}") from error


def _class_name(value) -> str:
    return type(value).__name__


def _require_class(value, role: str, expected: str) -> str:
    actual = _class_name(value)
    if actual != expected:
        raise LeafError(f"{role} is {actual}, expected the R79 class {expected}")
    return actual


def validate_activation(value, name: str = "activation") -> dict:
    """Fail closed unless ``value`` is an original Half NPU activation.

    Accepts ``(1, 2043, 5120)`` or ``(2043, 5120)``. Purely attribute-based so
    the stdlib tests can drive it with stand-in objects.
    """
    shape = _shape(value, name)
    dtype = str(getattr(value, "dtype", None))
    device = str(getattr(value, "device", None))
    if dtype != HALF_DTYPE_NAME:
        raise LeafError(f"{name}: dtype {dtype} is not {HALF_DTYPE_NAME}; refusing to "
                        "build a CPU FP16 leaf oracle for a non-Half activation")
    if not device.startswith(NPU_DEVICE_PREFIXES):
        raise LeafError(f"{name}: device {device} is not an Ascend NPU device; the "
                        "native-op side of a leaf control must run on the model device")
    if len(shape) == 3 and shape[0] == 1:
        tokens = shape[1:]
    elif len(shape) == 2:
        tokens = shape
    else:
        raise LeafError(f"{name}: shape {list(shape)} is neither (1,{SEQUENCE_LENGTH},"
                        f"{HIDDEN_SIZE}) nor ({SEQUENCE_LENGTH},{HIDDEN_SIZE})")
    if tokens != (SEQUENCE_LENGTH, HIDDEN_SIZE):
        raise LeafError(f"{name}: token shape {list(tokens)} != "
                        f"({SEQUENCE_LENGTH}, {HIDDEN_SIZE})")
    return {"name": name, "shape": list(shape), "dtype": dtype, "device": device}


def validate_linear_weight(weight, role: str, expected_shape: tuple) -> dict:
    """Fail closed unless ``weight`` is the original Half shard with exact shape."""
    shape = _shape(weight, role)
    dtype = str(getattr(weight, "dtype", None))
    if shape != tuple(expected_shape):
        raise LeafError(f"{role}: weight shape {list(shape)} != {list(expected_shape)}; "
                        f"note the actual checkpoint intermediate_size is "
                        f"{INTERMEDIATE_SIZE} (per-rank {INTERMEDIATE_PER_PARTITION}), "
                        f"not the stale {REJECTED_STALE_INTERMEDIATE}")
    if dtype != HALF_DTYPE_NAME:
        raise LeafError(f"{role}: weight dtype {dtype} is not {HALF_DTYPE_NAME}")
    return {"role": role, "shape": list(shape), "dtype": dtype}


def validate_layer_geometry(layer) -> dict:
    """Validate one real R79 late decoder block and report its leaf geometry.

    Pure python (attribute access only). Rejects: a different layer class, a
    wrong local MLP width (that is exactly how the Round84 K3456 slip happened),
    non-Half parameters, an added bias, a quantised linear method, and a
    different norm epsilon/weight length.
    """
    layer_class = _require_class(layer, "decoder layer", EXPECTED_CLASS_NAMES["decoder_layer"])
    norm = getattr(layer, "input_layernorm", None)
    mlp = getattr(layer, "mlp", None)
    if norm is None or mlp is None:
        raise LeafError("decoder layer has no input_layernorm/mlp")
    norm_class = _require_class(norm, "input_layernorm", EXPECTED_CLASS_NAMES["norm"])
    mlp_class = _require_class(mlp, "mlp", EXPECTED_CLASS_NAMES["mlp"])
    gate_up = getattr(mlp, "gate_up_proj", None)
    down = getattr(mlp, "down_proj", None)
    if gate_up is None or down is None:
        raise LeafError("mlp has no gate_up_proj/down_proj")
    gate_up_class = _require_class(gate_up, "gate_up_proj",
                                   EXPECTED_CLASS_NAMES["gate_up_proj"])
    down_class = _require_class(down, "down_proj", EXPECTED_CLASS_NAMES["down_proj"])

    ip = getattr(gate_up, "intermediate_size_per_partition", None)
    if ip is None:
        raise LeafError("gate_up_proj has no intermediate_size_per_partition")
    ip = int(ip)
    if ip != INTERMEDIATE_PER_PARTITION:
        raise LeafError(
            f"gate_up_proj.intermediate_size_per_partition is {ip}, expected "
            f"{INTERMEDIATE_PER_PARTITION} (actual intermediate_size {INTERMEDIATE_SIZE} / "
            f"world {WORLD_SIZE}); the stale {REJECTED_STALE_INTERMEDIATE} figure is wrong")

    for role, module, expected in (
            ("gate_up_proj", gate_up, GATE_UP_WEIGHT_SHAPE),
            ("down_proj", down, DOWN_WEIGHT_SHAPE)):
        bias = getattr(module, "bias", None)
        if bias is not None:
            raise LeafError(f"{role}: bias is not None; these H3 row/merged linears are "
                            "bias-free and a biased oracle would be a different operator")
        quant_method = getattr(module, "quant_method", None)
        if quant_method is None:
            raise LeafError(f"{role}: no quant_method to prove the linear is unquantised")
        actual_method = _class_name(quant_method)
        if actual_method != UNQUANTIZED_METHOD_NAME:
            raise LeafError(f"{role}: quant_method {actual_method} is not "
                            f"{UNQUANTIZED_METHOD_NAME}; refusing a quantised leaf")

    gate_up_weight = validate_linear_weight(getattr(gate_up, "weight", None),
                                            "gate_up_proj.weight", GATE_UP_WEIGHT_SHAPE)
    down_weight = validate_linear_weight(getattr(down, "weight", None),
                                         "down_proj.weight", DOWN_WEIGHT_SHAPE)
    norm_weight = getattr(norm, "weight", None)
    norm_shape = _shape(norm_weight, "input_layernorm.weight")
    if norm_shape != (HIDDEN_SIZE,):
        raise LeafError(f"input_layernorm.weight shape {list(norm_shape)} != ({HIDDEN_SIZE},)")
    if str(getattr(norm_weight, "dtype", None)) != HALF_DTYPE_NAME:
        raise LeafError("input_layernorm.weight is not float16; the two-round HF order "
                        "requires a Half gamma")
    eps = getattr(norm, "variance_epsilon", None)
    if eps is None:
        raise LeafError("input_layernorm has no variance_epsilon")
    eps = float(eps)
    if not math.isfinite(eps) or eps <= 0.0:
        raise LeafError(f"input_layernorm.variance_epsilon {eps!r} is not a positive finite value")

    hidden_size = getattr(mlp, "hidden_size", None)
    intermediate_size = getattr(mlp, "intermediate_size", None)
    if hidden_size is not None and int(hidden_size) != HIDDEN_SIZE:
        raise LeafError(f"mlp.hidden_size {hidden_size} != {HIDDEN_SIZE}")
    if intermediate_size is not None and int(intermediate_size) != INTERMEDIATE_SIZE:
        raise LeafError(f"mlp.intermediate_size {intermediate_size} != {INTERMEDIATE_SIZE}")

    return {
        "class_names": {"decoder_layer": layer_class, "norm": norm_class, "mlp": mlp_class,
                        "gate_up_proj": gate_up_class, "down_proj": down_class},
        "intermediate_size_per_partition": ip,
        "gate_up_weight": gate_up_weight,
        "down_weight": down_weight,
        "norm_weight_shape": list(norm_shape),
        "norm_weight_dtype": HALF_DTYPE_NAME,
        "variance_epsilon": eps,
        "bias_free_and_unquantised": True,
        "geometry": expected_geometry(),
    }


# ---------------------------------------------------------------------------
# Pinned R79 two-round FP16 norm order (source-of-truth check)
# ---------------------------------------------------------------------------


def r79_norm_branch_source(candidate_path) -> str:
    """Return the normalised source of the R79 FP16 two-round norm branch."""
    source_text = Path(candidate_path).read_text()
    tree = ast.parse(source_text)
    norm_class = None
    for node in tree.body:
        if isinstance(node, ast.ClassDef) and node.name == EXPECTED_CLASS_NAMES["norm"]:
            norm_class = node
            break
    if norm_class is None:
        raise LeafError(f"{EXPECTED_CLASS_NAMES['norm']} not found in {candidate_path}")
    forward_npu = None
    for child in norm_class.body:
        if isinstance(child, ast.FunctionDef) and child.name == "forward_npu":
            forward_npu = child
            break
    if forward_npu is None:
        raise LeafError("R79 norm has no forward_npu")
    first_if = None
    for statement in forward_npu.body:
        if isinstance(statement, ast.If):
            first_if = statement
            break
    if first_if is None:
        raise LeafError("R79 norm forward_npu no longer starts with the FP16 branch")
    segment = ast.get_source_segment(source_text, first_if)
    if not segment:
        raise LeafError("cannot read the R79 FP16 norm branch source")
    lines = [line.rstrip() for line in textwrap.dedent(segment).splitlines()]
    while lines and not lines[-1]:
        lines.pop()
    return "\n".join(lines) + "\n"


def r79_norm_branch_digest(candidate_path) -> str:
    return hashlib.sha256(r79_norm_branch_source(candidate_path).encode()).hexdigest()


def verify_pinned_norm_order(candidate_path) -> dict:
    """Fail closed unless the R79 two-round FP16 norm branch is byte-identical."""
    actual_file = hashlib.sha256(Path(candidate_path).read_bytes()).hexdigest()
    if actual_file != PINNED_R79_CANDIDATE_SHA256:
        raise LeafError(f"R79 candidate file changed: {actual_file} != "
                        f"{PINNED_R79_CANDIDATE_SHA256}")
    branch = r79_norm_branch_digest(candidate_path)
    if branch != PINNED_R79_NORM_FP16_BRANCH_SHA256:
        raise LeafError(f"R79 norm FP16 branch changed: {branch} != "
                        f"{PINNED_R79_NORM_FP16_BRANCH_SHA256}")
    return {"candidate_file_sha256": actual_file, "fp16_two_round_branch_sha256": branch}


# ---------------------------------------------------------------------------
# Explicit CPU FP16 oracles (lazy torch; called only by the root on the host)
# ---------------------------------------------------------------------------


def cpu_half_rmsnorm_reference(x_half, weight_half, eps: float):
    """HF ``Qwen3VLTextRMSNorm`` order: Float reduce, Half normalised, Half gamma.

    Two rounding points, matching ``MiniMaxH3Qwen3VLRMSNorm.forward_npu`` for the
    FP16 branch: (1) the FP32 normalised value is cast to Half, (2) the Half
    gamma product rounds again. Folding the gamma into the FP32 point math would
    be a *different* (one-round) arithmetic and is deliberately not done.
    """
    # Prove the operands are original Half *before* importing torch so the
    # failure mode is a LeafError (not a missing-module error) on any host.
    for role, tensor in (("input", x_half), ("gamma", weight_half)):
        if str(getattr(tensor, "dtype", None)) != HALF_DTYPE_NAME:
            raise LeafError(f"cpu_half_rmsnorm_reference requires a Half {role}; got "
                            f"{getattr(tensor, 'dtype', None)}")
    import torch

    xf = x_half.float()
    variance = xf.square().mean(-1, keepdim=True)
    normalized = (xf * torch.rsqrt(variance + float(eps))).to(torch.float16)
    return weight_half * normalized


def cpu_half_linear_reference(x_half, weight_half):
    """CPU Half ``F.linear``: FP32 accumulation inside ATen, one Half output round.

    This is the same declared Oracle used by the Round83/Round84 CPU Half GEMM
    controls. It is NOT a Float weight teacher: both operands are the original
    Half values.
    """
    for role, tensor in (("activation", x_half), ("weight", weight_half)):
        if str(getattr(tensor, "dtype", None)) != HALF_DTYPE_NAME:
            raise LeafError(f"cpu_half_linear_reference requires a Half {role}; got "
                            f"{getattr(tensor, 'dtype', None)}")
    import torch.nn.functional as functional

    return functional.linear(x_half, weight_half)


def _tensor_info_or_fail(tensor_info, value, name: str) -> dict:
    info = tensor_info(value)
    if not isinstance(info, dict) or "sha256" not in info:
        raise LeafError(f"root tensor_info did not return a sha256 for {name}: {info!r}")
    return info


def _compare_or_fail(compare, value, reference, name: str) -> dict:
    result = compare(value, reference)
    if not isinstance(result, dict) or "relative_rms" not in result:
        raise LeafError(f"root compare did not return relative_rms for {name}: {result!r}")
    if not result.get("finite", False) or not result.get("dtype_matches", False):
        raise LeafError(f"{name}: comparison is not finite/dtype-matched: {result!r}")
    return result


# ---------------------------------------------------------------------------
# The control itself
# ---------------------------------------------------------------------------


def run_controls(layer, block_input, mlp_input, compare, tensor_info) -> dict:
    """Isolate one real late decoder block's FP16 norm and MLP leaf operators.

    ``layer``      : the real R79 ``MiniMaxH3Qwen3VLTextDecoderLayer`` on the NPU
                     with its original Half parameters (never modified here).
    ``block_input``: actual Half block input ``(1,2043,5120)`` -- for block25 this
                     is the unchanged CPU FP16 raw layer24 state on the device.
    ``mlp_input``  : the actual Half input the native block feeds to ``mlp``,
                     captured from the first native ``mlp.forward`` pre-hook.
    ``compare``    : root helper ``compare(value, reference) -> dict`` returning
                     ``relative_rms``/``max_abs_error``/``raw_bits_equal``/bits.
    ``tensor_info``: root helper ``tensor_info(value) -> dict`` with ``sha256``.

    Returns plain JSON-serialisable data. No tolerance, no pass/fail verdict and
    no tensor is returned; the local ``down_proj`` collective is never executed.
    """
    import torch
    import torch.nn.functional as functional

    geometry = validate_layer_geometry(layer)
    activation_specs = {
        "block_input": validate_activation(block_input, "block_input"),
        "mlp_input": validate_activation(mlp_input, "mlp_input"),
    }

    norm = layer.input_layernorm
    mlp = layer.mlp
    gate_up_weight = mlp.gate_up_proj.weight
    down_weight = mlp.down_proj.weight
    norm_weight = norm.weight
    eps = geometry["variance_epsilon"]

    for role, tensor in (("gate_up_weight", gate_up_weight), ("down_weight", down_weight),
                         ("norm_weight", norm_weight)):
        device = str(getattr(tensor, "device", None))
        if not device.startswith(NPU_DEVICE_PREFIXES):
            raise LeafError(f"{role} is on {device}; the original Half parameter must be "
                            "resident on the model device")

    # Freeze the real captured values before any arithmetic touches them.
    frozen_targets = {
        "block_input": block_input,
        "mlp_input": mlp_input,
        "input_layernorm.weight": norm_weight,
        "gate_up_proj.weight": gate_up_weight,
        "down_proj.weight": down_weight,
    }
    frozen_before = {name: _tensor_info_or_fail(tensor_info, tensor, name)
                     for name, tensor in frozen_targets.items()}

    ip = INTERMEDIATE_PER_PARTITION
    gate_weight = gate_up_weight[0:ip]
    up_weight = gate_up_weight[ip:2 * ip]
    if tuple(int(v) for v in gate_weight.shape) != (ip, HIDDEN_SIZE):
        raise LeafError(f"gate weight slice shape {tuple(gate_weight.shape)} != ({ip},{HIDDEN_SIZE})")
    if tuple(int(v) for v in up_weight.shape) != (ip, HIDDEN_SIZE):
        raise LeafError(f"up weight slice shape {tuple(up_weight.shape)} != ({ip},{HIDDEN_SIZE})")

    cpu_mode = (torch.inference_mode() if not torch.is_inference_mode_enabled()
                else __import__("contextlib").nullcontext())

    results: dict = {}
    with cpu_mode:
        # ---- 1. input_layernorm: native FP16 two-round vs CPU FP16 two-round ----
        npu_norm = norm(block_input)
        npu_norm_repeat = norm(block_input)
        norm_repeat = _compare_or_fail(compare, npu_norm_repeat, npu_norm, "norm repeat")
        if not norm_repeat.get("raw_bits_equal", False):
            raise LeafError("native input_layernorm is not repeat-stable on identical input")

        cpu_norm = cpu_half_rmsnorm_reference(
            block_input.detach().cpu(), norm_weight.detach().cpu(), eps)
        results["input_layernorm"] = {
            "scope": "HF two-round FP16 order (Float reduce -> Half normalised -> Half gamma)",
            "variance_epsilon": eps,
            "source": "actual Half block_input and the original Half input_layernorm.weight",
            "native_vs_cpu_half": _compare_or_fail(compare, npu_norm, cpu_norm, "input_layernorm"),
            "native_repeat_bits": norm_repeat,
            "cpu_oracle": {"order": "float_reduce_then_half_normalised_then_half_gamma",
                           "rounding_points": 2,
                           "one_float_gamma_cast_substituted": False},
            "output": _tensor_info_or_fail(tensor_info, npu_norm, "npu input_layernorm output"),
        }

        # ---- 2. MLP gate/up local linear on the ACTUAL captured mlp input ----
        npu_gate = functional.linear(mlp_input, gate_weight)
        npu_up = functional.linear(mlp_input, up_weight)
        npu_gate_repeat = functional.linear(mlp_input, gate_weight)
        gate_repeat = _compare_or_fail(compare, npu_gate_repeat, npu_gate, "gate repeat")
        if not gate_repeat.get("raw_bits_equal", False):
            raise LeafError("native gate_up local linear is not repeat-stable")

        mlp_input_cpu = mlp_input.detach().cpu()
        gate_weight_cpu = gate_weight.detach().cpu()
        up_weight_cpu = up_weight.detach().cpu()
        cpu_gate = cpu_half_linear_reference(mlp_input_cpu, gate_weight_cpu)
        cpu_up = cpu_half_linear_reference(mlp_input_cpu, up_weight_cpu)
        results["mlp_gate_up_local_linear"] = {
            "scope": "local shard Half x Half F.linear only (no tensor-parallel collective)",
            "intermediate_size_per_partition": ip,
            "weight_rows_used": {"gate": [0, ip], "up": [ip, 2 * ip]},
            "gate_native_vs_cpu_half": _compare_or_fail(compare, npu_gate, cpu_gate,
                                                        "gate local linear"),
            "up_native_vs_cpu_half": _compare_or_fail(compare, npu_up, cpu_up,
                                                      "up local linear"),
            "native_gate_repeat_bits": gate_repeat,
            "cpu_oracle": {"operator": "functional.linear(Half,Half)",
                           "accumulation": "Aten CPU Half linear (FP32 accumulate, one Half round)",
                           "float_weight_teacher": False},
            "native_gate": _tensor_info_or_fail(tensor_info, npu_gate, "npu gate"),
            "native_up": _tensor_info_or_fail(tensor_info, npu_up, "npu up"),
        }

        # ---- 3. SiLU on the ACTUAL native Half gate (never a CPU Float teacher) ----
        npu_silu = functional.silu(npu_gate)
        npu_silu_repeat = functional.silu(npu_gate)
        silu_repeat = _compare_or_fail(compare, npu_silu_repeat, npu_silu, "silu repeat")
        if not silu_repeat.get("raw_bits_equal", False):
            raise LeafError("native F.silu is not repeat-stable")

        npu_gate_cpu = npu_gate.detach().cpu()
        # Same operator, CPU Half, on the SAME actual native Half gate values.
        cpu_silu_from_native_gate = functional.silu(npu_gate_cpu)
        # Allowed diagnostic only: activation promoted to Float, one Half round.
        # This is an activation widening, NOT an FP32 weight shadow.
        float_silu_diag = functional.silu(npu_gate.float()).to(torch.float16)
        results["silu_on_actual_native_gate"] = {
            "scope": "pointwise activation only; both sides consume the actual native Half gate",
            "native_vs_cpu_half_on_same_native_gate": _compare_or_fail(
                compare, npu_silu, cpu_silu_from_native_gate, "silu"),
            "native_repeat_bits": silu_repeat,
            "float_activation_diagnostic_vs_cpu_half": _compare_or_fail(
                compare, float_silu_diag, cpu_silu_from_native_gate, "float silu diagnostic"),
            "float_activation_diagnostic_note": (
                "diagnostic sizing of the pointwise rounding cost; the CPU FP16 leaf "
                "oracle stays Half and no FP32 weight shadow is created"),
            "output": _tensor_info_or_fail(tensor_info, npu_silu, "npu silu output"),
        }

        # ---- 4. Half elementwise multiply (SiLU(gate) * up) ----
        npu_product = npu_silu * npu_up
        npu_product_repeat = npu_silu * npu_up
        product_repeat = _compare_or_fail(compare, npu_product_repeat, npu_product,
                                          "multiply repeat")
        if not product_repeat.get("raw_bits_equal", False):
            raise LeafError("native Half multiply is not repeat-stable")
        cpu_product_from_native = cpu_silu_from_native_gate * npu_up.detach().cpu()
        cpu_product_from_cpu_linear = functional.silu(cpu_gate) * cpu_up
        results["silu_times_up"] = {
            "scope": "Half elementwise product of the actual native activations",
            "native_vs_cpu_half_from_native_values": _compare_or_fail(
                compare, npu_product, cpu_product_from_native, "silu*up"),
            "native_repeat_bits": product_repeat,
            "context_combining_cpu_linear_and_pointwise": _compare_or_fail(
                compare, npu_product, cpu_product_from_cpu_linear, "silu*up context"),
            "context_note": ("context only: mixes the gate/up linear deviation with the "
                             "pointwise one and therefore does not isolate either"),
            "output": _tensor_info_or_fail(tensor_info, npu_product, "npu silu*up product"),
        }

        # ---- 5. LOCAL down_proj shard linear only (no HCCL all-reduce) ----
        npu_down_local = functional.linear(npu_product, down_weight)
        npu_down_local_repeat = functional.linear(npu_product, down_weight)
        down_repeat = _compare_or_fail(compare, npu_down_local_repeat, npu_down_local,
                                       "down local repeat")
        if not down_repeat.get("raw_bits_equal", False):
            raise LeafError("native local down_proj linear is not repeat-stable")
        cpu_down_local = cpu_half_linear_reference(npu_product.detach().cpu(),
                                                   down_weight.detach().cpu())
        results["down_proj_local_linear"] = {
            "scope": ("local shard only: K=3200 -> 5120 on this rank's weight slice; the "
                      "row-parallel all-reduce is deliberately NOT executed here"),
            "collective_executed": False,
            "native_vs_cpu_half_on_same_product": _compare_or_fail(
                compare, npu_down_local, cpu_down_local, "down local linear"),
            "native_repeat_bits": down_repeat,
            "cpu_oracle": {"operator": "functional.linear(Half,Half)",
                           "float_weight_teacher": False},
            "output": _tensor_info_or_fail(tensor_info, npu_down_local,
                                           "npu local down output"),
        }

        del npu_norm_repeat, npu_gate_repeat, npu_silu_repeat, npu_product_repeat
        del npu_down_local_repeat

    # ---- freeze check: nothing we measured may have mutated real state ----
    frozen_after = {name: _tensor_info_or_fail(tensor_info, tensor, name)
                    for name, tensor in frozen_targets.items()}
    changed = [name for name in frozen_targets
               if frozen_before[name]["sha256"] != frozen_after[name]["sha256"]]
    if changed:
        raise LeafError(f"captured inputs/parameters changed during the leaf controls: {changed}")

    return {
        "scope": __doc__.split("WHAT THIS MEASURES")[0].strip(),
        "round": 86,
        "leaf_layer": "late decoder block (index assigned by the root driver)",
        "layer_geometry": geometry,
        "activation_specs": activation_specs,
        "controls": results,
        "frozen_before": frozen_before,
        "frozen_after": frozen_after,
        "frozen_unchanged": True,
        "same_original_half_source": True,
        "no_fp32_weight_shadow": True,
        "no_bfloat16_teacher": True,
        "no_parameter_mutation": True,
        "no_collective_executed": True,
        "numerical_tolerance": None,
        "numerical_pass_field_absent": True,
        "full_encoder_precision_accepted": False,
        "whole_goal_complete": False,
        "measured_only_one_block": True,
    }


def manifest() -> dict:
    """Describe this module and its pins without importing torch or touching NPU."""
    here = Path(__file__).resolve()
    return {
        "round": 86,
        "kind": "late-layer FP16 norm/MLP leaf controls (source only)",
        "file": here.name,
        "bytes": here.stat().st_size,
        "sha256": hashlib.sha256(here.read_bytes()).hexdigest(),
        "executes_npu": False,
        "author_ran_model": False,
        "imports_torch_at_module_scope": False,
        "proves_full_precision": False,
        "geometry": expected_geometry(),
        "pins": {
            "round79_candidate_file_sha256": PINNED_R79_CANDIDATE_SHA256,
            "round79_norm_fp16_two_round_branch_sha256": PINNED_R79_NORM_FP16_BRANCH_SHA256,
        },
        "entry_points": ["validate_activation", "validate_linear_weight",
                         "validate_layer_geometry", "verify_pinned_norm_order",
                         "cpu_half_rmsnorm_reference", "cpu_half_linear_reference",
                         "run_controls", "manifest"],
    }


if __name__ == "__main__":
    import json
    import sys

    print(json.dumps(manifest(), indent=2))
    if len(sys.argv) > 1:
        print(json.dumps(verify_pinned_norm_order(sys.argv[1]), indent=2))