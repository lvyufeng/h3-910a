#!/usr/bin/env python3
"""SOURCE ONLY original-Half CPU FP16 reference for the real H3 DiT conditioning stack.

STATUS: the author of this file never imported torch, never ran a model, never
touched CANN/SSH/NPU and produced no numerical result.  Everything below is a
reusable torch-CPU reference that the root executes on the 910A host.

What this reference is
----------------------
The genuine conditioning path of the checkpoint's ``MiniMaxH3DiTModel``
(original source: ``vllm_omni/diffusion/models/minimax_h3/minimax_h3_transformer.py``,
SHA256 54e45b10c1650822cb6896df7bb90d107c15d3eea52d47b57a078de64c410d13):

``condition_proj`` (biased ``nn.Linear`` text 5120 -> hidden 5376)
-> ``token_refiner`` = 2 pre-norm blocks ``x = x + attn(norm1(x));
x = x + mlp(norm2(x))`` with per-head q/k RMSNorm, **no RoPE, no causal mask,
no attention bias, 56 q heads == 56 kv heads**, then the *original*
``token_refiner.final_norm``.

Only the 19 original Half parameters of that stack are used.  Weights are the
official dense checkpoint values converted exactly once to the native FP16
alias (root verified the BF16 -> Half bytes); no BF16 tensor is ever read as an
output and no FP32 Linear weight teacher exists.

Numerics that are taken from the original source, not invented here
------------------------------------------------------------------
* ``RMSNorm.forward_native``: FP32 variance/rsqrt, FP32 gamma multiply, and a
  **single** cast back to the activation dtype -- no intermediate Half norm
  round (this is the H3 norm, not the HF Qwen two-round norm).
* ``SiluAndMul.forward_native``: ``F.silu(gate) * up`` on Half operands.
* attention: ``softmax_scale = head_dim ** -0.5``, unmasked non-causal SDPA in
  the MATH backend on Half q/k/v.
* residual adds and ``F.linear`` stay Half end to end; the receipt flags say
  ``numerical_tolerance: None``, ``full_encoder_precision_accepted: False`` and
  ``whole_goal_complete: False`` -- this reference proves nothing on its own
  about the encoder gap.

Module-scope imports are stdlib only; torch/numpy are imported lazily inside
the API functions so the source can be parsed and tested without a tensor
runtime.
"""

from __future__ import annotations

import argparse
import functools
import hashlib
import json
import os
import sys
import time
from pathlib import Path

# --------------------------------------------------------------------------- #
# Paths and fixed pins
# --------------------------------------------------------------------------- #
REPO_ROOT = next((p for p in Path(__file__).resolve().parents if (p / 'artifacts/diagnostics').is_dir()), Path(__file__).resolve().parent)
ROUND_DIR = Path(__file__).resolve().parent
DEFAULT_ROOT = Path('/tmp/h3_round111_cpu01')
DEFAULT_CHECKPOINT_DIR = Path('/home/mseco/models/MiniMax-H3-native-fp16-audit53/FL2VA/transformer')
DEFAULT_ORIGINAL_SOURCE = Path(
    '/home/mseco/vllm-omni/vllm_omni/diffusion/models/minimax_h3/minimax_h3_transformer.py'
)
DEFAULT_EXPECTED_PARAMETERS = REPO_ROOT / 'artifacts/diagnostics/round111/expected_half_parameters111.json'
DEFAULT_EXPECTED_INPUTS = REPO_ROOT / 'artifacts/diagnostics/round111/expected_inputs111.json'
DEFAULT_CHECKPOINT_METADATA = REPO_ROOT / 'artifacts/diagnostics/round111/checkpoint_conditioning111.json'
DEFAULT_NORM_ACTIVATION_ORDER = REPO_ROOT / 'artifacts/diagnostics/round111/norm_activation_order111.json'

REFERENCE_JSON = 'reference_CPUoriginal111.json'
FAILURE_JSON = 'CPU_conditioning_failure111.json'
STAGE_TENSOR_TEMPLATE = 'CPU_stages111_%s.pt'

# Fixed read-only identities (root verified them; this source only re-checks).
ORIGINAL_SOURCE_SHA256 = '54e45b10c1650822cb6896df7bb90d107c15d3eea52d47b57a078de64c410d13'
CHECKPOINT_CONFIG_SHA256 = 'f619093a231fcfbcc3d035bec26c50ad864e7331a500d5c519f5045dc1e50458'
CHECKPOINT_INDEX_SHA256 = 'fb457a26ffa6294660e249b0ddd03a337f2e5393f770b5c34c8b8f90a29a7efb'
NATIVE_HALF_FIRST_SHARD_SHA256 = '2a5bd5b54bb02cb168e4053e5db87aacb3feedc7c21e3bf06954de033e3b42d0'
NATIVE_HALF_FIRST_SHARD_NAME = 'model-00001-of-00013.safetensors'
EXPECTED_TORCH_PREFIX = '2.10'
DEFAULT_THREADS = 16
MAX_STAGE_FILE_BYTES = 100_000_000

# --------------------------------------------------------------------------- #
# Architecture facts of the actual checkpoint config (not class defaults)
# --------------------------------------------------------------------------- #
ARCH = {
    'hidden_size': 5376,
    'text_dim': 5120,
    'num_attention_heads': 56,
    'attention_head_dim': 128,
    'ffn_hidden_size': 14336,
    'token_refiner_num_layers': 2,
    'norm_eps': 1e-5,
    'qk_norm_eps': 1e-5,
    'final_norm_eps': 1e-5,
}
INNER_DIM = ARCH['num_attention_heads'] * ARCH['attention_head_dim']  # 7168
QKV_ROWS = 3 * INNER_DIM  # 21504
QKV_SPLIT = (INNER_DIM, INNER_DIM, INNER_DIM)
FC1_ROWS = 2 * ARCH['ffn_hidden_size']  # 28672
FC1_HALVES = (ARCH['ffn_hidden_size'], ARCH['ffn_hidden_size'])
SOFTMAX_SCALE = ARCH['attention_head_dim'] ** -0.5
CHECKPOINT_QKV_GROUP_STRIDE = 3 * ARCH['attention_head_dim']  # 384 rows per head group


def _block_names(index: int) -> tuple[str, ...]:
    prefix = 'token_refiner.blocks.%d' % index
    return (
        '%s.attn.q_norm.weight' % prefix,
        '%s.attn.k_norm.weight' % prefix,
        '%s.attn.qkv_proj.weight' % prefix,
        '%s.attn.out_proj.weight' % prefix,
        '%s.mlp.fc1.weight' % prefix,
        '%s.mlp.fc2.weight' % prefix,
        '%s.norm1.weight' % prefix,
        '%s.norm2.weight' % prefix,
    )


PARAMETER_NAMES = (
    'condition_proj.weight',
    'condition_proj.bias',
) + tuple(name for index in range(ARCH['token_refiner_num_layers']) for name in _block_names(index)) + (
    'token_refiner.final_norm.weight',
)

PARAMETER_SHAPES = {
    'condition_proj.weight': (ARCH['hidden_size'], ARCH['text_dim']),
    'condition_proj.bias': (ARCH['hidden_size'],),
    'token_refiner.final_norm.weight': (ARCH['hidden_size'],),
}
for _index in range(ARCH['token_refiner_num_layers']):
    _p = 'token_refiner.blocks.%d' % _index
    PARAMETER_SHAPES.update({
        '%s.attn.q_norm.weight' % _p: (ARCH['attention_head_dim'],),
        '%s.attn.k_norm.weight' % _p: (ARCH['attention_head_dim'],),
        '%s.attn.qkv_proj.weight' % _p: (QKV_ROWS, ARCH['hidden_size']),
        '%s.attn.out_proj.weight' % _p: (ARCH['hidden_size'], INNER_DIM),
        '%s.mlp.fc1.weight' % _p: (FC1_ROWS, ARCH['hidden_size']),
        '%s.mlp.fc2.weight' % _p: (ARCH['hidden_size'], ARCH['ffn_hidden_size']),
        '%s.norm1.weight' % _p: (ARCH['hidden_size'],),
        '%s.norm2.weight' % _p: (ARCH['hidden_size'],),
    })

MANDATORY_STAGES = ('projected', 'refiner0', 'refiner1', 'finalnorm')

_STAGE_LAST_DIM = {
    'norm1': ARCH['hidden_size'],
    'qkv': QKV_ROWS,
    'q': (ARCH['num_attention_heads'], ARCH['attention_head_dim']),
    'k': (ARCH['num_attention_heads'], ARCH['attention_head_dim']),
    'v': (ARCH['num_attention_heads'], ARCH['attention_head_dim']),
    'attn': INNER_DIM,
    'attn_proj': ARCH['hidden_size'],
    'res1': ARCH['hidden_size'],
    'norm2': ARCH['hidden_size'],
    'fc1': FC1_ROWS,
    'mlp_act': ARCH['ffn_hidden_size'],
    'mlp_out': ARCH['hidden_size'],
}


def detailed_stage_names() -> tuple[str, ...]:
    names = []
    for index in range(ARCH['token_refiner_num_layers']):
        for suffix in _STAGE_LAST_DIM:
            names.append('block%d.%s' % (index, suffix))
    return tuple(names)


DETAILED_STAGES = detailed_stage_names()
ALL_STAGES = MANDATORY_STAGES + DETAILED_STAGES


def expected_stage_shape(name: str, rows: int) -> tuple[int, ...]:
    """Shape the named stage must have for ``rows`` text tokens."""
    if not isinstance(rows, int) or isinstance(rows, bool) or rows <= 0:
        raise ValueError('rows must be a positive int, got %r' % (rows,))
    if name in MANDATORY_STAGES:
        return (rows, ARCH['hidden_size'])
    prefix, _, suffix = name.partition('.')
    if prefix.startswith('block') and suffix in _STAGE_LAST_DIM:
        tail = _STAGE_LAST_DIM[suffix]
        if isinstance(tail, tuple):
            return (rows,) + tail
        return (rows, tail)
    raise KeyError('unknown conditioning stage %r' % (name,))


# --------------------------------------------------------------------------- #
# Pure stdlib helpers (unit-testable without a tensor runtime)
# --------------------------------------------------------------------------- #
@functools.lru_cache(maxsize=1)
def qkv_reorder_order() -> tuple[int, ...]:
    """Checkpoint-row order of the model-layout ``[all Q | all K | all V]`` rows.

    The checkpoint stores ``[head_group, (1 q + 1 k + 1 v) * head_dim, hidden]``
    interleaved per head group.  The original loader reshapes to
    ``(num_heads, 3 * head_dim, -1)``, splits q/k/v per group and concatenates,
    i.e. the model parameter row ``r`` reads checkpoint row ``order[r]``.
    """
    heads = ARCH['num_attention_heads']
    head_dim = ARCH['attention_head_dim']
    order = []
    for part in range(3):  # 0 = all Q, 1 = all K, 2 = all V
        for group in range(heads):
            base = group * CHECKPOINT_QKV_GROUP_STRIDE + part * head_dim
            order.extend(base + offset for offset in range(head_dim))
    return tuple(order)


def text_rows_from_shape(shape) -> tuple[int, int]:
    """Accept the pipeline's exact 2D rows or a size-1 leading batch dim."""
    shape = tuple(shape)
    if len(shape) == 2:
        rows, width = shape
    elif len(shape) == 3:
        batch, rows, width = shape
        if batch != 1:
            raise ValueError(
                'conditioning reference runs one packed request; got batch dim %r '
                'in shape %r (squeeze is only allowed for a size-1 batch)' % (batch, shape))
    else:
        raise ValueError('expected rows [T, text_dim] or [1, T, text_dim], got shape %r' % (shape,))
    if width != ARCH['text_dim']:
        raise ValueError('expected text width %d, got %r in shape %r'
                         % (ARCH['text_dim'], width, shape))
    if rows <= 0:
        raise ValueError('expected a positive text row count, got %r' % (rows,))
    return int(rows), int(width)


def file_sha256(path) -> str:
    digest = hashlib.sha256()
    with open(path, 'rb') as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def tensor_info(tensor) -> dict:
    """Shape/dtype/byte identity of a tensor: SHA256 of the raw little-endian payload.

    ``.contiguous().view(torch.uint8)`` makes the identity the exact boxed bytes
    (FP16 = 2 bytes/element) instead of the pickle container, so the same rule
    can be reused for every stage tensor and every pinned weight.
    """
    detached = tensor.detach().to('cpu').contiguous()
    payload = detached.view(_torch_module().uint8)
    info = {
        'shape': [int(dim) for dim in detached.shape],
        'dtype': str(detached.dtype),
        'bytes': int(payload.numel()),
        'sha256': hashlib.sha256(payload.numpy().tobytes()).hexdigest(),
        'finite': bool(_torch_module().isfinite(detached).all()),
    }
    if detached.numel():
        info['max_abs'] = float(detached.abs().max())
    return info


def is_sha256(value) -> bool:
    if not isinstance(value, str) or len(value) != 64:
        return False
    return all(character in '0123456789abcdef' for character in value)


def load_expected_parameters(path=DEFAULT_EXPECTED_PARAMETERS) -> dict:
    """Load and validate the root-supplied 19-parameter Half pin map."""
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError('expected parameter pin map is missing: %s' % path)
    payload = json.loads(path.read_text())
    if payload.get('all19Halfvalues_exact_original_conversion') is not True:
        raise ValueError('%s does not certify the exact Half conversion' % path)
    parameters = payload.get('parameters')
    if not isinstance(parameters, dict):
        raise ValueError('%s has no "parameters" mapping' % path)
    if set(parameters) != set(PARAMETER_NAMES):
        missing = sorted(set(PARAMETER_NAMES) - set(parameters))
        extra = sorted(set(parameters) - set(PARAMETER_NAMES))
        raise ValueError('parameter pin map mismatch: missing=%r extra=%r' % (missing, extra))
    for name in PARAMETER_NAMES:
        entry = parameters[name]
        if tuple(entry.get('shape') or ()) != PARAMETER_SHAPES[name]:
            raise ValueError('%s pin shape %r != expected %r'
                             % (name, entry.get('shape'), PARAMETER_SHAPES[name]))
        if entry.get('dtype') != 'torch.float16':
            raise ValueError('%s pin dtype is %r, expected torch.float16'
                             % (name, entry.get('dtype')))
        if not is_sha256(entry.get('reference_sha256')):
            raise ValueError('%s pin has no usable reference_sha256' % name)
    return {
        'path': str(path),
        'file_sha256': file_sha256(path),
        'checkpoint_config_SHA256': payload.get('checkpoint_config_SHA256'),
        'checkpoint_index_SHA256': payload.get('checkpoint_index_SHA256'),
        'nativeHalf_firstshard_sha256': payload.get('nativeHalf_firstshard_sha256'),
        'parameters': parameters,
    }


_INPUT_LIST_KEYS = ('inputs', 'cases', 'tensors')


def _iter_input_entries(payload: dict):
    for key in _INPUT_LIST_KEYS:
        value = payload.get(key)
        if isinstance(value, list):
            return value
        if isinstance(value, dict):
            return [dict(entry, name=entry.get('name', name)) for name, entry in value.items()]
    mapping = {name: entry for name, entry in payload.items()
               if isinstance(entry, dict) and 'path' in entry}
    if mapping:
        return [dict(entry, name=entry.get('name', name)) for name, entry in mapping.items()]
    raise ValueError('expected inputs file needs an "inputs"/"cases" list or a name->path mapping')


def load_expected_inputs(path) -> list[dict]:
    """Load the root-supplied inference input pin map (schema documented in README)."""
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError('expected input pin map is missing: %s' % path)
    payload = json.loads(path.read_text())
    if not isinstance(payload, dict):
        raise ValueError('%s must contain a JSON object' % path)
    entries = _iter_input_entries(payload)
    if not entries:
        raise ValueError('%s declares no inputs' % path)
    cases, seen = [], set()
    for entry in entries:
        if not isinstance(entry, dict):
            raise ValueError('input entry must be an object, got %r' % (entry,))
        entry = dict(entry)
        for lower, upper in (('file_sha256', 'file_SHA256'), ('tensor_sha256', 'tensor_SHA256')):
            if lower not in entry and upper in entry: entry[lower] = entry[upper]
        name = entry.get('name')
        if not isinstance(name, str) or not name.strip():
            raise ValueError('input entry needs a non-empty "name": %r' % (entry,))
        if any(not (c.isalnum() or c in '_-') for c in name):
            raise ValueError('unsafe output case name %r' % name)
        if name in seen:
            raise ValueError('duplicate input case name %r' % name)
        seen.add(name)
        source = entry.get('path')
        if not isinstance(source, str) or not source:
            raise ValueError('input case %r needs a "path"' % name)
        if not is_sha256(entry.get('tensor_sha256')):
            raise ValueError('input case %r needs a pinned tensor_sha256' % name)
        file_sha = entry.get('file_sha256')
        if file_sha is not None and not is_sha256(file_sha):
            raise ValueError('input case %r has an invalid file_sha256' % name)
        shape = entry.get('shape')
        if shape is not None:
            if not isinstance(shape, list) or not shape:
                raise ValueError('input case %r has an invalid shape pin' % name)
            try:
                text_rows_from_shape(shape)
            except ValueError as error:
                raise ValueError('input case %r shape pin rejected: %s' % (name, error))
        cases.append({
            'name': name,
            'path': source,
            'file_sha256': file_sha,
            'tensor_sha256': entry['tensor_sha256'],
            'shape': shape,
            'role': entry.get('role'),
        })
    return cases


def load_gold_pins(path) -> dict:
    """Optional read-only gold file pins: {"files": {path: sha256}} or {path: sha256}."""
    if path is None:
        return {}
    path = Path(path)
    payload = json.loads(path.read_text())
    if isinstance(payload, dict) and isinstance(payload.get('files'), dict):
        payload = payload['files']
    if not isinstance(payload, dict):
        raise ValueError('%s must map file paths to sha256 values' % path)
    pins = {}
    for name, digest in payload.items():
        if not is_sha256(digest):
            raise ValueError('gold pin for %r is not a sha256 value' % name)
        pins[str(name)] = digest
    return pins


def declared_output_paths(root, case_names=()) -> list[Path]:
    root = Path(root)
    paths = [root / REFERENCE_JSON, root / FAILURE_JSON]
    paths.extend(root / (STAGE_TENSOR_TEMPLATE % name) for name in case_names)
    return paths


def refuse_existing_outputs(root, case_names=(), create=True) -> None:
    """Fail closed: never reuse a root that already holds a declared output."""
    root = Path(root)
    if root.exists() and not root.is_dir():
        raise FileExistsError('%s exists and is not a directory' % root)
    existing = [str(path) for path in declared_output_paths(root, case_names) if path.exists()]
    if existing:
        raise FileExistsError('refusing to overwrite existing conditioning outputs: %r' % existing)
    if create:
        root.mkdir(parents=True, exist_ok=True)


def _new_atomic_paths(path):
    destination = Path(path).absolute()
    parent = destination.parent.resolve(strict=True)
    if destination.parent != parent or destination.is_symlink() or destination.exists():
        raise FileExistsError('Refuse existing or symlinked output: %s' % destination)
    temporary = destination.with_name(destination.name + '.tmp')
    if temporary.exists() or temporary.is_symlink() or temporary.resolve() != temporary:
        raise FileExistsError('Refuse reused temporary output: %s' % temporary)
    return destination, temporary


def _checked_publish(destination, temporary):
    if destination.resolve() != destination or temporary.resolve(strict=True) != temporary:
        raise ValueError('Output endpoints changed before atomic publication')
    if destination.exists() or destination.is_symlink():
        raise FileExistsError('Output appeared before publication: %s' % destination)
    os.replace(temporary, destination)


def write_json_atomic(path, payload) -> None:
    destination, temporary = _new_atomic_paths(path)
    with temporary.open('x', encoding='utf-8') as f:
        f.write(json.dumps(payload, indent=2) + '\n'); f.flush(); os.fsync(f.fileno())
    _checked_publish(destination, temporary)


def save_tensors_atomic(path, mapping) -> None:
    destination, temporary = _new_atomic_paths(path)
    with temporary.open('xb') as f:
        _torch_module().save(mapping, f); f.flush(); os.fsync(f.fileno())
    _checked_publish(destination, temporary)


def stage_mapping_bytes(mapping) -> int:
    total = 0
    for tensor in mapping.values():
        total += int(tensor.numel()) * int(tensor.element_size())
    return total


def build_receipt() -> dict:
    return {
        'scope': __doc__,
        'round': 111,
        'kind': ('original-Half CPU FP16 DiT conditioning: biased condition_proj, 2 original '
                 'pre-norm token-refiner blocks with per-head q/k RMSNorm and unmasked non-causal '
                 'MATH SDPA, then the original token_refiner.final_norm'),
        'source_SHA256': file_sha256(Path(__file__)),
        'status': 'SOURCE ONLY at authoring time; the root executes this file',
        'completed': False,
        'cpu_conditioning_reference_completed': False,
        'not_a_new_oracle': True,
        'original_half_parameters_only': True,
        'half_parameters_only': True,
        'float_linear_weight_teacher': False,
        'bf16_gold_cast': False,
        'fake_encoder_final_norm': False,
        'original_dit_final_norm_applied_from_original_source': True,
        'hidden_conditioning_cache_used': False,
        'numerical_tolerance': None,
        'full_encoder_precision_accepted': False,
        'whole_goal_complete': False,
        'encoder_raw_gap_validated_here': False,
        'architecture': dict(ARCH),
        'derived': {
            'inner_dim': INNER_DIM,
            'qkv_rows': QKV_ROWS,
            'qkv_split': list(QKV_SPLIT),
            'fc1_rows': FC1_ROWS,
            'fc1_gate_then_up': list(FC1_HALVES),
            'softmax_scale': SOFTMAX_SCALE,
            'qkv_checkpoint_groups': ARCH['num_attention_heads'],
            'qkv_checkpoint_group_stride': CHECKPOINT_QKV_GROUP_STRIDE,
            'qkv_reorder': 'checkpoint [group, (q,k,v)*head_dim] -> model [all Q | all K | all V]',
            'attention': 'unmasked non-causal SDPA, MATH backend, Half q/k/v',
            'num_kv_heads': ARCH['num_attention_heads'],
            'rope': False,
            'adaln': False,
            'attention_bias': False,
        },
        'mandatory_stages': list(MANDATORY_STAGES),
        'detailed_stages': list(DETAILED_STAGES),
        'parameter_names': list(PARAMETER_NAMES),
    }


# --------------------------------------------------------------------------- #
# Lazy torch access
# --------------------------------------------------------------------------- #
def _torch_module():
    import torch  # noqa: PLC0415 - deliberately lazy: module scope stays stdlib-only

    return torch


def _torch_functional():
    import torch  # noqa: PLC0415
    import torch.nn.functional as F  # noqa: PLC0415

    return torch, F


def require_torch(threads=DEFAULT_THREADS, expected_torch_prefix=EXPECTED_TORCH_PREFIX):
    torch = _torch_module()
    if threads:
        if not isinstance(threads, int) or threads <= 0:
            raise ValueError('threads must be a positive int, got %r' % (threads,))
        torch.set_num_threads(threads)
    version = str(torch.__version__)
    if expected_torch_prefix and not version.startswith(expected_torch_prefix):
        raise RuntimeError('requires torch %s.* for the pinned CPU FP16 reference, got %s'
                           % (expected_torch_prefix, version))
    if torch.get_num_threads() != threads:
        raise RuntimeError('could not pin torch to %d threads (got %d)'
                           % (threads, torch.get_num_threads()))
    return torch


def math_sdpa_context(torch):
    """Force the deterministic MATH attention backend (fail closed if unavailable)."""
    attention = getattr(torch.nn, 'attention', None)
    if attention is not None and hasattr(attention, 'sdpa_kernel') and hasattr(attention, 'SDPBackend'):
        return attention.sdpa_kernel(attention.SDPBackend.MATH)
    backends = getattr(torch, 'backends', None)
    cuda_backend = getattr(backends, 'cuda', None) if backends is not None else None
    if cuda_backend is not None and hasattr(cuda_backend, 'sdp_kernel'):
        return cuda_backend.sdp_kernel(enable_flash=False, enable_math=True, enable_mem_efficient=False)
    raise RuntimeError('this torch build exposes no way to force SDPBackend.MATH')


def check_checkpoint_metadata(path=DEFAULT_CHECKPOINT_METADATA) -> dict:
    """Cross-check the pinned checkpoint metadata against ARCH (stdlib only)."""
    path = Path(path)
    payload = json.loads(path.read_text())
    roots = payload.get('roots') or []
    if not roots:
        raise ValueError('%s records no checkpoint roots' % path)
    checked = []
    for root in roots:
        config = (root.get('config') or {})
        value = config.get('value') or {}
        for key, expected in ARCH.items():
            actual = value.get(key)
            if actual != expected:
                raise ValueError('checkpoint config %r=%r != reference %r'
                                 % (key, actual, expected))
        parameters = root.get('conditioning_parameters') or {}
        if set(parameters) != set(PARAMETER_NAMES):
            raise ValueError('metadata parameter set mismatch for %s' % root.get('path'))
        for name in PARAMETER_NAMES:
            shape = tuple(parameters[name].get('shape') or ())
            if shape != PARAMETER_SHAPES[name]:
                raise ValueError('metadata shape %r != %r for %s'
                                 % (shape, PARAMETER_SHAPES[name], name))
        checked.append({
            'path': root.get('path'),
            'config_sha256': config.get('sha256'),
            'index_sha256': root.get('index_sha256'),
            'parameter_dtypes': sorted({parameters[name].get('dtype') for name in PARAMETER_NAMES}),
            'config_matches_reference_architecture': True,
            'parameter_shapes_match_reference': True,
        })
    return {'path': str(path), 'file_sha256': file_sha256(path), 'roots': checked}


def verify_original_sources(checkpoint_dir=DEFAULT_CHECKPOINT_DIR,
                            original_source=DEFAULT_ORIGINAL_SOURCE,
                            parameter_pins=None,
                            strict=True) -> dict:
    """Re-check the read-only config/index/original-source pins from this round."""
    pins = parameter_pins or {}
    expected_config = pins.get('checkpoint_config_SHA256') or CHECKPOINT_CONFIG_SHA256
    expected_index = pins.get('checkpoint_index_SHA256') or CHECKPOINT_INDEX_SHA256
    report = {'strict': bool(strict), 'missing': [], 'checks': {}}
    targets = {
        'config': Path(checkpoint_dir) / 'config.json',
        'index': Path(checkpoint_dir) / 'model.safetensors.index.json',
        'original_source': Path(original_source),
    }
    expected = {
        'config': expected_config,
        'index': expected_index,
        'original_source': ORIGINAL_SOURCE_SHA256,
    }
    for label, path in targets.items():
        entry = {'path': str(path), 'expected_sha256': expected[label]}
        if not path.is_file():
            entry['exists'] = False
            report['missing'].append(str(path))
            report['checks'][label] = entry
            if strict:
                raise FileNotFoundError('required read-only source is missing: %s' % path)
            continue
        entry['exists'] = True
        entry['actual_sha256'] = file_sha256(path)
        entry['matches_pin'] = entry['actual_sha256'] == expected[label]
        report['checks'][label] = entry
        if strict and not entry['matches_pin']:
            raise RuntimeError('%s changed: %s != %s'
                               % (label, entry['actual_sha256'], expected[label]))
    if report['checks']['config'].get('exists'):
        value = json.loads(targets['config'].read_text())
        for key, expected_value in ARCH.items():
            if value.get(key) != expected_value:
                raise ValueError('checkpoint config %r=%r != reference %r'
                                 % (key, value.get(key), expected_value))
        report['checks']['config']['value_matches_reference_architecture'] = True
    return report


# --------------------------------------------------------------------------- #
# Reference model
# --------------------------------------------------------------------------- #
class ConditioningCPU111:
    """Original-Half CPU FP16 conditioning stack with a stage trace.

    A plain class on purpose: the 19 parameters are ``torch.nn.Parameter`` Half
    tensors with ``requires_grad=False`` and there is no ``nn.Module.to()`` path
    that could silently build FP32 weight shadows.
    """

    def __init__(self, parameters: dict, receipt: dict | None = None):
        torch = _torch_module()
        if set(parameters) != set(PARAMETER_NAMES):
            raise ValueError('conditioning model needs exactly the 19 original parameters')
        stored = {}
        for name in PARAMETER_NAMES:
            tensor = parameters[name]
            if not torch.is_tensor(tensor):
                raise TypeError('%s is not a tensor' % name)
            if tensor.dtype != torch.float16:
                raise TypeError('%s must stay float16, got %s' % (name, tensor.dtype))
            if tuple(tensor.shape) != PARAMETER_SHAPES[name]:
                raise ValueError('%s has shape %r, expected %r'
                                 % (name, tuple(tensor.shape), PARAMETER_SHAPES[name]))
            stored[name] = torch.nn.Parameter(tensor.detach().to('cpu').contiguous().clone(),
                                              requires_grad=False)
        self._parameters_map = stored
        self.receipt = dict(receipt or {})

    # -- parameter surface -------------------------------------------------- #
    def named_parameters(self):
        for name in PARAMETER_NAMES:
            yield name, self._parameters_map[name]

    def parameters(self):
        return [self._parameters_map[name] for name in PARAMETER_NAMES]

    def parameter_info(self) -> dict:
        return {name: tensor_info(tensor) for name, tensor in self.named_parameters()}

    def parameter_sha(self) -> dict:
        return {name: info['sha256'] for name, info in self.parameter_info().items()}

    def parameter_dtypes(self) -> set:
        return {str(tensor.dtype) for tensor in self.parameters()}

    def no_float32_shadow(self) -> bool:
        torch = _torch_module()
        return all(tensor.dtype == torch.float16 for tensor in self.parameters())

    def _param(self, name: str):
        return self._parameters_map[name]

    # -- numerics ----------------------------------------------------------- #
    @staticmethod
    def rms_norm(x, weight, eps: float):
        """``RMSNorm.forward_native``: FP32 variance/rsqrt, FP32 gamma, one dtype cast."""
        torch = _torch_module()
        input_dtype = x.dtype
        upcast = x.to(torch.float32)
        variance = upcast.pow(2).mean(-1, keepdim=True)
        out = upcast * torch.rsqrt(variance + eps)
        out = weight.to(torch.float32) * out
        return out.to(input_dtype)

    @staticmethod
    def linear(x, weight, bias=None):
        torch, functional = _torch_functional()
        if x.dtype != torch.float16 or weight.dtype != torch.float16:
            raise TypeError('the CPU FP16 reference only does Half x Half linear')
        if bias is not None and bias.dtype != torch.float16:
            raise TypeError('the CPU FP16 reference only does Half bias')
        return functional.linear(x, weight, bias)

    def attention(self, x, block: int, record=None):
        torch, functional = _torch_functional()
        prefix = 'token_refiner.blocks.%d' % block
        query = self._param(prefix + '.attn.q_norm.weight')
        key = self._param(prefix + '.attn.k_norm.weight')
        rows = int(x.shape[0])
        qkv = self.linear(x, self._param(prefix + '.attn.qkv_proj.weight'))
        if record is not None:
            record('block%d.qkv' % block, qkv)
        q, k, v = qkv.split(list(QKV_SPLIT), dim=-1)
        heads, head_dim = ARCH['num_attention_heads'], ARCH['attention_head_dim']
        q = q.reshape(rows, heads, head_dim)
        k = k.reshape(rows, heads, head_dim)
        v = v.reshape(rows, heads, head_dim)
        q = self.rms_norm(q, query, ARCH['qk_norm_eps'])
        k = self.rms_norm(k, key, ARCH['qk_norm_eps'])
        if record is not None:
            record('block%d.q' % block, q)
            record('block%d.k' % block, k)
            record('block%d.v' % block, v)
        q = q.transpose(0, 1).unsqueeze(0).contiguous()  # [1, heads, rows, head_dim]
        k = k.transpose(0, 1).unsqueeze(0).contiguous()
        v = v.transpose(0, 1).unsqueeze(0).contiguous()
        with math_sdpa_context(torch):
            attended = functional.scaled_dot_product_attention(
                q, k, v,
                attn_mask=None,
                dropout_p=0.0,
                is_causal=False,
                scale=SOFTMAX_SCALE,
            )
        attended = attended.squeeze(0).transpose(0, 1).reshape(rows, INNER_DIM)
        if record is not None:
            record('block%d.attn' % block, attended)
        return self.linear(attended, self._param(prefix + '.attn.out_proj.weight'))

    def mlp(self, x, block: int, record=None):
        torch, functional = _torch_functional()
        prefix = 'token_refiner.blocks.%d' % block
        gate_up = self.linear(x, self._param(prefix + '.mlp.fc1.weight'))
        if record is not None:
            record('block%d.fc1' % block, gate_up)
        gate, up = gate_up.chunk(2, dim=-1)  # checkpoint rows: first gate, second up
        activated = functional.silu(gate) * up  # Half SiLU then Half product
        if record is not None:
            record('block%d.mlp_act' % block, activated)
        return self.linear(activated, self._param(prefix + '.mlp.fc2.weight'))

    def _text_rows(self, x):
        torch = _torch_module()
        if not torch.is_tensor(x):
            raise TypeError('conditioning input must be a tensor, got %r' % type(x))
        if x.device.type != 'cpu':
            raise ValueError('this reference runs on CPU only, got device %r' % (x.device,))
        if x.dtype != torch.float16:
            raise ValueError('conditioning input must be float16, got %s' % x.dtype)
        rows, width = text_rows_from_shape(tuple(x.shape))
        if not bool(torch.isfinite(x).all()):
            raise ValueError('conditioning input contains non-finite values')
        return x.reshape(rows, width).contiguous(), rows

    def forward_stages(self, x, trace=None, keep=()):
        """Run the original conditioning stack and return the requested stage tensors.

        ``keep`` selects extra per-block boundaries; ``keep='all'`` retains every
        documented boundary.  Everything returned is detached, CPU and
        contiguous Half.
        """
        if keep == 'all':
            keep = frozenset(ALL_STAGES)
        else:
            keep = frozenset(keep or ())
        unknown = sorted(set(keep) - set(ALL_STAGES))
        if unknown:
            raise ValueError('unknown stage names: %r' % unknown)
        stages = {} if trace is None else trace
        retain = frozenset(MANDATORY_STAGES) | keep

        def record(name, tensor):
            if name in retain:
                stages[name] = tensor.detach().to('cpu').contiguous()

        rows_hidden = self._text_rows(x)
        hidden, rows = rows_hidden
        hidden = self.linear(hidden, self._param('condition_proj.weight'),
                             self._param('condition_proj.bias'))
        record('projected', hidden)
        for block in range(ARCH['token_refiner_num_layers']):
            prefix = 'token_refiner.blocks.%d' % block
            normalized = self.rms_norm(hidden, self._param(prefix + '.norm1.weight'),
                                       ARCH['norm_eps'])
            record('block%d.norm1' % block, normalized)
            attention_out = self.attention(normalized, block, record)
            record('block%d.attn_proj' % block, attention_out)
            hidden = hidden + attention_out
            record('block%d.res1' % block, hidden)
            normalized = self.rms_norm(hidden, self._param(prefix + '.norm2.weight'),
                                       ARCH['norm_eps'])
            record('block%d.norm2' % block, normalized)
            mlp_out = self.mlp(normalized, block, record)
            record('block%d.mlp_out' % block, mlp_out)
            hidden = hidden + mlp_out
            record('refiner%d' % block, hidden)
        final = self.rms_norm(hidden, self._param('token_refiner.final_norm.weight'),
                              ARCH['final_norm_eps'])
        record('finalnorm', final)
        return stages


def stage_tensor_path(root, case_name: str) -> Path:
    """Exact stage-tensor file the root should expect for a case."""
    return Path(root) / (STAGE_TENSOR_TEMPLATE % case_name)


def build_model(checkpoint_dir=DEFAULT_CHECKPOINT_DIR,
                expected_parameters=DEFAULT_EXPECTED_PARAMETERS,
                original_source=DEFAULT_ORIGINAL_SOURCE,
                checkpoint_metadata=DEFAULT_CHECKPOINT_METADATA,
                verify_shard_sha=True,
                verify_sources=True):
    """Load the 19 original Half conditioning weights and build the reference model."""
    torch = _torch_module()
    from safetensors import safe_open  # noqa: PLC0415 - lazy, keeps module scope stdlib-only

    checkpoint_dir = Path(checkpoint_dir)
    pins = load_expected_parameters(expected_parameters)
    sources = verify_original_sources(checkpoint_dir, original_source, pins,
                                      strict=verify_sources) if verify_sources else {'checks': {}}
    metadata = check_checkpoint_metadata(checkpoint_metadata)
    index_path = checkpoint_dir / 'model.safetensors.index.json'
    index = json.loads(index_path.read_text())
    weight_map = index.get('weight_map') or {}
    shard_names = {}
    for name in PARAMETER_NAMES:
        shard = weight_map.get(name)
        if not isinstance(shard, str):
            raise ValueError('checkpoint index has no shard for %s' % name)
        shard_names[name] = shard
    shard_pins = {}
    for shard in sorted(set(shard_names.values())):
        path = checkpoint_dir / shard
        if not path.is_file():
            raise FileNotFoundError('checkpoint shard is missing: %s' % path)
        pin = None
        if verify_shard_sha:
            if shard == NATIVE_HALF_FIRST_SHARD_NAME:
                pin = pins.get('nativeHalf_firstshard_sha256') or NATIVE_HALF_FIRST_SHARD_SHA256
            entry = {'path': str(path), 'expected': pin, 'actual': file_sha256(path)}
            if pin is not None and entry['actual'] != pin:
                raise RuntimeError('checkpoint shard %s changed: %s != %s'
                                   % (shard, entry['actual'], pin))
            entry['matches_pin'] = pin is not None and entry['actual'] == pin
            shard_pins[shard] = entry
        else:
            shard_pins[shard] = {'path': str(path), 'verified': False}
    raw = {}
    parameter_receipt = {}
    handles = {}
    try:
        for name in PARAMETER_NAMES:
            shard = shard_names[name]
            if shard not in handles:
                handles[shard] = safe_open(str(checkpoint_dir / shard), framework='pt', device='cpu')
            tensor = handles[shard].get_tensor(name)
            expected_shape = PARAMETER_SHAPES[name]
            if tuple(tensor.shape) != expected_shape:
                raise ValueError('%s shape %r != pinned %r'
                                 % (name, tuple(tensor.shape), expected_shape))
            if tensor.dtype != torch.float16:
                raise TypeError('%s is %s, expected the native FP16 alias' % (name, tensor.dtype))
            info = tensor_info(tensor)
            pin = pins['parameters'][name]['reference_sha256']
            if info['sha256'] != pin:
                raise RuntimeError('%s Half bytes changed: %s != pinned %s'
                                   % (name, info['sha256'], pin))
            parameter_receipt[name] = dict(info, pinned_sha256=pin, matches_original_half_pin=True)
            raw[name] = tensor
    finally:
        for handle in handles.values():
            del handle
    model_layout = {}
    for name in PARAMETER_NAMES:
        tensor = raw[name]
        if name.endswith('.attn.qkv_proj.weight'):
            order = torch.tensor(qkv_reorder_order(), dtype=torch.long)
            tensor = tensor.index_select(0, order).contiguous()
            parameter_receipt[name]['checkpoint_grouped_sha256'] = parameter_receipt[name]['sha256']
            parameter_receipt[name]['model_layout_sha256'] = tensor_info(tensor)['sha256']
        model_layout[name] = tensor.detach().to('cpu').contiguous()
        raw[name] = None
    del raw
    receipt = {
        'checkpoint_dir': str(checkpoint_dir),
        'pins_path': pins['path'],
        'pins_file_sha256': pins['file_sha256'],
        'checkpoint_metadata': metadata,
        'source_pins': sources,
        'shards': shard_pins,
        'parameters': parameter_receipt,
        'qkv_reorder_rows': len(qkv_reorder_order()),
        'qkv_reorder_rule': ('checkpoint [head_group, (q,k,v)*head_dim] rows -> '
                             'model [all Q | all K | all V] rows'),
        'original_half_values_verified_before_build': True,
        'no_float32_weight_shadow': True,
    }
    return ConditioningCPU111(model_layout, receipt=receipt)


# --------------------------------------------------------------------------- #
# Execution driver (root runs this; the author never did)
# --------------------------------------------------------------------------- #
def _load_case_tensor(case: dict) -> dict:
    torch = _torch_module()
    path = Path(case['path'])
    if not path.is_file():
        raise FileNotFoundError('inference input is missing: %s' % path)
    file_before = file_sha256(path)
    if case.get('file_sha256') and file_before != case['file_sha256']:
        raise RuntimeError('input file %s changed: %s != pinned %s'
                           % (path, file_before, case['file_sha256']))
    payload = torch.load(path, map_location='cpu', weights_only=True)
    tensor = payload
    if isinstance(payload, dict):
        if 'pre_final_norm' in payload:
            tensor = payload['pre_final_norm']
        else:
            candidates = [value for value in payload.values() if torch.is_tensor(value)]
            if len(candidates) != 1:
                raise ValueError('%s must hold one tensor (or a single-tensor dict), got %d'
                                 % (path, len(candidates)))
            tensor = candidates[0]
    if not torch.is_tensor(tensor):
        raise TypeError('%s does not contain a tensor' % path)
    info = tensor_info(tensor)
    if info['sha256'] != case['tensor_sha256']:
        raise RuntimeError('input tensor %s changed: %s != pinned %s'
                           % (path, info['sha256'], case['tensor_sha256']))
    rows, _ = text_rows_from_shape(info['shape'])
    if case.get('shape') is not None:
        pinned_rows, _ = text_rows_from_shape(case['shape'])
        if pinned_rows != rows:
            raise ValueError('input %s has %d rows, pin says %d' % (path, rows, pinned_rows))
    if info['dtype'] != 'torch.float16':
        raise TypeError('input %s is %s, expected torch.float16' % (path, info['dtype']))
    if not info['finite']:
        raise ValueError('input %s contains non-finite values' % path)
    squared = tensor.reshape(rows, ARCH['text_dim'])
    return {
        'name': case['name'],
        'path': str(path),
        'file_sha256_before': file_before,
        'file_sha256_after': None,
        'tensor_sha256': info['sha256'],
        'raw_input': info,
        'rows': rows,
        'tensor': squared,
        'role': case.get('role'),
    }


def _validate_stage_infos(infos: dict, rows: int) -> None:
    missing = [name for name in MANDATORY_STAGES if name not in infos]
    if missing:
        raise RuntimeError('conditioning trace is missing mandatory stages: %r' % missing)
    for name, info in infos.items():
        expected = expected_stage_shape(name, rows)
        if tuple(info['shape']) != expected:
            raise RuntimeError('stage %s has shape %r, expected %r'
                               % (name, info['shape'], expected))
        if info['dtype'] != 'torch.float16':
            raise RuntimeError('stage %s is %s, expected float16' % (name, info['dtype']))
        if not info['finite']:
            raise RuntimeError('stage %s contains non-finite values' % name)


def run(root=DEFAULT_ROOT,
        *,
        checkpoint_dir=DEFAULT_CHECKPOINT_DIR,
        expected_parameters=DEFAULT_EXPECTED_PARAMETERS,
        expected_inputs=DEFAULT_EXPECTED_INPUTS,
        original_source=DEFAULT_ORIGINAL_SOURCE,
        checkpoint_metadata=DEFAULT_CHECKPOINT_METADATA,
        passes=2,
        threads=DEFAULT_THREADS,
        case_names=None,
        keep=(),
        save_stage_tensors=True,
        max_stage_file_bytes=MAX_STAGE_FILE_BYTES,
        gold_pins=None,
        dry_run=False):
    """Execute the CPU FP16 conditioning reference and write its receipt.

    ``dry_run=True`` validates pins/schema/root exclusivity without importing
    torch and without writing anything.
    """
    if not isinstance(passes, int) or passes < 2:
        raise ValueError('the reference needs at least two exact passes, got %r' % (passes,))
    root = Path(root)
    receipt = build_receipt()
    receipt['root'] = str(root)
    receipt['dry_run'] = bool(dry_run)
    receipt['passes_requested'] = passes
    receipt['threads_requested'] = threads
    stage = ['init']
    started = time.perf_counter()

    def write_failure(error):
        receipt['failure'] = {'stage': stage[0], 'type': type(error).__name__,
                             'message': str(error),
                             'source_SHA256': receipt['source_SHA256']}
        try:
            if not (root / REFERENCE_JSON).exists() and not (root / FAILURE_JSON).exists():
                write_json_atomic(root / FAILURE_JSON, receipt)
        except OSError:
            pass

    try:
        stage[0] = 'root'
        pins = load_expected_parameters(expected_parameters)
        parameter_file_before = pins['file_sha256']
        cases_meta = []
        inputs_missing = []
        if Path(expected_inputs).is_file():
            cases_meta = load_expected_inputs(expected_inputs)
        else:
            inputs_missing.append(str(expected_inputs))
        selected = cases_meta
        if case_names:
            wanted = list(case_names)
            known = {case['name'] for case in cases_meta}
            unknown = [name for name in wanted if name not in known]
            if unknown:
                raise ValueError('unknown case names %r; known=%r' % (unknown, sorted(known)))
            selected = [case for case in cases_meta if case['name'] in set(wanted)]
            if not selected:
                raise ValueError('no input cases selected')
        refuse_existing_outputs(
            root,
            case_names=[case['name'] for case in selected],
            create=not dry_run,
        )
        receipt['expected_parameters'] = {
            'path': pins['path'],
            'file_sha256': parameter_file_before,
            'all19Halfvalues_exact_original_conversion': True,
            'parameter_count': len(pins['parameters']),
        }
        receipt['expected_inputs'] = {
            'path': str(expected_inputs),
            'declared_cases': [case['name'] for case in selected],
            'schema': 'inputs[] with name/path/tensor_sha256 (+ optional file_sha256, shape, role)',
        }
        stage[0] = 'source_pins'
        sources = verify_original_sources(checkpoint_dir, original_source, pins,
                                          strict=not dry_run)
        receipt['source_pins'] = sources
        gold = load_gold_pins(gold_pins)
        gold_before = {path: file_sha256(path) for path in gold}
        for path, digest in gold.items():
            if not Path(path).is_file():
                raise FileNotFoundError('gold file is missing: %s' % path)
            if gold_before[path] != digest:
                raise RuntimeError('gold file %s changed before the run' % path)
        receipt['gold_files'] = {
            'pins': gold,
            'sha256_before': gold_before,
            'all_match_before': True,
        }
        if dry_run:
            receipt['missing_files'] = inputs_missing + list(sources.get('missing') or [])
            receipt['torch_imported'] = 'torch' in sys.modules
            receipt['dry_run_ok'] = True
            receipt['plan'] = {
                'checkpoint_dir': str(checkpoint_dir),
                'expected_parameters': str(expected_parameters),
                'expected_inputs': str(expected_inputs),
                'original_source': str(original_source),
                'cases': [case['name'] for case in selected],
                'passes': passes,
                'threads': threads,
                'keep': list(keep) if keep != 'all' else 'all',
                'save_stage_tensors': bool(save_stage_tensors),
                'declared_outputs': [str(path) for path in
                                     declared_output_paths(root, [case['name'] for case in selected])],
            }
            if inputs_missing and not cases_meta:
                receipt['warnings'] = ['expected inputs file not present yet: %s'
                                       % expected_inputs]
            return receipt

        if not cases_meta:
            raise FileNotFoundError('expected inputs pin map is missing: %s' % expected_inputs)

        stage[0] = 'torch'
        torch = require_torch(threads=threads)
        receipt['torch'] = {
            'version': str(torch.__version__),
            'threads': int(torch.get_num_threads()),
            'mkldnn_enabled': bool(torch.backends.mkldnn.enabled),
            'device': 'cpu',
            'math_sdpa_forced': True,
        }
        stage[0] = 'build_model'
        model = build_model(checkpoint_dir, expected_parameters, original_source=original_source,
                            checkpoint_metadata=checkpoint_metadata, verify_shard_sha=True,
                            verify_sources=True)
        receipt['checkpoint'] = {
            'checkpoint_dir': model.receipt['checkpoint_dir'],
            'checkpoint_metadata': model.receipt['checkpoint_metadata'],
            'shards': model.receipt['shards'],
        }
        receipt['parameters'] = model.receipt['parameters']
        receipt['qkv_reorder_rule'] = model.receipt['qkv_reorder_rule']
        receipt['qkv_reorder_rows'] = model.receipt['qkv_reorder_rows']
        receipt['original_half_values_verified_before_build'] = True
        parameter_sha_before = model.parameter_sha()
        receipt['parameters_before_run'] = parameter_sha_before
        receipt['parameter_dtypes'] = sorted(model.parameter_dtypes())
        receipt['no_float32_parameter_shadow'] = bool(model.no_float32_shadow())

        stage[0] = 'inputs'
        inputs = []
        for case in selected:
            loaded = _load_case_tensor(case)
            receipt.setdefault('inputs', {})[loaded['name']] = {
                'path': loaded['path'],
                'file_sha256': loaded['file_sha256_before'],
                'tensor_sha256': loaded['tensor_sha256'],
                'raw_shape': loaded['raw_input']['shape'],
                'raw_dtype': loaded['raw_input']['dtype'],
                'text_rows': loaded['rows'],
                'role': loaded['role'],
                'matches_pinned_tensor_sha256': True,
                'matches_pinned_file_sha256': bool(case.get('file_sha256')),
            }
            inputs.append(loaded)

        stage[0] = 'forward'
        receipt['passes'] = {'passes': passes, 'exact_repeat_required': True, 'cases': {}}
        receipt['cases'] = {}
        stage_files = {}
        for loaded in inputs:
            first_stages = None
            first_infos = None
            for pass_index in range(1, passes + 1):
                with torch.inference_mode():
                    trace = model.forward_stages(loaded['tensor'], keep=keep)
                infos = {name: tensor_info(tensor) for name, tensor in trace.items()}
                _validate_stage_infos(infos, loaded['rows'])
                if pass_index == 1:
                    first_stages, first_infos = trace, infos
                else:
                    if infos != first_infos:
                        raise RuntimeError('pass %d of case %r is not bit-identical to pass 1'
                                           % (pass_index, loaded['name']))
                    del trace, infos
            assert first_stages is not None and first_infos is not None
            case_receipt = {
                'text_rows': loaded['rows'],
                'stages': first_infos,
                'stage_count': len(first_infos),
                'stage_bytes': stage_mapping_bytes(first_stages),
                'exact_repeat_across_%d_passes' % passes: True,
                'uses_original_cpu_half_parameters_only': True,
            }
            if save_stage_tensors:
                target = root / (STAGE_TENSOR_TEMPLATE % loaded['name'])
                if target.exists():
                    raise FileExistsError('refusing to overwrite %s' % target)
                if case_receipt['stage_bytes'] > max_stage_file_bytes:
                    raise RuntimeError(
                        'stage tensors for %r need %d bytes > max_stage_file_bytes=%d; '
                        'narrow --keep or raise the limit explicitly'
                        % (loaded['name'], case_receipt['stage_bytes'], max_stage_file_bytes))
                save_tensors_atomic(target, first_stages)
                case_receipt['stage_tensor_file'] = {
                    'path': str(target),
                    'file_sha256': file_sha256(target),
                    'file_bytes': target.stat().st_size,
                    'contains': sorted(first_stages),
                }
                stage_files[loaded['name']] = case_receipt['stage_tensor_file']
            receipt['cases'][loaded['name']] = case_receipt
            receipt['passes']['cases'][loaded['name']] = {
                'stage_sha256': {name: infos['sha256'] for name, infos in first_infos.items()},
                'exact_repeat_across_%d_passes' % passes: True,
            }
            receipt['passes'][loaded['name']] = receipt['passes']['cases'][loaded['name']]
            del first_stages, first_infos

        stage[0] = 'final_checks'
        parameter_sha_after = model.parameter_sha()
        receipt['parameters_after_run'] = parameter_sha_after
        receipt['original_parameters_unchanged'] = parameter_sha_after == parameter_sha_before
        if not receipt['original_parameters_unchanged']:
            raise RuntimeError('original Half conditioning parameters changed during the run')
        for loaded in inputs:
            loaded['file_sha256_after'] = file_sha256(loaded['path'])
            entry = receipt['inputs'][loaded['name']]
            entry['file_sha256_after'] = loaded['file_sha256_after']
            entry['file_unchanged'] = loaded['file_sha256_before'] == loaded['file_sha256_after']
            if not entry['file_unchanged']:
                raise RuntimeError('input %s changed during the run' % loaded['path'])
        sources_after = verify_original_sources(checkpoint_dir, original_source, pins, strict=True)
        receipt['source_pins_after_run'] = sources_after
        receipt['original_sources_unchanged'] = True
        parameter_file_after = file_sha256(expected_parameters)
        receipt['expected_parameters_file_unchanged'] = parameter_file_after == parameter_file_before
        if not receipt['expected_parameters_file_unchanged']:
            raise RuntimeError('the Half oracle pin file changed during the run')
        receipt['gold_files']['sha256_after'] = {path: file_sha256(path) for path in gold}
        receipt['gold_files']['all_unchanged'] = (
            receipt['gold_files']['sha256_after'] == gold_before)
        if not receipt['gold_files']['all_unchanged']:
            raise RuntimeError('a read-only gold file changed during the run')
        for loaded in inputs:
            receipt['cases'][loaded['name']]['stage_tensor_file'] = stage_files.get(loaded['name'])
        receipt['completed'] = True
        receipt['cpu_conditioning_reference_completed'] = True
        receipt['runtime_seconds'] = time.perf_counter() - started
        stage[0] = 'save'
        write_json_atomic(root / REFERENCE_JSON, receipt)
        return receipt
    except BaseException as error:
        write_failure(error)
        raise


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument('--root', type=Path, default=DEFAULT_ROOT,
                        help='new private CPU root for stage tensors and the receipt')
    parser.add_argument('--checkpoint-dir', type=Path, default=DEFAULT_CHECKPOINT_DIR,
                        help='native FP16 H3 transformer checkpoint directory')
    parser.add_argument('--original-source', type=Path, default=DEFAULT_ORIGINAL_SOURCE,
                        help='read-only original H3 DiT source file to pin')
    parser.add_argument('--checkpoint-metadata', type=Path, default=DEFAULT_CHECKPOINT_METADATA,
                        help='round111 checkpoint metadata pin file')
    parser.add_argument('--expected-parameters', type=Path, default=DEFAULT_EXPECTED_PARAMETERS,
                        help='root-supplied 19 Half parameter pin map')
    parser.add_argument('--expected-inputs', type=Path, default=DEFAULT_EXPECTED_INPUTS,
                        help='root-supplied inference input pin map')
    parser.add_argument('--passes', type=int, default=2,
                        help='exact-repeat passes per case (>= 2)')
    parser.add_argument('--threads', type=int, default=DEFAULT_THREADS,
                        help='torch CPU threads (reference control uses 16)')
    parser.add_argument('--case', action='append', default=None, dest='cases',
                        help='restrict to a named case from the inputs pin map (repeatable)')
    parser.add_argument('--keep', action='append', default=None,
                        help='extra boundary stage to retain, e.g. block0.norm1 (repeatable), '
                             'or --keep all')
    parser.add_argument('--max-stage-bytes', type=int, default=MAX_STAGE_FILE_BYTES,
                        help='refuse to serialize a case stage dict larger than this')
    parser.add_argument('--gold-pins', type=Path, default=None,
                        help='optional read-only gold file pins {path: sha256}')
    parser.add_argument('--dry-run', action='store_true',
                        help='validate pins/schema/root exclusivity without importing torch')
    return parser


def keep_selection(values):
    if not values:
        return ()
    if 'all' in values:
        if len(values) > 1:
            raise ValueError("--keep 'all' cannot be combined with explicit stage names")
        return 'all'
    return tuple(values)


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    try:
        receipt = run(
            root=args.root,
            checkpoint_dir=args.checkpoint_dir,
            expected_parameters=args.expected_parameters,
            expected_inputs=args.expected_inputs,
            original_source=args.original_source,
            checkpoint_metadata=args.checkpoint_metadata,
            passes=args.passes,
            threads=args.threads,
            case_names=args.cases,
            keep=keep_selection(args.keep),
            max_stage_file_bytes=args.max_stage_bytes,
            gold_pins=args.gold_pins,
            dry_run=args.dry_run,
        )
    except FileExistsError as error:
        print(json.dumps({'ok': False, 'error': 'root_not_reusable', 'message': str(error)},
                         indent=2))
        return 3
    except BaseException as error:  # noqa: BLE001 - CLI surfaces every failure as JSON
        print(json.dumps({'ok': False, 'error': type(error).__name__, 'message': str(error)},
                         indent=2), file=sys.stderr)
        return 2
    if args.dry_run:
        print(json.dumps({'ok': True, 'dry_run': True, 'plan': receipt.get('plan'),
                          'missing_files': receipt.get('missing_files'),
                          'torch_imported': receipt.get('torch_imported')}, indent=2))
    else:
        print(json.dumps({'ok': True,
                          'reference': str(args.root / REFERENCE_JSON),
                          'cases': sorted(receipt.get('cases') or {}),
                          'completed': receipt.get('completed'),
                          'cpu_conditioning_reference_completed':
                              receipt.get('cpu_conditioning_reference_completed'),
                          'numerical_tolerance': receipt.get('numerical_tolerance'),
                          'full_encoder_precision_accepted':
                              receipt.get('full_encoder_precision_accepted')}, indent=2))
    return 0


if __name__ == '__main__':
    sys.exit(main())