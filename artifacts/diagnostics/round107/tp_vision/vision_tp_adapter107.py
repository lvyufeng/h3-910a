#!/usr/bin/env python3
"""Round107 -- private original-Half Vision TP8 adapter for the 27 Qwen3-VL vision blocks.

SOURCE ONLY.  Nothing here was compiled, launched or measured by its author: the root
agent supplies the GEMM backend, the local 2-head attention callable, the TP group and
the real model, and owns every hardware receipt.

Scope (deliberately bounded)
----------------------------
Only ``blocks[i].attn.forward`` and ``blocks[i].mlp.forward`` are replaced, and only as
*instance* attributes (never a class-level ``Linear.forward`` patch, which would collide
with the root's ``VisionBiasAdapter87``).  Patch embedding, the four merger families,
LayerNorm, rotary geometry, ``cu_seqlens`` construction, DeepStack and the 50 text layers
stay the executed code, so the Round90 true-erf merger contract (2016 x 4608) is
untouched.

Tensor-parallel decomposition (rank ``r`` of 8)
----------------------------------------------
* QKV: original weight ``[3456,1152]`` = three 1152-wide q/k/v segments; rank r keeps the
  144 *output rows* ``s*1152 + r*144 : s*1152 + (r+1)*144`` of each segment and
  concatenates them into a 432-wide local projection.  Bias sliced identically.
  Column mode: the backend accumulates in Float32 over the original Half input/weight,
  adds the exact original Half bias widened to Float32, and casts to Half once.
* Attention runs locally on 2 heads per rank, per image segment (4032 tokens), with the
  original scale ``72 ** -0.5`` and original ``cu_seqlens``; no causal mask.
* Out projection: original weight ``[1152,1152]``; rank r keeps input *columns*
  ``r*144:(r+1)*144`` (row mode).  The backend returns a **Float32 row partial** with no
  bias; the adapter sums the eight partials with ``group.all_reduce`` in Float32, adds
  the full original bias widened to Float32, then casts to Half once.
* MLP fc1: original ``[4304,1152]``; rank r keeps output rows ``r*538:(r+1)*538``
  (column mode, Half out) and the *original* ``nn.GELU(approximate="tanh")`` is applied
  to exactly those 538 logical channels (never to Cube padding).  fc2: original
  ``[1152,4304]``; rank r keeps input columns ``r*538:(r+1)*538`` (row mode, Float32
  partial) then the same Float32 collective + full bias + one Half cast.

Exactly 4304 = 8*538 and 1152 = 8*144, so the eight local chunks tile the real channels
with no gap, overlap or omission; the only extra channels are zero padding added *inside*
the backend to reach the Cube multiples of 128.

Honesty constraints
-------------------
* Every original weight/bias stays Half on the rank's NPU device; only Half weight
  slices are handed to the backend for priming and no Float weight shadow, activation
  cache, BF16 gold cast, fake norm, clamp or tolerance is introduced.
* ``full_encoder_precision_accepted``/``whole_goal_complete`` stay False and no 8x
  speedup is claimed: Float32 HCCL partials can change low bits, and the root must
  measure the real graph against the original CPU FP16 reference.
"""
from __future__ import annotations

import copy
import hashlib
import types
from pathlib import Path

# ---------------------------------------------------------------------------
# Checkpoint geometry (guarded against the loaded model, never assumed silently)
# ---------------------------------------------------------------------------
VISION_TOKENS = 8064              # two images x 4032 tokens
VISION_SEGMENT = 4032             # tokens per image
VISION_DIM = 1152
VISION_HEADS = 16
VISION_HEAD_DIM = 72              # 16 * 72 == 1152
VISION_MLP = 4304                 # 8 * 538
VISION_BLOCKS = 27
TP_WORLD = 8
CUBE_ALIGN = 128
SCALE = VISION_HEAD_DIM ** -0.5   # 0.11785113019775793, the original attention scale

PARAM_ORDER = ('weight', 'bias')


class VisionTPError(RuntimeError):
    """Refusal: geometry, dtype, ownership or backend contract violated."""


# ---------------------------------------------------------------------------
# Pure planning helpers (standard library only; the tests exercise these directly)
# ---------------------------------------------------------------------------
def pad_to_align(value: int, align: int = CUBE_ALIGN) -> int:
    """Round ``value`` up to the Cube alignment used by the root backend."""
    if isinstance(value, bool) or not isinstance(value, int):
        raise VisionTPError(f'expected an int, got {type(value).__name__}')
    if value <= 0:
        raise VisionTPError(f'expected a positive dimension, got {value}')
    return ((value + align - 1) // align) * align


def local_channel_spans(world_size: int = TP_WORLD, total: int = VISION_DIM) -> tuple:
    """Return the eight half-open output-channel spans, tiling ``[0,total)`` exactly."""
    if isinstance(world_size, bool) or not isinstance(world_size, int) or world_size <= 0:
        raise VisionTPError(f'world size must be a positive int, got {world_size!r}')
    if total % world_size:
        raise VisionTPError(f'{total} output channels are not divisible by {world_size}')
    step = total // world_size
    spans = tuple((rank * step, (rank + 1) * step) for rank in range(world_size))
    if spans[0][0] != 0 or spans[-1][1] != total:
        raise VisionTPError('channel spans do not tile the full width')
    for left, right in zip(spans, spans[1:]):
        if left[1] != right[0]:
            raise VisionTPError(f'channel spans are not contiguous: {left} -> {right}')
    return spans


def plan_local_geometry(rank: int, world_size: int = TP_WORLD) -> dict:
    """Every local dimension and slice this rank uses, with no torch dependency."""
    if isinstance(world_size, bool) or not isinstance(world_size, int) or world_size != TP_WORLD:
        raise VisionTPError(f'this adapter is specialised to TP world size {TP_WORLD}, got {world_size!r}')
    if isinstance(rank, bool) or not isinstance(rank, int) or not 0 <= rank < world_size:
        raise VisionTPError(f'rank must be an int in [0,{world_size}), got {rank!r}')
    segment = VISION_HEADS * VISION_HEAD_DIM
    if segment != VISION_DIM:
        raise VisionTPError(f'head geometry {VISION_HEADS}x{VISION_HEAD_DIM} != hidden {VISION_DIM}')

    head_spans = local_channel_spans(world_size, segment)
    mlp_spans = local_channel_spans(world_size, VISION_MLP)
    head_lo, head_hi = head_spans[rank]
    mlp_lo, mlp_hi = mlp_spans[rank]
    local_head_channels = head_hi - head_lo
    local_mlp = mlp_hi - mlp_lo
    local_heads = VISION_HEADS // world_size
    if local_head_channels != local_heads * VISION_HEAD_DIM:
        raise VisionTPError(f'{local_head_channels} local channels != {local_heads} heads')

    qkv_rows = tuple((segment * slot + head_lo, segment * slot + head_hi) for slot in range(3))
    qkv_n = 3 * local_head_channels
    geometries = {
        'attn.qkv': ('column', segment, qkv_n),
        'attn.proj': ('row', local_head_channels, VISION_DIM),
        'mlp.fc1': ('column', VISION_DIM, local_mlp),
        'mlp.fc2': ('row', local_mlp, VISION_DIM),
    }
    padded = {role: (pad_to_align(k), pad_to_align(n))
              for role, (_, k, n) in geometries.items()}
    return {
        'rank': rank, 'world_size': world_size, 'm': VISION_TOKENS,
        'segment': segment, 'segment_tokens': VISION_SEGMENT,
        'num_segments': VISION_TOKENS // VISION_SEGMENT,
        'heads': VISION_HEADS, 'head_dim': VISION_HEAD_DIM,
        'local_heads': local_heads, 'local_head_channels': local_head_channels,
        'local_mlp_channels': local_mlp,
        'head_channel_span': (head_lo, head_hi), 'mlp_channel_span': (mlp_lo, mlp_hi),
        'qkv_row_slices': qkv_rows,
        'out_proj_input_columns': (head_lo, head_hi),
        'mlp_fc1_output_rows': (mlp_lo, mlp_hi), 'mlp_fc2_input_columns': (mlp_lo, mlp_hi),
        'geometries': geometries, 'padded_geometries': padded,
    }


def padded_geometries(world_size: int = TP_WORLD) -> dict:
    """The four backend (mode, k, n, k_pad, n_pad) contracts; rank independent."""
    plan = plan_local_geometry(0, world_size)
    out = {}
    for role, (mode, k, n) in plan['geometries'].items():
        k_pad, n_pad = plan['padded_geometries'][role]
        out[role] = {'mode': mode, 'k': k, 'n': n, 'k_pad': k_pad, 'n_pad': n_pad}
    return out


def expected_counts_per_forward(num_segments: int = VISION_TOKENS // VISION_SEGMENT) -> dict:
    """Counters one enabled vision forward must produce (27 blocks, 2 image segments)."""
    if isinstance(num_segments, bool) or not isinstance(num_segments, int) or num_segments < 1:
        raise VisionTPError(f'num_segments must be a positive int, got {num_segments!r}')
    per_block_collectives = 2
    return {
        'attention_calls': VISION_BLOCKS,
        'mlp_calls': VISION_BLOCKS,
        'attention_segments': VISION_BLOCKS * num_segments,
        'all_reduce_calls': VISION_BLOCKS * per_block_collectives,
        'column_runs': VISION_BLOCKS * 2,
        'row_runs': VISION_BLOCKS * 2,
        'primed_weight_modules': VISION_BLOCKS * 4,
    }


# ---------------------------------------------------------------------------
# Small runtime helpers
# ---------------------------------------------------------------------------
def _torch():
    """Import torch lazily so this module stays importable without it."""
    import torch
    return torch


def _token(value) -> str:
    target = value if not isinstance(value, type) else value
    module = getattr(target, '__module__', None) or type(target).__module__
    qualname = getattr(target, '__qualname__', None) or getattr(target, '__name__', None)
    return f'{module}.{qualname}' if qualname else f'{type(target).__name__}'


def _source_sha256() -> str:
    try:
        return hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    except OSError:
        return 'unavailable'


def _safe_copy(value):
    try:
        return copy.deepcopy(value)
    except Exception as error:                                   # noqa: BLE001 - receipt only
        return {'deepcopy_failed': f'{type(error).__name__}: {error}',
                'value_type': type(value).__name__}


def _tensor_version(tensor) -> int:
    """Autograd version counter; inference tensors raise, so record -1 instead."""
    try:
        return int(tensor._version)
    except Exception:                                            # noqa: BLE001
        return -1


def _param_identity(param) -> tuple:
    """Cheap identity of an original parameter, without reading its 2 GiB of values."""
    return {
        'data_ptr': int(param.data_ptr()),
        'storage_offset': int(param.storage_offset()),
        'numel': int(param.numel()),
        'shape': tuple(int(d) for d in param.shape),
        'dtype': str(param.dtype),
        'device': str(param.device),
        'version': _tensor_version(param),
    }


class VisionTPAdapter107:
    """Private original-Half Vision TP8 for the 27 vision blocks' attention and MLP.

    Parameters
    ----------
    candidate : the loaded R79 candidate module (owns ``_apply_rotary_pos_emb_vision``).
    group : TP group coordinator; needs ``rank_in_group``, ``world_size`` and
        ``all_reduce(FloatTensor) -> FloatTensor``.
    backend : root GEMM backend; needs ``prime(name, weight_half, bias_half_or_None,
        mode, m=...) -> entry``, ``run(entry, x_half_2d)`` and ``snapshot()``.
    attention : standalone callable ``(q, k, v) -> 4D Half`` for 2 local heads per rank.
        It owns the original ``72 ** -0.5`` scale; nothing here replaces the global
        ``_vision_sdpa_bf16_compatible`` helper.
    m : sequence length of one vision forward (must be 8064: two 4032-token images).
    """

    def __init__(self, candidate, group, backend, attention, m: int = VISION_TOKENS):
        if not callable(attention):
            raise VisionTPError('attention must be a callable (q, k, v) -> Half 4D tensor')
        for name, methods in (('group', ('all_reduce',)),
                              ('backend', ('prime', 'run', 'snapshot'))):
            holder = {'group': group, 'backend': backend}[name]
            for method in methods:
                if not callable(getattr(holder, method, None)):
                    raise VisionTPError(f'{name} must provide a callable {method}()')
        rank = getattr(group, 'rank_in_group', None)
        world = getattr(group, 'world_size', None)
        if isinstance(rank, bool) or not isinstance(rank, int):
            raise VisionTPError(f'group.rank_in_group must be an int, got {rank!r}')
        if isinstance(world, bool) or not isinstance(world, int) or world != TP_WORLD:
            raise VisionTPError(f'group.world_size must be {TP_WORLD}, got {world!r}')
        if not 0 <= rank < world:
            raise VisionTPError(f'group.rank_in_group {rank} outside [0,{world})')
        if isinstance(m, bool) or not isinstance(m, int) or m != VISION_TOKENS:
            raise VisionTPError(f'this adapter is specialised to m={VISION_TOKENS}, got {m!r}')
        rotary = getattr(candidate, '_apply_rotary_pos_emb_vision', None)
        if not callable(rotary):
            raise VisionTPError('candidate must expose _apply_rotary_pos_emb_vision')

        self._candidate = candidate
        self._group = group
        self._backend = backend
        self._attention = attention
        self._rotary = rotary
        self._m = m
        self._rank = rank
        self._world = world
        self._geometry = plan_local_geometry(rank, world)
        self._segments = expected_counts_per_forward(self._geometry['num_segments'])
        self._primed = False
        self._enabled = False
        self._installed = False
        self._prepared = []          # plain dicts only: safe to deepcopy
        self._signatures = {}        # (index, label, 'weight'|'bias') -> identity dict
        self._roles = {}             # (index, role) -> entry dict (holds the opaque entry)
        self._patched = []           # patch records holding the live modules
        self._stats = {
            'primed_weight_modules': 0, 'parameter_identity_checks': 0,
            'enabled_attention_calls': 0, 'enabled_mlp_calls': 0,
            'disabled_attention_calls': 0, 'disabled_mlp_calls': 0,
            'attention_segments': 0, 'column_runs': 0, 'row_runs': 0,
            'all_reduce_calls': 0, 'bias_widenings': 0, 'half_outputs': 0,
            'activation_half_checks': 0, 'dense_copies': 0,
        }
        self._attention_token = _token(attention)
        self._rotary_token = _token(rotary)

    # -- introspection ----------------------------------------------------
    def __repr__(self) -> str:
        return (f'{type(self).__name__}(rank={self._rank}, m={self._m}, '
                f'primed={self._primed}, enabled={self._enabled})')

    @property
    def geometry(self) -> dict:
        return copy.deepcopy(self._geometry)

    # -- guards -----------------------------------------------------------
    def _check_parameter(self, param, expected_shape, label, torch) -> None:
        if param is None:
            raise VisionTPError(f'{label} is missing; every TP-sliced linear needs its bias')
        if not isinstance(param, torch.Tensor):
            raise VisionTPError(f'{label} is not a tensor')
        if param.dtype != torch.float16:
            raise VisionTPError(f'{label} must stay original Half, got {param.dtype}')
        if str(param.device.type) != 'npu':
            raise VisionTPError(f'{label} must live on this rank NPU, got {param.device}')
        if int(param.device.index if param.device.index is not None else -1) != self._rank:
            raise VisionTPError(f'{label} device {param.device} is not rank {self._rank}')
        shape = tuple(int(d) for d in param.shape)
        if shape != tuple(expected_shape):
            raise VisionTPError(f'{label} shape {shape} != expected {tuple(expected_shape)}')

    def _check_hidden(self, torch, value, where: str) -> int:
        if not isinstance(value, torch.Tensor):
            raise VisionTPError(f'{where}: expected a tensor, got {type(value).__name__}')
        if value.dtype != torch.float16:
            raise VisionTPError(f'{where}: expected Half activations, got {value.dtype}')
        if value.dim() != 2:
            raise VisionTPError(f'{where}: expected a 2D (m,1152) activation')
        shape = tuple(int(d) for d in value.shape)
        if shape != (self._m, VISION_DIM):
            raise VisionTPError(f'{where}: shape {shape} != {(self._m, VISION_DIM)}')
        index = value.device.index
        if str(value.device.type) != 'npu' or index is None or int(index) != self._rank:
            raise VisionTPError(f'{where}: activation on {value.device}, expected npu:{self._rank}')
        return shape[0]

    def _segment_lengths(self, torch, cu_seqlens, seq: int) -> list:
        if not isinstance(cu_seqlens, torch.Tensor):
            raise VisionTPError('cu_seqlens must be a tensor')
        if cu_seqlens.dim() != 1:
            raise VisionTPError('cu_seqlens must be 1D')
        values = [int(v) for v in cu_seqlens.tolist()]
        if len(values) < 2:
            raise VisionTPError(f'cu_seqlens needs at least two offsets, got {values}')
        if values[0] != 0 or values[-1] != seq:
            raise VisionTPError(f'cu_seqlens {values} does not span [0,{seq})')
        if values != sorted(values):
            raise VisionTPError(f'cu_seqlens {values} is not monotonic')
        lengths = [right - left for left, right in zip(values, values[1:])]
        if any(length != VISION_SEGMENT for length in lengths):
            raise VisionTPError(f'every image segment must be {VISION_SEGMENT} tokens, got {lengths}')
        if sum(lengths) != seq:
            raise VisionTPError(f'segment lengths {lengths} do not sum to {seq}')
        return lengths

    def _verify_linears(self, index: int, modules) -> None:
        for label, module in modules:
            for name in PARAM_ORDER:
                key = (index, label, name)
                expected = self._signatures.get(key)
                if expected is None:
                    raise VisionTPError(f'parameter {key} was never primed')
                param = getattr(module, name, None)
                if param is None:
                    raise VisionTPError(f'parameter {key} disappeared')
                if _param_identity(param) != expected:
                    raise VisionTPError(f'parameter {key} changed identity; refusing to run')
                self._stats['parameter_identity_checks'] += 1

    # -- backend plumbing -------------------------------------------------
    def _dense(self, torch, activation):
        """Return a dense, offset-zero view without changing any value.

        A contiguous tensor can still be a view with a nonzero storage offset (the
        executed vision attention guard hit exactly that); clone only in that case so
        the normal freshly-allocated norm/backend output path never pays a copy.
        """
        dense = activation.contiguous()
        if int(dense.storage_offset()) != 0:
            dense = dense.clone(memory_format=torch.contiguous_format)
            self._stats['dense_copies'] += 1
        if int(dense.storage_offset()) != 0:
            raise VisionTPError('could not produce an offset-zero dense activation')
        return dense

    def _run_column(self, index: int, role: str, activation):
        torch = _torch()
        entry = self._roles[(index, role)]
        out = self._backend.run(entry['entry'], self._dense(torch, activation))
        if not isinstance(out, torch.Tensor):
            raise VisionTPError(f'{role}: backend returned {type(out).__name__}')
        if out.dtype != torch.float16:
            raise VisionTPError(f'{role}: column mode must return Half, got {out.dtype}')
        shape = tuple(int(d) for d in out.shape)
        if shape != (self._m, entry['n']):
            raise VisionTPError(f'{role}: column output {shape} != {(self._m, entry["n"])}')
        self._stats['column_runs'] += 1
        return out

    def _run_row(self, index: int, role: str, activation):
        torch = _torch()
        entry = self._roles[(index, role)]
        out = self._backend.run(entry['entry'], self._dense(torch, activation))
        if not isinstance(out, torch.Tensor):
            raise VisionTPError(f'{role}: backend returned {type(out).__name__}')
        if out.dtype != torch.float32:
            raise VisionTPError(
                f'{role}: row mode must return a Float32 partial before the collective, '
                f'got {out.dtype}; a Half partial would lose low bits before HCCL')
        shape = tuple(int(d) for d in out.shape)
        if shape != (self._m, entry['n']):
            raise VisionTPError(f'{role}: row partial {shape} != {(self._m, entry["n"])}')
        self._stats['row_runs'] += 1
        return out

    def _all_reduce(self, partial, role: str, index: int):
        torch = _torch()
        if partial.dtype != torch.float32:
            raise VisionTPError(f'{role}: refusing a {partial.dtype} collective')
        expected = tuple(int(d) for d in partial.shape)
        reduced = self._group.all_reduce(partial)
        if not isinstance(reduced, torch.Tensor):
            raise VisionTPError(f'{role}: all_reduce returned {type(reduced).__name__}')
        if reduced.dtype != torch.float32:
            raise VisionTPError(f'{role}: all_reduce must stay Float32, got {reduced.dtype}')
        shape = tuple(int(d) for d in reduced.shape)
        if shape != expected:
            raise VisionTPError(f'{role}: all_reduce shape {shape} != {expected}')
        self._stats['all_reduce_calls'] += 1
        return reduced

    def _full_bias(self, torch, bias, width: int):
        if bias is None:
            raise VisionTPError('the four vision linears all carry an original Half bias')
        if bias.dtype != torch.float16:
            raise VisionTPError(f'original bias must stay Half, got {bias.dtype}')
        if tuple(int(d) for d in bias.shape) != (width,):
            raise VisionTPError(f'original bias shape {tuple(bias.shape)} != {(width,)}')
        widened = bias.detach().float()
        if widened.dtype != torch.float32:
            raise VisionTPError('bias widening must produce Float32')
        self._stats['bias_widenings'] += 1
        return widened

    # -- priming ----------------------------------------------------------
    @staticmethod
    def _resolve_vision(vision):
        if hasattr(vision, 'blocks'):
            return vision
        inner = getattr(vision, 'vision', None)
        if inner is not None and hasattr(inner, 'blocks'):
            return inner
        raise VisionTPError('prime() needs the vision tower (an object with .blocks)')

    @staticmethod
    def _linears_of(kind: str, module):
        """The two linears of this submodule under their plan geometry role names."""
        if kind == 'attn':
            return (('attn.qkv', module.qkv), ('attn.proj', module.proj))
        return (('mlp.fc1', module.linear_fc1), ('mlp.fc2', module.linear_fc2))

    def _slice_rows(self, torch, param, row_slices, label: str):
        base = param.detach()
        parts = [base[low:high] for low, high in row_slices]
        if len(parts) == 1:
            sliced = parts[0]
        else:
            sliced = torch.cat(parts, dim=0)
        expected = sum(high - low for low, high in row_slices)
        if int(sliced.shape[0]) != expected:
            raise VisionTPError(f'{label}: row slice produced {sliced.shape[0]} != {expected}')
        return sliced.contiguous()

    def _slice_columns(self, torch, param, span, label: str):
        low, high = span
        sliced = param.detach()[:, low:high]
        if int(sliced.shape[1]) != high - low:
            raise VisionTPError(f'{label}: column slice produced {sliced.shape[1]} != {high - low}')
        return sliced.contiguous()

    def _entry_name(self, index: int, role: str, mode: str, k: int, n: int, spec: str) -> str:
        return f'blocks.{index:02d}.{role}:{mode}:k{k}:n{n}:{spec}'

    def _prime_block(self, index: int, block) -> None:
        torch = _torch()
        attn = getattr(block, 'attn', None)
        mlp = getattr(block, 'mlp', None)
        if attn is None or mlp is None:
            raise VisionTPError(f'block {index} is missing .attn or .mlp')
        for attribute, expected in (('num_heads', VISION_HEADS), ('head_dim', VISION_HEAD_DIM),
                                    ('dim', VISION_DIM)):
            actual = getattr(attn, attribute, None)
            if isinstance(actual, bool) or not isinstance(actual, int) or actual != expected:
                raise VisionTPError(f'block {index} attn.{attribute}={actual!r}, expected {expected}')
        act_fn = getattr(mlp, 'act_fn', None)
        if act_fn is None or getattr(act_fn, 'approximate', None) != 'tanh':
            raise VisionTPError(
                f'block {index} mlp.act_fn must be the original nn.GELU(approximate="tanh")')

        geometry = self._geometry
        segment = geometry['segment']
        local_head_channels = geometry['local_head_channels']
        local_mlp = geometry['local_mlp_channels']
        qkv_rows = geometry['qkv_row_slices']
        head_span = geometry['out_proj_input_columns']
        mlp_span = geometry['mlp_fc1_output_rows']

        linears = self._linears_of('attn', attn) + self._linears_of('mlp', mlp)
        for label, module in linears:
            weight = getattr(module, 'weight', None)
            bias = getattr(module, 'bias', None)
            if label == 'attn.qkv':
                expected = ((3 * segment, VISION_DIM), (3 * segment,))
            elif label == 'attn.proj':
                expected = ((VISION_DIM, VISION_DIM), (VISION_DIM,))
            elif label == 'mlp.fc1':
                expected = ((VISION_MLP, VISION_DIM), (VISION_MLP,))
            else:
                expected = ((VISION_DIM, VISION_MLP), (VISION_DIM,))
            self._check_parameter(weight, expected[0], f'blocks.{index}.{label}.weight', torch)
            self._check_parameter(bias, expected[1], f'blocks.{index}.{label}.bias', torch)
            self._signatures[(index, label, 'weight')] = _param_identity(weight)
            self._signatures[(index, label, 'bias')] = _param_identity(bias)

        # Four local geometries; only Half slices and the original bias cross the boundary.
        qkv_weight = self._slice_rows(torch, attn.qkv.weight, qkv_rows, f'blocks.{index}.qkv.weight')
        qkv_bias = self._slice_rows(torch, attn.qkv.bias, qkv_rows, f'blocks.{index}.qkv.bias')
        proj_weight = self._slice_columns(torch, attn.proj.weight, head_span,
                                          f'blocks.{index}.proj.weight')
        fc1_weight = self._slice_rows(torch, mlp.linear_fc1.weight, (mlp_span,),
                                      f'blocks.{index}.fc1.weight')
        fc1_bias = self._slice_rows(torch, mlp.linear_fc1.bias, (mlp_span,),
                                    f'blocks.{index}.fc1.bias')
        fc2_weight = self._slice_columns(torch, mlp.linear_fc2.weight, mlp_span,
                                         f'blocks.{index}.fc2.weight')

        jobs = (
            ('attn.qkv', qkv_weight, qkv_bias, 'column',
             f'rows{list(qkv_rows)}'),
            ('attn.proj', proj_weight, None, 'row', f'cols{list(head_span)}'),
            ('mlp.fc1', fc1_weight, fc1_bias, 'column', f'rows{list(mlp_span)}'),
            ('mlp.fc2', fc2_weight, None, 'row', f'cols{list(mlp_span)}'),
        )
        for role, weight, bias, mode, spec in jobs:
            expected_mode, k, n = geometry['geometries'][role]
            if mode != expected_mode or int(weight.shape[1]) != k or int(weight.shape[0]) != n:
                raise VisionTPError(
                    f'blocks.{index}.{role}: sliced {(weight.shape[0], weight.shape[1])} '
                    f'does not match {mode} k={k} n={n}')
            if (bias is None) != (mode == 'row'):
                raise VisionTPError(f'blocks.{index}.{role}: bias is added after the collective only')
            k_pad, n_pad = geometry['padded_geometries'][role]
            name = self._entry_name(index, role, mode, k, n, spec)
            entry = self._backend.prime(name, weight, bias, mode, m=self._m)
            if entry is None:
                raise VisionTPError(f'backend.prime returned no entry for {name}')
            record = {
                'index': index, 'role': role, 'name': name, 'mode': mode,
                'k': k, 'n': n, 'k_pad': k_pad, 'n_pad': n_pad,
                'bias_included_in_backend': bias is not None,
                'bias_added_after_float_collective': bias is None,
                'weight_slice': spec,
                'weight_shape': [n, k],
                'entry_type': type(entry).__name__,
                'entry': entry,
            }
            self._roles[(index, role)] = record
            self._prepared.append({key: value for key, value in record.items() if key != 'entry'})
            self._stats['primed_weight_modules'] += 1

    # -- instance-scoped patching -----------------------------------------
    def _install_wrapper(self, module, kind: str, index: int):
        record = {
            'module': module, 'kind': kind, 'index': index,
            'had_instance_forward': 'forward' in vars(module),
            'saved_instance_forward': vars(module).get('forward'),
            # What would have been called if this adapter had never been installed.
            'original_callable': module.forward,
        }
        if kind == 'attn':
            def forward(this, hidden_states, cu_seqlens=None, position_embeddings=None, **kwargs):
                if not self._enabled:
                    self._stats['disabled_attention_calls'] += 1
                    return record['original_callable'](
                        hidden_states, cu_seqlens=cu_seqlens,
                        position_embeddings=position_embeddings, **kwargs)
                if kwargs:
                    raise VisionTPError(f'block {record["index"]} attn got unexpected kwargs {sorted(kwargs)}')
                return self._attention_forward(
                    record['index'], record['module'], hidden_states,
                    cu_seqlens, position_embeddings)
        else:
            def forward(this, hidden_states, **kwargs):
                if not self._enabled:
                    self._stats['disabled_mlp_calls'] += 1
                    return record['original_callable'](hidden_states, **kwargs)
                if kwargs:
                    raise VisionTPError(f'block {record["index"]} mlp got unexpected kwargs {sorted(kwargs)}')
                return self._mlp_forward(record['index'], record['module'], hidden_states)
        bound = types.MethodType(forward, module)
        record['wrapper'] = bound
        module.forward = bound
        return record

    def _restore_wrapper(self, record, force: bool = False) -> None:
        module = record['module']
        current = vars(module).get('forward')
        if current is not record['wrapper']:
            if force:
                return
            raise VisionTPError(
                f'refusing to clobber block {record["index"]} {record["kind"]}.forward: '
                'the instance attribute is no longer this adapter wrapper')
        if record['had_instance_forward']:
            module.forward = record['saved_instance_forward']
        elif 'forward' in vars(module):
            del module.forward

    # -- enabled forward paths --------------------------------------------
    def _attention_forward(self, index: int, attn_module, hidden_states,
                           cu_seqlens, position_embeddings):
        torch = _torch()
        seq = self._check_hidden(torch, hidden_states, f'block{index}.attn')
        self._verify_linears(index, self._linears_of('attn', attn_module))
        if position_embeddings is None:
            raise VisionTPError('enabled vision TP attention needs (cos, sin) position_embeddings')
        if not isinstance(position_embeddings, (tuple, list)) or len(position_embeddings) != 2:
            raise VisionTPError('position_embeddings must be a (cos, sin) pair')
        cos, sin = position_embeddings
        lengths = self._segment_lengths(torch, cu_seqlens, seq)

        geometry = self._geometry
        local_heads = geometry['local_heads']
        local_channels = geometry['local_head_channels']
        head_dim = VISION_HEAD_DIM

        qkv = self._run_column(index, 'attn.qkv', hidden_states)          # Half (m, 3*144)
        expected_qkv = (seq, 3 * local_channels)
        if tuple(int(d) for d in qkv.shape) != expected_qkv:
            raise VisionTPError(f'block{index}.attn.qkv gave {tuple(qkv.shape)} != {expected_qkv}')
        q, k, v = qkv.reshape(seq, 3, local_heads, head_dim).permute(1, 0, 2, 3).unbind(0)
        q, k = self._rotary(q, k, cos, sin)
        q = q.transpose(0, 1).unsqueeze(0)
        k = k.transpose(0, 1).unsqueeze(0)
        v = v.transpose(0, 1).unsqueeze(0)

        pieces = [torch.split(tensor, lengths, dim=2) for tensor in (q, k, v)]
        outputs = []
        for query, key, value in zip(*pieces):
            expected = (1, local_heads, VISION_SEGMENT, head_dim)
            if tuple(int(d) for d in query.shape) != expected:
                raise VisionTPError(f'local attention query {tuple(query.shape)} != {expected}')
            for name, tensor in (('q', query), ('k', key), ('v', value)):
                if tensor.dtype != torch.float16:
                    raise VisionTPError(f'local attention {name} must stay Half, got {tensor.dtype}')
            out = self._attention(query, key, value)
            if not isinstance(out, torch.Tensor):
                raise VisionTPError(f'attention callable returned {type(out).__name__}')
            if tuple(int(d) for d in out.shape) != expected:
                raise VisionTPError(f'attention output {tuple(out.shape)} != {expected}')
            if out.dtype != torch.float16:
                raise VisionTPError(f'attention output must be Half, got {out.dtype}')
            outputs.append(out)
            self._stats['attention_segments'] += 1

        attn_output = outputs[0] if len(outputs) == 1 else torch.cat(outputs, dim=2)
        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.reshape(seq, local_channels).contiguous()
        partial = self._run_row(index, 'attn.proj', attn_output)          # Float (m, 1152)
        reduced = self._all_reduce(partial, 'attn.proj', index)
        bias = self._full_bias(torch, attn_module.proj.bias, VISION_DIM)
        out = (reduced + bias).to(torch.float16)
        self._check_final(torch, out, 'attn')
        self._stats['enabled_attention_calls'] += 1
        return out

    def _mlp_forward(self, index: int, mlp_module, hidden_states):
        torch = _torch()
        seq = self._check_hidden(torch, hidden_states, f'block{index}.mlp')
        self._verify_linears(index, self._linears_of('mlp', mlp_module))
        local_mlp = self._geometry['local_mlp_channels']

        column = self._run_column(index, 'mlp.fc1', hidden_states)         # Half (m, 538)
        expected = (seq, local_mlp)
        if tuple(int(d) for d in column.shape) != expected:
            raise VisionTPError(f'block{index}.fc1 gave {tuple(column.shape)} != {expected}')
        activated = mlp_module.act_fn(column)          # the ORIGINAL GELU(tanh), 538 channels
        if not isinstance(activated, torch.Tensor):
            raise VisionTPError('mlp.act_fn returned a non-tensor')
        if tuple(int(d) for d in activated.shape) != expected:
            raise VisionTPError(f'activation shape {tuple(activated.shape)} != {expected}')
        if activated.dtype != torch.float16:
            raise VisionTPError(f'activation must stay Half, got {activated.dtype}')
        self._stats['activation_half_checks'] += 1
        partial = self._run_row(index, 'mlp.fc2', activated)               # Float (m, 1152)
        reduced = self._all_reduce(partial, 'mlp.fc2', index)
        bias = self._full_bias(torch, mlp_module.linear_fc2.bias, VISION_DIM)
        out = (reduced + bias).to(torch.float16)
        self._check_final(torch, out, 'mlp')
        self._stats['enabled_mlp_calls'] += 1
        return out

    def _check_final(self, torch, out, where: str) -> None:
        shape = tuple(int(d) for d in out.shape)
        if shape != (self._m, VISION_DIM):
            raise VisionTPError(f'{where} output {shape} != {(self._m, VISION_DIM)}')
        if out.dtype != torch.float16:
            raise VisionTPError(f'{where} output must be a single Half cast, got {out.dtype}')
        self._stats['half_outputs'] += 1

    # -- public API -------------------------------------------------------
    def prime(self, vision):
        """Slice each block's four local geometries and install the instance wrappers.

        Half slices are handed straight to ``backend.prime()``; the adapter keeps no
        tensor reference of its own, and a mid-way failure rolls every patch back.
        """
        if self._primed:
            raise VisionTPError('prime() was already called; construct a new adapter instead')
        torch = _torch()
        tower = self._resolve_vision(vision)
        blocks = getattr(tower, 'blocks', None)
        if blocks is None or len(blocks) != VISION_BLOCKS:
            raise VisionTPError(f'vision tower must expose {VISION_BLOCKS} blocks, got {blocks!r}')
        patched = []
        try:
            for index, block in enumerate(blocks):
                self._prime_block(index, block)
                patched.append(self._install_wrapper(block.attn, 'attn', index))
                patched.append(self._install_wrapper(block.mlp, 'mlp', index))
            if self._stats['primed_weight_modules'] != self._segments['primed_weight_modules']:
                raise VisionTPError(
                    f'primed {self._stats["primed_weight_modules"]} weight modules, '
                    f'expected {self._segments["primed_weight_modules"]} (27 blocks x 4 linears)')
        except BaseException:
            # Fail closed: never leave a partially installed instance patch behind.
            for record in reversed(patched):
                self._restore_wrapper(record, force=True)
            raise
        self._patched = patched
        self._installed = True
        self._primed = True
        self._enabled = False
        return self.snapshot()

    def set_enabled(self, value: bool):
        if not isinstance(value, bool):
            raise VisionTPError(f'set_enabled expects a bool, got {type(value).__name__}')
        if not self._primed:
            raise VisionTPError('prime() must run before set_enabled()')
        self._enabled = value
        return value

    def snapshot(self) -> dict:
        data = {
            'round': 107, 'source_SHA256': _source_sha256(),
            'primed': self._primed, 'enabled': self._enabled, 'installed': self._installed,
            'm': self._m, 'rank_in_group': self._rank, 'world_size': self._world,
            'geometry': copy.deepcopy(self._geometry),
            'prepared': copy.deepcopy(self._prepared),
            'patched': [{'index': record['index'], 'kind': record['kind'],
                         'had_instance_forward': record['had_instance_forward']}
                        for record in self._patched],
            'stats': dict(self._stats),
            'expected_counts_per_forward': dict(self._segments),
            'attention_token': self._attention_token, 'rotary_token': self._rotary_token,
            'attention_scale_from_original_head_dim': SCALE,
            'backend_snapshot': _safe_copy(self._backend.snapshot()),
            'no_Float_weight_shadow': True, 'no_cached_activations': True,
            'no_BF16_gold_cast': True, 'no_fake_final_norm': True, 'no_clamp': True,
            'original_Half_weights_and_biases_untouched': True,
            'numerical_tolerance': None,
            'full_encoder_precision_accepted': False, 'whole_goal_complete': False,
            'eight_times_speedup_claimed': False,
        }
        return copy.deepcopy(data)

    def uninstall(self) -> dict:
        result = self.snapshot()
        for record in reversed(self._patched):
            self._restore_wrapper(record)
        self._patched = []
        self._roles = {}
        self._signatures = {}
        self._prepared = []
        self._enabled = False
        self._installed = False
        self._primed = False
        result['uninstalled'] = True
        result['enabled'] = False
        result['installed'] = False
        result['primed'] = False
        return copy.deepcopy(result)