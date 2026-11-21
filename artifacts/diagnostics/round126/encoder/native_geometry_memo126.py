"""R126 instance-local memoization of TWO original R124 vision geometry methods.

SOURCE ONLY: no Torch import, model construction, compilation or device action.
The lead supplies the live encoder, original-source path and runtime-state callback.
Misses call R124 itself. Hits return fresh clones of private geometry masters; no
pixel, hidden state, Q/K/V, DeepStack, text or sin/cos result is cached/recomputed.
The original vision.forward still computes cos/sin and all 27 blocks every call.

One grid value read at vision entry replaces pointer-based keys; it is NOT a
per-layer read. Lifecycle hooks conservatively discard device caches on every
load/offload/_apply. A fully staged service may therefore see NO hits; report that
rather than extrapolating resident speed. Loader .data and direct stager rebinding
outside these hooks require explicit invalidate() by the owning lead integration.
"""
from __future__ import annotations

import ast
from collections import OrderedDict
import hashlib
import inspect
import json
from pathlib import Path
import threading
from types import MethodType

BASELINE_SHA256 = '0959895b778fc0900d785250b71279055d3940d975f3523909856072960f5026'
CLASS_NAME = 'MiniMaxH3Qwen3VLVisionModel'
GEOMETRY_METHODS = ('fast_pos_embed_interpolate', 'rot_pos_emb')
RUNTIME_KEYS = ('default_dtype', 'default_device', 'autocast_enabled',
                'autocast_dtype', 'matmul_allow_hf32', 'conv_allow_hf32', 'stream_id')
DEFAULT_MAX_BYTES = 32 * 1024 * 1024
DEFAULT_MAX_ENTRIES = 4


class GeometryMemoError(RuntimeError):
    """Refuse stale geometry, unsupported precision or unsafe ownership."""


def verify_source_file(path):
    path = Path(path).resolve(strict=True)
    raw = path.read_bytes()
    digest = hashlib.sha256(raw).hexdigest()
    if digest != BASELINE_SHA256:
        raise GeometryMemoError('Requires EXACT R124 executed encoder, not R79/BF16/new math')
    tree = ast.parse(raw.decode())
    cls = next((n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == CLASS_NAME), None)
    if cls is None:
        raise GeometryMemoError('Missing original vision class')
    names = set(GEOMETRY_METHODS) | {'forward'}
    nodes = {n.name: n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name in names}
    if set(nodes) != names:
        raise GeometryMemoError('Missing original geometry/vision methods')
    return {'path': str(path), 'sha256': digest,
            'method_sha256': {name: hashlib.sha256(ast.get_source_segment(raw.decode(), node).encode()).hexdigest()
                              for name, node in nodes.items()}}


def _verify_bound_source(vision, audit):
    for name in (*GEOMETRY_METHODS, 'forward'):
        bound = getattr(vision, name)
        fn = getattr(bound, '__func__', None)
        if getattr(bound, '__self__', None) is not vision or fn is None:
            raise GeometryMemoError('Original method must be instance-bound: ' + name)
        source = inspect.getsourcefile(fn)
        if source is None or str(Path(source).resolve(strict=True)) != audit['path']:
            raise GeometryMemoError('Bound method is not the pinned executed source: ' + name)
        segment = inspect.getsource(fn)
        # inspect returns class indentation; the AST/source audit above pins the
        # entire file, while identity to its class method rejects a monkeypatch.
        if not segment or getattr(type(vision), name) is not fn:
            raise GeometryMemoError('Unknown class/instance override: ' + name)


def _version(tensor):
    try:
        return int(tensor._version)
    except RuntimeError as error:
        if 'Inference tensors do not track version counter' not in str(error):
            raise
        return None


def tensor_signature(tensor):
    return (int(tensor.untyped_storage().data_ptr()), int(tensor.storage_offset()),
            tuple(int(v) for v in tensor.shape), tuple(int(v) for v in tensor.stride()),
            str(tensor.dtype), str(tensor.device), _version(tensor))


def _layout(tensor):
    return (tuple(int(v) for v in tensor.shape), tuple(int(v) for v in tensor.stride()),
            str(tensor.dtype), str(tensor.device))


def _frozen(value):
    """Only immutable scalar metadata can enter a key; never tensor contents."""
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        if not (-float('inf') < value < float('inf')):
            raise GeometryMemoError('Nonfinite metadata')
        return value
    if isinstance(value, (list, tuple)):
        return tuple(_frozen(v) for v in value)
    if isinstance(value, dict):
        return tuple(sorted((str(k), _frozen(v)) for k, v in value.items()))
    raise GeometryMemoError('Runtime/config key must contain scalar metadata only')


def _private_clone(tensor):
    clone = tensor.detach().clone()
    if _layout(clone) != _layout(tensor):
        raise GeometryMemoError('Clone changed native shape/stride/dtype/device')
    if clone.untyped_storage().data_ptr() == tensor.untyped_storage().data_ptr():
        raise GeometryMemoError('Cache/return clone aliases original storage')
    return clone


class NativeFP16GeometryMemo126:
    """Disabled-by-default cache for one live encoder; no module-global patches.

    runtime_state() must provide scalar RUNTIME_KEYS, including current stream ID.
    Default dtype/device and autocast matter because the original interpolation
    uses torch.linspace without an explicit dtype. Stream IDs prevent a cache
    producer from being cloned on an unrelated stream without a dependency.

    Private masters are NEVER returned via snapshot or forward. Each hit clones
    afresh; first-call callers receive the genuine original output. No .contiguous
    or format conversion is inserted. Physical-format equality is a lead NPU gate.
    """
    def __init__(self, encoder, source_path, runtime_state, *,
                 max_bytes=DEFAULT_MAX_BYTES, max_entries=DEFAULT_MAX_ENTRIES):
        if not callable(runtime_state):
            raise GeometryMemoError('An explicit lead-owned runtime-state callback is required')
        for value in (max_bytes, max_entries):
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise GeometryMemoError('Positive integer cache budgets required')
        self.encoder = encoder
        self.vision = encoder.vision
        self.source = verify_source_file(source_path)
        _verify_bound_source(self.vision, self.source)
        self.runtime_state = runtime_state
        self.max_bytes, self.max_entries = max_bytes, max_entries
        self._lock = threading.RLock()
        self._entries = OrderedDict()
        self._active = None
        self._epoch = 0
        self._last_state = None
        self._enabled = False
        self._bindings = []
        self._originals = {name: getattr(self.vision, name) for name in (*GEOMETRY_METHODS, 'forward')}
        self.stats = {'vision_calls': 0, 'grid_value_reads': 0, 'misses': 0,
                      'hits': 0, 'private_clones': 0, 'return_clones': 0,
                      'budget_bypasses': 0, 'evictions': 0, 'invalidations': 0,
                      'outside_scope_calls': 0, 'original_geometry_calls': 0}
        self.invalidation_reasons = []
        targets = [(self.vision, name) for name in (*GEOMETRY_METHODS, 'forward', '_apply')]
        targets += [(encoder, name) for name in ('load_weights', 'load_to_device', 'offload_to_cpu', '_apply')]
        for owner, name in targets:
            if name in vars(owner) or not callable(getattr(owner, name, None)):
                raise GeometryMemoError('Refusing existing/unknown instance override: ' + name)
        # Validate architecture/FP16 ownership before touching a live instance.
        self._state()
        try:
            for name in GEOMETRY_METHODS:
                self._bind(self.vision, name, self._geometry_wrapper(name))
            self._bind(self.vision, 'forward', self._forward_wrapper())
            for owner, name in targets:
                if name not in GEOMETRY_METHODS and name != 'forward':
                    self._bind(owner, name, self._lifecycle_wrapper(owner, name))
        except BaseException:
            self.uninstall()
            raise

    def _bind(self, owner, name, function):
        original = getattr(owner, name)
        wrapped = MethodType(function, owner)
        setattr(owner, name, wrapped)
        self._bindings.append((owner, name, original, wrapped))

    def _geometry_wrapper(self, name):
        def wrapped(owner, grid_thw):
            return self._dispatch(name, owner, grid_thw)
        return wrapped

    def _forward_wrapper(self):
        original = self._originals['forward']
        def wrapped(owner, *args, **kwargs):
            if not self._enabled:
                return original(*args, **kwargs)
            grid = kwargs.get('grid_thw', args[1] if len(args) > 1 else None)
            with self._lock:
                if self._active is not None:
                    raise GeometryMemoError('Reentrant vision calls are unsupported')
                self._check_geometry_ownership()
                state = self._state()
                if self._last_state is not None and state != self._last_state:
                    self.invalidate('parameter/config/runtime drift')
                    state = self._state()  # invalidate advanced the lifecycle epoch.
                self._last_state = state
                key, signature = self._grid_key(grid, state)
                self._active = {'grid': grid, 'signature': signature, 'state': state, 'key': key}
                self.stats['vision_calls'] += 1
                try:
                    return original(*args, **kwargs)
                finally:
                    self._active = None
        return wrapped

    def _lifecycle_wrapper(self, owner, name):
        original = getattr(owner, name)
        def wrapped(current, *args, **kwargs):
            # This intentionally also clears a no-op load/.to call: an arbitrary
            # _apply callable or .data loader need not increment tensor versions.
            self.invalidate(type(owner).__name__ + '.' + name)
            return original(*args, **kwargs)
        return wrapped

    def _state(self):
        vision = self.vision
        weight = vision.pos_embed.weight
        sig = tensor_signature(weight)
        if str(weight.dtype) != 'torch.float16' or weight.device.type not in ('npu', 'privateuseone'):
            raise GeometryMemoError('Only the original live NPU FP16 position parameters are supported')
        if tuple(weight.shape) != (2304, 1152):
            raise GeometryMemoError('Unexpected original position-embedding geometry')
        rotary = vision.rotary_pos_emb
        inv = rotary._inv_freq
        if str(inv.dtype) != 'torch.float32' or inv.device.type != 'cpu' or tuple(inv.shape) != (18,):
            raise GeometryMemoError('Require original plain CPU Float32 inverse-frequency metadata')
        # 72 CPU bytes; no NPU transfer, no trig and no learned tensor is hashed.
        inv_sha = hashlib.sha256(inv.detach().contiguous().numpy().tobytes()).hexdigest()
        config = vision.config
        attrs = {name: getattr(config, name) for name in ('hidden_size', 'num_heads', 'intermediate_size',
                                                         'depth', 'num_position_embeddings', 'spatial_merge_size')}
        if attrs != {'hidden_size': 1152, 'num_heads': 16, 'intermediate_size': 4304,
                     'depth': 27, 'num_position_embeddings': 2304, 'spatial_merge_size': 2}:
            raise GeometryMemoError('Not the original R124 vision architecture')
        state = dict(self.runtime_state())
        if set(state) != set(RUNTIME_KEYS):
            raise GeometryMemoError('Incomplete/unknown runtime precision/stream state')
        for name in ('autocast_enabled', 'matmul_allow_hf32', 'conv_allow_hf32'):
            if not isinstance(state[name], bool):
                raise GeometryMemoError('Runtime precision flags must be bool')
        return (self.source['sha256'], sig, tensor_signature(inv), inv_sha,
                _frozen(attrs), int(vision.num_grid_per_side), int(vision.spatial_merge_size),
                int(vision.patch_size), int(rotary.dim), float(rotary.theta), _frozen(state), self._epoch)

    def _grid_key(self, grid, state):
        if grid is None or str(grid.dtype) != 'torch.int64' or grid.device.type not in ('npu', 'privateuseone'):
            raise GeometryMemoError('Expected original NPU int64 grid')
        if len(grid.shape) != 2 or grid.shape[1] != 3 or grid.shape[0] <= 0:
            raise GeometryMemoError('Expected nonempty [images,3] grid')
        if str(grid.device) != str(self.vision.pos_embed.weight.device):
            raise GeometryMemoError('Grid/device parameter ownership mismatch')
        # Exactly ONE value read per complete vision invocation, never per block.
        values = tuple(tuple(v for v in row) for row in grid.detach().tolist())
        self.stats['grid_value_reads'] += 1
        if any(len(row) != 3 or any(isinstance(v, bool) or not isinstance(v, int) or v <= 0 for v in row) for row in values):
            raise GeometryMemoError('Positive integer geometry values required')
        merge = int(self.vision.spatial_merge_size)
        if any(h % merge or w % merge for _, h, w in values):
            raise GeometryMemoError('Grid does not satisfy the native spatial merge')
        length = sum(t * h * w for t, h, w in values)
        signature = tensor_signature(grid)
        return (values, length, str(grid.dtype), str(grid.device), tuple(grid.shape), state), signature

    def _dispatch(self, name, owner, grid):
        with self._lock:
            if not self._enabled or self._active is None:
                self.stats['outside_scope_calls'] += 1
                self.stats['original_geometry_calls'] += 1
                return self._originals[name](grid)
            self._check_geometry_ownership()
            active = self._active
            if owner is not self.vision or grid is not active['grid'] or tensor_signature(grid) != active['signature']:
                raise GeometryMemoError('Grid changed inside a vision call; mutation is unsupported')
            if self._state() != active['state']:
                raise GeometryMemoError('Geometry parameters/runtime changed inside a vision call')
            key = (name, active['key'])
            if key in self._entries:
                entry = self._entries.pop(key)
                if tensor_signature(entry['master']) != entry['signature']:
                    self._entries.clear()
                    raise GeometryMemoError('Private geometry master mutated')
                self._entries[key] = entry
                result = _private_clone(entry['master'])
                self.stats['hits'] += 1
                self.stats['return_clones'] += 1
                return result
            self.stats['misses'] += 1
            self.stats['original_geometry_calls'] += 1
            result = self._originals[name](grid)  # EXACT original R124 computation.
            tokens = active['key'][1]
            width = 1152 if name == 'fast_pos_embed_interpolate' else 36
            dtype = 'torch.float16' if name == 'fast_pos_embed_interpolate' else 'torch.float32'
            if tuple(result.shape) != (tokens, width) or str(result.dtype) != dtype or result.device != grid.device:
                raise GeometryMemoError('Unexpected original geometry output contract')
            if self._state() != active['state'] or tensor_signature(grid) != active['signature']:
                raise GeometryMemoError('Original geometry function mutated its inputs/state')
            byte_count = int(result.numel()) * int(result.element_size())
            if byte_count > self.max_bytes:
                self.stats['budget_bypasses'] += 1
                return result
            while self._entries and (len(self._entries) >= self.max_entries or self._bytes() + byte_count > self.max_bytes):
                self._entries.popitem(last=False)
                self.stats['evictions'] += 1
            master = _private_clone(result)
            self._entries[key] = {'master': master, 'signature': tensor_signature(master),
                                  'bytes': byte_count, 'method': name, 'grid_values': active['key'][0]}
            self.stats['private_clones'] += 1
            return result  # The first caller gets the genuine original output.

    def _bytes(self):
        return sum(entry['bytes'] for entry in self._entries.values())

    def _check_geometry_ownership(self):
        for owner, name, original, wrapper in self._bindings:
            if owner is self.vision and name in GEOMETRY_METHODS:
                current = getattr(owner, name)
                if getattr(current, '__self__', None) is not owner or getattr(current, '__func__', None) is not wrapper.__func__:
                    raise GeometryMemoError('Geometry method replaced by another owner: ' + name)

    def set_enabled(self, enabled):
        with self._lock:
            if not isinstance(enabled, bool) or self._active is not None or not self._bindings:
                raise GeometryMemoError('Toggle only an installed cache with bool outside active vision calls')
            self._enabled = enabled
        return self.snapshot()

    def invalidate(self, reason='explicit parameter/stager update'):
        with self._lock:
            if self._active is not None:
                raise GeometryMemoError('Do not reload/offload or mutate parameters during vision execution')
            self._entries.clear()
            self._epoch += 1
            self._last_state = None
            self.stats['invalidations'] += 1
            self.invalidation_reasons.append(str(reason))

    def snapshot(self):
        # No tensor/master reference or large CPU hash escapes here.
        return {'source': dict(self.source), 'enabled': self._enabled, 'epoch': self._epoch,
                'entries': len(self._entries), 'cached_geometry_bytes': self._bytes(),
                'max_bytes': self.max_bytes, 'max_entries': self.max_entries,
                'stats': dict(self.stats), 'invalidation_reasons': list(self.invalidation_reasons),
                'cached_methods': [e['method'] for e in self._entries.values()],
                'ownership': 'Private geometry masters; misses return original output; every hit returns a fresh nonaliasing clone.',
                'no_hidden_features_KV_or_sin_cos_cache': True, 'no_arithmetic_or_dtype_replacement': True,
                'numerical_acceptance': 'Requires lead actual full same-input raw-bit + zero numeric difference gate.',
                'NPU_executed_by_author': False, 'production_deployed': False, 'whole_goal_complete': False}

    def uninstall(self):
        with self._lock:
            if self._active is not None:
                raise GeometryMemoError('Cannot uninstall during execution')
            # Validate ALL owners before changing ANY attribute; no partial restore.
            for owner, name, original, wrapper in self._bindings:
                current = getattr(owner, name)
                if getattr(current, '__self__', None) is not owner or getattr(current, '__func__', None) is not wrapper.__func__:
                    raise GeometryMemoError('Uninstall refuses a third-party wrapper: ' + name)
            for owner, name, original, wrapper in reversed(self._bindings):
                delattr(owner, name)
            self._bindings.clear()
            self._enabled = False
            self._entries.clear()
            self._last_state = None


def install(encoder, source_path, runtime_state, **budgets):
    """Lead-only factory. Installs disabled; first enabled live call primes misses."""
    return NativeFP16GeometryMemo126(encoder, source_path, runtime_state, **budgets)
