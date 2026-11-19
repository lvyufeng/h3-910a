#!/usr/bin/env python3
"""Focused standard-library tests for artifacts/diagnostics/round107/tp_vision.

NO real torch, torch_npu, CANN, NPU, GPU, model or checkpoint is imported or executed.
The tensor/backend/group objects below are *record-only* fakes: they carry shapes,
dtypes and a call log so the adapter's index math, dtype contract, counters and
instance-patch ownership can be asserted without hardware.  These are structural
tests, not an accuracy, speed or compiler result.
"""
import ast
import importlib.util
import json
import sys
import types
import unittest
from pathlib import Path

HERE = Path(__file__).resolve().parent
SOURCE = HERE / 'vision_tp_adapter107.py'
_REAL_TORCH = sys.modules.get('torch')
_MISSING = object()

# ---------------------------------------------------------------------------
# Record-only fake torch
# ---------------------------------------------------------------------------
_LOG = []
_ORIGINAL = []
_ROTARY = []
_UIDS = [0]


class FakeDType:
    def __init__(self, name):
        self.name = name

    def __repr__(self):
        return 'torch.' + self.name


DT_HALF = FakeDType('float16')
DT_FLOAT = FakeDType('float32')


class FakeDevice:
    def __init__(self, kind='npu', index=0):
        self.type = kind
        self.index = index

    def __repr__(self):
        return f'{self.type}:{self.index}'


class FakeTensor:
    """Shape/dtype/device bookkeeping plus an operation log.  Holds no data."""

    def __init__(self, shape, dtype, device, values=None, offset=0):
        self.shape = tuple(int(d) for d in shape)
        self.dtype = dtype
        self.device = device
        self._values = values
        self._offset = offset
        self._uid = _UIDS[0]
        _UIDS[0] += 1

    # -- parameter identity API ------------------------------------------
    def dim(self):
        return len(self.shape)

    def numel(self):
        total = 1
        for size in self.shape:
            total *= size
        return total

    def data_ptr(self):
        return 100000 + self._uid

    def storage_offset(self):
        return self._offset

    def clone(self, memory_format=None):
        _LOG.append(('clone', self._uid, self.shape))
        return FakeTensor(self.shape, self.dtype, self.device)

    def detach(self):
        _LOG.append(('detach', self._uid, self.shape, repr(self.dtype)))
        twin = FakeTensor(self.shape, self.dtype, self.device)
        twin._uid = self._uid     # detach shares storage, like torch
        return twin

    def tolist(self):
        if self._values is None:
            raise AssertionError('tolist() on a tensor without recorded values')
        return list(self._values)

    # -- operations used by the adapter ----------------------------------
    def contiguous(self):
        _LOG.append(('contiguous', self._uid, self.shape))
        return self

    def reshape(self, *shape):
        if len(shape) == 1 and isinstance(shape[0], (tuple, list)):
            shape = tuple(shape[0])
        shape = tuple(int(d) for d in shape)
        if -1 in shape:
            known = 1
            for size in shape:
                if size != -1:
                    known *= size
            shape = tuple(self.numel() // known if size == -1 else size for size in shape)
        if _product(shape) != self.numel():
            raise AssertionError(f'reshape {self.shape} -> {shape} changes numel')
        _LOG.append(('reshape', self._uid, self.shape, shape))
        return FakeTensor(shape, self.dtype, self.device)

    def permute(self, *dims):
        if len(dims) == 1 and isinstance(dims[0], (tuple, list)):
            dims = tuple(dims[0])
        dims = tuple(int(d) for d in dims)
        if sorted(dims) != list(range(self.dim())):
            raise AssertionError(f'permute {dims} is not a permutation of {self.dim()} axes')
        shape = tuple(self.shape[d] for d in dims)
        _LOG.append(('permute', self._uid, self.shape, dims))
        return FakeTensor(shape, self.dtype, self.device)

    def unbind(self, dim):
        size = self.shape[dim]
        shape = self.shape[:dim] + self.shape[dim + 1:]
        _LOG.append(('unbind', self._uid, self.shape, dim, size))
        return tuple(FakeTensor(shape, self.dtype, self.device) for _ in range(size))

    def transpose(self, first, second):
        shape = list(self.shape)
        shape[first], shape[second] = shape[second], shape[first]
        _LOG.append(('transpose', self._uid, self.shape, (first, second)))
        return FakeTensor(shape, self.dtype, self.device)

    def unsqueeze(self, dim):
        shape = list(self.shape)
        shape.insert(dim, 1)
        _LOG.append(('unsqueeze', self._uid, tuple(shape), dim))
        return FakeTensor(shape, self.dtype, self.device)

    def to(self, dtype):
        _LOG.append(('to', self._uid, repr(self.dtype), repr(dtype)))
        return FakeTensor(self.shape, dtype, self.device)

    def float(self):
        _LOG.append(('float', self._uid, repr(self.dtype)))
        return FakeTensor(self.shape, DT_FLOAT, self.device)

    def __getitem__(self, index):
        if not isinstance(index, tuple):
            index = (index,)
        shape = list(self.shape)
        offset = self._offset
        for axis, part in enumerate(index):
            if isinstance(part, slice):
                start, stop, step = part.indices(shape[axis])
                if step != 1:
                    raise AssertionError('only unit-step slices are modelled')
                shape[axis] = stop - start
                if axis == 0:
                    offset = offset + start      # slicing row 0 shifts the view
                _LOG.append(('slice', self._uid, tuple(self.shape), axis, start, stop))
        return FakeTensor(shape, self.dtype, self.device, offset=offset)

    def __add__(self, other):
        left = list(self.shape)
        right = list(other.shape)
        if len(right) > len(left):
            left, right = right, left
        for offset in range(1, len(right) + 1):
            a, b = left[-offset], right[-offset]
            if a != b and a != 1 and b != 1:
                raise AssertionError(f'cannot broadcast {tuple(left)} with {tuple(right)}')
            left[-offset] = max(a, b)
        dtype = DT_FLOAT if DT_FLOAT in (self.dtype, other.dtype) else self.dtype
        _LOG.append(('add', self._uid, self.shape, other._uid, other.shape, tuple(left)))
        return FakeTensor(left, dtype, self.device)


def _product(values):
    total = 1
    for item in values:
        total *= item
    return total


class FakeTorch(types.ModuleType):
    def __init__(self):
        super().__init__('torch')
        self.float16 = DT_HALF
        self.float32 = DT_FLOAT
        self.Tensor = FakeTensor
        self.contiguous_format = 'contiguous_format'

    def split(self, tensor, sizes, dim):
        _LOG.append(('split', tensor._uid, tensor.shape, tuple(sizes), dim))
        pieces = []
        for size in sizes:
            shape = list(tensor.shape)
            shape[dim] = size
            pieces.append(FakeTensor(shape, tensor.dtype, tensor.device))
        return tuple(pieces)

    def cat(self, tensors, dim=0):
        shape = list(tensors[0].shape)
        shape[dim] = sum(item.shape[dim] for item in tensors)
        _LOG.append(('cat', [item._uid for item in tensors], tuple(shape), dim))
        return FakeTensor(shape, tensors[0].dtype, tensors[0].device)


# ---------------------------------------------------------------------------
# Record-only fake model / backend / group / attention
# ---------------------------------------------------------------------------
def param(shape, device):
    return FakeTensor(shape, DT_HALF, device)


class FakeLinear:
    def __init__(self, out_features, in_features, device):
        self.weight = param((out_features, in_features), device)
        self.bias = param((out_features,), device)


class FakeGelu:
    approximate = 'tanh'

    def __call__(self, value):
        _LOG.append(('gelu', value._uid, value.shape, repr(value.dtype)))
        return FakeTensor(value.shape, value.dtype, value.device)


class FakeAttn:
    def __init__(self, device):
        self.dim = 1152
        self.num_heads = 16
        self.head_dim = 72
        self.qkv = FakeLinear(3456, 1152, device)
        self.proj = FakeLinear(1152, 1152, device)

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)

    def forward(self, hidden_states, cu_seqlens=None, position_embeddings=None):
        _ORIGINAL.append(('attn', hidden_states.shape))
        return FakeTensor(hidden_states.shape, hidden_states.dtype, hidden_states.device)


class FakeMLP:
    def __init__(self, device):
        self.linear_fc1 = FakeLinear(4304, 1152, device)
        self.linear_fc2 = FakeLinear(1152, 4304, device)
        self.act_fn = FakeGelu()

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)

    def forward(self, hidden_states):
        _ORIGINAL.append(('mlp', hidden_states.shape))
        return FakeTensor(hidden_states.shape, hidden_states.dtype, hidden_states.device)


class FakeBlock:
    def __init__(self, device):
        self.attn = FakeAttn(device)
        self.mlp = FakeMLP(device)


def fake_vision(device, blocks=27):
    tower = types.SimpleNamespace()
    tower.blocks = [FakeBlock(device) for _ in range(blocks)]
    return tower


def _rotary(q, k, cos, sin):
    _ROTARY.append((q.shape, k.shape, cos.shape, sin.shape))
    return q, k


def fake_candidate():
    module = types.SimpleNamespace()
    module._apply_rotary_pos_emb_vision = _rotary
    return module


class FakeGroup:
    def __init__(self, rank=0, world=8, reduce_dtype=None):
        self.rank_in_group = rank
        self.world_size = world
        self._reduce_dtype = reduce_dtype
        self.reduces = []

    def all_reduce(self, tensor):
        self.reduces.append((tensor.shape, repr(tensor.dtype)))
        return FakeTensor(tensor.shape, self._reduce_dtype or DT_FLOAT, tensor.device)


class FakeBackend:
    def __init__(self, row_dtype=None, column_dtype=None):
        self.row_dtype = row_dtype or DT_FLOAT
        self.column_dtype = column_dtype or DT_HALF
        self.primed = []
        self.runs = []

    def prime(self, name, weight, bias, mode, m=8064):
        _LOG.append(('prime', name, weight.shape,
                     None if bias is None else bias.shape, mode, m))
        entry = {'name': name, 'mode': mode, 'n': weight.shape[0], 'k': weight.shape[1]}
        self.primed.append({'name': name, 'weight': weight.shape,
                            'bias': None if bias is None else bias.shape,
                            'mode': mode, 'entry': entry})
        return entry

    def run(self, entry, activation):
        self.runs.append({'name': entry['name'], 'mode': entry['mode'],
                          'shape': activation.shape, 'dtype': repr(activation.dtype)})
        dtype = self.column_dtype if entry['mode'] == 'column' else self.row_dtype
        return FakeTensor((activation.shape[0], entry['n']), dtype, activation.device)

    def snapshot(self):
        return {'primed': len(self.primed), 'runs': len(self.runs)}


class FakeAttention:
    def __init__(self, dtype=None, shape_shift=0):
        self.dtype = dtype or DT_HALF
        self.shape_shift = shape_shift
        self.calls = []

    def __call__(self, q, k, v):
        self.calls.append({'q': q.shape, 'k': k.shape, 'v': v.shape,
                           'dtype': repr(q.dtype)})
        shape = list(q.shape)
        shape[-1] += self.shape_shift
        return FakeTensor(shape, self.dtype, q.device)


def hidden(device, m=8064):
    return FakeTensor((m, 1152), DT_HALF, device)


def position(device, m=8064):
    return (param((m, 72), device), param((m, 72), device))


def cu_seqlens(values):
    return FakeTensor((len(values),), None, FakeDevice('cpu'), values=values)


# ---------------------------------------------------------------------------
# Harness
# ---------------------------------------------------------------------------
_SPEC = importlib.util.spec_from_file_location('vision_tp_adapter107_under_test', SOURCE)
mod = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(mod)


class Harness:
    def __init__(self, rank=0, world=8, blocks=27):
        sys.modules['torch'] = FakeTorch()
        _LOG.clear()
        _ORIGINAL.clear()
        _ROTARY.clear()
        self.device = FakeDevice('npu', rank)
        self.vision = fake_vision(self.device, blocks)
        self.candidate = fake_candidate()
        self.group = FakeGroup(rank, world)
        self.backend = FakeBackend()
        self.attention = FakeAttention()

    def adapter(self, **kwargs):
        return mod.VisionTPAdapter107(self.candidate, self.group, self.backend,
                                      kwargs.pop('attention', self.attention),
                                      kwargs.pop('m', 8064))

    @staticmethod
    def close():
        if _REAL_TORCH is None:
            sys.modules.pop('torch', None)
        else:
            sys.modules['torch'] = _REAL_TORCH


class FakeTorchCase(unittest.TestCase):
    def setUp(self):
        self.harness = Harness()

    def tearDown(self):
        Harness.close()

    def restart(self, rank=0, world=8, blocks=27):
        self.harness = Harness(rank, world, blocks)
        return self.harness

    def primed(self, **kwargs):
        adapter = self.harness.adapter(**kwargs)
        report = adapter.prime(self.harness.vision)
        return adapter, report


def slices_for(shape, axis=None):
    out = {}
    for op in _LOG:
        if op[0] != 'slice':
            continue
        _, uid, parent, ax, start, stop = op
        if parent == shape and (axis is None or ax == axis):
            out.setdefault(uid, []).append((ax, start, stop))
    return out


# ---------------------------------------------------------------------------
# 1. Pure geometry
# ---------------------------------------------------------------------------
class PureGeometryTests(unittest.TestCase):
    def test_channel_spans_tile_both_widths_exactly(self):
        for total in (1152, 4304):
            spans = mod.local_channel_spans(8, total)
            self.assertEqual(len(spans), 8)
            self.assertEqual(spans[0][0], 0)
            self.assertEqual(spans[-1][1], total)
            covered = []
            for low, high in spans:
                covered.extend(range(low, high))
            self.assertEqual(covered, list(range(total)))
        self.assertEqual(mod.local_channel_spans(8, 4304)[3], (1614, 2152))

    def test_local_plan_matches_the_task_geometry(self):
        for rank in range(8):
            plan = mod.plan_local_geometry(rank)
            self.assertEqual(plan['local_heads'], 2)
            self.assertEqual(plan['local_head_channels'], 144)
            self.assertEqual(plan['local_mlp_channels'], 538)
            self.assertEqual(plan['head_channel_span'], (rank * 144, (rank + 1) * 144))
            self.assertEqual(plan['mlp_channel_span'], (rank * 538, (rank + 1) * 538))
            self.assertEqual(plan['qkv_row_slices'],
                             ((rank * 144, (rank + 1) * 144),
                              (1152 + rank * 144, 1152 + (rank + 1) * 144),
                              (2304 + rank * 144, 2304 + (rank + 1) * 144)))
            self.assertEqual(plan['num_segments'], 2)
            self.assertEqual(plan['segment_tokens'], 4032)

    def test_four_backend_geometries_and_128_padding(self):
        table = mod.padded_geometries()
        self.assertEqual(table['attn.qkv'],
                         {'mode': 'column', 'k': 1152, 'n': 432,
                          'k_pad': 1152, 'n_pad': 512})
        self.assertEqual(table['attn.proj'],
                         {'mode': 'row', 'k': 144, 'n': 1152,
                          'k_pad': 256, 'n_pad': 1152})
        self.assertEqual(table['mlp.fc1'],
                         {'mode': 'column', 'k': 1152, 'n': 538,
                          'k_pad': 1152, 'n_pad': 640})
        self.assertEqual(table['mlp.fc2'],
                         {'mode': 'row', 'k': 538, 'n': 1152,
                          'k_pad': 640, 'n_pad': 1152})

    def test_expected_counts_match_54_collectives_per_forward(self):
        self.assertEqual(mod.expected_counts_per_forward(),
                         {'attention_calls': 27, 'mlp_calls': 27,
                          'attention_segments': 54, 'all_reduce_calls': 54,
                          'column_runs': 54, 'row_runs': 54,
                          'primed_weight_modules': 108})

    def test_pure_helpers_fail_closed(self):
        for value in (0, -1, True, '128', 128.0):
            with self.assertRaises(mod.VisionTPError):
                mod.pad_to_align(value)
        for args in ((7, 1152), (8, 1151), (0, 1152)):
            with self.assertRaises(mod.VisionTPError):
                mod.local_channel_spans(*args)
        for world in (3, 4, '8', True, 8.0):
            with self.assertRaises(mod.VisionTPError):
                mod.plan_local_geometry(0, world)
        for rank in (-1, 8, True, '0', 1.0):
            with self.assertRaises(mod.VisionTPError):
                mod.plan_local_geometry(rank)
        with self.assertRaises(mod.VisionTPError):
            mod.expected_counts_per_forward(0)
        self.assertEqual(mod.SCALE, 72 ** -0.5)


# ---------------------------------------------------------------------------
# 2. Priming: slice index math and the 108 backend entries
# ---------------------------------------------------------------------------
class PrimingTests(FakeTorchCase):
    def test_rank0_qkv_slices_three_segments_and_concatenates(self):
        self.primed()
        found = slices_for((3456, 1152), axis=0)
        self.assertEqual(len(found), 27)
        expected = [(0, 0, 144), (0, 1152, 1296), (0, 2304, 2448)]
        for uid, slices in found.items():
            self.assertEqual(slices, expected, f'qkv rows for tensor {uid}')
        cats = [op for op in _LOG if op[0] == 'cat' and op[2] == (432, 1152)]
        self.assertEqual(len(cats), 27)
        self.assertTrue(all(op[3] == 0 for op in cats))

    def test_rank5_local_slices_are_offset_correctly(self):
        self.restart(rank=5)
        self.primed()
        expected_qkv = [(0, 720, 864), (0, 1872, 2016), (0, 3024, 3168)]
        for slices in slices_for((3456, 1152), axis=0).values():
            self.assertEqual(slices, expected_qkv)
        for slices in slices_for((1152, 1152), axis=1).values():
            self.assertEqual(slices, [(1, 720, 864)])
        for slices in slices_for((4304, 1152), axis=0).values():
            self.assertEqual(slices, [(0, 2690, 3228)])
        for slices in slices_for((1152, 4304), axis=1).values():
            self.assertEqual(slices, [(1, 2690, 3228)])

    def test_every_block_primes_four_geometries_with_row_bias_omitted(self):
        adapter, report = self.primed()
        self.assertEqual(len(self.harness.backend.primed), 108)
        self.assertEqual(report['stats']['primed_weight_modules'], 108)
        expected = {
            'qkv': ('column', (432, 1152), (432,)),
            'proj': ('row', (1152, 144), None),
            'fc1': ('column', (538, 1152), (538,)),
            'fc2': ('row', (1152, 538), None),
        }
        seen = {role: 0 for role in expected}
        for item in self.harness.backend.primed:
            role = item['name'].split(':')[0].rsplit('.', 1)[1]
            mode, weight, bias = expected[role]
            seen[role] += 1
            self.assertEqual(item['mode'], mode, item['name'])
            self.assertEqual(item['weight'], weight, item['name'])
            self.assertEqual(item['bias'], bias, item['name'])
            # row mode must not fold a bias into a partial that is summed 8 times
            self.assertEqual(item['bias'] is None, mode == 'row', item['name'])
        self.assertEqual(seen, {'qkv': 27, 'proj': 27, 'fc1': 27, 'fc2': 27})

    def test_prepared_metadata_records_padded_contract_for_every_entry(self):
        _, report = self.primed()
        self.assertEqual(len(report['prepared']), 108)
        table = mod.padded_geometries()
        for entry in report['prepared']:
            role = entry['role']
            self.assertEqual((entry['k_pad'], entry['n_pad']),
                             (table[role]['k_pad'], table[role]['n_pad']))
            self.assertEqual(entry['mode'], table[role]['mode'])
            self.assertEqual(entry['bias_added_after_float_collective'],
                             role in ('attn.proj', 'mlp.fc2'))

    def test_only_instance_forward_is_patched_on_all_54_modules(self):
        adapter, report = self.primed()
        self.assertEqual(len(report['patched']), 54)
        self.assertTrue(all(not item['had_instance_forward'] for item in report['patched']))
        for block in self.harness.vision.blocks:
            self.assertIn('forward', vars(block.attn))
            self.assertIn('forward', vars(block.mlp))
        # the class-level methods must stay untouched (no global Linear/attn patch)
        self.assertIsInstance(FakeAttn.forward, types.FunctionType)
        self.assertIsInstance(FakeMLP.forward, types.FunctionType)

    def test_prime_returns_a_json_serialisable_snapshot(self):
        _, report = self.primed()
        json.dumps(report)
        self.assertTrue(report['primed'])
        self.assertFalse(report['enabled'])
        self.assertIsNone(report['numerical_tolerance'])
        self.assertFalse(report['full_encoder_precision_accepted'])
        self.assertFalse(report['whole_goal_complete'])
        self.assertFalse(report['eight_times_speedup_claimed'])
        self.assertTrue(report['no_Float_weight_shadow'])
        self.assertTrue(report['no_cached_activations'])
        self.assertTrue(report['original_Half_weights_and_biases_untouched'])
        self.assertEqual(report['attention_scale_from_original_head_dim'], 72 ** -0.5)


# ---------------------------------------------------------------------------
# 3. Enabled forward paths
# ---------------------------------------------------------------------------
class EnabledForwardTests(FakeTorchCase):
    def test_one_attention_call_uses_two_local_heads_and_five_counters(self):
        adapter, _ = self.primed()
        adapter.set_enabled(True)
        out = self.harness.vision.blocks[0].attn(
            hidden(self.harness.device),
            cu_seqlens=cu_seqlens([0, 4032, 8064]),
            position_embeddings=position(self.harness.device))
        self.assertEqual(out.shape, (8064, 1152))
        self.assertIs(out.dtype, DT_HALF)

        stats = adapter.snapshot()['stats']
        self.assertEqual(stats['enabled_attention_calls'], 1)
        self.assertEqual(stats['attention_segments'], 2)
        self.assertEqual(stats['column_runs'], 1)
        self.assertEqual(stats['row_runs'], 1)
        self.assertEqual(stats['all_reduce_calls'], 1)
        self.assertEqual(stats['bias_widenings'], 1)
        self.assertEqual(stats['half_outputs'], 1)

        self.assertEqual(len(self.harness.attention.calls), 2)
        for call in self.harness.attention.calls:
            self.assertEqual(call['q'], (1, 2, 4032, 72))
            self.assertEqual(call['k'], (1, 2, 4032, 72))
            self.assertEqual(call['v'], (1, 2, 4032, 72))
            self.assertEqual(call['dtype'], repr(DT_HALF))

        self.assertEqual(self.harness.backend.runs,
                         [{'name': 'blocks.00.attn.qkv:column:k1152:n432:'
                                    'rows[(0, 144), (1152, 1296), (2304, 2448)]',
                           'mode': 'column', 'shape': (8064, 1152), 'dtype': repr(DT_HALF)},
                          {'name': 'blocks.00.attn.proj:row:k144:n1152:cols[0, 144]',
                           'mode': 'row', 'shape': (8064, 144), 'dtype': repr(DT_HALF)}])
        self.assertEqual(self.harness.group.reduces, [((8064, 1152), repr(DT_FLOAT))])

    def test_one_mlp_call_applies_original_gelu_to_538_logical_channels(self):
        adapter, _ = self.primed()
        adapter.set_enabled(True)
        out = self.harness.vision.blocks[0].mlp(hidden(self.harness.device))
        self.assertEqual(out.shape, (8064, 1152))
        self.assertIs(out.dtype, DT_HALF)

        gelu = [op for op in _LOG if op[0] == 'gelu']
        self.assertEqual(len(gelu), 1)
        self.assertEqual(gelu[0][2], (8064, 538))
        self.assertEqual(gelu[0][3], repr(DT_HALF))

        stats = adapter.snapshot()['stats']
        self.assertEqual(stats['enabled_mlp_calls'], 1)
        self.assertEqual(stats['activation_half_checks'], 1)
        self.assertEqual([(run['mode'], run['shape']) for run in self.harness.backend.runs],
                         [('column', (8064, 1152)), ('row', (8064, 538))])
        self.assertEqual(self.harness.group.reduces, [((8064, 1152), repr(DT_FLOAT))])

    def test_full_vision_forward_hits_every_documented_counter(self):
        adapter, _ = self.primed()
        adapter.set_enabled(True)
        offsets = cu_seqlens([0, 4032, 8064])
        cos_sin = position(self.harness.device)
        activation = hidden(self.harness.device)
        for block in self.harness.vision.blocks:
            block.attn(activation, cu_seqlens=offsets, position_embeddings=cos_sin)
            block.mlp(activation)
        report = adapter.snapshot()
        stats = report['stats']
        expected = mod.expected_counts_per_forward()
        self.assertEqual(stats['enabled_attention_calls'], expected['attention_calls'])
        self.assertEqual(stats['enabled_mlp_calls'], expected['mlp_calls'])
        self.assertEqual(stats['attention_segments'], expected['attention_segments'])
        self.assertEqual(stats['all_reduce_calls'], expected['all_reduce_calls'])
        self.assertEqual(stats['column_runs'], expected['column_runs'])
        self.assertEqual(stats['row_runs'], expected['row_runs'])
        self.assertEqual(len(self.harness.attention.calls), 54)
        self.assertEqual(len(self.harness.group.reduces), 54)
        self.assertEqual(len(self.harness.backend.runs), 108)
        self.assertEqual(stats['parameter_identity_checks'], 216)
        self.assertEqual(stats['half_outputs'], 54)
        self.assertEqual(stats['bias_widenings'], 54)
        self.assertEqual(stats['enabled_attention_calls'] + stats['enabled_mlp_calls'], 54)

    def test_disabled_wrapper_delegates_to_the_original_method(self):
        adapter, _ = self.primed()
        block = self.harness.vision.blocks[0]
        block.attn(hidden(self.harness.device), cu_seqlens=cu_seqlens([0, 4032, 8064]),
                   position_embeddings=position(self.harness.device))
        block.mlp(hidden(self.harness.device))
        self.assertEqual(_ORIGINAL, [('attn', (8064, 1152)), ('mlp', (8064, 1152))])
        self.assertEqual(self.harness.backend.runs, [])
        self.assertEqual(self.harness.group.reduces, [])
        stats = adapter.snapshot()['stats']
        self.assertEqual(stats['disabled_attention_calls'], 1)
        self.assertEqual(stats['disabled_mlp_calls'], 1)
        self.assertEqual(stats['enabled_attention_calls'], 0)
        self.assertEqual(stats['all_reduce_calls'], 0)


# ---------------------------------------------------------------------------
# 4. Fail-closed guards
# ---------------------------------------------------------------------------
class FailClosedTests(FakeTorchCase):
    def test_constructor_rejects_bad_m_attention_and_group_arguments(self):
        for kwargs in ({'m': 4032}, {'m': True}, {'m': '8064'}):
            with self.assertRaises(mod.VisionTPError):
                self.harness.adapter(**kwargs)
        with self.assertRaises(mod.VisionTPError):
            mod.VisionTPAdapter107(self.harness.candidate, self.harness.group,
                                   self.harness.backend, 'not callable')
        for world in (1, 4, 16, True, '8'):
            harness = self.restart()
            harness.group.world_size = world
            with self.assertRaises(mod.VisionTPError):
                mod.VisionTPAdapter107(harness.candidate, harness.group,
                                       harness.backend, harness.attention)
        harness = self.restart()
        harness.group.rank_in_group = 8
        with self.assertRaises(mod.VisionTPError):
            mod.VisionTPAdapter107(harness.candidate, harness.group,
                                   harness.backend, harness.attention)
        harness = self.restart()
        harness.group.rank_in_group = True
        with self.assertRaises(mod.VisionTPError):
            mod.VisionTPAdapter107(harness.candidate, harness.group,
                                   harness.backend, harness.attention)

    def test_set_enabled_requires_prime_and_a_bool_and_rejects_double_prime(self):
        adapter = self.harness.adapter()
        with self.assertRaises(mod.VisionTPError):
            adapter.set_enabled(True)
        adapter.prime(self.harness.vision)
        for bad in (1, 0, 'yes', None):
            with self.assertRaises(mod.VisionTPError):
                adapter.set_enabled(bad)
        self.assertTrue(adapter.set_enabled(True))
        self.assertFalse(adapter.set_enabled(False))
        with self.assertRaises(mod.VisionTPError):
            adapter.prime(self.harness.vision)

    def test_wrong_block_count_is_refused_without_installing_anything(self):
        adapter = self.harness.adapter()
        with self.assertRaises(mod.VisionTPError):
            adapter.prime(fake_vision(self.harness.device, blocks=26))
        self.assertEqual(adapter.snapshot()['patched'], [])

    def test_float_weight_is_refused_and_partial_prime_is_rolled_back(self):
        vision = self.harness.vision
        vision.blocks[5].attn.qkv.weight = FakeTensor((3456, 1152), DT_FLOAT,
                                                      self.harness.device)
        with self.assertRaises(mod.VisionTPError):
            self.harness.adapter().prime(vision)
        for block in vision.blocks:
            self.assertNotIn('forward', vars(block.attn))
            self.assertNotIn('forward', vars(block.mlp))

    def test_missing_bias_and_wrong_device_index_are_refused(self):
        self.harness.vision.blocks[2].mlp.linear_fc1.bias = None
        with self.assertRaises(mod.VisionTPError):
            self.harness.adapter().prime(self.harness.vision)
        harness = self.restart()
        harness.vision.blocks[2].attn.proj.weight = FakeTensor(
            (1152, 1152), DT_HALF, FakeDevice('npu', 1))
        with self.assertRaises(mod.VisionTPError):
            harness.adapter().prime(harness.vision)

    def test_wrong_activation_function_and_attention_metadata_are_refused(self):
        self.harness.vision.blocks[0].mlp.act_fn = FakeGelu()
        self.harness.vision.blocks[0].mlp.act_fn.approximate = 'none'
        with self.assertRaises(mod.VisionTPError):
            self.harness.adapter().prime(self.harness.vision)
        harness = self.restart()
        harness.vision.blocks[0].attn.num_heads = 8
        with self.assertRaises(mod.VisionTPError):
            harness.adapter().prime(harness.vision)

    def test_half_row_partial_is_refused_before_any_collective(self):
        adapter, _ = self.primed()
        adapter.set_enabled(True)
        self.harness.backend.row_dtype = DT_HALF
        with self.assertRaises(mod.VisionTPError) as caught:
            self.harness.vision.blocks[0].mlp(hidden(self.harness.device))
        self.assertIn('Half partial', str(caught.exception))
        self.assertEqual(self.harness.group.reduces, [])

    def test_column_returning_float_and_half_collective_are_refused(self):
        adapter, _ = self.primed()
        adapter.set_enabled(True)
        self.harness.backend.column_dtype = DT_FLOAT
        with self.assertRaises(mod.VisionTPError):
            self.harness.vision.blocks[0].mlp(hidden(self.harness.device))
        self.harness.backend.column_dtype = DT_HALF
        self.harness.group._reduce_dtype = DT_HALF
        with self.assertRaises(mod.VisionTPError):
            self.harness.vision.blocks[0].mlp(hidden(self.harness.device))

    def test_bad_attention_output_dtype_or_shape_is_refused(self):
        adapter, _ = self.primed(attention=FakeAttention(dtype=DT_FLOAT))
        adapter.set_enabled(True)
        with self.assertRaises(mod.VisionTPError):
            self.harness.vision.blocks[0].attn(
                hidden(self.harness.device), cu_seqlens=cu_seqlens([0, 4032, 8064]),
                position_embeddings=position(self.harness.device))
        harness = self.restart()
        adapter, _ = harness.adapter(attention=FakeAttention(shape_shift=8)), None
        adapter.prime(harness.vision)
        adapter.set_enabled(True)
        with self.assertRaises(mod.VisionTPError):
            harness.vision.blocks[0].attn(
                hidden(harness.device), cu_seqlens=cu_seqlens([0, 4032, 8064]),
                position_embeddings=position(harness.device))

    def test_bad_segments_hidden_dtype_and_missing_position_are_refused(self):
        adapter, _ = self.primed()
        adapter.set_enabled(True)
        block = self.harness.vision.blocks[0]
        good = position(self.harness.device)
        with self.assertRaises(mod.VisionTPError):
            block.attn(hidden(self.harness.device), cu_seqlens=cu_seqlens([0, 8064]),
                       position_embeddings=good)
        with self.assertRaises(mod.VisionTPError):
            block.attn(FakeTensor((8064, 1152), DT_FLOAT, self.harness.device),
                       cu_seqlens=cu_seqlens([0, 4032, 8064]), position_embeddings=good)
        with self.assertRaises(mod.VisionTPError):
            block.attn(hidden(self.harness.device), cu_seqlens=cu_seqlens([0, 4032, 8064]),
                       position_embeddings=None)
        with self.assertRaises(mod.VisionTPError):
            block.attn(hidden(self.harness.device), cu_seqlens=cu_seqlens([0, 4032, 8064]),
                       position_embeddings=good, unexpected=True)
        with self.assertRaises(mod.VisionTPError):
            block.mlp(hidden(self.harness.device), unexpected=True)
        with self.assertRaises(mod.VisionTPError):
            block.mlp(FakeTensor((4032, 1152), DT_HALF, self.harness.device))
        with self.assertRaises(mod.VisionTPError):
            block.mlp(FakeTensor((8064, 1152), DT_HALF, FakeDevice('npu', 1)))

    def test_changed_parameter_identity_is_refused(self):
        adapter, _ = self.primed()
        adapter.set_enabled(True)
        block = self.harness.vision.blocks[0]
        block.mlp.linear_fc2.weight = FakeTensor((1152, 4304), DT_HALF, self.harness.device)
        with self.assertRaises(mod.VisionTPError) as caught:
            block.mlp(hidden(self.harness.device))
        self.assertIn('changed identity', str(caught.exception))


# ---------------------------------------------------------------------------
# 5. Instance-patch ownership and restore
# ---------------------------------------------------------------------------
class PatchOwnershipTests(FakeTorchCase):
    def test_uninstall_deletes_only_our_instance_attribute(self):
        adapter, _ = self.primed()
        blocks = list(self.harness.vision.blocks)
        result = adapter.uninstall()
        self.assertTrue(result['uninstalled'])
        self.assertFalse(result['primed'])
        for block in blocks:
            self.assertNotIn('forward', vars(block.attn))
            self.assertNotIn('forward', vars(block.mlp))
        _ORIGINAL.clear()
        blocks[0].attn(hidden(self.harness.device))
        self.assertEqual(_ORIGINAL, [('attn', (8064, 1152))])

    def test_preexisting_instance_forward_is_restored_identically(self):
        module = self.harness.vision.blocks[0].attn
        sentinel = types.MethodType(lambda this, value, **kwargs: value, module)
        module.forward = sentinel
        adapter, report = self.primed()
        self.assertTrue(report['patched'][0]['had_instance_forward'])
        self.assertIsNot(vars(module)['forward'], sentinel)
        adapter.uninstall()
        self.assertIs(vars(module)['forward'], sentinel)

    def test_uninstall_refuses_to_clobber_a_foreign_wrapper(self):
        adapter, _ = self.primed()
        blocks = self.harness.vision.blocks
        foreign = object()
        blocks[3].attn.forward = foreign
        with self.assertRaises(mod.VisionTPError) as caught:
            adapter.uninstall()
        self.assertIn('refusing to clobber', str(caught.exception))
        self.assertIs(blocks[3].attn.forward, foreign)

    def test_snapshot_survives_uninstall_and_is_deep_copied(self):
        adapter, _ = self.primed()
        adapter.set_enabled(True)
        self.harness.vision.blocks[0].mlp(hidden(self.harness.device))
        before = adapter.snapshot()
        json.dumps(before)

        adapter.uninstall()
        self.assertTrue(before['primed'])
        self.assertEqual(len(before['prepared']), 108)
        self.assertEqual(before['stats']['enabled_mlp_calls'], 1)

        # mutating a returned snapshot must not reach the adapter's own state
        before['prepared'][0]['n'] = -1
        before['stats']['enabled_mlp_calls'] = 999
        before['geometry']['local_mlp_channels'] = -1
        after = adapter.snapshot()
        self.assertEqual(after['prepared'], [])
        self.assertEqual(after['patched'], [])
        self.assertFalse(after['primed'])
        self.assertEqual(after['stats']['enabled_mlp_calls'], 1)
        self.assertEqual(after['geometry']['local_mlp_channels'], 538)

    def test_nonzero_storage_offset_input_is_copied_dense_value_preserving(self):
        adapter, _ = self.primed()
        adapter.set_enabled(True)
        view = FakeTensor((8064, 1152), DT_HALF, self.harness.device, offset=42)
        out = self.harness.vision.blocks[0].mlp(view)
        self.assertEqual(out.shape, (8064, 1152))
        self.assertEqual(adapter.snapshot()['stats']['dense_copies'], 1)
        self.assertEqual([op for op in _LOG if op[0] == 'clone'][0][2], (8064, 1152))

    def test_adapter_keeps_no_tensor_or_activation_after_prime(self):
        adapter, _ = self.primed()
        for name in ('_signatures', '_prepared', '_stats', '_roles'):
            self.assertFalse(_holds_tensor(getattr(adapter, name)), name)
        self.assertEqual(adapter.snapshot()['stats']['primed_weight_modules'], 108)


def _holds_tensor(value, depth=0):
    if isinstance(value, FakeTensor):
        return True
    if depth > 4 or isinstance(value, (str, bytes, int, float, bool, type(None), type)):
        return False
    if isinstance(value, dict):
        return any(_holds_tensor(item, depth + 1) for item in value.values())
    if isinstance(value, (list, tuple)):
        return any(_holds_tensor(item, depth + 1) for item in value)
    return False


# ---------------------------------------------------------------------------
# 6. Source discipline
# ---------------------------------------------------------------------------
class SourceDisciplineTests(unittest.TestCase):
    def setUp(self):
        self.source = SOURCE.read_text()
        self.tree = ast.parse(self.source)

    def test_module_scope_imports_are_standard_library_only(self):
        module_level = set()
        for node in self.tree.body:
            if isinstance(node, ast.Import):
                module_level.update(alias.name.split('.')[0] for alias in node.names)
            elif isinstance(node, ast.ImportFrom) and node.module:
                module_level.add(node.module.split('.')[0])
        self.assertLessEqual(module_level, {'copy', 'hashlib', 'types', 'pathlib', '__future__'})

    def test_torch_is_imported_only_inside_the_lazy_helper(self):
        imports = [node for node in ast.walk(self.tree)
                   if isinstance(node, ast.Import)
                   and any(alias.name == 'torch' for alias in node.names)]
        self.assertEqual(len(imports), 1)
        self.assertEqual(imports[0].col_offset, 4)  # only inside _torch()

    def test_never_touches_the_global_sdpa_helper_or_narrows_types(self):
        for forbidden in ('bfloat16', '.clamp(', 'scaled_dot_product_attention',
                          'autocast', 'setattr('):
            self.assertNotIn(forbidden, self.source)
        # the global helper may only be *named* in the module docstring, never called
        self.assertEqual(self.source.count('_vision_sdpa_bf16_compatible'), 1)
        self.assertIn('nothing here replaces the global', self.source)
        self.assertNotIn('cls.forward', self.source)
        self.assertIn('types.MethodType', self.source)

    def test_public_api_and_no_claim_fields(self):
        for name in ('prime', 'set_enabled', 'snapshot', 'uninstall'):
            self.assertIn(f'def {name}(', self.source)
        self.assertIn("'full_encoder_precision_accepted': False", self.source)
        self.assertIn("'whole_goal_complete': False", self.source)
        self.assertIn("'numerical_tolerance': None", self.source)
        self.assertIn("'eight_times_speedup_claimed': False", self.source)


if __name__ == '__main__':
    unittest.main(verbosity=2)