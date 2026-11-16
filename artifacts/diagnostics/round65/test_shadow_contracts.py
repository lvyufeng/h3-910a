"""Portable cache lifecycle and extracted timing contracts, no NPU emulation."""
import ast
import gc
import importlib.util
from pathlib import Path
import sys
from types import ModuleType, SimpleNamespace
import unittest
from unittest.mock import patch
import weakref

ROOT = Path(__file__).resolve().parent
fake_torch = ModuleType('torch')
fake_torch.grad_enabled = False
fake_torch.is_grad_enabled = lambda: fake_torch.grad_enabled
spec = importlib.util.spec_from_file_location('shadow65_contracts', ROOT / 'shadow_cache65.py')
cache_module = importlib.util.module_from_spec(spec)
with patch.dict(sys.modules, {'torch': fake_torch}):
    spec.loader.exec_module(cache_module)


class Storage:
    serial = 0
    def __init__(self, values):
        Storage.serial += 1
        self._cdata = Storage.serial
        self.values = list(values)
        self.version = 0
    def nbytes(self):
        return len(self.values) * 2


class Tensor:
    def __init__(self, values=(1, 2, 3, 4), storage=None, offset=0, shape=(2, 2), stride=(2, 1), inference=False):
        self.storage = Storage(values) if storage is None else storage
        self.offset, self.shape, self._stride = offset, shape, stride
        self.dtype, self.device = 'bf16', 'npu:0'
        self.inference = inference
    @property
    def _version(self):
        if self.inference:
            raise RuntimeError('Inference tensors do not track version counter')
        return self.storage.version
    def untyped_storage(self):
        return self.storage
    def data_ptr(self):
        return 10000 + self.storage._cdata * 1000 + self.offset * 2
    def storage_offset(self):
        return self.offset
    def stride(self):
        return self._stride
    def numel(self):
        total = 1
        for dimension in self.shape:
            total *= dimension
        return total
    def element_size(self):
        return 4 if self.dtype == 'float32' else 2
    def float(self):
        # Simple values only: these tests verify lifetime/invalidation mechanics,
        # not hardware rounding, GEMM results or a substitute arithmetic oracle.
        result = Tensor(self.storage.values[self.offset:self.offset + self.numel()], shape=self.shape, stride=self._stride)
        result.dtype = 'float32'
        return result


class CacheLifecycle(unittest.TestCase):
    def setUp(self):
        fake_torch.grad_enabled = False
    def test_hits_across_fresh_views_of_identical_storage(self):
        cache = cache_module.ShadowWeights()
        x = Tensor()
        a = cache.get('q', x)
        b = Tensor(storage=x.storage)
        self.assertIs(cache.get('q', b), a)
        self.assertEqual(cache.stats()['hits'], 1)
        self.assertEqual(cache.stats()['cached_fp32_bytes'], 16)
    def test_versioned_mutation_recomputes(self):
        cache = cache_module.ShadowWeights()
        x = Tensor()
        prior = cache.get('q', x)
        x.storage.values[0] = 9
        x.storage.version += 1
        current = cache.get('q', x)
        self.assertIsNot(prior, current)
        self.assertEqual(current.storage.values[0], 9)
        self.assertEqual(cache.stats()['invalidations'], 1)
    def test_data_write_requires_explicit_clear_not_a_fake_automatic_guard(self):
        cache = cache_module.ShadowWeights()
        x = Tensor()
        prior = cache.get('q', x)
        x.storage.values[0] = 7  # models .data.copy_ bypassing source _version
        self.assertIs(cache.get('q', x), prior)
        cache.clear()
        current = cache.get('q', x)
        self.assertEqual(current.storage.values[0], 7)
        self.assertIsNot(prior, current)
    def test_rebind_offset_stride_dtype_device_each_invalidate(self):
        cache = cache_module.ShadowWeights()
        x = Tensor(values=range(12))
        previous = cache.get('slot', x)
        for update in (lambda: setattr(x, 'storage', Storage(range(12))),
                       lambda: setattr(x, 'offset', 2), lambda: setattr(x, '_stride', (1, 2)),
                       lambda: setattr(x, 'dtype', 'float16'), lambda: setattr(x, 'device', 'npu:1')):
            update()
            value = cache.get('slot', x)
            self.assertIsNot(value, previous)
            previous = value
        self.assertEqual(cache.stats()['invalidations'], 5)
    def test_strong_source_reference_and_clear_release(self):
        cache = cache_module.ShadowWeights()
        x = Tensor()
        pointer = weakref.ref(x)
        cache.get('slot', x)
        del x
        gc.collect()
        self.assertIsNotNone(pointer())
        cache.clear()
        gc.collect()
        self.assertIsNone(pointer())
    def test_unversioned_or_gradient_enabled_weights_are_not_cached(self):
        cache = cache_module.ShadowWeights()
        x = Tensor(inference=True)
        self.assertIsNot(cache.get('slot', x), cache.get('slot', x))
        self.assertEqual(cache.stats()['entries'], 0)
        fake_torch.grad_enabled = True
        cache.get('ordinary', Tensor())
        self.assertEqual(cache.stats()['entries'], 0)
    def test_entry_and_byte_caps_fail_instead_of_silent_thrash(self):
        for cache in (cache_module.ShadowWeights(max_entries=1), cache_module.ShadowWeights(max_bytes=16)):
            cache.get('a', Tensor())
            with self.assertRaises(RuntimeError):
                cache.get('b', Tensor())
            self.assertEqual(cache.stats()['entries'], 1)


class ActivationLifetime(unittest.TestCase):
    def test_one_scope_same_object_only_and_exception_cleanup(self):
        cache = cache_module.SharedActivation()
        x = Tensor()
        with self.assertRaises(ValueError):
            with cache.scope(x):
                first = cache.get(x)
                self.assertIs(cache.get(x), first)
                other = Tensor(storage=x.storage)
                self.assertIsNot(cache.get(other), first)
                raise ValueError('simulated attention failure')
        self.assertIsNone(cache.active)
        self.assertIsNot(cache.get(x), first)
    def test_inference_activation_can_reuse_within_readonly_scope(self):
        cache = cache_module.SharedActivation()
        x = Tensor(inference=True)
        with cache.scope(x):
            self.assertIs(cache.get(x), cache.get(x))
        self.assertIsNone(cache.active)
    def test_versioned_mutation_or_rebind_in_scope_rejected(self):
        for mutate in (lambda x: setattr(x.storage, 'version', 1), lambda x: setattr(x, 'storage', Storage(range(4)))):
            cache, x = cache_module.SharedActivation(), Tensor()
            with cache.scope(x):
                mutate(x)
                with self.assertRaises(RuntimeError):
                    cache.get(x)
            self.assertIsNone(cache.active)
    def test_nested_scope_rejected_without_leaking_outer(self):
        cache, x = cache_module.SharedActivation(), Tensor()
        with cache.scope(x):
            with self.assertRaises(RuntimeError):
                with cache.scope(x):
                    pass
            self.assertIsNotNone(cache.active)
        self.assertIsNone(cache.active)


class ReplayBoundary(unittest.TestCase):
    def test_actual_timer_excludes_cpu_copy_and_drains_npu(self):
        tree = ast.parse((ROOT / 'text_shadow_tp8.py').read_text())
        function = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == 'once')
        clock = {'now': 0., 'pending': 0.}
        class Result:
            def detach(self): return self
            def cpu(self): clock['now'] += 100.; return self
            def contiguous(self): return self
        def sync():
            clock['now'] += clock['pending']; clock['pending'] = 0.
        def model(*args, **kwargs):
            clock['now'] += 2.; clock['pending'] = 8.; return Result()
        def barrier(): clock['now'] += 50.
        torch = SimpleNamespace(distributed=SimpleNamespace(barrier=barrier), npu=SimpleNamespace(synchronize=sync),
                                inference_mode=lambda: __import__('contextlib').nullcontext())
        namespace = {'torch': torch, 'time': SimpleNamespace(perf_counter=lambda: clock['now'])}
        exec(compile(ast.fix_missing_locations(ast.Module(body=[function], type_ignores=[])), '<actual timer>', 'exec'), namespace)
        out, seconds = namespace['once'](model, None, None, None, None)
        self.assertEqual(seconds, 10.)
        self.assertEqual(clock['now'], 160.)


if __name__ == '__main__':
    unittest.main()
