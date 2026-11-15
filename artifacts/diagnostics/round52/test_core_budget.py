"""Host-only regression tests for safe core-count detection and cache isolation."""
import ctypes
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import types
import unittest
from unittest.mock import patch

PATH = Path(__file__).with_name("candidate_fa910a.py")


def module():
    spec = importlib.util.spec_from_file_location("tested_fa910a", PATH)
    result = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(result)
    return result


class Query:
    def __init__(self, counts, rc=0):
        self.counts, self.rc, self.calls = counts, rc, []

    def __call__(self, device, attr, pointer):
        self.calls.append((device, attr))
        ctypes.cast(pointer, ctypes.POINTER(ctypes.c_int64))[0] = self.counts[device]
        return self.rc


class CoreBudgetTests(unittest.TestCase):
    def setUp(self):
        self.fa = module()

    def bind(self, counts, rc=0):
        query = Query(counts, rc)
        self.fa._lib = lambda: types.SimpleNamespace(aclrtGetDeviceInfo=query)
        return query

    def test_runtime_reports_thirty_not_compile_default(self):
        q = self.bind({0: 30})
        self.assertEqual(self.fa._query_runtime_aicore_count(0), 30)
        self.assertEqual(q.calls, [(0, 101)])

    def test_failure_is_not_silently_replaced_by_thirtytwo(self):
        self.bind({0: 30}, rc=7)
        with self.assertRaises(self.fa.FlashAttn910AError):
            self.fa._query_runtime_aicore_count(0)

    def test_invalid_count_rejected(self):
        for count in (0, -1, 65):
            self.bind({0: count})
            with self.assertRaises(self.fa.FlashAttn910AError):
                self.fa._query_runtime_aicore_count(0)

    def test_runtime_without_device_info_getter_rejected(self):
        self.fa._lib = lambda: object()
        with self.assertRaises(self.fa.FlashAttn910AError):
            self.fa._query_runtime_aicore_count(0)

    def test_budget_cached_per_logical_device(self):
        q = self.bind({0: 30, 1: 32})
        device = [0]
        torch = types.SimpleNamespace(npu=types.SimpleNamespace(current_device=lambda: device[0]))
        with patch.dict(sys.modules, {"torch": torch}):
            self.assertEqual(self.fa._runtime_core_budget(), 30)
            self.assertEqual(self.fa._runtime_core_budget(), 30)
            device[0] = 1
            self.assertEqual(self.fa._runtime_core_budget(), 32)
        self.assertEqual(q.calls, [(0, 101), (1, 101)])

    def test_core_budget_changes_disk_cache_identity(self):
        args = (1, 7, 63232, 128, 128, 304)
        c30 = self.fa._kernel_cache_key(*args, 30)
        c32 = self.fa._kernel_cache_key(*args, 32)
        self.assertNotEqual(c30, c32)
        self.assertTrue(c30.endswith("_c30"))
        self.assertNotEqual(c30, "b1_h7_n63232_d128_br128_bc304")

    def cached_build(self, block_dim):
        temp = tempfile.TemporaryDirectory(prefix="h3_core_budget_unittest_")
        self.addCleanup(temp.cleanup)
        root = Path(temp.name).resolve()
        assert root.name.startswith("h3_core_budget_unittest_")
        self.fa._KERNEL_DIR = str(root)
        self.fa._runtime_core_budget = lambda: 30
        key = self.fa._kernel_cache_key(1, 7, 63232, 128, 128, 304, 30)
        obj = root / key / (key + ".o")
        obj.parent.mkdir()
        obj.write_bytes(b"fake compiled object")
        obj.with_suffix(".json").write_text(json.dumps({"blockDim": block_dim}))
        return self.fa._build(1, 7, 63232, 128)

    def test_oversubscribed_cache_rejected(self):
        with self.assertRaises(self.fa.FlashAttn910AError):
            self.cached_build(32)

    def test_valid_cached_budget_preserved(self):
        obj, name, cores = self.cached_build(30)
        self.assertEqual(cores, 30)
        self.assertIn("_c30", obj)
        self.assertTrue(name.endswith("_c30__kernel0"))


if __name__ == "__main__":
    unittest.main()
