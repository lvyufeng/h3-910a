#!/usr/bin/env python3
"""Plan/key-cache tests require no accelerator or model assets."""
import importlib.util
import json
import sys
import types
from unittest.mock import patch
from pathlib import Path
import tempfile
import unittest

PATH = Path(__file__).with_name("candidate_fa910a.py")


def module():
    spec = importlib.util.spec_from_file_location("fa52_test", PATH)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


class PaddingPlanTests(unittest.TestCase):
    def setUp(self):
        self.fa = module()

    def test_measured_bad_shape_aligns_to_large_tiles(self):
        n = self.fa._kernel_sequence_length(41792, 128)
        self.assertEqual(n, 43776)
        self.assertEqual((self.fa.pick_tile(n,128),self.fa.pick_tile(n,304)),(128,304))
        self.assertLessEqual(n-41792,41792*.1)

    def test_good_shapes_and_small_shapes_are_not_expanded(self):
        for n in (64,128,512,2048,63232):
            self.assertEqual(self.fa._kernel_sequence_length(n,128),n)

    def test_unverified_head_width_and_explicit_optout_preserved(self):
        self.assertEqual(self.fa._kernel_sequence_length(41792,64),41792)
        self.assertEqual(self.fa._kernel_sequence_length(41792,128,enabled=False),41792)

    def test_invalid_sequence_lengths_fail_closed(self):
        for n in (0,-16,17,41792.0):
            with self.assertRaises(self.fa.FlashAttn910AError):
                self.fa._kernel_sequence_length(n,128)

    def test_masked_cache_cannot_alias_the_old_unmasked_prototype(self):
        args = (1,7,43776,128,128,304,30)
        masked = self.fa._kernel_cache_key(*args,key_seq_limit=41792)
        unmasked = self.fa._kernel_cache_key(*args)
        other = self.fa._kernel_cache_key(*args,key_seq_limit=41920)
        self.assertNotEqual(masked,unmasked)
        self.assertNotEqual(masked,other)
        self.assertTrue(masked.endswith("_c30_k41792"))

    def cached(self, extra):
        work = tempfile.TemporaryDirectory(prefix="h3_keymask_test_")
        self.addCleanup(work.cleanup)
        self.fa._KERNEL_DIR = work.name
        self.fa._runtime_core_budget = lambda: 30
        key = self.fa._kernel_cache_key(1,7,43776,128,128,304,30,41792)
        obj = Path(work.name)/key/(key+".o")
        obj.parent.mkdir()
        obj.write_bytes(b"test object")
        obj.with_suffix(".json").write_text(json.dumps({"blockDim":30,**extra}))
        return self.fa._build(1,7,43776,128,key_seq_limit=41792)

    def test_missing_or_wrong_mask_metadata_rejected(self):
        for extra in ({},{"fa910a_key_seq_limit":41920,"fa910a_kernel_version":2},
                      {"fa910a_key_seq_limit":41792,"fa910a_kernel_version":1}):
            with self.assertRaises(self.fa.FlashAttn910AError):
                self.cached(extra)

    def test_fresh_build_publishes_mask_metadata_before_object(self):
        # Exercise the uncached compilation/publication branch with a fake CANN
        # builder. This catches local-import initialization errors that cached
        # metadata-only regressions do not exercise.
        work = tempfile.TemporaryDirectory(prefix="h3_fresh_build_test_")
        self.addCleanup(work.cleanup)
        self.fa._KERNEL_DIR = str(Path(work.name).resolve())
        self.fa._runtime_core_budget = lambda: 30
        calls = []
        class FakeFA:
            def __init__(self, **kwargs):
                self.name = kwargs["kernel_name"]
                self.core_num = 32
            def compute_process(self):
                calls.append(self.name)
                root = Path("kernel_meta")
                root.mkdir()
                (root/(self.name+".o")).write_bytes(b"fresh fake object")
                (root/(self.name+".json")).write_text(json.dumps({"blockDim":self.core_num}))
        names = ["tbe", "tbe.common", "tbe.common.platform", "flash_attention", "flash_attention.ops",
                 "flash_attention.ops.flash_attention", "flash_attention.ops.flash_attention.flash_attention_fwd",
                 "flash_attention.ops.flash_attention.tiling_strategy",
                 "flash_attention.ops.flash_attention.tiling_strategy.strategy"]
        modules = {n:types.ModuleType(n) for n in names}
        for n,m in modules.items():
            m.__path__ = []
            if "." in n:
                parent,child=n.rsplit(".",1)
                setattr(modules[parent],child,m)
        modules["tbe.common.platform"].set_current_compile_soc_info=lambda soc:None
        modules["flash_attention.ops.flash_attention.flash_attention_fwd"].FlashAttentionFwd=FakeFA
        modules["flash_attention.ops.flash_attention.tiling_strategy.strategy"].TilingStrategy=types.SimpleNamespace(from_strategy_name=lambda n:None)
        with patch.dict(sys.modules,modules), patch.object(self.fa,"_install_key_sequence_limit",lambda fa,n:None):
            obj,name,cores=self.fa._build(1,7,43776,128,key_seq_limit=41792)
            meta=json.loads(Path(obj).with_suffix(".json").read_text())
            self.assertEqual(meta["fa910a_key_seq_limit"],41792)
            self.assertEqual(meta["fa910a_kernel_version"],2)
            self.assertEqual(cores,30)
            self.assertTrue(Path(obj).exists())
            self.assertEqual(self.fa._build(1,7,43776,128,key_seq_limit=41792),(obj,name,cores))
        self.assertEqual(len(calls),1)

    def test_valid_masked_cache_preserved(self):
        obj,name,cores = self.cached({"fa910a_key_seq_limit":41792,"fa910a_kernel_version":2})
        self.assertEqual(cores,30)
        self.assertIn("_k41792",obj)
        self.assertTrue(name.endswith("_k41792__kernel0"))


if __name__ == "__main__":
    unittest.main(verbosity=2)
