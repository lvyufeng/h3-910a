"""Regression checks for shared-storage K/V slices and bound-method cleanup."""
import ast
from copy import deepcopy
from pathlib import Path
import sys
import types
import unittest

HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE))
import kv_padding81 as kv


class ReviewRegressionTests(unittest.TestCase):
    def stale_slice_checker(self):
        tree=ast.parse((HERE/'kv_padding81.py').read_text())
        cls=next(node for node in tree.body if isinstance(node,ast.ClassDef)and node.name=='KVPaddingAdapter')
        method=next(node for node in cls.body if isinstance(node,ast.FunctionDef)and node.name=='_cached_weight')
        loop=deepcopy(next(node for node in method.body if isinstance(node,ast.For)))
        loop.iter=ast.Name(id='cache',ctx=ast.Load())
        function=ast.FunctionDef(name='check',args=ast.arguments(posonlyargs=[],args=[ast.arg(arg='cache'),ast.arg(arg='key')],kwonlyargs=[],kw_defaults=[],defaults=[]),body=[loop],decorator_list=[])
        module=ast.Module(body=[function],type_ignores=[]);ast.fix_missing_locations(module)
        namespace={'KVPaddingError':kv.KVPaddingError};exec(compile(module,'<stale-slice-check>','exec'),namespace)
        return namespace['check']
    def test_K_and_V_share_storage_but_different_offsets_are_valid(self):
        check=self.stale_slice_checker()
        K=(4096,5242880,(128,5120),(5120,1),'torch.float16','npu:0',1)
        V=(4096,5898240,(128,5120),(5120,1),'torch.float16','npu:0',1)
        check({K:{}},V)
    def test_same_logical_slice_changed_version_still_fails_closed(self):
        check=self.stale_slice_checker()
        K=(4096,5242880,(128,5120),(5120,1),'torch.float16','npu:0',1)
        altered=K[:-1]+(2,)
        with self.assertRaises(kv.KVPaddingError):check({K:{}},altered)
    def test_uninstall_removes_the_installed_bound_helper(self):
        def original(self):return None
        cls=type('FakeAttention',(),{'forward':original})
        candidate=types.SimpleNamespace(MiniMaxH3Qwen3VLTextAttention=cls)
        adapter=kv.KVPaddingAdapter(candidate)
        adapter._class=cls;adapter._original_forward=original;adapter._installed=True
        adapter._injected_helper_name='_r81_kv_linear';adapter._previous_helper=None
        candidate._r81_kv_linear=adapter._kv_linear
        self.assertIsNot(candidate._r81_kv_linear,adapter._kv_linear)
        adapter.uninstall()
        self.assertFalse(hasattr(candidate,'_r81_kv_linear'))
        self.assertIs(cls.forward,original)


if __name__=='__main__':unittest.main()
