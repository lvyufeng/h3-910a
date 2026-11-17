"""Regression for checkpoint manifest names shadowed by layer-loop indexes."""
import ast
from pathlib import Path
import unittest

ROOT=Path(__file__).resolve().parent


class CheckpointIndexNameContracts(unittest.TestCase):
    def test_fixed_driver_keeps_manifest_separate_from_layer_index(self):
        text=(ROOT/'encode_fp16_padded_kv_fixed81.py').read_text();tree=ast.parse(text)
        self.assertIn("checkpoint_index = json.loads((MODEL / 'model.safetensors.index.json').read_text())['weight_map']",text)
        self.assertIn('MODEL / checkpoint_index[embedding_key]',text)
        self.assertIn('for key, name in checkpoint_index.items()',text)
        self.assertIn('sorted(set(checkpoint_index.values()))',text)
        self.assertIn("ROOT = Path('/tmp/h3_round81_full_fp16_pad02')",text)
        for node in ast.walk(tree):
            if isinstance(node,ast.For):
                for child in ast.walk(node.target):
                    if isinstance(child,ast.Name):self.assertNotEqual(child.id,'checkpoint_index')
    def test_first_failed_source_preserved_for_diagnosis(self):
        text=(ROOT/'encode_fp16_padded_kv81.py').read_text()
        self.assertIn('MODEL / index[embedding_key]',text)
        self.assertIn("ROOT = Path('/tmp/h3_round81_full_fp16_pad01')",text)
        self.assertIn("'int' object is not subscriptable",(ROOT/'run.log').read_text())


if __name__=='__main__':unittest.main()
