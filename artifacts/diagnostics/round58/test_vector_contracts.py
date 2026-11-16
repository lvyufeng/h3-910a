#!/usr/bin/env python3
"""CPU structural regressions for actual native-codec vector kernel contracts."""
import ast
from pathlib import Path
import unittest

ROOT = Path(__file__).resolve().parent


class VectorContracts(unittest.TestCase):
    def test_codec_transpose_schedule_preserves_channel_order(self):
        a = [[r * 16 + c for c in range(16)] for r in range(16)]
        transposed = [list(x) for x in zip(*a)]
        words = [[None] * 32 for _ in range(16)]
        for half in (0, 1):
            src = []
            for c in range(8):
                src.extend([[0] * 16, transposed[half * 8 + c]])
            part = [list(x) for x in zip(*src)]
            for r in range(16):
                words[r][half * 16:half * 16 + 16] = part[r]
        self.assertEqual([[words[r][2*c+1] for c in range(16)] for r in range(16)], a)
        self.assertTrue(all(words[r][2*c] == 0 for r in range(16) for c in range(16)))
        # Inverse two-half transpose writes odd output rows into the full
        # transposed matrix, then a final transpose restores ordinary rows.
        extracted = [[None] * 16 for _ in range(16)]
        for half in (0, 1):
            part = [list(x) for x in zip(*[row[half*16:half*16+16] for row in words])]
            for c in range(8):
                extracted[half*8+c] = part[2*c+1]
        self.assertEqual([list(x) for x in zip(*extracted)], a)

    def test_all_finite_bf16_midpoint_integer_rounding(self):
        count = 0
        for h in range(65536):
            if h & 0x7f80 == 0x7f80:
                continue
            for lo in (0x7fff, 0x8000, 0x8001):
                bits = (h << 16) | lo
                lsb = (bits & 0x00010000) // 65536
                software = ((bits + 32767 + lsb) & 0xffffffff) & 0xffff0000
                expected = h + int(lo > 0x8000 or (lo == 0x8000 and h & 1))
                self.assertEqual(software, (expected & 65535) << 16)
                count += 1
        self.assertEqual(count, 195840)

    def test_npu_zero_behavior_is_not_replaced_by_cpu_product_sign(self):
        text = (ROOT / 'native_bf16_vector58.py').read_text()
        tree = ast.parse(text)
        parents = {}
        for node in ast.walk(tree):
            for child in ast.iter_child_nodes(node):
                parents[child] = node
        calls = [n for n in ast.walk(tree) if isinstance(n, ast.Call) and isinstance(n.func, ast.Name) and n.func.id == '_fix_product_zero']
        self.assertEqual(len(calls), 1)
        node = calls[0]
        while not isinstance(node, ast.If):
            node = parents[node]
        self.assertIn("kind in ('rms', 'addrms')", ast.unparse(node.test))

    def test_native_api_has_no_external_fp32_casts(self):
        tree = ast.parse((ROOT / 'native_bf16_vector58.py').read_text())
        node = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'native_launch')
        self.assertFalse(any(isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute) and n.func.attr in ('float', 'half', 'to') for n in ast.walk(node)))
        text = ast.unparse(node)
        self.assertIn('get_npu_format', text)
        self.assertIn('weight.numel() == width', text)
        self.assertIn('residual.shape == x.shape', text)

    def test_true_width_controls_mean_and_output(self):
        text = (ROOT / 'native_bf16_vector58.py').read_text()
        self.assertIn('padded = ((width + 255) // 256) * 256', text)
        self.assertIn('1.0 / width', text)
        self.assertIn('width // 16', text)
        self.assertIn('x_gm[row, width]', text)

    def test_compiler_cache_source_identity_and_shared_lock(self):
        text = (ROOT / 'native_bf16_vector58.py').read_text()
        self.assertIn("Path(__file__).read_text() + _SOURCE_HASH", text)
        self.assertIn("'.compile.lock'", text)
        self.assertIn('fcntl.LOCK_EX', text)
        self.assertIn('min(core_budget(), rows)', text)

    def test_saturation_counterexample_is_handled_and_not_overclaimed(self):
        text = (ROOT / 'native_bf16_vector58.py').read_text()
        self.assertIn('raw_sum >= 3.4028234663852886e38', text)
        self.assertIn('Exact legitimately-maximal sums are not certified', text)
        self.assertIn('t.vector_dup(64, inv, 0.0', text)

    def test_complete_stack_references_are_not_in_warm_timing(self):
        text = (ROOT / 'text_vectors_tp8.py').read_text()
        self.assertIn('VERIFY_ACT = False', text)
        self.assertIn("'first_forward_includes_act_reference_checks': use_act", text)
        self.assertIn('assert len(ACT_CHECKS) == 50', text)
        self.assertIn('assert count == 200', text)


if __name__ == '__main__':
    unittest.main(verbosity=2)
