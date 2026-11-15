#!/usr/bin/env python3
"""Host-only workload-contract checks against the inspected production source AST."""
import ast
from pathlib import Path
import types
import unittest

ROOT = Path(__file__).with_name("source")


class WorkloadContractTests(unittest.TestCase):
    def test_seconds_is_form_field_but_duration_is_not(self):
        tree = ast.parse((ROOT / "helpers.py").read_text())
        fn = next(x for x in ast.walk(tree) if isinstance(x, ast.AsyncFunctionDef) and x.name == "_parse_video_form")
        names = [x.arg for x in fn.args.args]
        self.assertIn("seconds", names)
        self.assertIn("input_references", names)
        self.assertIn("size", names)
        self.assertNotIn("duration", names)

    def test_five_seconds_snaps_to_native_124_frame_boundary(self):
        tree = ast.parse((ROOT / "time_request.py").read_text())
        fn = next(x for x in tree.body if isinstance(x, ast.FunctionDef) and x.name == "_align_frame_count")
        isolated = ast.Module(body=[fn], type_ignores=[])
        scope = {}
        exec(compile(isolated, "production-frame-alignment", "exec"), scope)
        align = scope["_align_frame_count"]
        self.assertEqual(align(5 * 24), 124)
        self.assertEqual(align(209), 209)
        self.assertAlmostEqual(align(5 * 24) / 24, 5.1666666667)

    def test_missing_duration_has_209_frame_fallback(self):
        tree = ast.parse((ROOT / "encoder_processing.py").read_text())
        fn = next(x for x in tree.body if isinstance(x, ast.FunctionDef) and x.name == "resolve_minimax_h3_shape")
        values = [x.value for x in ast.walk(fn) if isinstance(x, ast.Constant)]
        self.assertIn(209, values)
        self.assertIn(124, values)
        self.assertIn("duration", values)
        self.assertIn("duration_seconds", values)

    def test_fl2va_requires_actual_image_conditioning(self):
        source = (ROOT / "encoder_processing.py").read_text()
        self.assertIn('elif task == "fl2va":', source)
        self.assertIn('raise OmniClientError("fl2va requires image conditions only")', source)
        self.assertIn('raise OmniClientError("fl2va accepts at most first and last images")', source)


if __name__ == "__main__":
    unittest.main()
