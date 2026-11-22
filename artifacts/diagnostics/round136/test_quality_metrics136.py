#!/usr/bin/env python3
"""Small, pure-NumPy R136 metric/source units; never execute Torch or a model.

No temporary files, model weights, activation archives, NPU/remote calls or
quality-acceptance thresholds are used. Floating assertions below test metric
formula arithmetic only, not an allowance for any candidate output error.
"""
from __future__ import annotations

import ast
import copy
import hashlib
import io
import json
import math
from pathlib import Path
import unittest
from unittest.mock import patch

import numpy as np

from quality_metrics136 import (
    MAX_CHUNK_WORDS,
    _paired_chunks,
    compare_audio_float32,
    compare_float_arrays,
    compare_video_uint8,
    sha256_file,
)
from compare_quality136 import CHECKS_PER_REQUEST, LABELS, TOTAL_CHECKS, run_quality_comparison, validate_quality_receipts


class FloatTests(unittest.TestCase):
    def test_exact_fp32_and_full_word_schema(self):
        reference = np.arange(12, dtype=np.float32).reshape(3, 4)
        result = compare_float_arrays(reference, reference.copy(), chunk_words=3)
        self.assertEqual(result["total_words"], 12)
        self.assertEqual(result["word_bits"], 32)
        self.assertEqual(result["different_words"], 0)
        self.assertTrue(result["all_words_equal"])
        self.assertTrue(result["complete_logical_word_range_compared"])
        self.assertEqual(result["relative_RMS_ratio"], 0.0)
        self.assertEqual(result["first_changed_coordinates"], [])

    def test_fp64_formulas_and_percent_not_word_percentage(self):
        reference = np.array([1, -2, 3, -4], dtype=np.float32)
        actual = np.array([2, -2, 1, -4], dtype=np.float32)
        result = compare_float_arrays(actual, reference, chunk_words=1)
        self.assertEqual(result["different_words"], 2)
        self.assertEqual(result["first_changed_coordinates"], [[0], [2]])
        self.assertEqual(result["squared_L2_difference"], 5.0)
        self.assertEqual(result["squared_L2_reference"], 30.0)
        self.assertEqual(result["max_abs_difference"], 2.0)
        self.assertEqual(result["mean_absolute_error"], 0.75)
        self.assertAlmostEqual(result["root_mean_square_error"], math.sqrt(5.0 / 4))
        self.assertAlmostEqual(result["relative_RMS_ratio"], math.sqrt(5.0 / 30))
        self.assertEqual(result["relative_RMS_percent"], 100 * result["relative_RMS_ratio"])
        self.assertNotEqual(result["relative_RMS_percent"], 50.0)

    def test_signed_zero_is_bit_difference_not_numeric_error(self):
        for dtype, bits in ((np.float16, 16), (np.float32, 32)):
            with self.subTest(dtype=dtype):
                actual = np.array([-0.0, 1.0], dtype=dtype)
                reference = np.array([0.0, 1.0], dtype=dtype)
                result = compare_float_arrays(actual, reference, chunk_words=1)
                self.assertEqual(result["word_bits"], bits)
                self.assertEqual(result["different_words"], 1)
                self.assertFalse(result["all_words_equal"])
                self.assertTrue(result["numerically_perfect"])
                self.assertEqual(result["max_abs_difference"], 0.0)
                self.assertEqual(result["relative_RMS_ratio"], 0.0)

    def test_half_not_fp32_words_and_no_half_squared_overflow(self):
        reference = np.array([65504, 65504], dtype=np.float16)
        actual = np.array([65472, 65504], dtype=np.float16)
        result = compare_float_arrays(actual, reference, chunk_words=1)
        self.assertEqual(result["word_bits"], 16)
        self.assertEqual(result["total_words"], 2)
        self.assertEqual(result["different_words"], 1)
        self.assertEqual(result["max_abs_difference"], 32.0)
        self.assertEqual(result["squared_L2_difference"], 1024.0)
        self.assertTrue(math.isfinite(result["squared_L2_reference"]))

    def test_strided_negative_views_and_readonly_inputs(self):
        reference = np.arange(48, dtype=np.float32).reshape(6, 8)[:, ::-2]
        actual = reference.copy()
        actual[2, 1] += 0.5
        actual = actual[:, ::-1]
        reference = reference[:, ::-1]
        before = reference.copy()
        actual.flags.writeable = False
        reference.flags.writeable = False
        result = compare_float_arrays(actual, reference, chunk_words=2)
        self.assertEqual(result["total_words"], 24)
        self.assertEqual(result["different_words"], 1)
        self.assertEqual(result["first_changed_coordinates"], [[2, 2]])
        self.assertTrue(np.array_equal(reference, before))
        chunks = []
        for a, b in _paired_chunks(actual, reference, 2):
            self.assertLessEqual(a.size, 2)
            chunks.append(a.copy())
        self.assertEqual(sum(c.size for c in chunks), actual.size)
        self.assertTrue(np.array_equal(np.concatenate(chunks), actual.reshape(-1)))

    def test_big_endian_fp32_logical_words(self):
        reference = np.array([1, 2, 3], dtype=">f4")
        actual = np.array([1, 4, 3], dtype=">f4")
        result = compare_float_arrays(actual, reference, chunk_words=1)
        self.assertEqual(result["different_words"], 1)
        self.assertEqual(result["max_abs_difference"], 2.0)

    def test_zero_reference_clamp_and_json_finite(self):
        result = compare_float_arrays(np.array([1.0], np.float32), np.array([0.0], np.float32))
        self.assertTrue(result["reference_norm_clamped_at_1e_minus30"])
        self.assertAlmostEqual(result["relative_RMS_ratio"] / 1e30, 1.0)
        json.dumps(result, allow_nan=False)

    def test_finite_shape_dtype_and_chunk_gates(self):
        good = np.ones(4, dtype=np.float32)
        for value in (np.nan, np.inf, -np.inf):
            bad = good.copy()
            bad[-1] = value
            with self.assertRaises(ValueError):
                compare_float_arrays(bad, good, chunk_words=1)
            with self.assertRaises(ValueError):
                compare_float_arrays(good, bad, chunk_words=1)
        with self.assertRaises(ValueError):
            compare_float_arrays(good, good.reshape(2, 2))
        with self.assertRaises(ValueError):
            compare_float_arrays(good, good.astype(np.float16))
        with self.assertRaises(TypeError):
            compare_float_arrays(good.astype(np.float64), good.astype(np.float64))
        with self.assertRaises(ValueError):
            compare_float_arrays(good[:0], good[:0])
        for size in (0, MAX_CHUNK_WORDS + 1):
            with self.assertRaises(ValueError):
                compare_float_arrays(good, good, chunk_words=size)
        with self.assertRaises(TypeError):
            compare_float_arrays(good, good, chunk_words=True)
        with self.assertRaises(ValueError):
            compare_float_arrays(good, good, max_coordinates=17)

    def test_first_coordinates_bounded(self):
        reference = np.zeros((10, 3), dtype=np.float32)
        actual = np.ones((10, 3), dtype=np.float32)
        result = compare_float_arrays(actual, reference, chunk_words=2, max_coordinates=3)
        self.assertEqual(result["different_words"], 30)
        self.assertEqual(result["first_changed_coordinates"], [[0, 0], [0, 1], [0, 2]])


class VideoTests(unittest.TestCase):
    def test_perfect_frames_json_safe_infinite_psnr(self):
        frames = np.zeros((2, 2, 3, 3), dtype=np.uint8)
        result = compare_video_uint8(frames, frames.copy(), chunk_words=2)
        self.assertEqual(result["total_RGB_pixels"], 12)
        self.assertEqual(result["total_channel_values"], 36)
        self.assertEqual(result["different_channel_values"], 0)
        self.assertEqual(result["different_RGB_pixels"], 0)
        self.assertIsNone(result["psnr_db"])
        self.assertTrue(result["psnr_perfect"])
        self.assertTrue(result["mean_per_frame_psnr_infinite"])
        self.assertIsNone(result["mean_per_frame_psnr_db"])
        self.assertEqual(result["worst_frame_indices"], [])
        json.dumps(result, allow_nan=False)

    def test_uint8_subtraction_before_squared_error_and_pixel_count(self):
        reference = np.array([[[[255, 0, 0]]]], dtype=np.uint8)
        actual = np.array([[[[0, 255, 0]]]], dtype=np.uint8)
        unchanged = actual.copy()
        result = compare_video_uint8(actual, reference, chunk_words=1)
        self.assertEqual(result["different_channel_values"], 2)
        self.assertEqual(result["different_RGB_pixels"], 1)
        self.assertEqual(result["max_abs_difference"], 255.0)
        self.assertEqual(result["mean_squared_error"], 2 * 255 ** 2 / 3)
        self.assertAlmostEqual(result["psnr_db"], 10 * math.log10(1.5))
        self.assertTrue(np.array_equal(actual, unchanged))

    def test_rgb_grouping_across_chunk_and_frame_boundaries(self):
        reference = np.zeros((2, 1, 2, 3), dtype=np.uint8)
        actual = reference.copy()
        actual[0, 0, 0, 2] = 1
        actual[0, 0, 1, 0] = 2
        actual[1, 0, 1, 1:] = 3
        for size in (1, 2, 4, 5):
            with self.subTest(chunk_words=size):
                result = compare_video_uint8(actual, reference, chunk_words=size)
                self.assertEqual(result["different_channel_values"], 4)
                self.assertEqual(result["different_RGB_pixels"], 3)
                self.assertEqual(result["per_frame"][0]["different_RGB_pixels"], 2)
                self.assertEqual(result["per_frame"][1]["different_RGB_pixels"], 1)
                self.assertEqual(result["first_changed_coordinates"][0], [0, 0, 0, 2])

    def test_partial_perfect_mean_not_substituted_for_overall_psnr(self):
        reference = np.zeros((2, 1, 1, 3), dtype=np.uint8)
        actual = reference.copy()
        actual[1, 0, 0, 0] = 1
        result = compare_video_uint8(actual, reference, chunk_words=2)
        self.assertFalse(result["psnr_perfect"])
        self.assertEqual(result["perfect_frame_count"], 1)
        self.assertTrue(result["mean_per_frame_psnr_infinite"])
        self.assertIsNone(result["mean_per_frame_psnr_db"])
        self.assertEqual(result["worst_frame_indices"], [1])
        self.assertAlmostEqual(result["psnr_db"], 10 * math.log10(255 ** 2 / (1 / 6)))
        self.assertAlmostEqual(result["finite_only_frame_psnr_mean_db"], 10 * math.log10(255 ** 2 / (1 / 3)))
        self.assertNotEqual(result["psnr_db"], result["finite_only_frame_psnr_mean_db"])

    def test_worst_frames_bounded_and_ordered(self):
        reference = np.zeros((13, 1, 1, 3), dtype=np.uint8)
        actual = reference.copy()
        actual[:, 0, 0, 0] = np.arange(1, 14, dtype=np.uint8)
        result = compare_video_uint8(actual, reference, chunk_words=2)
        self.assertEqual(result["worst_frame_indices"], list(range(12, 0, -1)))
        self.assertLessEqual(len(result["worst_frame_indices"]), 12)
        with self.assertRaises(ValueError):
            compare_video_uint8(actual, reference, worst_frames=13)

    def test_video_dtype_and_shape_no_implicit_normalization(self):
        good = np.zeros((1, 2, 2, 3), dtype=np.uint8)
        with self.assertRaises(TypeError):
            compare_video_uint8(good.astype(np.float32), good.astype(np.float32))
        with self.assertRaises(TypeError):
            compare_video_uint8(good[..., :2], good[..., :2])


class AudioTests(unittest.TestCase):
    def test_raw_gain_and_offset_not_fitted_away(self):
        reference = np.array([[1, 2, -1, -2], [2, -2, 1, -1]], dtype=np.float32)
        actual = reference + np.float32(0.5)
        result = compare_audio_float32(actual, reference, chunk_words=3)
        self.assertEqual(result["different_words"], 8)
        self.assertFalse(result["numerically_perfect"])
        self.assertAlmostEqual(result["snr_db"], 10.0)
        self.assertAlmostEqual(result["pearson_correlation"], 1.0)
        self.assertAlmostEqual(result["actual_mean"], 0.5)
        self.assertEqual(len(result["channels"]), 2)
        doubled = compare_audio_float32(reference * 2, reference, chunk_words=1)
        self.assertAlmostEqual(doubled["pearson_correlation"], 1.0)
        self.assertAlmostEqual(doubled["snr_db"], 0.0)
        self.assertGreater(doubled["root_mean_square_error"], 0.0)

    def test_perfect_snr_and_undefined_constant_correlation_json_safe(self):
        reference = np.zeros((2, 3), dtype=np.float32)
        actual = reference.copy()
        actual[0, 0] = np.float32(-0.0)
        result = compare_audio_float32(actual, reference, chunk_words=1)
        self.assertEqual(result["different_words"], 1)
        self.assertFalse(result["all_words_equal"])
        self.assertTrue(result["snr_perfect"])
        self.assertIsNone(result["snr_db"])
        self.assertIsNone(result["pearson_correlation"])
        self.assertFalse(result["correlation_defined"])
        self.assertEqual(result["correlation_undefined_reason"], "constant_actual_and_reference")
        json.dumps(result, allow_nan=False)

    def test_zero_signal_snr_negative_infinity_explicit(self):
        reference = np.zeros((2, 2), dtype=np.float32)
        actual = np.ones((2, 2), dtype=np.float32)
        result = compare_audio_float32(actual, reference, chunk_words=1)
        self.assertIsNone(result["snr_db"])
        self.assertFalse(result["snr_perfect"])
        self.assertTrue(result["snr_reference_zero"])
        self.assertEqual(result["snr_status"], "negative_infinity_zero_reference")
        json.dumps(result, allow_nan=False)

    def test_correlation_no_alignment_search_or_clamping_data(self):
        reference = np.array([[1, 2, 3, 4], [-1, -2, -3, -4]], dtype=np.float32)
        inverted = compare_audio_float32(-reference, reference, chunk_words=1)
        self.assertAlmostEqual(inverted["pearson_correlation"], -1.0)
        shifted = compare_audio_float32(np.roll(reference, 1, axis=1), reference, chunk_words=3)
        self.assertGreater(shifted["different_words"], 0)
        self.assertFalse(shifted["snr_perfect"])
        self.assertTrue(shifted["no_alignment_gain_normalization_or_clipping"])

    def test_audio_f32_stereo_finite_gates(self):
        good = np.ones((2, 3), dtype=np.float32)
        with self.assertRaises(TypeError):
            compare_audio_float32(good.astype(np.float16), good.astype(np.float16))
        with self.assertRaises(TypeError):
            compare_audio_float32(good[:1], good[:1])
        bad = good.copy()
        bad[1, 2] = np.nan
        with self.assertRaises(ValueError):
            compare_audio_float32(bad, good, chunk_words=1)


class RunnerContractTests(unittest.TestCase):
    def receipts(self):
        process = {"exit_code": 0, "all_private_children_exited": True, "cohort_monitor_errors": [], "private_children_after_parent_exit": []}
        lifecycle = {"completed": True, "strict_checks_allranks": TOTAL_CHECKS, "requests": []}
        for index, label in enumerate(LABELS, start=1):
            lifecycle["requests"].append({
                "index": index, "label": label, "completed": True,
                "actual_DiT_evaluations": 49, "strict_checks_allranks": CHECKS_PER_REQUEST,
                "rawframes_sha256": "a" * 64, "rawaudio_sha256": "b" * 64,
                "final_latents_sha256": "c" * 64, "encoder_hidden_file_sha256": "d" * 64,
            })
        return process, lifecycle

    def test_exact_full49_receipt_gate(self):
        process, lifecycle = self.receipts()
        rows = validate_quality_receipts(process, lifecycle)
        self.assertEqual([r["label"] for r in rows], list(LABELS))
        self.assertEqual(TOTAL_CHECKS, 182280)
        self.assertEqual(CHECKS_PER_REQUEST, 60760)

    def test_no_pending_or_failed_model_postprocessing(self):
        process, lifecycle = self.receipts()
        for key, value in (("exit_code", None), ("exit_code", 1), ("exit_code", False), ("all_private_children_exited", False), ("completion_unknown", True), ("cleanup_failure", {"message": "failed"}), ("cohort_monitor_errors", ["error"]), ("private_children_after_parent_exit", [1])):
            with self.subTest(key=key, value=value):
                changed = dict(process)
                changed[key] = value
                with self.assertRaises(ValueError):
                    validate_quality_receipts(changed, lifecycle)

    def test_no_partial_or_relabeled_lifecycle(self):
        process, lifecycle = self.receipts()
        for operation in (
            lambda x: x.update(completed=False),
            lambda x: x.update(strict_checks_allranks=181000),
            lambda x: x["requests"].pop(),
            lambda x: x["requests"][0].update(label=LABELS[1]),
            lambda x: x["requests"][0].update(index=True),
            lambda x: x["requests"][1].update(actual_DiT_evaluations=48),
            lambda x: x["requests"][2].update(strict_checks_allranks=155),
        ):
            changed = copy.deepcopy(lifecycle)
            operation(changed)
            with self.assertRaises(ValueError):
                validate_quality_receipts(process, changed)

    def test_all_raw_artifact_hashes_required_including_own_live_encoder(self):
        process, lifecycle = self.receipts()
        for key in ("rawframes_sha256", "rawaudio_sha256", "final_latents_sha256", "encoder_hidden_file_sha256"):
            changed = copy.deepcopy(lifecycle)
            del changed["requests"][0][key]
            with self.assertRaises(ValueError):
                validate_quality_receipts(process, changed)
        changed = copy.deepcopy(lifecycle)
        changed["requests"][0]["rawframes_sha256"] = "not-a-sha"
        with self.assertRaises(ValueError):
            validate_quality_receipts(process, changed)

    def test_invalid_chunk_rejected_before_file_or_torch_loading(self):
        with patch("compare_quality136._read_json") as reader, patch("compare_quality136._load_latents_cpu") as loader:
            for value in (0, True, MAX_CHUNK_WORDS + 1):
                with self.assertRaises(ValueError):
                    run_quality_comparison(chunk_words=value)
            reader.assert_not_called()
            loader.assert_not_called()

    def test_streaming_whole_file_hash_without_writing_a_file(self):
        value = b"NPY header included\x00whole file bytes" * 11
        with patch.object(Path, "open", return_value=io.BytesIO(value)):
            digest = sha256_file("synthetic.npy", chunk_bytes=3)
        self.assertEqual(digest, hashlib.sha256(value).hexdigest())

    def test_torch_only_lazy_cpu_deserialization_source_contract(self):
        source = (Path(__file__).resolve().parent / "compare_quality136.py").read_text()
        tree = ast.parse(source)
        self.assertFalse(any(isinstance(n, (ast.Import, ast.ImportFrom)) and (
            any(a.name == "torch" or a.name == "torch_npu" for a in n.names) if isinstance(n, ast.Import) else n.module in ("torch", "torch_npu")
        ) for n in tree.body))
        loads = [n for n in ast.walk(tree) if isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute) and isinstance(n.func.value, ast.Name) and n.func.value.id == "torch" and n.func.attr == "load"]
        self.assertEqual(len(loads), 2)
        for call in loads:
            keywords = {k.arg: k.value for k in call.keywords}
            self.assertEqual(ast.literal_eval(keywords["map_location"]), "cpu")
            self.assertIs(ast.literal_eval(keywords["weights_only"]), True)
        self.assertNotIn("import torch_npu", source)
        self.assertNotIn(".generate(", source)
        self.assertIn('"full49_candidate_repeatability_tested": False', source)
        self.assertIn('"own_live_encoder_conditioning_not_teacher_features": True', source)


if __name__ == "__main__":
    unittest.main()
