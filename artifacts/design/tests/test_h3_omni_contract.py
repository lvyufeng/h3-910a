#!/usr/bin/env python3
"""Protocol tests for the vLLM-Omni compatible H3 service.

These run without aiohttp or ComfyUI: they pin the request rules that
OpenDirector's ``minimax_h3`` adapter depends on.
"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "scripts"))

import h3_omni_contract as contract  # noqa: E402


def image(name: str = "first.png", content_type: str = "image/png", size: int = 1024) -> contract.Upload:
    return contract.Upload(
        field="input_reference", filename=name, content_type=content_type, data=b"x" * size
    )


class FrameGridTest(unittest.TestCase):
    def test_grid_offsets(self):
        for step in range(6, 21):
            frames = contract.GRID * step + contract.GRID_OFFSET
            self.assertIsNone(contract.validate_frames(frames))

    def test_seconds_map_inside_the_grid(self):
        for seconds in range(4, 16):
            frames = contract.frames_for_seconds(seconds)
            self.assertEqual((frames - 5) % 17, 0)
            self.assertLessEqual(abs(frames / 24 - seconds), 0.75)

    def test_seconds_are_clamped(self):
        self.assertEqual(contract.frames_for_seconds(1), 107)
        self.assertEqual(contract.frames_for_seconds(99), 345)

    def test_off_grid_frames_are_refused(self):
        with self.assertRaises(contract.RequestError):
            contract.validate_frames(120)


class SizeTest(unittest.TestCase):
    def test_parse(self):
        self.assertEqual(contract.parse_size("864x480"), (864, 480))
        self.assertEqual(contract.parse_size("1280*736"), (1280, 736))

    def test_unaligned_is_refused(self):
        for value in ("865x480", "864x481", "100x100", "4096x4096"):
            with self.assertRaises(contract.RequestError):
                contract.parse_size(value)

    def test_ratio_defaults(self):
        self.assertEqual(contract.size_for_ratio("9:16"), (480, 864))
        self.assertEqual(contract.size_for_ratio(None), contract.DEFAULT_SIZE)
        with self.assertRaises(contract.RequestError):
            contract.size_for_ratio("7:3")


class ExtraParamsTest(unittest.TestCase):
    def test_empty(self):
        self.assertEqual(contract.parse_extra_params(None), {})

    def test_invalid_json(self):
        with self.assertRaises(contract.RequestError):
            contract.parse_extra_params("{not json")

    def test_unknown_task(self):
        with self.assertRaises(contract.RequestError):
            contract.parse_extra_params('{"task": "i2va"}')

    def test_continuation_is_refused(self):
        with self.assertRaises(contract.RequestError) as caught:
            contract.parse_extra_params('{"continue_from": "vid_1", "task": "t2va"}')
        self.assertEqual(caught.exception.code, "unsupported_continuation")


class BuildRequestTest(unittest.TestCase):
    def test_text_to_video(self):
        spec = contract.build_request(
            {"prompt": "a dragon", "num_frames": "124"}, []
        )
        self.assertEqual(spec.task, "t2va")
        self.assertEqual((spec.width, spec.height, spec.frames), (864, 480, 124))
        self.assertIsNone(spec.image)
        overrides = spec.overrides(filename_prefix="h3-omni/x")
        self.assertEqual(overrides["first_image"], "none")
        self.assertEqual(overrides["last_image"], "none")

    def test_first_frame_upload(self):
        spec = contract.build_request(
            {"prompt": "a dragon", "size": "1280x736", "num_frames": "192"},
            [image()],
        )
        self.assertEqual(spec.task, "fl2va")
        self.assertEqual((spec.width, spec.height), (1280, 736))
        self.assertEqual(spec.overrides(filename_prefix="p")["first_image"], "__uploaded__")

    def test_seconds_are_accepted_when_frames_are_absent(self):
        spec = contract.build_request({"prompt": "x", "seconds": "7"}, [])
        self.assertEqual(spec.frames, contract.frames_for_seconds(7))

    def test_prompt_is_required(self):
        with self.assertRaises(contract.RequestError):
            contract.build_request({"prompt": "   "}, [])

    def test_fps_is_pinned(self):
        with self.assertRaises(contract.RequestError):
            contract.build_request({"prompt": "x", "fps": "30"}, [])

    def test_reference_count_is_limited(self):
        with self.assertRaises(contract.RequestError) as caught:
            contract.build_request(
                {"prompt": "x", "extra_params": '{"task": "ref2va"}'},
                [image("a.png"), image("b.png")],
            )
        self.assertEqual(caught.exception.code, "unsupported_reference_count")

    def test_reference_task_needs_an_image(self):
        with self.assertRaises(contract.RequestError):
            contract.build_request({"prompt": "x", "extra_params": '{"task": "fl2va"}'}, [])

    def test_text_task_rejects_images(self):
        with self.assertRaises(contract.RequestError):
            contract.build_request({"prompt": "x", "extra_params": '{"task": "t2va"}'}, [image()])

    def test_only_image_fields_are_accepted(self):
        audio = contract.Upload(field="audio", filename="a.wav", content_type="audio/wav", data=b"x")
        with self.assertRaises(contract.RequestError):
            contract.build_request({"prompt": "x"}, [audio])

    def test_oversized_image_is_refused(self):
        big = image(size=30 * 1024 * 1024 + 1)
        with self.assertRaises(contract.RequestError):
            contract.build_request({"prompt": "x"}, [big])

    def test_seed_is_optional(self):
        self.assertIsNone(contract.build_request({"prompt": "x"}, []).seed)
        self.assertEqual(contract.build_request({"prompt": "x", "seed": "42"}, []).seed, 42)


class DocumentTest(unittest.TestCase):
    def test_capabilities_match_the_opendirector_contract(self):
        payload = contract.models_payload("minimax-h3")
        self.assertEqual(payload["object"], "list")
        model = payload["data"][0]
        self.assertEqual(model["id"], "minimax-h3")
        capabilities = model["videoCapabilities"]
        self.assertEqual(capabilities["dimensionStep"], 32)
        self.assertTrue(4 <= capabilities["defaultDuration"] <= 15)
        names = [item["resolution"] for item in capabilities["outputs"]]
        self.assertEqual(len(names), len(set(names)))
        self.assertIn(capabilities["defaultResolution"], names)
        for item in capabilities["outputs"]:
            self.assertEqual(item["width"] % capabilities["dimensionStep"], 0)
            self.assertEqual(item["height"] % capabilities["dimensionStep"], 0)
            self.assertTrue(item["durations"])
            for duration in item["durations"]:
                self.assertIsInstance(duration, int)
                self.assertTrue(1 <= duration <= 60)

    def test_status_payload(self):
        self.assertEqual(
            contract.status_payload("vid_1", contract.STATUS_RUNNING),
            {"id": "vid_1", "status": "in_progress"},
        )
        failed = contract.status_payload("vid_1", contract.STATUS_FAILED, error="boom")
        self.assertEqual(failed["error"]["message"], "boom")

    def test_error_payload_carries_both_spellings(self):
        error = contract.RequestError("nope", status=422, code="bad")
        payload = error.payload()
        self.assertEqual(payload["detail"], "nope")
        self.assertEqual(payload["error"], {"code": "bad", "message": "nope"})


if __name__ == "__main__":
    unittest.main()