#!/usr/bin/env python3
"""UI-to-API conversion must keep widget values when their link is null.

A widget input in a ComfyUI UI export carries ``"link": null`` because nothing
is wired into it; its value lives in ``widgets_values``.  Treating that null as
"disconnected" silently produced prompts with missing required inputs.
"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "scripts"))

from h3_workflow import ui_to_api  # noqa: E402


def widget(name: str) -> dict:
    return {"name": name, "type": "STRING", "link": None, "widget": {"name": name}}


OBJECT_INFO = {
    "CLIPLoader": {
        "input": {
            "required": {"clip_name": [["a.safetensors"]], "type": [["minimax"]]},
            "optional": {"device": [["default", "cpu"]]},
        }
    },
    "MiniMaxH3ImageToVideo": {
        "input": {
            "required": {
                "clip": ["CLIP"],
                "prompt": ["STRING", {"default": ""}],
                "width": ["INT", {"default": 864}],
                "height": ["INT", {"default": 480}],
                "length": ["INT", {"default": 124}],
            },
            "optional": {"first_frame": ["IMAGE"]},
        }
    },
}


def workflow() -> dict:
    return {
        "nodes": [
            {
                "id": 1,
                "type": "CLIPLoader",
                "mode": 0,
                "inputs": [widget("clip_name"), widget("type"), widget("device")],
                "widgets_values": ["a.safetensors", "minimax", "default"],
            },
            {
                "id": 2,
                "type": "MiniMaxH3ImageToVideo",
                "mode": 0,
                "inputs": [
                    {"name": "clip", "type": "CLIP", "link": 7},
                    {"name": "prompt", "type": "STRING", "link": None, "widget": {"name": "prompt"}},
                    {"name": "width", "type": "INT", "link": None, "widget": {"name": "width"}},
                    {"name": "height", "type": "INT", "link": None, "widget": {"name": "height"}},
                    {"name": "length", "type": "INT", "link": None, "widget": {"name": "length"}},
                    # Optional image input with a null link: the user removed it.
                    {"name": "first_frame", "type": "IMAGE", "link": None},
                ],
                "widgets_values": ["a prompt", 608, 352, 107],
            },
        ],
        "links": [[7, 1, 0, 2, 0, "CLIP"]],
    }


class UiToApiTest(unittest.TestCase):
    def test_widget_values_survive_null_link(self):
        prompt = ui_to_api(workflow(), OBJECT_INFO)
        self.assertEqual(
            prompt["1"]["inputs"],
            {"clip_name": "a.safetensors", "type": "minimax", "device": "default"},
        )
        self.assertEqual(prompt["2"]["inputs"]["prompt"], "a prompt")
        self.assertEqual(prompt["2"]["inputs"]["width"], 608)
        self.assertEqual(prompt["2"]["inputs"]["length"], 107)

    def test_disconnected_non_widget_input_is_omitted(self):
        prompt = ui_to_api(workflow(), OBJECT_INFO)
        self.assertNotIn("first_frame", prompt["2"]["inputs"])

    def test_connected_input_becomes_a_link(self):
        prompt = ui_to_api(workflow(), OBJECT_INFO)
        self.assertEqual(prompt["2"]["inputs"]["clip"], ["1", 0])


if __name__ == "__main__":
    unittest.main()