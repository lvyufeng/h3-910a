#!/usr/bin/env python3
"""Local HTTP-only tests for scripts/benchmark_h3_api.py.

The tests use a synthetic ComfyUI server and a fake ffprobe. They exercise the
client protocol, terminal-state handling, media contract, dual submission, and
result files without touching an NPU or a real model.
"""

from __future__ import annotations

import json
import os
import socket
import subprocess
import sys
import tempfile
import threading
import time
import unittest
import urllib.parse
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[3]
BENCHMARK = ROOT / "scripts" / "benchmark_h3_api.py"


def api_workflow() -> dict[str, Any]:
    return {
        "1": {
            "class_type": "UNETLoader",
            "inputs": {"unet_name": "original-dit.safetensors"},
        },
        "2": {
            "class_type": "CLIPLoader",
            "inputs": {"clip_name": "original-clip.safetensors"},
        },
        "3": {
            "class_type": "VAELoader",
            "inputs": {"vae_name": "original-video-vae.safetensors"},
        },
        "4": {
            "class_type": "VAELoader",
            "inputs": {"vae_name": "original-audio-vae.safetensors"},
        },
        "5": {
            "class_type": "MiniMaxH3ImageToVideo",
            "inputs": {
                "prompt": "original",
                "width": 832,
                "height": 480,
                "length": 82,
                "first_frame": ["6", 0],
                "last_frame": ["7", 0],
            },
        },
        "6": {
            "class_type": "LoadImage",
            "inputs": {"image": "first.png"},
        },
        "7": {
            "class_type": "LoadImage",
            "inputs": {"image": "last.png"},
        },
        "8": {
            "class_type": "MultiNPUParallelConfig",
            "inputs": {
                "npu_count": "2",
                "video_vae": ["3", 0],
                "audio_vae": ["4", 0],
            },
        },
        "9": {
            "class_type": "BasicScheduler",
            "inputs": {"steps": 8},
        },
        "10": {
            "class_type": "RandomNoise",
            "inputs": {"noise_seed": 1},
        },
        "11": {
            "class_type": "SaveVideo",
            "inputs": {"filename_prefix": "original"},
        },
        "12": {
            "class_type": "MiniMaxH3TurboLoRA",
            "inputs": {"lora_name": "original-lora.safetensors"},
        },
    }


def ui_workflow() -> dict[str, Any]:
    return {
        "nodes": [
            {
                "id": 1,
                "type": "UNETLoader",
                "inputs": [{"name": "unet_name", "widget": {"name": "unet_name"}}],
                "widgets_values": ["original-dit.safetensors"],
            },
            {
                "id": 2,
                "type": "CLIPLoader",
                "inputs": [{"name": "clip_name", "widget": {"name": "clip_name"}}],
                "widgets_values": ["original-clip.safetensors"],
            },
            {
                "id": 3,
                "type": "VAELoader",
                "inputs": [{"name": "vae_name", "widget": {"name": "vae_name"}}],
                "widgets_values": ["original-video-vae.safetensors"],
            },
            {
                "id": 4,
                "type": "VAELoader",
                "inputs": [{"name": "vae_name", "widget": {"name": "vae_name"}}],
                "widgets_values": ["original-audio-vae.safetensors"],
            },
            {
                "id": 5,
                "type": "MiniMaxH3ImageToVideo",
                "inputs": [
                    {"name": "prompt", "widget": {"name": "prompt"}},
                    {"name": "width", "widget": {"name": "width"}},
                    {"name": "height", "widget": {"name": "height"}},
                    {"name": "length", "widget": {"name": "length"}},
                    {"name": "first_frame", "link": 1},
                    {"name": "last_frame", "link": 2},
                ],
                "widgets_values": ["", 832, 480, 82],
            },
            {
                "id": 6,
                "type": "LoadImage",
                "inputs": [{"name": "image", "widget": {"name": "image"}}],
                "outputs": [{"name": "IMAGE", "type": "IMAGE", "links": [1]}],
                "widgets_values": ["first.png", "image"],
            },
            {
                "id": 7,
                "type": "LoadImage",
                "inputs": [{"name": "image", "widget": {"name": "image"}}],
                "outputs": [{"name": "IMAGE", "type": "IMAGE", "links": [2]}],
                "widgets_values": ["last.png", "image"],
            },
            {
                "id": 8,
                "type": "MultiNPUParallelConfig",
                "inputs": [{"name": "npu_count", "widget": {"name": "npu_count"}}],
                "widgets_values": ["2"],
            },
            {
                "id": 9,
                "type": "BasicScheduler",
                "inputs": [{"name": "steps", "widget": {"name": "steps"}}],
                "widgets_values": ["beta", 8, 1.0],
            },
            {
                "id": 10,
                "type": "RandomNoise",
                "inputs": [{"name": "noise_seed", "widget": {"name": "noise_seed"}}],
                "widgets_values": [1, "fixed"],
            },
            {
                "id": 11,
                "type": "SaveVideo",
                "inputs": [
                    {"name": "filename_prefix", "widget": {"name": "filename_prefix"}}
                ],
                "widgets_values": ["original"],
            },
        ],
        "links": [
            [1, 6, 0, 5, 4, "IMAGE"],
            [2, 7, 0, 5, 5, "IMAGE"],
        ],
    }


def object_info() -> dict[str, Any]:
    specs: dict[str, dict[str, Any]] = {}
    for node_type, required in {
        "UNETLoader": {"unet_name": [[], {}]},
        "CLIPLoader": {"clip_name": [[], {}]},
        "VAELoader": {"vae_name": [[], {}]},
        "MiniMaxH3ImageToVideo": {
            "prompt": ["STRING", {}],
            "width": ["INT", {}],
            "height": ["INT", {}],
            "length": ["INT", {}],
            "first_frame": ["IMAGE", {}],
            "last_frame": ["IMAGE", {}],
        },
        "MultiNPUParallelConfig": {"npu_count": [["1", "2", "4"], {}]},
        "BasicScheduler": {"steps": ["INT", {}]},
        "RandomNoise": {"noise_seed": ["INT", {}]},
        "SaveVideo": {"filename_prefix": ["STRING", {}]},
        "LoadImage": {"image": [[], {}]},
    }.items():
        specs[node_type] = {"input": {"required": required}}
    return specs


FAKE_FFPROBE = r"""#!/usr/bin/env python3
import json
import sys
from pathlib import Path

name = Path(sys.argv[-1]).name
if "probe-fail" in name:
    print("synthetic ffprobe failure", file=sys.stderr)
    raise SystemExit(1)

video = {
    "index": 0,
    "codec_type": "video",
    "codec_name": "h264",
    "width": 864,
    "height": 480,
    "r_frame_rate": "24/1",
    "avg_frame_rate": "24/1",
    "nb_frames": "124",
    "nb_read_frames": "124",
}
audio = {
    "index": 1,
    "codec_type": "audio",
    "codec_name": "aac",
    "sample_rate": "32000",
    "channels": 2,
    "channel_layout": "stereo",
}
if "wrong-width" in name:
    video["width"] = 832
if "wrong-frames" in name:
    video["nb_frames"] = "123"
    video["nb_read_frames"] = "123"
if "wrong-fps" in name:
    video["r_frame_rate"] = "25/1"
    video["avg_frame_rate"] = "25/1"
if "mono" in name:
    audio["channels"] = 1
    audio["channel_layout"] = "mono"
streams = [video, audio]
if "no-audio" in name:
    streams = [video]
print(json.dumps({
    "format": {"format_name": "mov,mp4", "duration": "5.166667"},
    "streams": streams,
}))
"""


class FakeComfyState:
    def __init__(self) -> None:
        self.history_delay = 0.02
        self.history_status = "success"
        self.history_available = True
        self.prompt_status = 200
        self.node_errors: dict[str, Any] | None = None
        self.artifact_filenames = ["good.mp4"]
        self.view_status = 200
        self.view_body = b"synthetic-video-bytes"
        self.object_info_status = 200
        self.prompts: list[dict[str, Any]] = []
        self.prompt_receive_times: list[float] = []
        self.view_requests: list[str] = []
        self.created_at = time.monotonic()
        self.next_id = 0
        self.lock = threading.Lock()
        self.history: dict[str, dict[str, Any]] = {}
        self.history_available_at: dict[str, float] = {}

    def new_prompt_id(self) -> str:
        with self.lock:
            self.next_id += 1
            return f"prompt-{self.next_id:04d}"


class FakeComfyHandler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"
    server_version = "FakeComfy/1"

    @property
    def state(self) -> FakeComfyState:
        return self.server.fake_state  # type: ignore[attr-defined]

    def log_message(self, _format: str, *_args: Any) -> None:
        return

    def send_json(self, status: int, payload: Any) -> None:
        body = json.dumps(payload).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Connection", "close")
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self) -> None:
        parsed = urllib.parse.urlsplit(self.path)
        if parsed.path == "/object_info":
            if self.state.object_info_status != 200:
                self.send_json(self.state.object_info_status, {"error": "object_info"})
            else:
                self.send_json(200, object_info())
            return
        if parsed.path == "/queue":
            running = []
            for prompt_id, available_at in self.state.history_available_at.items():
                if time.monotonic() < available_at:
                    running.append([0, prompt_id, {}, {}, []])
            if not self.state.history_available:
                running = []
            self.send_json(200, {"queue_running": running, "queue_pending": []})
            return
        if parsed.path.startswith("/history/"):
            prompt_id = urllib.parse.unquote(parsed.path.rsplit("/", 1)[1])
            available_at = self.state.history_available_at.get(prompt_id)
            if (
                not self.state.history_available
                or available_at is None
                or time.monotonic() < available_at
            ):
                self.send_json(200, {})
                return
            history = self.state.history.get(prompt_id, {})
            self.send_json(200, {prompt_id: history})
            return
        if parsed.path == "/view":
            query = urllib.parse.parse_qs(parsed.query)
            self.state.view_requests.append(query.get("filename", [""])[0])
            if self.state.view_status != 200:
                self.send_json(self.state.view_status, {"error": "view"})
                return
            body = self.state.view_body
            self.send_response(200)
            self.send_header("Content-Type", "video/mp4")
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Connection", "close")
            self.end_headers()
            self.wfile.write(body)
            return
        self.send_json(404, {"error": "not found", "path": parsed.path})

    def do_POST(self) -> None:
        parsed = urllib.parse.urlsplit(self.path)
        length = int(self.headers.get("Content-Length", "0"))
        raw = self.rfile.read(length)
        if parsed.path != "/prompt":
            self.send_json(404, {"error": "not found"})
            return
        payload = json.loads(raw.decode("utf-8"))
        with self.state.lock:
            self.state.prompt_receive_times.append(time.monotonic())
            self.state.prompts.append(payload)
        if self.state.prompt_status != 200:
            self.send_json(self.state.prompt_status, {"error": "prompt rejected"})
            return
        prompt_id = self.state.new_prompt_id()
        response: dict[str, Any] = {"prompt_id": prompt_id, "number": 1}
        if self.state.node_errors is not None:
            response["node_errors"] = self.state.node_errors
        status = {"status_str": self.state.history_status, "completed": True}
        if self.state.history_status == "success":
            status = {"status_str": "success", "completed": True, "messages": []}
        elif self.state.history_status == "statusless":
            status = {}
        elif self.state.history_status == "completed-false":
            status = {"status_str": "running", "completed": False}
        elif self.state.history_status == "interrupted":
            status = {"status_str": "interrupted", "completed": False}
        outputs = {
            "92": {
                "videos": [
                    {"filename": name, "subfolder": "", "type": "output"}
                    for name in self.state.artifact_filenames
                ]
            }
        }
        self.state.history[prompt_id] = {"status": status, "outputs": outputs}
        self.state.history_available_at[prompt_id] = (
            time.monotonic() + self.state.history_delay
        )
        self.send_json(200, response)


class FakeComfy:
    def __init__(self) -> None:
        self.state = FakeComfyState()
        self.httpd = ThreadingHTTPServer(("127.0.0.1", 0), FakeComfyHandler)
        self.httpd.fake_state = self.state  # type: ignore[attr-defined]
        self.thread = threading.Thread(target=self.httpd.serve_forever, daemon=True)

    @property
    def url(self) -> str:
        host, port = self.httpd.server_address[:2]
        return f"http://{host}:{port}"

    def start(self) -> "FakeComfy":
        self.thread.start()
        return self

    def close(self) -> None:
        self.httpd.shutdown()
        self.httpd.server_close()
        self.thread.join(timeout=5)


class BenchmarkHTTPTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.fake_bin = tempfile.TemporaryDirectory(prefix="h3-fake-bin-")
        fake_probe = Path(cls.fake_bin.name) / "ffprobe"
        fake_probe.write_text(FAKE_FFPROBE, encoding="utf-8")
        fake_probe.chmod(0o755)

    @classmethod
    def tearDownClass(cls) -> None:
        cls.fake_bin.cleanup()

    def setUp(self) -> None:
        self.temp = tempfile.TemporaryDirectory(prefix="h3-benchmark-test-")
        self.temp_path = Path(self.temp.name)
        self.api_path = self.temp_path / "api-workflow.json"
        self.ui_path = self.temp_path / "ui-workflow.json"
        self.api_path.write_text(json.dumps(api_workflow()), encoding="utf-8")
        self.ui_path.write_text(json.dumps(ui_workflow()), encoding="utf-8")
        self.servers: list[FakeComfy] = []
        self.counter = 0

    def tearDown(self) -> None:
        for server in self.servers:
            server.close()
        self.temp.cleanup()

    def server(self) -> FakeComfy:
        server = FakeComfy().start()
        self.servers.append(server)
        return server

    def run_benchmark(
        self,
        workflow: Path,
        servers: list[tuple[str, str]],
        *,
        mode: str = "single",
        extra: list[str] | None = None,
        timeout: float = 0.8,
    ) -> tuple[subprocess.CompletedProcess[str], dict[str, Any], Path]:
        self.counter += 1
        run_label = f"case-{self.counter}"
        output_root = self.temp_path / "output"
        command = [
            sys.executable,
            str(BENCHMARK),
            "--workflow",
            str(workflow),
            "--output-root",
            str(output_root),
            "--run-label",
            run_label,
            "--mode",
            mode,
            "--timeout",
            str(timeout),
            "--request-timeout",
            "2",
            "--poll-interval",
            "0.01",
            "--width",
            "864",
            "--height",
            "480",
            "--frames",
            "124",
            "--steps",
            "4",
            "--npu-count",
            "4",
            "--seed",
            "7",
            "--prompt",
            "synthetic test prompt",
            "--first-image",
            "none",
            "--last-image",
            "none",
        ]
        for label, url in servers:
            command.extend(["--server", f"{label}={url}"])
        command.extend(extra or [])
        env = os.environ.copy()
        env["PYTHONDONTWRITEBYTECODE"] = "1"
        env["PATH"] = str(Path(self.fake_bin.name)) + os.pathsep + env.get("PATH", "")
        completed = subprocess.run(
            command,
            cwd=ROOT,
            env=env,
            capture_output=True,
            text=True,
            timeout=20,
            check=False,
        )
        run_dir = output_root / run_label
        report = json.loads((run_dir / "result.json").read_text(encoding="utf-8"))
        return completed, report, run_dir

    def test_single_api_success_and_result_contract(self) -> None:
        server = self.server()
        completed, report, run_dir = self.run_benchmark(
            self.api_path,
            [("a", server.url)],
            extra=[
                "--model",
                "chosen-dit.safetensors",
                "--clip",
                "chosen-clip.safetensors",
                "--video-vae",
                "chosen-video-vae.safetensors",
                "--audio-vae",
                "chosen-audio-vae.safetensors",
                "--lora",
                "chosen-lora.safetensors",
            ],
        )

        self.assertEqual(completed.returncode, 0, completed.stderr)
        self.assertTrue(report["overall_success"])
        self.assertEqual(len(report["results"]), 1)
        result = report["results"][0]
        self.assertTrue(result["success"])
        self.assertTrue(result["completed"])
        self.assertEqual(result["history_state"], "success")
        self.assertTrue(result["media"]["evaluated"])
        self.assertTrue(result["media"]["video_ok"])
        self.assertTrue(result["media"]["audio_ok"])
        self.assertTrue(result["media"]["audio_format_ok"])
        self.assertEqual(
            result["media"]["matching_video"],
            {"width": 864, "height": 480, "frames": 124, "fps": 24.0},
        )
        self.assertEqual(result["media"]["audio_observed"][0]["sample_rate"], 32000)
        self.assertEqual(result["media"]["audio_observed"][0]["channels"], 2)
        self.assertEqual(len(result["artifacts"]), 1)
        self.assertGreater(result["artifacts"][0]["size"], 0)
        self.assertRegex(result["artifacts"][0]["sha256"], r"^[0-9a-f]{64}$")
        self.assertTrue((run_dir / "manifest.json").is_file())
        self.assertTrue((run_dir / "result.json").is_file())
        self.assertTrue((run_dir / "single" / "a" / "result.json").is_file())

        submitted = json.loads(
            (run_dir / "single" / "a" / "submitted-workflow.json").read_text(
                encoding="utf-8"
            )
        )
        video_inputs = submitted["5"]["inputs"]
        self.assertEqual(
            (
                video_inputs["width"],
                video_inputs["height"],
                video_inputs["length"],
            ),
            (864, 480, 124),
        )
        self.assertNotIn("first_frame", video_inputs)
        self.assertNotIn("last_frame", video_inputs)
        self.assertEqual(submitted["8"]["inputs"]["npu_count"], "4")
        self.assertEqual(submitted["9"]["inputs"]["steps"], 4)
        self.assertEqual(
            submitted["1"]["inputs"]["unet_name"], "chosen-dit.safetensors"
        )
        self.assertEqual(
            submitted["3"]["inputs"]["vae_name"],
            "chosen-video-vae.safetensors",
        )
        self.assertEqual(
            submitted["4"]["inputs"]["vae_name"],
            "chosen-audio-vae.safetensors",
        )
        self.assertEqual(server.state.prompts[0]["prompt"], submitted)
        self.assertEqual(server.state.view_requests, ["good.mp4"])

    def test_ui_workflow_uses_object_info_and_converts(self) -> None:
        server = self.server()
        completed, report, run_dir = self.run_benchmark(
            self.ui_path, [("a", server.url)]
        )

        self.assertEqual(completed.returncode, 0, completed.stderr)
        self.assertTrue(report["overall_success"])
        result = report["results"][0]
        self.assertEqual(result["workflow"]["conversion"], "ui-to-api")
        paths = [record["path"] for record in result["metrics"]["requests"]]
        self.assertIn("/object_info", paths)
        self.assertIn("/prompt", paths)
        submitted = json.loads(
            (run_dir / "single" / "a" / "submitted-workflow.json").read_text(
                encoding="utf-8"
            )
        )
        self.assertTrue(
            all("class_type" in node for node in submitted.values())
        )
        self.assertEqual(submitted["5"]["inputs"]["width"], 864)
        self.assertNotIn("first_frame", submitted["5"]["inputs"])
        self.assertNotIn("last_frame", submitted["5"]["inputs"])

    def test_node_errors_fail_even_with_prompt_id(self) -> None:
        server = self.server()
        server.state.node_errors = {
            "5": {
                "errors": [
                    {
                        "type": "value_not_valid",
                        "message": "synthetic invalid model",
                    }
                ],
                "class_type": "MiniMaxH3ImageToVideo",
            }
        }
        completed, report, _ = self.run_benchmark(
            self.api_path, [("a", server.url)]
        )

        self.assertEqual(completed.returncode, 1)
        result = report["results"][0]
        self.assertFalse(result["success"])
        self.assertEqual(result["prompt_id"], "prompt-0001")
        self.assertTrue(
            any("node_errors" in error for error in result["errors"]),
            result["errors"],
        )
        self.assertFalse(
            any(record["path"] == "/history/prompt-0001" for record in result["metrics"]["requests"])
        )

    def test_media_contract_rejects_bad_observations(self) -> None:
        cases = {
            "wrong-width.mp4": "width/height/frames/fps",
            "wrong-frames.mp4": "width/height/frames/fps",
            "wrong-fps.mp4": "width/height/frames/fps",
            "mono.mp4": "32000 Hz / 2 channels / stereo",
            "no-audio.mp4": "no ffprobe-confirmed audio stream",
            "probe-fail.mp4": "ffprobe failed",
        }
        for filename, expected_error in cases.items():
            with self.subTest(filename=filename):
                server = self.server()
                server.state.artifact_filenames = [filename]
                completed, report, _ = self.run_benchmark(
                    self.api_path, [("a", server.url)]
                )
                self.assertEqual(completed.returncode, 1)
                result = report["results"][0]
                self.assertFalse(result["success"], result)
                errors = "\n".join(result["errors"])
                self.assertIn(expected_error, errors)
                self.assertFalse(result["media_ok"])

    def test_no_ffprobe_is_explicitly_degraded_and_cannot_pass(self) -> None:
        server = self.server()
        completed, report, _ = self.run_benchmark(
            self.api_path,
            [("a", server.url)],
            extra=[
                "--no-ffprobe",
                "--no-require-video",
                "--no-require-audio",
                "--no-require-media-metadata",
            ],
        )

        self.assertEqual(completed.returncode, 1)
        result = report["results"][0]
        self.assertTrue(result["completed"])
        self.assertTrue(result["artifact_ok"])
        self.assertFalse(result["media"]["evaluated"])
        self.assertFalse(result["media_ok"])
        self.assertTrue(result["contract_only"])
        self.assertFalse(result["success"])
        self.assertTrue(
            any("degraded HTTP-only mode" in warning for warning in result["warnings"])
        )

    def test_history_record_without_explicit_success_times_out(self) -> None:
        server = self.server()
        server.state.history_status = "statusless"
        completed, report, _ = self.run_benchmark(
            self.api_path, [("a", server.url)], timeout=0.15
        )

        self.assertEqual(completed.returncode, 1)
        result = report["results"][0]
        self.assertFalse(result["completed"])
        self.assertEqual(result["history_state"], "pending")
        self.assertGreater(result["poll_count"], 1)
        self.assertTrue(
            any("explicit success" in error for error in result["errors"]),
            result["errors"],
        )

    def test_history_error_terminal_is_rejected(self) -> None:
        server = self.server()
        server.state.history_status = "interrupted"
        completed, report, _ = self.run_benchmark(
            self.api_path, [("a", server.url)]
        )

        self.assertEqual(completed.returncode, 1)
        result = report["results"][0]
        self.assertFalse(result["completed"])
        self.assertEqual(result["history_state"], "error")
        self.assertTrue(
            any("execution error" in error for error in result["errors"])
        )

    def test_zero_history_grace_fails_after_queue_disappears(self) -> None:
        server = self.server()
        server.state.history_available = False
        completed, report, _ = self.run_benchmark(
            self.api_path,
            [("a", server.url)],
            timeout=2,
            extra=["--history-grace", "0"],
        )

        self.assertEqual(completed.returncode, 1)
        result = report["results"][0]
        self.assertTrue(result["history_grace_used"])
        self.assertFalse(result["completed"])
        self.assertTrue(
            any("grace window" in error for error in result["errors"]),
            result["errors"],
        )

    def test_timeout_while_queue_is_active_is_recorded(self) -> None:
        server = self.server()
        server.state.history_delay = 10
        completed, report, _ = self.run_benchmark(
            self.api_path, [("a", server.url)], timeout=0.12
        )

        self.assertEqual(completed.returncode, 1)
        result = report["results"][0]
        self.assertFalse(result["completed"])
        self.assertFalse(result["history_grace_used"])
        self.assertEqual(result["final_queue_state"], "queue_running")
        self.assertTrue(result["poll_events"])

    def test_connection_failure_still_writes_valid_json(self) -> None:
        with socket.socket() as sock:
            sock.bind(("127.0.0.1", 0))
            unused_port = sock.getsockname()[1]
        completed, report, run_dir = self.run_benchmark(
            self.api_path, [("a", f"http://127.0.0.1:{unused_port}")]
        )

        self.assertEqual(completed.returncode, 1)
        self.assertFalse(report["overall_success"])
        result = report["results"][0]
        self.assertFalse(result["success"])
        self.assertTrue(result["errors"])
        self.assertTrue((run_dir / "single" / "a" / "result.json").is_file())

    def test_download_failure_is_failure_evidence(self) -> None:
        server = self.server()
        server.state.view_status = 503
        completed, report, _ = self.run_benchmark(
            self.api_path, [("a", server.url)]
        )

        self.assertEqual(completed.returncode, 1)
        result = report["results"][0]
        self.assertTrue(result["completed"])
        self.assertFalse(result["transport_ok"])
        self.assertFalse(result["artifact_ok"])
        self.assertEqual(result["artifacts"][0]["download_status"], "FAIL")
        self.assertFalse(result["success"])

    def test_single_with_two_servers_runs_in_server_order(self) -> None:
        first = self.server()
        second = self.server()
        completed, report, _ = self.run_benchmark(
            self.api_path,
            [("a", first.url), ("b", second.url)],
            mode="single",
        )

        self.assertEqual(completed.returncode, 0, completed.stderr)
        self.assertEqual(len(report["results"]), 2)
        self.assertTrue(all(result["success"] for result in report["results"]))
        self.assertGreater(second.state.prompt_receive_times[0], first.state.prompt_receive_times[0])

    def test_dual_submission_has_barrier_skew_and_overlap(self) -> None:
        a = self.server()
        b = self.server()
        completed, report, run_dir = self.run_benchmark(
            self.api_path,
            [("a", a.url), ("b", b.url)],
            mode="dual",
        )

        self.assertEqual(completed.returncode, 0, completed.stderr)
        self.assertTrue(report["overall_success"])
        self.assertEqual(len(report["results"]), 2)
        for result in report["results"]:
            self.assertTrue(result["success"])
            self.assertIsNotNone(result["submission_skew_seconds"])
            self.assertLess(result["submission_skew_seconds"], 0.5)
            self.assertGreater(result["dual_overlap_seconds"], 0)
            self.assertIsNotNone(result["dual_wall_seconds"])
            self.assertTrue(
                (run_dir / "dual" / result["label"] / "result.json").is_file()
            )
        receive_skew = abs(
            a.state.prompt_receive_times[0] - b.state.prompt_receive_times[0]
        )
        self.assertLess(receive_skew, 0.5)
        self.assertIsNotNone(report["dual_wall_seconds"])

    def test_dual_preparation_failure_releases_peer_barrier(self) -> None:
        a = self.server()
        b = self.server()
        a.state.object_info_status = 500
        completed, report, _ = self.run_benchmark(
            self.ui_path,
            [("a", a.url), ("b", b.url)],
            mode="dual",
        )

        self.assertEqual(completed.returncode, 1)
        self.assertFalse(report["overall_success"])
        self.assertEqual(len(report["results"]), 2)
        by_label = {result["label"]: result for result in report["results"]}
        self.assertTrue(
            any("HTTP 500" in error for error in by_label["a"]["errors"]),
            by_label["a"]["errors"],
        )
        self.assertTrue(
            any("barrier" in error for error in by_label["b"]["errors"]),
            by_label["b"]["errors"],
        )
        self.assertFalse(any(result["success"] for result in report["results"]))
        self.assertIsNone(report["dual_wall_seconds"])

    def test_dual_prompt_http_failure_terminates_both_workers(self) -> None:
        a = self.server()
        b = self.server()
        a.state.prompt_status = 500
        completed, report, _ = self.run_benchmark(
            self.api_path,
            [("a", a.url), ("b", b.url)],
            mode="dual",
        )

        self.assertEqual(completed.returncode, 1)
        self.assertEqual(len(report["results"]), 2)
        by_label = {result["label"]: result for result in report["results"]}
        self.assertFalse(by_label["a"]["success"])
        self.assertTrue(by_label["b"]["success"], by_label["b"]["errors"])
        self.assertTrue(
            any("HTTP 500" in error for error in by_label["a"]["errors"])
        )
        self.assertIsNotNone(report["dual_wall_seconds"])


if __name__ == "__main__":
    unittest.main(verbosity=2)
