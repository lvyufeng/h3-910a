#!/usr/bin/env python3
"""Submit MiniMax H3 ComfyUI workflows and measure one- and two-instance runs.

The script intentionally uses only the Python standard library. It does not
start ComfyUI or inspect a remote filesystem.

Examples:
  python3 scripts/benchmark_h3_api.py \
    --server one=http://127.0.0.1:8801 \
    --server two=http://127.0.0.1:8802 \
    --workflow /mnt/workspace/h3-ascend/ts217/instance-1/workflows/base.json \
    --mode both --width 864 --height 480 --frames 124 --steps 4 \
    --output-root /mnt/workspace/h3-ascend/ts217/runs/0.4mp

  python3 scripts/benchmark_h3_api.py \
    --server one=http://127.0.0.1:8801 \
    --workflow workflow.json --mode single --npu-count 4

Distinguish execution success from contract success:
  - completed: ComfyUI returned a terminal history record without an error
  - artifact_ok: at least one non-empty output was downloaded and verified
  - transport_ok: all artifact downloads completed
  - media_ok: required video/audio streams and requested media metadata match
  - success: completed AND artifact_ok AND transport_ok AND media_ok

--no-ffprobe is an explicit fixture/contract-only mode. It requires all three
--no-require-* flags, sets media_ok=false, and reports contract_only=true when
the HTTP submission, terminal history, and downloaded file contract pass.
Contract-only success is not a real media acceptance result.

For UI workflows the script reads each server's /object_info and converts the
workflow independently. An API workflow is submitted as-is after overrides.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from h3_workflow import (
    Overrides,
    WorkflowError,
    apply_overrides,
    detect_format,
    dump_workflow,
    load_workflow,
    ui_to_api,
    validate_api,
)


DEFAULT_PROMPT = (
    "A single static cinematic shot of a red ceramic cup on a wooden table. "
    "Soft window light, no text, no camera movement."
)
ARTIFACT_KEYS = ("images", "gifs", "videos", "audio", "files")


class BenchmarkError(RuntimeError):
    """Raised for a failed benchmark step."""


class HTTPClient:
    def __init__(self, base_url: str, timeout: float) -> None:
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self.records: list[dict[str, Any]] = []

    def _record(
        self,
        *,
        method: str,
        path: str,
        kind: str,
        started_monotonic: float,
        started_epoch: float,
        http_status: int | None,
        error: str | None,
        bytes_received: int | None = None,
    ) -> None:
        self.records.append(
            {
                "method": method,
                "path": path,
                "kind": kind,
                "started_monotonic": round(started_monotonic, 6),
                "started_epoch": round(started_epoch, 6),
                "duration_seconds": round(
                    time.monotonic() - started_monotonic, 6
                ),
                "http_status": http_status,
                "outcome": "OK" if error is None else "FAIL",
                "error": error,
                "bytes_received": bytes_received,
            }
        )

    def request(
        self,
        method: str,
        path: str,
        *,
        payload: dict[str, Any] | None = None,
        accept_json: bool = True,
    ) -> Any:
        url = self.base_url + path
        data = None
        headers = {"Accept": "application/json" if accept_json else "*/*"}
        if payload is not None:
            data = json.dumps(payload, ensure_ascii=False).encode("utf-8")
            headers["Content-Type"] = "application/json"
        request = urllib.request.Request(url, data=data, headers=headers, method=method)
        started_monotonic = time.monotonic()
        started_epoch = time.time()
        http_status: int | None = None
        bytes_received: int | None = None
        error_text: str | None = None
        try:
            with urllib.request.urlopen(request, timeout=self.timeout) as response:
                http_status = getattr(response, "status", 200)
                body = response.read()
                bytes_received = len(body)
                if not accept_json:
                    return body
                return json.loads(body.decode("utf-8"))
        except urllib.error.HTTPError as exc:
            http_status = exc.code
            detail = exc.read().decode("utf-8", errors="replace")
            error_text = f"HTTP {exc.code}: {detail[:2000]}"
            raise BenchmarkError(
                f"{method} {url} -> HTTP {exc.code}: {detail[:2000]}"
            ) from exc
        except (urllib.error.URLError, TimeoutError, json.JSONDecodeError) as exc:
            error_text = f"{type(exc).__name__}: {exc}"
            raise BenchmarkError(f"{method} {url} failed: {exc}") from exc
        finally:
            self._record(
                method=method,
                path=path,
                kind="json",
                started_monotonic=started_monotonic,
                started_epoch=started_epoch,
                http_status=http_status,
                error=error_text,
                bytes_received=bytes_received,
            )

    def download(self, path: str, target: Path) -> int:
        url = self.base_url + path
        target.parent.mkdir(parents=True, exist_ok=True)
        partial = target.with_name(target.name + ".part")
        started_monotonic = time.monotonic()
        started_epoch = time.time()
        http_status: int | None = None
        bytes_received = 0
        error_text: str | None = None
        try:
            with urllib.request.urlopen(url, timeout=self.timeout) as response:
                http_status = getattr(response, "status", 200)
                with partial.open("wb") as output:
                    while True:
                        chunk = response.read(1024 * 1024)
                        if not chunk:
                            break
                        output.write(chunk)
                        bytes_received += len(chunk)
            partial.replace(target)
        except urllib.error.HTTPError as exc:
            partial.unlink(missing_ok=True)
            http_status = exc.code
            detail = exc.read().decode("utf-8", errors="replace")
            error_text = f"HTTP {exc.code}: {detail[:2000]}"
            raise BenchmarkError(
                f"GET {url} -> HTTP {exc.code}: {detail[:2000]}"
            ) from exc
        except (urllib.error.URLError, TimeoutError, OSError) as exc:
            partial.unlink(missing_ok=True)
            error_text = f"{type(exc).__name__}: {exc}"
            raise BenchmarkError(f"GET {url} failed: {exc}") from exc
        finally:
            self._record(
                method="GET",
                path=path,
                kind="download",
                started_monotonic=started_monotonic,
                started_epoch=started_epoch,
                http_status=http_status,
                error=error_text,
                bytes_received=bytes_received,
            )
        return target.stat().st_size

    def metrics_summary(self) -> dict[str, Any]:
        durations = [float(record["duration_seconds"]) for record in self.records]
        return {
            "request_count": len(self.records),
            "request_duration_seconds_summary": summarize_numbers(durations),
            "requests": self.records,
        }


def parse_server(spec: str) -> tuple[str, str]:
    if "=" not in spec:
        raise argparse.ArgumentTypeError(
            f"server must be LABEL=URL, got {spec!r}"
        )
    label, url = spec.split("=", 1)
    label = label.strip()
    url = url.strip().rstrip("/")
    if not label or not re.fullmatch(r"[A-Za-z0-9_.-]+", label):
        raise argparse.ArgumentTypeError(f"invalid server label: {label!r}")
    if not re.match(r"^https?://", url):
        raise argparse.ArgumentTypeError(f"server URL must use http(s): {url!r}")
    return label, url


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def summarize_numbers(values: list[float]) -> dict[str, Any]:
    if not values:
        return {
            "count": 0,
            "min": None,
            "median": None,
            "mean": None,
            "p95": None,
            "max": None,
        }
    ordered = sorted(values)
    count = len(ordered)
    p95_index = min(count - 1, max(0, int(round(0.95 * count)) - 1))
    return {
        "count": count,
        "min": round(ordered[0], 6),
        "median": round(ordered[count // 2], 6),
        "mean": round(sum(ordered) / count, 6),
        "p95": round(ordered[p95_index], 6),
        "max": round(ordered[-1], 6),
    }


def write_json_atomic(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="H3 ComfyUI API acceptance and concurrency benchmark."
    )
    parser.add_argument(
        "--server",
        action="append",
        required=True,
        type=parse_server,
        metavar="LABEL=URL",
        help="ComfyUI endpoint; provide twice for --mode dual/both",
    )
    parser.add_argument(
        "--workflow",
        default=os.getenv("H3_WORKFLOW"),
        help="UI/API workflow JSON; defaults to H3_WORKFLOW",
    )
    parser.add_argument("--mode", choices=("single", "dual", "both"), default="both")
    parser.add_argument(
        "--output-root",
        default=os.getenv("H3_OUTPUT_ROOT"),
        help="run output root; defaults to H3_OUTPUT_ROOT",
    )
    parser.add_argument("--run-label")
    parser.add_argument("--timeout", type=float, default=7200)
    parser.add_argument("--request-timeout", type=float, default=120)
    parser.add_argument("--poll-interval", type=float, default=2)
    parser.add_argument(
        "--history-grace",
        type=float,
        default=30,
        help=(
            "seconds to keep polling /history after the prompt leaves /queue "
            "(default: 30)"
        ),
    )
    parser.add_argument(
        "--min-artifact-bytes",
        type=int,
        default=1,
        help="minimum accepted size for each downloaded artifact (default: 1)",
    )
    parser.add_argument("--barrier-timeout", type=float, default=60)
    parser.add_argument("--prompt", default=DEFAULT_PROMPT)
    parser.add_argument("--width", type=int)
    parser.add_argument("--height", type=int)
    parser.add_argument("--frames", type=int)
    parser.add_argument("--steps", type=int)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--model", default=os.getenv("H3_MODEL_DIFFUSION"))
    parser.add_argument("--clip", default=os.getenv("H3_MODEL_CLIP"))
    parser.add_argument("--video-vae", default=os.getenv("H3_MODEL_VIDEO_VAE"))
    parser.add_argument("--audio-vae", default=os.getenv("H3_MODEL_AUDIO_VAE"))
    parser.add_argument("--lora", default=os.getenv("H3_MODEL_LORA"))
    parser.add_argument("--npu-count", type=int, choices=(1, 2, 4))
    parser.add_argument("--first-image")
    parser.add_argument("--last-image")
    parser.add_argument(
        "--require-video",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="require ffprobe-confirmed video output (default: true)",
    )
    parser.add_argument(
        "--require-audio",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="require ffprobe-confirmed audio stream (default: true)",
    )
    parser.add_argument(
        "--require-media-metadata",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="require width/height/frame count/fps to match requested values",
    )
    parser.add_argument(
        "--expected-fps",
        type=float,
        default=24.0,
        help="expected rounded video frame rate (default: 24)",
    )
    parser.add_argument(
        "--fps-tolerance",
        type=float,
        default=0.05,
        help="allowed absolute FPS difference (default: 0.05)",
    )
    parser.add_argument(
        "--no-ffprobe",
        action="store_true",
        help="Do not run ffprobe on downloaded video artifacts",
    )
    args = parser.parse_args()
    if not args.workflow:
        parser.error("--workflow is required when H3_WORKFLOW is unset")
    if not args.output_root:
        parser.error("--output-root is required when H3_OUTPUT_ROOT is unset")
    return args


def artifact_entries(history: dict[str, Any]) -> list[dict[str, Any]]:
    outputs = history.get("outputs")
    if not isinstance(outputs, dict):
        return []
    entries: list[dict[str, Any]] = []
    for node_id, node_output in outputs.items():
        if not isinstance(node_output, dict):
            continue
        for key in ARTIFACT_KEYS:
            values = node_output.get(key)
            if not isinstance(values, list):
                continue
            for value in values:
                if not isinstance(value, dict) or not value.get("filename"):
                    continue
                entry = dict(value)
                entry["_node_id"] = str(node_id)
                entry["_kind"] = key
                entries.append(entry)
    return entries


def history_error(history: dict[str, Any]) -> str | None:
    status = history.get("status")
    if isinstance(status, dict):
        status_str = str(status.get("status_str") or "").lower()
        if status_str in {
            "error",
            "failed",
            "failure",
            "interrupted",
            "cancelled",
            "canceled",
        }:
            messages = status.get("messages")
            return f"execution error: {json.dumps(messages, ensure_ascii=False)[:4000]}"
        if status.get("completed") is False:
            return "history reports completed=false"
    if history.get("execution_interrupted"):
        return "execution_interrupted=true"
    return None


def history_state(history: dict[str, Any]) -> str:
    """Return an explicit state; presence in /history is not success."""
    status = history.get("status")
    if not isinstance(status, dict):
        return "pending"
    status_str = str(status.get("status_str") or "").lower()
    if (
        history.get("execution_interrupted")
        or status.get("completed") is False
        or status_str
        in {
            "error",
            "failed",
            "failure",
            "interrupted",
            "cancelled",
            "canceled",
        }
    ):
        return "error"
    if status_str == "success" or status.get("completed") is True:
        return "success"
    return "pending"


def queue_contains(queue: dict[str, Any], prompt_id: str) -> tuple[bool, str]:
    for state in ("queue_running", "queue_pending"):
        items = queue.get(state)
        if not isinstance(items, list):
            continue
        for item in items:
            if isinstance(item, list) and len(item) > 1 and item[1] == prompt_id:
                return True, state
    return False, "missing"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        while chunk := source.read(8 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def ffprobe_artifact(path: Path) -> dict[str, Any]:
    executable = shutil.which("ffprobe")
    if executable is None:
        return {"available": False, "status": "SKIPPED"}
    try:
        process = subprocess.run(
            [
                executable,
                "-v",
                "error",
                "-count_frames",
                "-show_entries",
                (
                    "format=duration,format_name:"
                    "stream=index,codec_type,codec_name,width,height,r_frame_rate,"
                    "avg_frame_rate,nb_frames,nb_read_frames,sample_rate,channels,"
                    "channel_layout,duration"
                ),
                "-of",
                "json",
                str(path),
            ],
            capture_output=True,
            text=True,
            check=False,
            timeout=120,
        )
    except (OSError, subprocess.SubprocessError) as exc:
        return {"available": True, "status": "FAIL", "error": str(exc)}
    if process.returncode:
        return {
            "available": True,
            "status": "FAIL",
            "error": process.stderr.strip()[:4000],
        }
    try:
        payload = json.loads(process.stdout)
    except json.JSONDecodeError as exc:
        return {"available": True, "status": "FAIL", "error": str(exc)}
    streams = payload.get("streams") or []
    video_streams = [
        stream for stream in streams if stream.get("codec_type") == "video"
    ]
    audio_streams = [
        stream for stream in streams if stream.get("codec_type") == "audio"
    ]
    return {
        "available": True,
        "status": "OK" if video_streams else "FAIL",
        "format": payload.get("format"),
        "video_streams": video_streams,
        "audio_streams": audio_streams,
    }


def safe_filename(filename: str) -> str:
    name = Path(filename).name
    if not name or name in {".", ".."}:
        raise BenchmarkError(f"unsafe artifact filename: {filename!r}")
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", name)


def download_artifacts(
    client: HTTPClient,
    history: dict[str, Any],
    target_dir: Path,
    *,
    do_ffprobe: bool,
    min_artifact_bytes: int,
) -> tuple[list[dict[str, Any]], bool, list[str]]:
    entries = artifact_entries(history)
    records: list[dict[str, Any]] = []
    transport_ok = True
    errors: list[str] = []
    target_dir.mkdir(parents=True, exist_ok=True)
    for index, entry in enumerate(entries):
        filename = safe_filename(str(entry["filename"]))
        unique_name = f"{index:02d}_{filename}"
        target = target_dir / unique_name
        query = urllib.parse.urlencode(
            {
                "filename": entry["filename"],
                "subfolder": entry.get("subfolder", ""),
                "type": entry.get("type", "output"),
            }
        )
        record = {
            "node_id": entry["_node_id"],
            "kind": entry["_kind"],
            "filename": entry["filename"],
            "subfolder": entry.get("subfolder", ""),
            "server_type": entry.get("type", "output"),
            "local_path": str(target),
        }
        try:
            size = client.download(f"/view?{query}", target)
            record["size"] = size
            record["sha256"] = sha256_file(target)
            record["download_status"] = "OK"
            if size < min_artifact_bytes:
                record["download_status"] = "FAIL"
                errors.append(
                    f"artifact is smaller than {min_artifact_bytes} bytes: "
                    f"{filename} ({size} bytes)"
                )
            elif (
                do_ffprobe
                and target.suffix.lower() in {".mp4", ".mov", ".webm", ".mkv"}
            ):
                record["ffprobe"] = ffprobe_artifact(target)
        except (BenchmarkError, OSError) as exc:
            record["download_status"] = "FAIL"
            record["error"] = str(exc)
            errors.append(str(exc))
        if record["download_status"] != "OK":
            transport_ok = False
        records.append(record)
    return records, transport_ok, errors


def _stream_number(stream: dict[str, Any], field: str) -> float | None:
    value = stream.get(field)
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _rate_number(stream: dict[str, Any], field: str) -> float | None:
    value = stream.get(field)
    if value is None:
        return None
    text = str(value).strip()
    if "/" in text:
        numerator, denominator = text.split("/", 1)
        try:
            denominator_value = float(denominator)
            if denominator_value == 0:
                return None
            return float(numerator) / denominator_value
        except (TypeError, ValueError):
            return None
    try:
        return float(text)
    except (TypeError, ValueError):
        return None


def _frame_count(stream: dict[str, Any]) -> int | None:
    for field in ("nb_read_frames", "nb_frames"):
        value = stream.get(field)
        if value is None or str(value).upper() == "N/A":
            continue
        try:
            return int(value)
        except (TypeError, ValueError):
            continue
    return None


def validate_media_contract(
    artifacts: list[dict[str, Any]], args: argparse.Namespace
) -> dict[str, Any]:
    probes = [
        record.get("ffprobe")
        for record in artifacts
        if isinstance(record.get("ffprobe"), dict)
    ]
    available = [probe for probe in probes if probe.get("available")]
    videos = [
        stream
        for probe in available
        for stream in (probe.get("video_streams") or [])
        if isinstance(stream, dict)
    ]
    audio_streams = [
        stream
        for probe in available
        for stream in (probe.get("audio_streams") or [])
        if isinstance(stream, dict)
    ]
    contract_errors: list[str] = []
    for probe in probes:
        if probe.get("status") == "FAIL":
            contract_errors.append(
                f"ffprobe failed: {probe.get('error', 'unknown error')}"
            )

    expected: dict[str, Any] = {
        "width": args.width,
        "height": args.height,
        "frames": args.frames,
        "fps": args.expected_fps,
        "sample_rate": 32000,
        "channels": 2,
        "audio_layout": "stereo",
    }
    matches_expected = False
    matching_stream: dict[str, Any] | None = None
    observed_video: list[dict[str, Any]] = []
    for stream in videos:
        width = _stream_number(stream, "width")
        height = _stream_number(stream, "height")
        frames = _frame_count(stream)
        fps = _rate_number(stream, "avg_frame_rate")
        if fps is None:
            fps = _rate_number(stream, "r_frame_rate")
        observed = {
            "width": int(width) if width is not None else None,
            "height": int(height) if height is not None else None,
            "frames": frames,
            "fps": round(fps, 6) if fps is not None else None,
        }
        observed_video.append(observed)
        stream_matches = True
        if args.width is not None:
            stream_matches = stream_matches and width == args.width
        if args.height is not None:
            stream_matches = stream_matches and height == args.height
        if args.frames is not None:
            stream_matches = stream_matches and frames == args.frames
        if args.expected_fps > 0:
            stream_matches = (
                stream_matches
                and fps is not None
                and abs(fps - args.expected_fps) <= args.fps_tolerance
            )
        if stream_matches:
            matches_expected = True
            matching_stream = observed
            break

    video_ok = bool(videos) and any(
        probe.get("status") == "OK" for probe in available
    )
    audio_ok = bool(audio_streams)
    audio_format_ok = False
    observed_audio: list[dict[str, Any]] = []
    for stream in audio_streams:
        sample_rate = _stream_number(stream, "sample_rate")
        channels = _stream_number(stream, "channels")
        layout = str(stream.get("channel_layout") or "").lower()
        observed = {
            "sample_rate": int(sample_rate) if sample_rate is not None else None,
            "channels": int(channels) if channels is not None else None,
            "channel_layout": layout or None,
        }
        observed_audio.append(observed)
        if (
            sample_rate == expected["sample_rate"]
            and channels == expected["channels"]
            and (not layout or layout == expected["audio_layout"])
        ):
            audio_format_ok = True

    if args.no_ffprobe:
        metadata_ok: bool | None = None
    elif args.require_media_metadata:
        metadata_ok = matches_expected if videos else False
    else:
        metadata_ok = None
    if not args.no_ffprobe and not available:
        contract_errors.append("ffprobe did not produce a machine-readable result")
    if not args.no_ffprobe and args.require_video and not video_ok:
        contract_errors.append("no ffprobe-confirmed video stream")
    if not args.no_ffprobe and args.require_audio and not audio_ok:
        contract_errors.append("no ffprobe-confirmed audio stream")
    if (
        not args.no_ffprobe
        and args.require_media_metadata
        and not matches_expected
    ):
        contract_errors.append(
            "no video stream matched width/height/frames/fps contract"
        )
    if (
        not args.no_ffprobe
        and args.require_audio
        and audio_ok
        and not audio_format_ok
    ):
        contract_errors.append(
            "audio stream did not match 32000 Hz / 2 channels / stereo"
        )

    return {
        "evaluated": not args.no_ffprobe,
        "probe_available": bool(available),
        "probe_count": len(probes),
        "video_ok": video_ok,
        "audio_ok": audio_ok,
        "audio_format_ok": audio_format_ok,
        "metadata_ok": metadata_ok,
        "expected": expected,
        "video_stream_count": len(videos),
        "audio_stream_count": len(audio_streams),
        "video_observed": observed_video,
        "audio_observed": observed_audio,
        "matching_video": matching_stream,
        "errors": contract_errors,
    }


def prepare_workflow_for_server(
    source_workflow: dict[str, Any],
    client: HTTPClient,
    overrides: Overrides,
    output_path: Path,
) -> tuple[dict[str, Any], dict[str, Any]]:
    patched, summary = apply_overrides(source_workflow, overrides)
    if detect_format(patched) == "ui":
        object_info = client.request("GET", "/object_info")
        if not isinstance(object_info, dict):
            raise BenchmarkError("/object_info did not return an object")
        api_workflow = ui_to_api(patched, object_info)
        summary["conversion"] = "ui-to-api"
    else:
        api_workflow = patched
        summary["conversion"] = "api-passthrough"
    validate_api(api_workflow)
    dump_workflow(api_workflow, output_path)
    return api_workflow, summary


def make_overrides(
    args: argparse.Namespace, server_label: str, run_name: str
) -> Overrides:
    return Overrides(
        model=args.model,
        clip=args.clip,
        video_vae=args.video_vae,
        audio_vae=args.audio_vae,
        lora=args.lora,
        width=args.width,
        height=args.height,
        frames=args.frames,
        steps=args.steps,
        npu_count=args.npu_count,
        prompt=args.prompt,
        seed=args.seed,
        first_image=args.first_image,
        last_image=args.last_image,
        filename_prefix=f"h3-accept/{run_name}/{server_label}",
    )


def submit_and_wait(
    label: str,
    url: str,
    source_workflow: dict[str, Any],
    args: argparse.Namespace,
    server_dir: Path,
    run_name: str,
    run_kind: str,
    ready_barrier: threading.Barrier | None,
) -> dict[str, Any]:
    call_started_monotonic = time.monotonic()
    call_started_epoch = time.time()
    result: dict[str, Any] = {
        "label": label,
        "url": url,
        "run_kind": run_kind,
        "phase": run_kind,
        "output_dir": str(server_dir),
        "call_started_monotonic": round(call_started_monotonic, 6),
        "call_started_epoch": round(call_started_epoch, 6),
        "call_started_at": utc_now(),
        "preparation_started_monotonic": round(call_started_monotonic, 6),
        "preparation_started_epoch": round(call_started_epoch, 6),
        "success": False,
        "completed": False,
        "artifact_ok": False,
        "transport_ok": False,
        "media_ok": False,
        "history_grace_used": False,
        "errors": [],
        "warnings": [],
        "artifacts": [],
    }
    client = HTTPClient(url, args.request_timeout)
    barrier_released = False
    history: dict[str, Any] | None = None
    prompt_id: str | None = None
    poll_events: list[dict[str, Any]] = []
    queue_observations: list[dict[str, Any]] = []
    queue_state = "not-queried"
    queue_left_at: float | None = None
    try:
        workflow, summary = prepare_workflow_for_server(
            source_workflow,
            client,
            make_overrides(args, label, run_name),
            server_dir / "submitted-workflow.json",
        )
        result["workflow"] = summary
        preparation_finished = time.monotonic()
        result["preparation_finished_monotonic"] = round(
            preparation_finished, 6
        )
        result["preparation_finished_epoch"] = round(time.time(), 6)
        result["preparation_seconds"] = round(
            preparation_finished - call_started_monotonic, 6
        )
        client_id = str(uuid.uuid4())
        submit_payload = {"prompt": workflow, "client_id": client_id}
        if ready_barrier is not None:
            try:
                ready_barrier.wait(timeout=args.barrier_timeout)
            except threading.BrokenBarrierError as exc:
                raise BenchmarkError(
                    "dual submission barrier broke before POST; peer preparation failed"
                ) from exc
            barrier_released = True

        submit_start = time.monotonic()
        submit_start_epoch = time.time()
        result["submit_started_monotonic"] = round(submit_start, 6)
        result["submit_started_epoch"] = round(submit_start_epoch, 6)
        result["submit_started_at"] = datetime.fromtimestamp(
            submit_start_epoch, timezone.utc
        ).isoformat()
        response = client.request("POST", "/prompt", payload=submit_payload)
        submit_end = time.monotonic()
        submit_end_epoch = time.time()
        result["submit_finished_monotonic"] = round(submit_end, 6)
        result["submit_finished_epoch"] = round(submit_end_epoch, 6)
        result["submit_finished_at"] = datetime.fromtimestamp(
            submit_end_epoch, timezone.utc
        ).isoformat()
        result["submit_seconds"] = round(submit_end - submit_start, 6)
        if not isinstance(response, dict):
            raise BenchmarkError("/prompt response is not an object")
        result["prompt_response"] = response
        prompt_candidate = response.get("prompt_id")
        if isinstance(prompt_candidate, str) and prompt_candidate:
            prompt_id = prompt_candidate
            result["prompt_id"] = prompt_id
        node_errors = response.get("node_errors")
        if node_errors not in (None, {}, []):
            raise BenchmarkError(
                "/prompt returned node_errors: "
                + json.dumps(node_errors, ensure_ascii=False)[:6000]
            )
        if not isinstance(prompt_candidate, str) or not prompt_candidate:
            raise BenchmarkError(f"/prompt returned no prompt_id: {response}")

        deadline = submit_end + args.timeout
        history_deadline: float | None = None
        poll_count = 0
        last_poll_at: float | None = None
        while True:
            loop_started = time.monotonic()
            active_deadline = history_deadline or deadline
            if loop_started >= active_deadline:
                break
            poll_count += 1
            history_response = client.request(
                "GET", f"/history/{urllib.parse.quote(prompt_id)}"
            )
            if not isinstance(history_response, dict):
                raise BenchmarkError("/history response is not an object")
            candidate = history_response.get(prompt_id)
            if isinstance(candidate, dict):
                history = candidate
                current_history_state = history_state(candidate)
                result["history_state"] = current_history_state
                poll_events.append(
                    {
                        "poll": poll_count,
                        "at_monotonic": round(loop_started, 6),
                        "at": utc_now(),
                        "result": (
                            "history-terminal"
                            if current_history_state != "pending"
                            else "history-pending"
                        ),
                        "history_state": current_history_state,
                        "queue_state": queue_state,
                        "poll_lag_seconds": (
                            round(loop_started - last_poll_at, 6)
                            if last_poll_at is not None
                            else None
                        ),
                    }
                )
                if current_history_state != "pending":
                    break
                last_poll_at = loop_started
                remaining = (history_deadline or deadline) - time.monotonic()
                if remaining > 0:
                    time.sleep(min(args.poll_interval, remaining))
                continue
            try:
                queue_response = client.request("GET", "/queue")
                if isinstance(queue_response, dict):
                    present, observed_state = queue_contains(
                        queue_response, prompt_id
                    )
                    queue_state = observed_state if present else "not-in-queue"
                    if not present:
                        if queue_left_at is None:
                            queue_left_at = time.monotonic()
                            history_deadline = queue_left_at + args.history_grace
                            result["history_grace_used"] = True
                            result["queue_left_monotonic"] = round(
                                queue_left_at, 6
                            )
                            result["queue_left_epoch"] = round(time.time(), 6)
                            result["queue_left_at"] = utc_now()
                            result["history_grace_deadline_monotonic"] = round(
                                history_deadline, 6
                            )
                    if (
                        not queue_observations
                        or queue_observations[-1]["state"] != queue_state
                    ):
                        queue_observations.append(
                            {
                                "poll": poll_count,
                                "at_monotonic": round(time.monotonic(), 6),
                                "at": utc_now(),
                                "state": queue_state,
                            }
                        )
                else:
                    queue_state = "queue-response-not-object"
            except BenchmarkError as exc:
                queue_state = f"queue-error: {exc}"
            poll_events.append(
                {
                    "poll": poll_count,
                    "at_monotonic": round(loop_started, 6),
                    "at": utc_now(),
                    "result": "not-ready",
                    "queue_state": queue_state,
                    "poll_lag_seconds": (
                        round(loop_started - last_poll_at, 6)
                        if last_poll_at is not None
                        else None
                    ),
                }
            )
            last_poll_at = loop_started
            active_deadline = history_deadline or deadline
            remaining = active_deadline - time.monotonic()
            if remaining > 0:
                time.sleep(min(args.poll_interval, remaining))

        completed_at = time.monotonic()
        result["poll_count"] = poll_count
        result["queue_state_at_timeout"] = queue_state
        result["final_queue_state"] = queue_state
        result["queue_observations"] = queue_observations
        result["poll_events"] = poll_events
        poll_lags = [
            float(event["poll_lag_seconds"])
            for event in poll_events
            if event.get("poll_lag_seconds") is not None
        ]
        result["poll_interval_seconds_summary"] = summarize_numbers(poll_lags)
        result["execution_finished_monotonic"] = round(completed_at, 6)
        result["execution_finished_epoch"] = round(time.time(), 6)
        if history is None:
            grace_detail = (
                f"; prompt left /queue and /history did not appear within the "
                f"{args.history_grace}s grace window"
                if queue_left_at is not None
                else ""
            )
            raise BenchmarkError(
                f"prompt {prompt_id} did not reach /history within the "
                f"configured deadline; queue_state={queue_state}{grace_detail}"
            )

        result["history"] = history
        result["history_state"] = history_state(history)
        result["execution_finished_at"] = utc_now()
        result["execution_seconds"] = round(completed_at - submit_start, 6)
        error = history_error(history)
        if error:
            raise BenchmarkError(error)
        if result["history_state"] != "success":
            raise BenchmarkError(
                "history did not report an explicit success terminal state "
                "(status.status_str=success or status.completed=true)"
            )
        result["completed"] = True
        artifacts, transport_ok, download_errors = download_artifacts(
            client,
            history,
            server_dir / "artifacts",
            do_ffprobe=not args.no_ffprobe,
            min_artifact_bytes=args.min_artifact_bytes,
        )
        result["artifacts"] = artifacts
        nonempty_artifacts = any(
            record.get("download_status") == "OK"
            and record.get("size", 0) >= args.min_artifact_bytes
            for record in artifacts
        )
        result["artifact_ok"] = bool(nonempty_artifacts and transport_ok)
        result["transport_ok"] = transport_ok
        result["errors"].extend(download_errors)
        if not artifacts:
            result["errors"].append("history contained no downloadable artifacts")
        media = validate_media_contract(artifacts, args)
        result["media"] = media
        contract_only = bool(args.no_ffprobe)
        if contract_only:
            media_ok = False
            result["warnings"].append(
                "contract-only mode (degraded HTTP-only mode): HTTP submission, "
                "terminal history, and "
                "downloaded file checks passed without ffprobe; this is not a "
                "real media acceptance result"
            )
        else:
            media_ok = not media["errors"]
        result["errors"].extend(media["errors"])
        result["media_ok"] = media_ok
        result["contract_only"] = contract_only
        result["success"] = bool(
            result["completed"]
            and result["artifact_ok"]
            and result["transport_ok"]
            and not contract_only
            and result["media_ok"]
        )
    except Exception as exc:
        result["errors"].append(f"{type(exc).__name__}: {exc}")
        if ready_barrier is not None and not barrier_released:
            ready_barrier.abort()
    finally:
        finished_monotonic = time.monotonic()
        result["finished_monotonic"] = round(finished_monotonic, 6)
        result["finished_epoch"] = round(time.time(), 6)
        result["finished_at"] = utc_now()
        result["metrics"] = client.metrics_summary()
        submit_start = result.get("submit_started_monotonic")
        if isinstance(submit_start, (int, float)):
            result["completion_seconds"] = round(
                finished_monotonic - submit_start, 6
            )
        try:
            write_json_atomic(server_dir / "result.json", result)
        except OSError as exc:
            result["errors"].append(
                f"failed to write per-server result.json: {type(exc).__name__}: {exc}"
            )
            result["success"] = False
    return result


def run_single(
    source_workflow: dict[str, Any],
    args: argparse.Namespace,
    output_root: Path,
    run_name: str,
) -> list[dict[str, Any]]:
    results: list[dict[str, Any]] = []
    for label, url in args.server:
        print(f"[{run_name}] single baseline on {label} ({url})", flush=True)
        result = submit_and_wait(
            label,
            url,
            source_workflow,
            args,
            output_root / "single" / label,
            f"{run_name}-single",
            "single",
            None,
        )
        print(
            f"[{run_name}] {label}: success={result['success']} "
            f"completion_s={result.get('completion_seconds')}",
            flush=True,
        )
        results.append(result)
    return results


def run_dual(
    source_workflow: dict[str, Any],
    args: argparse.Namespace,
    output_root: Path,
    run_name: str,
) -> list[dict[str, Any]]:
    if len(args.server) < 2:
        raise BenchmarkError("--mode dual requires two --server arguments")
    barrier = threading.Barrier(2)
    results: list[dict[str, Any] | None] = [None, None]

    def worker(index: int, label: str, url: str) -> None:
        try:
            results[index] = submit_and_wait(
                label,
                url,
                source_workflow,
                args,
                output_root / "dual" / label,
                f"{run_name}-dual",
                "dual",
                barrier,
            )
        except Exception as exc:
            now_monotonic = time.monotonic()
            now_epoch = time.time()
            results[index] = {
                "label": label,
                "url": url,
                "run_kind": "dual",
                "phase": "dual",
                "output_dir": str(output_root / "dual" / label),
                "success": False,
                "completed": False,
                "artifact_ok": False,
                "transport_ok": False,
                "media_ok": False,
                "finished_monotonic": round(now_monotonic, 6),
                "finished_epoch": round(now_epoch, 6),
                "finished_at": utc_now(),
                "errors": [
                    f"dual worker raised {type(exc).__name__}: {exc}"
                ],
                "warnings": [],
                "artifacts": [],
            }
            try:
                write_json_atomic(
                    output_root / "dual" / label / "result.json",
                    results[index],
                )
            except OSError:
                pass
            try:
                barrier.abort()
            except threading.BrokenBarrierError:
                pass

    print(
        f"[{run_name}] dual submit: {args.server[0][0]} + {args.server[1][0]}",
        flush=True,
    )
    threads = [
        threading.Thread(
            target=worker,
            args=(index, label, url),
            name=f"h3-{label}",
            daemon=False,
        )
        for index, (label, url) in enumerate(args.server[:2])
    ]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    if any(result is None for result in results):
        raise BenchmarkError("one dual worker did not return a result")
    final_results = [result for result in results if result is not None]

    starts = [
        float(result["submit_started_monotonic"])
        for result in final_results
        if result.get("submit_started_monotonic") is not None
    ]
    ends = [
        float(result["finished_monotonic"])
        for result in final_results
        if result.get("finished_monotonic") is not None
    ]
    submission_skew = max(starts) - min(starts) if len(starts) == 2 else None
    dual_wall = max(ends) - min(starts) if starts and ends else None
    overlap_start = max(starts) if len(starts) == 2 else None
    overlap_end = min(ends) if len(ends) == 2 else None
    for result in final_results:
        result["submission_skew_seconds"] = (
            round(submission_skew, 6) if submission_skew is not None else None
        )
        result["dual_wall_seconds"] = (
            round(dual_wall, 6) if dual_wall is not None else None
        )
        result["dual_overlap_start_monotonic"] = (
            round(overlap_start, 6) if overlap_start is not None else None
        )
        result["dual_overlap_end_monotonic"] = (
            round(overlap_end, 6) if overlap_end is not None else None
        )
        result["dual_overlap_seconds"] = (
            round(max(0.0, overlap_end - overlap_start), 6)
            if overlap_start is not None and overlap_end is not None
            else None
        )
        try:
            write_json_atomic(
                output_root / "dual" / result["label"] / "result.json", result
            )
        except OSError as exc:
            result["errors"].append(
                f"failed to update dual result.json: {type(exc).__name__}: {exc}"
            )
            result["success"] = False
        print(
            f"[{run_name}] {result['label']}: success={result['success']} "
            f"completion_s={result.get('completion_seconds')}",
            flush=True,
        )
    print(
        f"[{run_name}] dual wall_s={dual_wall} submit_skew_s={submission_skew}",
        flush=True,
    )
    return final_results


def add_wall_clock_fields(result: dict[str, Any]) -> None:
    if not isinstance(result.get("finished_epoch"), (int, float)):
        result["finished_epoch"] = time.time()
    start_monotonic = result.get("submit_started_monotonic")
    if isinstance(start_monotonic, (int, float)) and not isinstance(
        result.get("submit_started_epoch"), (int, float)
    ):
        now_epoch = time.time()
        monotonic_now = time.monotonic()
        result["submit_started_epoch"] = now_epoch - (monotonic_now - start_monotonic)


def print_summary(results: list[dict[str, Any]]) -> None:
    print("")
    print(
        f"{'label':<12} {'run_kind':<8} {'success':<8} {'completed':<10} "
        f"{'contract':<8} {'history':<8} {'media':<7} {'artifacts':<9} "
        f"{'submit_s':>10} {'execute_s':>11} {'complete_s':>12} "
        f"{'dual_s':>10} {'skew_s':>9}"
    )
    for result in results:
        print(
            f"{result['label']:<12} {result.get('run_kind', '-'):<8} "
            f"{str(result['success']):<8} {str(result['completed']):<10} "
            f"{str(result.get('contract_only', False)):<8} "
            f"{str(result.get('history_state', '-')):<8} "
            f"{str(result.get('media_ok', False)):<7} "
            f"{len(result.get('artifacts', [])):<9} "
            f"{format_number(result.get('submit_seconds')):>10} "
            f"{format_number(result.get('execution_seconds')):>11} "
            f"{format_number(result.get('completion_seconds')):>12} "
            f"{format_number(result.get('dual_wall_seconds')):>10} "
            f"{format_number(result.get('submission_skew_seconds')):>9}"
        )
        for error in result.get("errors", []):
            print(f"  error: {error}")


def format_number(value: Any) -> str:
    return f"{value:.3f}" if isinstance(value, (int, float)) else "-"


def new_run_name() -> str:
    return datetime.now().strftime("%Y%m%dT%H%M%S")


def main() -> int:
    args = parse_args()
    Overrides(
        width=args.width,
        height=args.height,
        frames=args.frames,
        steps=args.steps,
        npu_count=args.npu_count,
        seed=args.seed,
    ).validate()
    if args.mode in {"dual", "both"} and len(args.server) < 2:
        raise BenchmarkError("--mode dual/both requires at least two servers")
    if args.mode in {"single", "dual"} and len(args.server) < 1:
        raise BenchmarkError("at least one server is required")
    if len(args.server) > 2:
        raise BenchmarkError("at most two --server arguments are supported")
    if args.mode in {"dual", "both"} and len(args.server) != 2:
        raise BenchmarkError("--mode dual/both requires exactly two servers")
    labels = [label for label, _ in args.server]
    urls = [url for _, url in args.server]
    if len(labels) != len(set(labels)):
        raise BenchmarkError("server labels must be unique")
    if args.mode in {"dual", "both"} and len(urls) != len(set(urls)):
        raise BenchmarkError("dual mode requires two distinct server URLs")
    if (
        args.poll_interval <= 0
        or args.timeout <= 0
        or args.request_timeout <= 0
        or args.barrier_timeout <= 0
    ):
        raise BenchmarkError("timeouts and poll interval must be positive")
    if args.history_grace < 0:
        raise BenchmarkError("--history-grace must be zero or positive")
    if args.min_artifact_bytes < 1:
        raise BenchmarkError("--min-artifact-bytes must be positive")
    if args.expected_fps < 0 or args.fps_tolerance < 0:
        raise BenchmarkError(
            "--expected-fps and --fps-tolerance must be zero or positive"
        )
    if args.no_ffprobe and (
        args.require_video
        or args.require_audio
        or args.require_media_metadata
    ):
        raise BenchmarkError(
            "--no-ffprobe is a degraded HTTP-only media mode and requires "
            "--no-require-video --no-require-audio "
            "--no-require-media-metadata"
        )

    if args.run_label is not None and (
        args.run_label in {".", ".."}
        or re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*", args.run_label) is None
    ):
        raise BenchmarkError(
            "--run-label must match [A-Za-z0-9][A-Za-z0-9_.-]*"
        )

    run_name = args.run_label or new_run_name()
    output_root = Path(args.output_root).resolve() / run_name
    output_root.mkdir(parents=True, exist_ok=True)
    source_workflow = load_workflow(args.workflow)
    models = {
        "diffusion": args.model,
        "clip": args.clip,
        "video_vae": args.video_vae,
        "audio_vae": args.audio_vae,
        "lora": args.lora,
    }
    manifest = {
        "schema": 1,
        "started_at": utc_now(),
        "mode": args.mode,
        "run_name": run_name,
        "servers": [
            {"label": label, "url": url} for label, url in args.server
        ],
        "workflow_source": str(Path(args.workflow).resolve()),
        "workflow_format": detect_format(source_workflow),
        "contract_only": bool(args.no_ffprobe),
        "models": models,
        "parameters": {
            "width": args.width,
            "height": args.height,
            "frames": args.frames,
            "steps": args.steps,
            "seed": args.seed,
            "npu_count": args.npu_count,
            "prompt": args.prompt,
            "first_image": args.first_image,
            "last_image": args.last_image,
            "timeout_seconds": args.timeout,
            "request_timeout_seconds": args.request_timeout,
            "poll_interval_seconds": args.poll_interval,
            "history_grace_seconds": args.history_grace,
            "expected_fps": args.expected_fps,
            "fps_tolerance": args.fps_tolerance,
            "require_video": args.require_video,
            "require_audio": args.require_audio,
            "require_media_metadata": args.require_media_metadata,
            "min_artifact_bytes": args.min_artifact_bytes,
        },
    }
    (output_root / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    results: list[dict[str, Any]] = []
    if args.mode in {"single", "both"}:
        results.extend(run_single(source_workflow, args, output_root, run_name))
    if args.mode in {"dual", "both"}:
        results.extend(run_dual(source_workflow, args, output_root, run_name))

    for result in results:
        add_wall_clock_fields(result)
    started = min(
        (
            result["submit_started_epoch"]
            for result in results
            if result.get("submit_started_epoch") is not None
        ),
        default=None,
    )
    finished = max(
        (
            result["finished_epoch"]
            for result in results
            if result.get("finished_epoch") is not None
        ),
        default=None,
    )
    dual_results = [
        result for result in results if result.get("run_kind") == "dual"
    ]
    dual_wall = None
    if dual_results:
        dual_starts = [
            result["submit_started_epoch"]
            for result in dual_results
            if result.get("submit_started_epoch") is not None
        ]
        dual_ends = [
            result["finished_epoch"]
            for result in dual_results
            if result.get("finished_epoch") is not None
        ]
        if dual_starts and dual_ends:
            dual_wall = round(max(dual_ends) - min(dual_starts), 6)

    overall_success = bool(results) and all(
        result.get("success") for result in results
    )
    report = {
        "schema": 1,
        "run_name": run_name,
        "finished_at": utc_now(),
        "overall_success": overall_success,
        "manifest": manifest,
        "wall_seconds": round(finished - started, 6)
        if started is not None and finished is not None
        else None,
        "dual_wall_seconds": dual_wall,
        "results": results,
    }
    report_path = output_root / "result.json"
    report_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print_summary(results)
    print(f"\nreport: {report_path}")
    return 0 if overall_success else 1


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (BenchmarkError, WorkflowError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        raise SystemExit(2)
