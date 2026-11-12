#!/usr/bin/env python3
"""vLLM-Omni compatible request contract for the H3-910A service.

OpenDirector talks to a self-hosted video service through the vLLM-Omni
``/videos`` job API.  This module is the protocol half of that service: it
turns one multipart submission into a validated request, and it renders the
job state back into the three status words OpenDirector understands
(``queued``/``in_progress``/``completed``).

Nothing here imports aiohttp, so the rules can be tested without a running
service.
"""

from __future__ import annotations

import json
import math
import re
from dataclasses import dataclass, field
from typing import Any, Iterable, Mapping

FPS = 24
MIN_SECONDS = 4
MAX_SECONDS = 15
GRID = 17
GRID_OFFSET = 5

#: Output size used when the client sends neither ``size`` nor a ratio.
DEFAULT_SIZE = (864, 480)

#: Aspect ratios OpenDirector offers for the native H3 adapter.
SIZE_BY_RATIO = {
    "16:9": (864, 480),
    "9:16": (480, 864),
    "1:1": (640, 640),
    "4:3": (640, 480),
    "3:4": (480, 640),
}

#: vLLM-Omni task names OpenDirector sends in ``extra_params.task``.
TASKS = {"t2va", "fl2va", "ref2va"}

#: Job states OpenDirector polls for.  It fails the task on anything else.
STATUS_QUEUED = "queued"
STATUS_RUNNING = "in_progress"
STATUS_COMPLETED = "completed"
STATUS_FAILED = "failed"
STATUS_CANCELLED = "cancelled"
STATUS_EXPIRED = "expired"

_TERMINAL = {STATUS_COMPLETED, STATUS_FAILED, STATUS_CANCELLED, STATUS_EXPIRED}
_SIZE_RE = re.compile(r"^\s*(\d{2,5})\s*[xX*]\s*(\d{2,5})\s*$")


class RequestError(ValueError):
    """A submission the service refuses, carrying the HTTP status to use."""

    def __init__(self, message: str, status: int = 400, code: str = "invalid_request"):
        super().__init__(message)
        self.message = message
        self.status = status
        self.code = code

    def payload(self) -> dict[str, Any]:
        # Both spellings on purpose: OpenDirector reads ``error.message`` and
        # generic clients read ``detail``.
        return {
            "detail": self.message,
            "error": {"code": self.code, "message": self.message},
        }


def is_terminal(status: str) -> bool:
    return status in _TERMINAL


def frames_for_seconds(seconds: float) -> int:
    """Nearest H3 length on the 17n+5 frame grid, clamped to 4-15 seconds."""
    if isinstance(seconds, bool) or not isinstance(seconds, (int, float)):
        raise RequestError("seconds 必须是数字。")
    if not math.isfinite(seconds) or seconds <= 0:
        raise RequestError("seconds 必须为正数。")
    seconds = min(max(float(seconds), MIN_SECONDS), MAX_SECONDS)
    steps = round((seconds * FPS - GRID_OFFSET) / GRID)
    return GRID * min(max(steps, 6), 20) + GRID_OFFSET


def validate_frames(frames: int) -> int:
    if isinstance(frames, bool) or not isinstance(frames, int):
        raise RequestError("num_frames 必须是整数。")
    if frames < GRID * 6 + GRID_OFFSET or frames > GRID * 20 + GRID_OFFSET:
        raise RequestError(
            f"num_frames 须在 4-15 秒之间（{GRID * 6 + GRID_OFFSET}-{GRID * 20 + GRID_OFFSET} 帧）。"
        )
    if (frames - GRID_OFFSET) % GRID:
        raise RequestError("num_frames 必须落在 H3 的 17n+5 帧网格上。")


def parse_size(text: str) -> tuple[int, int]:
    match = _SIZE_RE.match(str(text))
    if not match:
        raise RequestError("size 格式应为 宽x高，例如 864x480。")
    width, height = int(match.group(1)), int(match.group(2))
    for name, value in (("宽", width), ("高", height)):
        if value % 32:
            raise RequestError(f"{name}必须是 32 的整数倍（H3 的 patch 对齐）。")
        if not 256 <= value <= 2048:
            raise RequestError(f"{name}必须在 256-2048 像素之间。")
    return width, height


def size_for_ratio(ratio: str | None) -> tuple[int, int]:
    if not ratio:
        return DEFAULT_SIZE
    ratio = str(ratio).strip()
    if ratio not in SIZE_BY_RATIO:
        raise RequestError(
            "aspect_ratio 必须是 " + "、".join(SIZE_BY_RATIO) + " 之一。"
        )
    return SIZE_BY_RATIO[ratio]


def parse_extra_params(raw: str | None) -> dict[str, Any]:
    """``extra_params`` is a JSON object in one multipart field."""
    if raw in (None, ""):
        return {}
    try:
        value = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise RequestError(f"extra_params 不是合法 JSON：{exc}") from None
    if not isinstance(value, dict):
        raise RequestError("extra_params 必须是 JSON 对象。")
    task = value.get("task")
    if task is not None and task not in TASKS:
        raise RequestError("extra_params.task 只能是 t2va、fl2va 或 ref2va。")
    if value.get("continue_from"):
        # Continuation needs the cached latent tail of an earlier clip; the
        # 910A path keeps no such cache, so refuse instead of silently
        # generating an unrelated clip.
        raise RequestError(
            "该服务不支持 continue_from 接续（需要上一段视频的 latent 缓存）。",
            status=400,
            code="unsupported_continuation",
        )
    return value


@dataclass
class Upload:
    """One multipart file part."""

    field: str
    filename: str
    content_type: str
    data: bytes


@dataclass
class VideoRequest:
    """A validated submission, in the shape the workflow editor needs."""

    prompt: str
    task: str
    width: int
    height: int
    frames: int
    seed: int | None = None
    image: Upload | None = None
    fps: int = FPS
    extras: dict[str, Any] = field(default_factory=dict)

    @property
    def seconds(self) -> float:
        return round(self.frames / self.fps, 3)

    def overrides(self, *, filename_prefix: str) -> dict[str, Any]:
        """Keyword arguments for ``h3_workflow.Overrides``.

        ``none`` is the editor's documented way to drop an image input, which
        is what makes text-to-video and first-frame-to-video the same graph.
        """
        if self.image is None:
            first, last = "none", "none"
        elif self.task == "fl2va":
            first, last = "__uploaded__", "none"
        else:
            # ref2va with a single still is the same conditioning as fl2va.
            first, last = "__uploaded__", "none"
        return {
            "prompt": self.prompt,
            "width": self.width,
            "height": self.height,
            "frames": self.frames,
            "seed": self.seed,
            "first_image": first,
            "last_image": last,
            "filename_prefix": filename_prefix,
        }


def _field(fields: Mapping[str, str], *names: str) -> str | None:
    for name in names:
        if name in fields and fields[name] != "":
            return fields[name]
    return None


def build_request(fields: Mapping[str, str], files: Iterable[Upload]) -> VideoRequest:
    """Validate one ``POST /videos`` body.

    Field names follow the vLLM-Omni schema OpenDirector emits:
    ``model``, ``prompt``, ``fps``, ``num_frames``, optional ``size`` and
    ``extra_params`` (JSON with ``task``/``aspect_ratio``).
    """
    prompt = (fields.get("prompt") or "").strip()
    if not prompt:
        raise RequestError("prompt 不能为空。")
    if len(prompt) > 8000:
        raise RequestError("prompt 超过 8000 字符。")

    extras = parse_extra_params(fields.get("extra_params"))
    uploaded = list(files)
    images = [item for item in uploaded if item.field in {"input_reference", "input_references"}]
    others = [item for item in uploaded if item.field not in {"input_reference", "input_references"}]
    if others:
        raise RequestError(
            "暂不支持的文件字段：" + "、".join(sorted({item.field for item in others}))
        )

    task = extras.get("task")
    if task is None:
        task = "fl2va" if images else "t2va"
    if task == "ref2va" and len(images) > 1:
        raise RequestError(
            "当前 910A 服务只支持单张参考图（首帧）参考；多素材 ref2va 尚未接通。",
            code="unsupported_reference_count",
        )
    if task == "t2va" and images:
        raise RequestError("task=t2va 时不应上传图片。")
    if task in {"fl2va", "ref2va"} and not images:
        raise RequestError("task=%s 需要一张图片。" % task)
    if len(images) > 1:
        raise RequestError("当前 910A 服务一次只接受一张图片参考。")

    for item in images:
        if item.content_type not in {"image/png", "image/jpeg", "image/webp"}:
            raise RequestError("参考图须为 PNG、JPEG 或 WEBP。")
        if len(item.data) > 30 * 1024 * 1024:
            raise RequestError("参考图不得超过 30 MB。")

    if fields.get("size"):
        width, height = parse_size(fields["size"])
    else:
        width, height = size_for_ratio(extras.get("aspect_ratio"))

    if fields.get("num_frames"):
        try:
            frames = int(str(fields["num_frames"]))
        except ValueError:
            raise RequestError("num_frames 必须是整数。") from None
        validate_frames(frames)
    elif fields.get("seconds"):
        try:
            frames = frames_for_seconds(float(str(fields["seconds"])))
        except ValueError:
            raise RequestError("seconds 必须是数字。") from None
    else:
        frames = frames_for_seconds(5)

    fps = FPS
    if fields.get("fps"):
        try:
            fps = int(str(fields["fps"]))
        except ValueError:
            raise RequestError("fps 必须是整数。") from None
        if fps != FPS:
            raise RequestError(f"该服务固定 {FPS} fps（H3 视频网格要求）。")

    seed = None
    if fields.get("seed") not in (None, ""):
        try:
            seed = int(str(fields["seed"]))
        except ValueError:
            raise RequestError("seed 必须是整数。") from None
        if seed < 0:
            raise RequestError("seed 必须为非负数。")

    return VideoRequest(
        prompt=prompt,
        task=task,
        width=width,
        height=height,
        frames=frames,
        seed=seed,
        image=images[0] if images else None,
        fps=fps,
        extras=extras,
    )


def capabilities(model_id: str) -> dict[str, Any]:
    """The ``videoCapabilities`` block OpenDirector reads from ``GET /models``.

    Declaring it lets the UI fill resolution/duration without hand-typing.
    """
    durations = [seconds for seconds in range(MIN_SECONDS, MAX_SECONDS + 1)]
    return {
        "id": model_id,
        "object": "model",
        "owned_by": "h3-910a",
        "videoCapabilities": {
            "outputs": [
                {"resolution": "480P", "durations": durations, "width": 864, "height": 480},
                {"resolution": "768P", "durations": durations, "width": 1280, "height": 736},
            ],
            "defaultResolution": "480P",
            "defaultDuration": 5,
            "dimensionStep": 32,
        },
    }


def models_payload(model_id: str) -> dict[str, Any]:
    return {"object": "list", "data": [capabilities(model_id)]}


def status_payload(job_id: str, status: str, *, error: str | None = None,
                   progress: float | None = None) -> dict[str, Any]:
    """One ``GET /videos/{id}`` body."""
    body: dict[str, Any] = {"id": job_id, "status": status}
    if progress is not None:
        body["progress"] = round(float(progress), 4)
    if error:
        body["error"] = {"message": error}
    return body