#!/usr/bin/env python3
"""A vLLM-Omni compatible ``/videos`` service in front of the H3 ComfyUI graph.

OpenDirector's "vLLM-Omni 兼容 / 视频接口" connection type speaks three
endpoints: submit ``POST /videos`` (multipart), poll ``GET /videos/{id}`` and
download ``GET /videos/{id}/content``.  This module serves exactly those, and
turns each accepted job into one ComfyUI ``/prompt`` submission of the H3
workflow, so ComfyUI stays a pure execution backend.

Run it next to a live instance, one process per four-card group::

    python scripts/h3_vllm_omni_server.py \
        --comfy http://127.0.0.1:8801 \
        --workflow artifacts/design/workflows/instance-1-target.json \
        --port 9101
"""

from __future__ import annotations

import argparse
import asyncio
import copy
import json
import logging
import os
import sys
import tempfile
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import aiohttp
from aiohttp import web

sys.path.insert(0, str(Path(__file__).resolve().parent))

import h3_omni_contract as contract  # noqa: E402
from h3_workflow import (  # noqa: E402
    Overrides,
    WorkflowError,
    apply_overrides,
    detect_format,
    load_workflow,
    ui_to_api,
    validate_api,
)

LOG = logging.getLogger("h3-omni")
MAX_BODY = 160 * 1024 * 1024
MAX_ARTIFACT = 512 * 1024 * 1024


# --------------------------------------------------------------------------
# ComfyUI client
# --------------------------------------------------------------------------


class ComfyError(RuntimeError):
    pass


class ComfyClient:
    """The four ComfyUI calls this service needs, with a shared session."""

    def __init__(self, base: str, session: aiohttp.ClientSession, timeout: float = 60.0):
        self.base = base.rstrip("/")
        self.session = session
        self.timeout = aiohttp.ClientTimeout(total=timeout)

    async def _json(self, method: str, path: str, **kwargs: Any) -> Any:
        async with self.session.request(method, self.base + path, timeout=self.timeout, **kwargs) as response:
            body = await response.read()
            if response.status >= 400:
                raise ComfyError(
                    f"{method} {path} -> HTTP {response.status}: {body[:400]!r}"
                )
            return json.loads(body) if body else None

    async def object_info(self) -> dict[str, Any]:
        value = await self._json("GET", "/object_info")
        if not isinstance(value, dict):
            raise ComfyError("/object_info did not return an object")
        return value

    async def upload_image(self, upload: contract.Upload, subfolder: str) -> str:
        form = aiohttp.FormData()
        form.add_field(
            "image",
            upload.data,
            filename=upload.filename or "reference.png",
            content_type=upload.content_type or "image/png",
        )
        form.add_field("overwrite", "true")
        if subfolder:
            form.add_field("subfolder", subfolder)
        async with self.session.post(
            self.base + "/upload/image", data=form, timeout=self.timeout
        ) as response:
            body = await response.read()
            if response.status >= 400:
                raise ComfyError(f"/upload/image -> HTTP {response.status}: {body[:400]!r}")
        value = json.loads(body)
        name = value.get("name")
        if not isinstance(name, str) or not name:
            raise ComfyError(f"/upload/image returned no name: {value!r}")
        sub = value.get("subfolder") or ""
        return f"{sub}/{name}" if sub else name

    async def submit(self, prompt: dict[str, Any]) -> str:
        value = await self._json(
            "POST", "/prompt", json={"prompt": prompt, "client_id": uuid.uuid4().hex}
        )
        node_errors = (value or {}).get("node_errors")
        if node_errors:
            raise ComfyError("/prompt node_errors: " + json.dumps(node_errors, ensure_ascii=False)[:1500])
        prompt_id = (value or {}).get("prompt_id")
        if not isinstance(prompt_id, str) or not prompt_id:
            raise ComfyError(f"/prompt returned no prompt_id: {value!r}")
        return prompt_id

    async def queue(self) -> dict[str, Any]:
        return await self._json("GET", "/queue") or {}

    async def history(self, prompt_id: str) -> dict[str, Any] | None:
        value = await self._json("GET", f"/history/{prompt_id}")
        entry = (value or {}).get(prompt_id)
        return entry if isinstance(entry, dict) else None

    async def artifact(self, filename: str, subfolder: str, folder: str) -> bytes:
        params = {"filename": filename, "subfolder": subfolder, "type": folder}
        async with self.session.get(
            self.base + "/view", params=params, timeout=aiohttp.ClientTimeout(total=600)
        ) as response:
            if response.status >= 400:
                raise ComfyError(f"/view -> HTTP {response.status}")
            data = await response.read()
        if len(data) > MAX_ARTIFACT:
            raise ComfyError("产物超过 512 MB")
        return data

    async def delete_queued(self, prompt_id: str) -> None:
        try:
            await self._json("POST", "/queue", json={"delete": [prompt_id]})
        except ComfyError as exc:  # best effort
            LOG.warning("queue delete failed: %s", exc)

    async def interrupt(self) -> None:
        try:
            await self._json("POST", "/interrupt", json={})
        except ComfyError as exc:  # best effort
            LOG.warning("interrupt failed: %s", exc)


# --------------------------------------------------------------------------
# Jobs
# --------------------------------------------------------------------------


@dataclass
class Job:
    id: str
    status: str = contract.STATUS_QUEUED
    created: float = field(default_factory=time.time)
    updated: float = field(default_factory=time.time)
    prompt_id: str | None = None
    error: str | None = None
    artifact_path: Path | None = None
    cancel_requested: bool = False
    task: asyncio.Task[None] | None = None

    def touch(self, status: str) -> None:
        self.status = status
        self.updated = time.time()


class Service:
    def __init__(self, args: argparse.Namespace) -> None:
        self.args = args
        self.comfy: ComfyClient | None = None
        self.session: aiohttp.ClientSession | None = None
        self.jobs: dict[str, Job] = {}
        self.workflow = load_workflow(args.workflow)
        if detect_format(self.workflow) != "ui":
            raise SystemExit("--workflow must be the ComfyUI UI JSON, not an API prompt")
        self._object_info: dict[str, Any] | None = None
        self._object_info_at = 0.0
        self.output_dir = Path(args.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    # -- lifecycle ---------------------------------------------------------
    async def start(self) -> None:
        self.session = aiohttp.ClientSession()
        self.comfy = ComfyClient(self.args.comfy, self.session)

    async def stop(self) -> None:
        for job in self.jobs.values():
            if job.task and not job.task.done():
                job.task.cancel()
        if self.session:
            await self.session.close()

    # -- helpers -----------------------------------------------------------
    async def object_info(self, force: bool = False) -> dict[str, Any]:
        if force or self._object_info is None or time.time() - self._object_info_at > 300:
            self._object_info = await self.comfy.object_info()
            self._object_info_at = time.time()
        return self._object_info

    def _state(self, job_id: str) -> Job | None:
        return self.jobs.get(job_id)

    # -- submit ------------------------------------------------------------
    async def submit(self, request: web.Request) -> web.Response:
        fields: dict[str, str] = {}
        uploads: list[contract.Upload] = []
        reader = await request.multipart()
        size = 0
        async for part in reader:
            size += await self._read_part(part, fields, uploads)
            if size > MAX_BODY:
                raise contract.RequestError("请求体超过 160 MB。", status=413, code="payload_too_large")

        spec = contract.build_request(fields, uploads)
        job = Job(id="vid_" + uuid.uuid4().hex)
        self.jobs[job.id] = job
        try:
            await self._launch(job, spec)
        except (WorkflowError, ComfyError) as exc:
            job.touch(contract.STATUS_FAILED)
            job.error = str(exc)
            raise contract.RequestError(f"提交到 ComfyUI 失败：{exc}", status=502, code="backend_error") from None
        job.touch(contract.STATUS_QUEUED)
        LOG.info("job %s queued as prompt %s", job.id, job.prompt_id)
        return web.json_response(
            {"id": job.id, "status": job.status, "created": int(job.created)}, status=202
        )

    async def _read_part(
        self, part: Any, fields: dict[str, str], uploads: list[contract.Upload]
    ) -> int:
        if part.name in {"input_reference", "input_references"} and part.filename:
            data = await part.read()
            uploads.append(
                contract.Upload(
                    field=part.name,
                    filename=part.filename,
                    content_type=(part.headers.get("Content-Type") or "").split(";")[0].strip(),
                    data=data,
                )
            )
            return len(data)
        fields[part.name] = await part.text()
        return 0

    async def _launch(self, job: Job, spec: contract.VideoRequest) -> None:
        comfy = self.comfy
        assert comfy is not None

        overrides = spec.overrides(filename_prefix=f"h3-omni/{job.id}")
        if self.args.npu_count:
            # Debug switch: 1 runs the graph single-device and skips the
            # sequence-parallel collectives entirely.
            overrides["npu_count"] = int(self.args.npu_count)
        if spec.image is not None:
            name = await comfy.upload_image(spec.image, subfolder=f"h3-omni/{job.id}")
            overrides["first_image"] = name

        workflow, _ = apply_overrides(self.workflow, Overrides(**overrides))
        api = ui_to_api(workflow, await self.object_info())
        validate_api(api)

        job.prompt_id = await comfy.submit(api)
        job.touch(contract.STATUS_QUEUED)
        job.task = asyncio.create_task(self._watch(job))

    # -- polling -----------------------------------------------------------
    async def _watch(self, job: Job) -> None:
        assert self.comfy is not None and job.prompt_id
        deadline = time.time() + self.args.timeout
        running_at: float | None = None
        while time.time() < deadline:
            if job.cancel_requested:
                job.touch(contract.STATUS_CANCELLED)
                return
            try:
                history = await self.comfy.history(job.prompt_id)
                if history is not None:
                    await self._finish(job, history)
                    return
                if running_at is None:
                    running_at = await self._running(job.prompt_id)
                    if running_at is not None:
                        job.touch(contract.STATUS_RUNNING)
            except ComfyError as exc:
                LOG.warning("poll %s failed: %s", job.id, exc)
            await asyncio.sleep(self.args.poll)
        job.touch(contract.STATUS_FAILED)
        job.error = f"生成超时（{self.args.timeout:.0f} 秒）"

    async def _running(self, prompt_id: str) -> float | None:
        assert self.comfy is not None
        queue = await self.comfy.queue()
        for item in queue.get("queue_running") or []:
            if isinstance(item, list) and len(item) > 1 and item[1] == prompt_id:
                return time.time()
        return None

    async def _finish(self, job: Job, history: dict[str, Any]) -> None:
        status = history.get("status") or {}
        if isinstance(status, dict) and status.get("status_str") == "error":
            job.touch(contract.STATUS_FAILED)
            job.error = _error_text(history) or "ComfyUI 执行失败"
            return
        entry = _find_artifact(history)
        if entry is None:
            job.touch(contract.STATUS_FAILED)
            job.error = "ComfyUI 完成但没有返回视频文件"
            return
        assert self.comfy is not None
        try:
            data = await self.comfy.artifact(
                str(entry["filename"]), str(entry.get("subfolder") or ""), str(entry.get("type") or "output")
            )
        except ComfyError as exc:
            job.touch(contract.STATUS_FAILED)
            job.error = f"下载产物失败：{exc}"
            return
        path = self.output_dir / f"{job.id}.mp4"
        path.write_bytes(data)
        job.artifact_path = path
        job.touch(contract.STATUS_COMPLETED)
        LOG.info("job %s completed (%d bytes)", job.id, len(data))

    # -- routes ------------------------------------------------------------
    async def get_job(self, request: web.Request) -> web.Response:
        job = self._state(request.match_info["job_id"])
        if job is None:
            raise web.HTTPNotFound(text=json.dumps({"detail": "任务不存在"}, ensure_ascii=False),
                                   content_type="application/json")
        return web.json_response(
            contract.status_payload(job.id, job.status, error=job.error)
        )

    async def get_content(self, request: web.Request) -> web.StreamResponse:
        job = self._state(request.match_info["job_id"])
        if job is None or job.artifact_path is None or not job.artifact_path.exists():
            raise web.HTTPNotFound(text=json.dumps({"detail": "产物不存在"}, ensure_ascii=False),
                                   content_type="application/json")
        return web.FileResponse(job.artifact_path, headers={"Content-Type": "video/mp4"})

    async def delete_job(self, request: web.Request) -> web.Response:
        job = self._state(request.match_info["job_id"])
        if job is None:
            raise web.HTTPNotFound(text=json.dumps({"detail": "任务不存在"}, ensure_ascii=False),
                                   content_type="application/json")
        if not contract.is_terminal(job.status):
            job.cancel_requested = True
            assert self.comfy is not None and job.prompt_id
            await self.comfy.delete_queued(job.prompt_id)
            if job.status == contract.STATUS_RUNNING:
                await self.comfy.interrupt()
            job.touch(contract.STATUS_CANCELLED)
        return web.json_response({"id": job.id, "status": job.status})

    async def models(self, request: web.Request) -> web.Response:
        return web.json_response(contract.models_payload(self.args.model))

    async def health(self, request: web.Request) -> web.Response:
        return web.json_response({"status": "ok", "comfy": self.args.comfy})


def _find_artifact(history: dict[str, Any]) -> dict[str, Any] | None:
    outputs = history.get("outputs")
    if not isinstance(outputs, dict):
        return None
    for node_output in outputs.values():
        if not isinstance(node_output, dict):
            continue
        for entries in node_output.values():
            if not isinstance(entries, list):
                continue
            for entry in entries:
                if isinstance(entry, dict) and isinstance(entry.get("filename"), str):
                    if str(entry["filename"]).lower().endswith((".mp4", ".webm", ".mkv", ".mov")):
                        return entry
    return None


def _error_text(history: dict[str, Any]) -> str:
    status = history.get("status") or {}
    for message in status.get("messages") or []:
        if isinstance(message, list) and len(message) > 1:
            detail = message[1]
            if isinstance(detail, dict) and detail.get("exception_message"):
                return str(detail["exception_message"])[:800]
    return ""


# --------------------------------------------------------------------------
# Wiring
# --------------------------------------------------------------------------


@web.middleware
async def error_middleware(request: web.Request, handler: Any) -> web.StreamResponse:
    """Render request/backend failures as JSON instead of aiohttp's HTML 500."""
    try:
        return await handler(request)
    except contract.RequestError as exc:
        return web.json_response(exc.payload(), status=exc.status)
    except (WorkflowError, ComfyError) as exc:
        return web.json_response(
            {"detail": f"后端错误：{exc}",
             "error": {"code": "backend_error", "message": str(exc)}},
            status=502,
        )


def build_app(service: Service) -> web.Application:
    app = web.Application(client_max_size=MAX_BODY, middlewares=[error_middleware])
    for prefix in ("", "/v1"):
        app.router.add_get(f"{prefix}/health", service.health)
        app.router.add_get(f"{prefix}/models", service.models)
        app.router.add_post(f"{prefix}/videos", service.submit)
        app.router.add_get(f"{prefix}/videos/{{job_id}}", service.get_job)
        app.router.add_get(f"{prefix}/videos/{{job_id}}/content", service.get_content)
        app.router.add_delete(f"{prefix}/videos/{{job_id}}", service.delete_job)

    async def _start(app: web.Application) -> None:
        await service.start()

    async def _stop(app: web.Application) -> None:
        await service.stop()

    app.on_startup.append(_start)
    app.on_cleanup.append(_stop)
    return app


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--comfy", default="http://127.0.0.1:8801", help="ComfyUI base URL")
    parser.add_argument("--workflow", required=True, help="ComfyUI UI workflow JSON")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=9101)
    parser.add_argument("--model", default="minimax-h3", help="Model ID advertised on /models")
    parser.add_argument("--npu-count", type=int, default=0,
                        help="Override the workflow's parallel NPU count (0 keeps the workflow value)")
    parser.add_argument("--output-dir", default=os.path.join(tempfile.gettempdir(), "h3-omni"))
    parser.add_argument("--poll", type=float, default=2.0, help="ComfyUI poll interval (s)")
    parser.add_argument("--timeout", type=float, default=3600.0, help="Job timeout (s)")
    parser.add_argument("--log-level", default="INFO")
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=getattr(logging, str(args.log_level).upper(), logging.INFO),
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )
    service = Service(args)
    web.run_app(build_app(service), host=args.host, port=args.port, print=None)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())