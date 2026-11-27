"""MiniMax-H3 FL2VA inference service for 8x Ascend 910A, OpenDirector-compatible.

Two API surfaces are exposed from one resident engine:

``/v1/videos`` (the surface OpenDirector's ``vllm_omni`` protocol speaks):
    POST /v1/videos                  multipart submit -> {"id": "..."}
    GET  /v1/videos/{id}             job metadata, ``status`` in
                                     queued/in_progress/completed/failed/cancelled/expired
    GET  /v1/videos/{id}/content     the finished MP4 (video+audio)
    GET  /v1/models                  served model list

``/v1/generate`` (a small native facade used for smoke tests and scripts) plus
``/healthz``.

Why a gateway: this vllm-omni build (0.30.0 dev) ships the video request models
and multipart helpers, but does not register the ``/v1/videos`` routes, so the
routes are implemented here on top of the same resident ``Omni`` engine. The
numeric path stays the verified R124/R126 one; every speedup comes from
:mod:`h3_optimizations` and is installed in each worker by ``sitecustomize``.

OpenDirector mapping (Settings -> model connections -> video):
    access mode : self-hosted service
    protocol    : vLLM-Omni
    base URL    : http://127.0.0.1:8091/v1
    model id    : minimax-h3   (the adapter is selected by this exact name)
    auth        : none
"""
import base64
import io
import json
import os
import queue
import threading
import time
import traceback
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel

SERVED_MODEL_NAME = os.environ.get("H3_SERVED_MODEL_NAME", "minimax-h3")
DEFAULT_STEPS = int(os.environ.get("H3_DEFAULT_STEPS", "49"))
DEFAULT_FPS = 24
SUPPORTED_SIZES = {(1344, 768), (768, 1344), (832, 480), (480, 832), (1024, 1024), (1280, 720), (720, 1280)}


def logger(message: str) -> None:
    print(f"[h3-service {time.strftime('%H:%M:%S')}] {message}", flush=True)


# ---------------------------------------------------------------------------
# Jobs
# ---------------------------------------------------------------------------
@dataclass
class Job:
    job_id: str
    prompt: str
    params: dict[str, Any]
    status: str = "queued"          # queued | running | done | error
    created_at: float = field(default_factory=time.time)
    started_at: float | None = None
    finished_at: float | None = None
    error: str | None = None
    error_message: str | None = None
    assets: dict[str, str] = field(default_factory=dict)
    metrics: dict[str, Any] = field(default_factory=dict)

    def public_status(self) -> str:
        return {"queued": "queued", "running": "in_progress", "done": "completed",
                "error": "failed"}.get(self.status, self.status)

    def progress(self) -> int:
        if self.status in ("done",):
            return 100
        if self.status == "running":
            return 50
        return 0

    def as_dict(self) -> dict[str, Any]:
        data: dict[str, Any] = {
            "id": self.job_id,
            "object": "video",
            "model": SERVED_MODEL_NAME,
            "created_at": int(self.created_at),
            "status": self.public_status(),
            "progress": self.progress(),
            "metrics": self.metrics,
            "params": _json_safe(self.params),
        }
        if self.started_at:
            data["started_at"] = int(self.started_at)
        if self.finished_at:
            data["finished_at"] = int(self.finished_at)
            data["elapsed_s"] = round(self.finished_at - self.started_at, 2) if self.started_at else None
        if self.status == "error":
            data["error"] = {"message": self.error_message or "generation failed",
                             "type": "server_error", "code": "generation_failed"}
        return data


def _json_safe(value: Any) -> Any:
    """Drop engine-only payloads (e.g. decoded PIL images) from public metadata."""
    if isinstance(value, dict):
        return {key: _json_safe(item) for key, item in value.items() if _json_safe_scalar(item)}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    return value if _json_safe_scalar(value) else str(type(value).__name__)


def _json_safe_scalar(value: Any) -> bool:
    return isinstance(value, (str, int, float, bool, type(None), list, tuple, dict))


class JobError(Exception):
    """A request-level rejection that maps onto an HTTP 4xx answer."""

    def __init__(self, message: str, status_code: int = 400) -> None:
        super().__init__(message)
        self.message = message
        self.status_code = status_code


# ---------------------------------------------------------------------------
# Engine
# ---------------------------------------------------------------------------
class EngineHolder:
    def __init__(self, checkpoint: str, output_root: Path) -> None:
        self.checkpoint = checkpoint
        self.output_root = output_root
        self.omni = None
        self.error: str | None = None
        self.ready = threading.Event()
        self.init_s: float | None = None

    def start(self) -> None:
        threading.Thread(target=self._build, name="h3-engine-init", daemon=True).start()

    def _build(self) -> None:
        try:
            import h3_service_bootstrap

            h3_service_bootstrap.install_parent_alignment()
            import torch
            import torch_npu  # noqa: F401
            from vllm_omni.entrypoints.omni import Omni

            from h3_optimizations import install as install_optimizations

            install_optimizations(verbose=True)
            torch.npu.matmul.allow_hf32 = False
            torch.npu.conv.allow_hf32 = False
            t0 = time.perf_counter()
            self.omni = Omni(model=self.checkpoint, **_engine_kwargs())
            self.init_s = round(time.perf_counter() - t0, 2)
            logger(f"engine ready in {self.init_s}s")
            if os.environ.get("H3_SERVICE_PREWARM", "0").strip().lower() in ("1", "true", "on", "yes"):
                self._prewarm()
        except Exception:
            self.error = traceback.format_exc()
            logger("engine init failed:\n" + self.error)
        finally:
            self.ready.set()

    def _prewarm(self) -> None:
        """Run one throwaway generation so the first real request is warm.

        The engine pays a one-time cost on its first request (VAE cold decode and
        first-call kernel setup). Prewarming moves that cost into startup.
        """
        try:
            from PIL import Image
            from vllm_omni.inputs.data import OmniDiffusionSamplingParams

            frame = Image.new("RGB", (1344, 768), (128, 128, 128))
            sampling = OmniDiffusionSamplingParams(
                seed=1234, height=768, width=1344, num_frames=120, fps=24, frame_rate=24,
                num_inference_steps=DEFAULT_STEPS, quality="lossless", output_type="np",
                return_frames=False, save_output=False,
                extra_args={"task": "fl2va", "duration_seconds": 5, "audio_mode": "native",
                            "frame_indices": [0, -1]},
            )
            t0 = time.perf_counter()
            logger("prewarm generation started")
            self.omni.generate({"prompt": "prewarm", "multi_modal_data": {"image": [frame, frame]}},
                               sampling_params_list=sampling, use_tqdm=False)
            logger(f"prewarm finished in {time.perf_counter() - t0:.1f}s")
        except Exception as exc:  # prewarm must never take the service down
            logger(f"prewarm failed (ignored): {exc!r}")


def _engine_kwargs() -> dict[str, Any]:
    """The verified R126 engine configuration (TP2 DiT, USP4, live FP16 encoder TP8)."""
    return dict(
        trust_remote_code=True,
        task_type="fl2va",
        num_gpus=8,
        tensor_parallel_size=2,
        text_encoder_tp_size=8,
        ulysses_degree=4,
        ring_degree=1,
        dtype="half",
        enforce_eager=True,
        enable_layerwise_offload=True,
        vae_use_tiling=True,
        vae_parallel_mode="tile",
        vae_patch_parallel_size=8,
        diffusion_attention_backend="TORCH_SDPA",
        stage_init_timeout=2400,
        init_timeout=2400,
    )


# ---------------------------------------------------------------------------
# Worker
# ---------------------------------------------------------------------------
class Worker(threading.Thread):
    def __init__(self, engine: EngineHolder, jobs: dict[str, Job], jobs_lock: threading.Lock,
                 inbox: "queue.Queue[str]") -> None:
        super().__init__(name="h3-worker", daemon=True)
        self.engine = engine
        self.jobs = jobs
        self.jobs_lock = jobs_lock
        self.inbox = inbox

    def run(self) -> None:
        while True:
            job_id = self.inbox.get()
            job = self.jobs[job_id]
            job.status = "running"
            job.started_at = time.time()
            try:
                self._run_job(job)
                job.status = "done"
            except Exception as exc:
                job.status = "error"
                job.error = traceback.format_exc()
                job.error_message = str(exc).splitlines()[0][:500] or "generation failed"
                logger(f"job {job_id} failed: {job.error_message}")
            finally:
                job.finished_at = time.time()
                self.inbox.task_done()

    def _run_job(self, job: Job) -> None:
        import numpy as np
        from vllm_omni.inputs.data import OmniDiffusionSamplingParams

        from media_helpers124 import export_av, find_audio, find_frames

        params = job.params
        sampling = OmniDiffusionSamplingParams(
            seed=int(params.get("seed", 1234)),
            height=int(params["height"]),
            width=int(params["width"]),
            num_frames=int(params["num_frames"]),
            fps=int(params.get("fps", DEFAULT_FPS)),
            frame_rate=int(params.get("fps", DEFAULT_FPS)),
            num_inference_steps=int(params.get("num_inference_steps", DEFAULT_STEPS)),
            quality="lossless",
            output_type="np",
            return_frames=True,
            save_output=False,
            extra_args={"task": "fl2va",
                        "duration_seconds": int(params.get("duration_seconds", 5)),
                        "audio_mode": "native",
                        "frame_indices": [0, -1]},
        )
        request = {"prompt": job.prompt,
                   "multi_modal_data": {"image": [params["first_frame_image"], params["last_frame_image"]]}}
        t0 = time.perf_counter()
        outputs = self.engine.omni.generate(request, sampling_params_list=sampling, use_tqdm=False)
        generate_s = time.perf_counter() - t0
        if len(outputs) != 1 or not outputs[0].finished or outputs[0].error:
            raise RuntimeError(f"generation did not finish cleanly: {outputs!r}")
        output = outputs[0]
        frames = find_frames(output.multimodal_output) or find_frames(output.images)
        audio = find_audio(output.multimodal_output)
        if frames is None or audio is None:
            raise RuntimeError("generation produced no frames or no audio")

        out_dir = self.engine.output_root / job.job_id
        out_dir.mkdir(parents=True, exist_ok=True)
        video_path = out_dir / "video.mp4"
        t1 = time.perf_counter()
        export_av(frames, audio, video_path, fps=int(params.get("fps", DEFAULT_FPS)), sample_rate=32000)
        mux_s = time.perf_counter() - t1
        job.assets = {"video": str(video_path)}
        job.metrics = {
            "engine_init_s": self.engine.init_s,
            "generate_s": round(generate_s, 2),
            "mux_s": round(mux_s, 2),
            "frames": int(frames.shape[0]),
            "frame_hw": [int(frames.shape[1]), int(frames.shape[2])],
            "audio_samples": int(audio.shape[-1]),
        }
        logger(f"job {job.job_id} done in {generate_s:.1f}s (+{mux_s:.1f}s mux)")


# ---------------------------------------------------------------------------
# Request translation: OpenDirector /v1/videos multipart -> engine job
# ---------------------------------------------------------------------------
def _decode_upload(data: bytes):
    from PIL import Image

    return Image.open(io.BytesIO(data)).convert("RGB")


def translate_video_request(fields: dict[str, str], uploads: list[tuple[str, bytes, str]]) -> tuple[str, dict[str, Any]]:
    """Map the vLLM-Omni multipart form onto the engine's fl2va parameters.

    Raises :class:`JobError` for anything this checkpoint cannot honour, with a
    message OpenDirector can surface verbatim.
    """
    prompt = fields.get("prompt", "")
    if not prompt.strip():
        raise JobError("prompt is required")
    extra: dict[str, Any] = {}
    if fields.get("extra_params"):
        try:
            extra = json.loads(fields["extra_params"])
        except json.JSONDecodeError as exc:
            raise JobError(f"extra_params is not valid JSON: {exc}") from exc
    task = str(extra.get("task", "fl2va"))
    if task != "fl2va":
        if task == "ref2va":
            raise JobError("this 910A deployment serves the FL2VA checkpoint only; "
                           "reference-to-video (ref2va) is not available", 400)
        if "continue_from" in extra:
            raise JobError("continue_from is not available: this deployment keeps no cached "
                           "latents across requests", 409)
        raise JobError(f"unsupported task {task!r}; only fl2va is served", 400)

    images = [payload for kind, payload, _mime in uploads if kind in ("input_reference", "input_references")]
    others = [kind for kind, _payload, _mime in uploads if kind not in ("input_reference", "input_references")]
    if others:
        raise JobError(f"unsupported reference uploads: {sorted(set(others))}", 400)
    if not images:
        raise JobError("fl2va requires a first frame (input_reference) and a last frame "
                       "(input_references)", 400)
    if len(images) > 2:
        raise JobError("fl2va takes exactly two frames (first + last); "
                       "reference-to-video is not available on this deployment", 400)

    size = fields.get("size") or ""
    width = height = None
    if "x" in size:
        try:
            width, height = (int(v) for v in size.lower().split("x", 1))
        except ValueError as exc:
            raise JobError(f"size must be WIDTHxHEIGHT, got {size!r}") from exc
    if width is None:
        width = int(fields.get("width") or 1344)
        height = int(fields.get("height") or 768)
    if width < 32 or height < 32 or width > 4096 or height > 4096:
        raise JobError(f"size {width}x{height} is outside the supported range", 400)
    if (width, height) not in SUPPORTED_SIZES:
        logger(f"size {width}x{height} is outside the validated buckets; passing through")

    fps = int(float(fields.get("fps") or DEFAULT_FPS))
    num_frames = int(float(fields.get("num_frames") or round(5 * fps)))
    if num_frames < 17 or num_frames > 400:
        raise JobError(f"num_frames={num_frames} is outside the supported range", 400)
    steps = int(float(fields.get("num_inference_steps") or DEFAULT_STEPS))
    if not 1 <= steps <= 200:
        raise JobError(f"num_inference_steps={steps} is outside the supported range", 400)

    seed_value = fields.get("seed")
    seed = int(float(seed_value)) if seed_value not in (None, "") else 1234

    first_image = _decode_upload(images[0])
    last_image = _decode_upload(images[1]) if len(images) > 1 else first_image.copy()
    params = {
        "width": width,
        "height": height,
        "fps": fps,
        "num_frames": num_frames,
        "num_inference_steps": steps,
        "seed": seed,
        "duration_seconds": int(round(num_frames / max(fps, 1))),
        "first_frame_image": first_image,
        "last_frame_image": last_image,
        "requested_task": task,
        "requested_extra": extra,
    }
    return prompt, params


# ---------------------------------------------------------------------------
# HTTP app
# ---------------------------------------------------------------------------
def build_app(engine: EngineHolder, jobs: dict[str, Job], jobs_lock: threading.Lock,
              inbox: "queue.Queue[str]"):
    class GenerateRequest(BaseModel):
        prompt: str
        first_frame: str | None = None
        last_frame: str | None = None
        seed: int = 1234
        height: int = 768
        width: int = 1344
        num_frames: int = 120
        fps: int = 24
        num_inference_steps: int = DEFAULT_STEPS

    app = FastAPI(title="MiniMax-H3 FL2VA (910A)", version="1.0")

    def submit(prompt: str, params: dict[str, Any]) -> str:
        job_id = uuid.uuid4().hex[:16]
        with jobs_lock:
            jobs[job_id] = Job(job_id=job_id, prompt=prompt, params=params)
        inbox.put(job_id)
        logger(f"job {job_id} queued")
        return job_id

    @app.get("/healthz")
    def healthz():
        from h3_optimizations import applied

        return {
            "status": "ok" if engine.omni is not None else ("error" if engine.error else "loading"),
            "engine_ready": engine.omni is not None,
            "engine_error": engine.error,
            "engine_init_s": engine.init_s,
            "queue_len": inbox.qsize(),
            "served_model": SERVED_MODEL_NAME,
            "optimizations": applied(),
        }

    @app.get("/v1/models")
    def list_models():
        return {"object": "list", "data": [{
            "id": SERVED_MODEL_NAME,
            "object": "model",
            "created": int(time.time()),
            "owned_by": "local",
            "root": SERVED_MODEL_NAME,
        }]}

    @app.post("/v1/videos")
    async def create_video(request: Request):
        if engine.omni is None:
            raise HTTPException(status_code=503, detail="engine is not ready")
        content_type = request.headers.get("content-type", "")
        fields: dict[str, str] = {}
        uploads: list[tuple[str, bytes, str]] = []
        if content_type.startswith("multipart/form-data"):
            form = await request.form()
            for key, value in form.multi_items():
                if hasattr(value, "read"):
                    uploads.append((key, await value.read(), value.content_type or "application/octet-stream"))
                else:
                    fields[key] = str(value)
        elif content_type.startswith("application/json"):
            fields = {key: ("" if value is None else str(value)) for key, value in (await request.json()).items()}
        else:
            raise HTTPException(status_code=415, detail="expected multipart/form-data")
        requested_model = fields.get("model")
        if requested_model and requested_model != SERVED_MODEL_NAME:
            raise HTTPException(status_code=404,
                                detail=f"model {requested_model!r} is not served here; use {SERVED_MODEL_NAME!r}")
        try:
            prompt, params = translate_video_request(fields, uploads)
        except JobError as exc:
            return JSONResponse(status_code=exc.status_code,
                                content={"error": {"message": exc.message,
                                                   "type": "invalid_request_error",
                                                   "code": "unsupported_request"}})
        job_id = submit(prompt, params)
        return JSONResponse(status_code=200, content={
            "id": job_id, "object": "video", "model": SERVED_MODEL_NAME,
            "created_at": int(time.time()), "status": "queued", "progress": 0,
        })

    @app.get("/v1/videos/{job_id}")
    def video_status(job_id: str):
        job = jobs.get(job_id)
        if job is None:
            raise HTTPException(status_code=404, detail="unknown video id")
        return job.as_dict()

    @app.get("/v1/videos/{job_id}/content")
    def video_content(job_id: str):
        job = jobs.get(job_id)
        if job is None:
            raise HTTPException(status_code=404, detail="unknown video id")
        if job.status != "done" or "video" not in job.assets:
            raise HTTPException(status_code=409, detail=f"video is {job.public_status()}")
        path = Path(job.assets["video"])
        return FileResponse(str(path), media_type="video/mp4", filename=f"{job_id}.mp4")

    # -- native facade ------------------------------------------------------
    @app.post("/v1/generate")
    def generate(req: GenerateRequest):
        if engine.omni is None:
            raise HTTPException(status_code=503, detail="engine is not ready")
        if not req.first_frame:
            raise HTTPException(status_code=400, detail="first_frame is required")
        first = _decode_image_payload(req.first_frame)
        last = _decode_image_payload(req.last_frame) if req.last_frame else first.copy()
        params = {"width": req.width, "height": req.height, "fps": req.fps,
                  "num_frames": req.num_frames, "num_inference_steps": req.num_inference_steps,
                  "seed": req.seed, "duration_seconds": int(round(req.num_frames / max(req.fps, 1))),
                  "first_frame_image": first, "last_frame_image": last}
        return {"job_id": submit(req.prompt, params), "status": "queued"}

    @app.get("/v1/jobs/{job_id}")
    def job_status(job_id: str):
        job = jobs.get(job_id)
        if job is None:
            raise HTTPException(status_code=404, detail="unknown job")
        return job.as_dict()

    @app.get("/v1/jobs/{job_id}/video")
    def job_video(job_id: str):
        return video_content(job_id)

    return app


def _decode_image_payload(payload: str):
    from PIL import Image

    if payload.startswith("data:"):
        payload = payload.split(",", 1)[1]
    return Image.open(io.BytesIO(base64.b64decode(payload))).convert("RGB")


def main() -> int:
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="/tmp/h3_align12401/original_metadata124f/FL2VA")
    parser.add_argument("--output-root", default="/var/lib/h3-service/outputs")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8091)
    args = parser.parse_args()

    os.environ.setdefault("H3_R126_GENERATION_OUT", args.output_root)
    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    jobs: dict[str, Job] = {}
    jobs_lock = threading.Lock()
    inbox: "queue.Queue[str]" = queue.Queue()

    engine = EngineHolder(args.model, output_root)
    engine.start()
    Worker(engine, jobs, jobs_lock, inbox).start()

    app = build_app(engine, jobs, jobs_lock, inbox)
    import uvicorn

    logger(f"serving {SERVED_MODEL_NAME} on {args.host}:{args.port} (OpenDirector: protocol vLLM-Omni, base .../v1)")
    uvicorn.run(app, host=args.host, port=args.port, log_level="info")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())