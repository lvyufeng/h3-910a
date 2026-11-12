#!/usr/bin/env python3
"""Submit one job to the vLLM-Omni compatible H3 service and record the run.

    python scripts/smoke_vllm_omni.py --base http://127.0.0.1:9101 \
        --prompt "a red fox" --frames 107 --size 608x352 --out /tmp/smoke.mp4

Uses only the standard library so it runs both on the dev machine and on
ts-217.
"""

from __future__ import annotations

import argparse
import json
import mimetypes
import os
import subprocess
import sys
import time
import urllib.error
import urllib.request
import uuid
from pathlib import Path


def _multipart(fields: dict[str, str], files: list[tuple[str, Path]]) -> tuple[bytes, str]:
    boundary = "----h3omni" + uuid.uuid4().hex
    lines: list[bytes] = []
    for name, value in fields.items():
        lines.append(f"--{boundary}".encode())
        lines.append(f'Content-Disposition: form-data; name="{name}"'.encode())
        lines.append(b"")
        lines.append(str(value).encode())
    for name, path in files:
        mime = mimetypes.guess_type(path.name)[0] or "application/octet-stream"
        lines.append(f"--{boundary}".encode())
        lines.append(
            f'Content-Disposition: form-data; name="{name}"; filename="{path.name}"'.encode()
        )
        lines.append(f"Content-Type: {mime}".encode())
        lines.append(b"")
        lines.append(path.read_bytes())
    lines.append(f"--{boundary}--".encode())
    lines.append(b"")
    return b"\r\n".join(lines), f"multipart/form-data; boundary={boundary}"


def _request(url: str, method: str = "GET", body: bytes | None = None,
             content_type: str | None = None, timeout: float = 300) -> tuple[int, bytes]:
    request = urllib.request.Request(url, data=body, method=method)
    if content_type:
        request.add_header("Content-Type", content_type)
    request.add_header("Accept", "application/json")
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return response.status, response.read()
    except urllib.error.HTTPError as exc:
        return exc.code, exc.read()


def _probe(path: Path) -> dict | None:
    try:
        out = subprocess.run(
            ["ffprobe", "-v", "error", "-print_format", "json",
             "-show_format", "-show_streams", str(path)],
            capture_output=True, timeout=60,
        )
    except (OSError, subprocess.TimeoutExpired):
        return None
    if out.returncode:
        return {"error": out.stderr.decode()[:300]}
    data = json.loads(out.stdout or b"{}")
    return {
        "size_bytes": path.stat().st_size,
        "duration": float(data.get("format", {}).get("duration") or 0),
        "streams": [
            {"codec_type": s.get("codec_type"), "codec": s.get("codec_name"),
             "width": s.get("width"), "height": s.get("height"),
             "r_frame_rate": s.get("r_frame_rate")}
            for s in data.get("streams", [])
        ],
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", default="http://127.0.0.1:9101")
    parser.add_argument("--prompt", default="A red fox running through snow, cinematic")
    parser.add_argument("--frames", default="107", help="H3 17n+5 frame count")
    parser.add_argument("--size", default="608x352")
    parser.add_argument("--fps", default="24")
    parser.add_argument("--task", default="t2va", choices=["t2va", "fl2va", "ref2va"])
    parser.add_argument("--image", default="", help="first-frame image for fl2va/ref2va")
    parser.add_argument("--seed", default="")
    parser.add_argument("--out", default="", help="where to save the mp4")
    parser.add_argument("--timeout", type=float, default=3600)
    parser.add_argument("--poll", type=float, default=5)
    args = parser.parse_args(argv)

    fields = {
        "model": "minimax-h3",
        "prompt": args.prompt,
        "num_frames": args.frames,
        "size": args.size,
        "fps": args.fps,
        "extra_params": json.dumps({"task": args.task}),
    }
    if args.seed:
        fields["seed"] = args.seed
    files: list[tuple[str, Path]] = []
    if args.image:
        files.append(("input_reference", Path(args.image)))

    body, content_type = _multipart(fields, files)
    started = time.time()
    status, payload = _request(args.base.rstrip("/") + "/videos", "POST", body, content_type)
    if status not in (200, 201, 202):
        print(json.dumps({"stage": "submit", "http": status,
                          "body": payload.decode(errors="replace")[:2000]}, ensure_ascii=False))
        return 1
    job = json.loads(payload)
    job_id = job["id"]
    print(f"submitted {job_id} (http {status})", flush=True)

    deadline = started + args.timeout
    state = job
    first_seen: dict[str, float] = {}
    while time.time() < deadline:
        status, payload = _request(args.base.rstrip("/") + f"/videos/{job_id}")
        state = json.loads(payload) if payload else {"status": f"http_{status}"}
        current = state.get("status")
        first_seen.setdefault(current, round(time.time() - started, 1))
        print(f"  {time.time() - started:7.1f}s {current}", flush=True)
        if current in {"completed", "failed", "cancelled", "expired"}:
            break
        time.sleep(args.poll)

    summary = {
        "job": job_id,
        "status": state.get("status"),
        "seconds_to_first_state": first_seen,
        "total_seconds": round(time.time() - started, 1),
        "error": (state.get("error") or {}).get("message") if isinstance(state.get("error"), dict) else state.get("error"),
    }
    if state.get("status") == "completed":
        status, data = _request(args.base.rstrip("/") + f"/videos/{job_id}/content", timeout=900)
        target = Path(args.out or f"/tmp/h3-omni-{job_id}.mp4")
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(data)
        summary["artifact"] = str(target)
        summary["http"] = status
        summary["probe"] = _probe(target)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0 if summary["status"] == "completed" else 1


if __name__ == "__main__":
    raise SystemExit(main())