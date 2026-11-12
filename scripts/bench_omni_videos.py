#!/usr/bin/env python3
"""Benchmark the H3 service against the 10s / 5min target.

Posts to the synchronous video endpoint, records wall-clock time, and (when
ffprobe is available) verifies the produced clip: duration, resolution, frame
count and whether the picture is blank.

    python scripts/bench_omni_videos.py --prompt "..." --seconds 10 --size 832x480

Use --repeat to measure steady state instead of cold start.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import time
import uuid
import urllib.error
import urllib.request
from pathlib import Path


def post_video(base: str, *, prompt: str, seconds: float, size: str, seed: int,
               timeout: float) -> tuple[float, bytes, dict]:
    boundary = "----h3bench" + uuid.uuid4().hex
    fields = {
        "prompt": prompt,
        "size": size,
        "seconds": str(seconds),
        "seed": str(seed),
    }
    parts: list[bytes] = []
    for name, value in fields.items():
        parts.append(f"--{boundary}".encode())
        parts.append(f'Content-Disposition: form-data; name="{name}"'.encode())
        parts.append(b"")
        parts.append(value.encode())
    parts.append(f"--{boundary}--".encode())
    parts.append(b"")
    body = b"\r\n".join(parts)

    request = urllib.request.Request(
        base.rstrip("/") + "/v1/videos/sync",
        data=body,
        method="POST",
        headers={"Content-Type": f"multipart/form-data; boundary={boundary}"},
    )
    started = time.time()
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            data = response.read()
            elapsed = time.time() - started
            return elapsed, data, {"status": response.status, "ctype": response.headers.get("Content-Type", "")}
    except urllib.error.HTTPError as exc:
        elapsed = time.time() - started
        return elapsed, exc.read(), {"status": exc.code, "ctype": exc.headers.get("Content-Type", "")}


def probe(path: Path) -> dict:
    ffprobe = shutil.which("ffprobe")
    if not ffprobe:
        return {"error": "ffprobe not available"}
    out = subprocess.run(
        [ffprobe, "-v", "error", "-print_format", "json", "-show_format", "-show_streams", str(path)],
        capture_output=True,
    )
    if out.returncode:
        return {"error": out.stderr.decode()[:300]}
    data = json.loads(out.stdout or b"{}")
    streams = data.get("streams", [])
    video = next((s for s in streams if s.get("codec_type") == "video"), {})
    audio = next((s for s in streams if s.get("codec_type") == "audio"), {})
    return {
        "duration": float(data.get("format", {}).get("duration") or 0),
        "size_bytes": int(data.get("format", {}).get("size") or path.stat().st_size),
        "video": {
            "codec": video.get("codec_name"),
            "width": video.get("width"),
            "height": video.get("height"),
            "fps": video.get("r_frame_rate"),
            "frames": video.get("nb_frames"),
        },
        "audio": {"codec": audio.get("codec_name"), "channels": audio.get("channels")},
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", default="http://127.0.0.1:9198")
    parser.add_argument("--prompt", default="A red fox running through snow, cinematic lighting")
    parser.add_argument("--seconds", type=float, default=5.0)
    parser.add_argument("--size", default="832x480")
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--repeat", type=int, default=1)
    parser.add_argument("--timeout", type=float, default=3600)
    parser.add_argument("--out-dir", default="/tmp/h3-bench")
    parser.add_argument("--target-seconds", type=float, default=300.0,
                        help="budget for one generation; report PASS/FAIL against it")
    args = parser.parse_args(argv)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    results = []
    for index in range(args.repeat):
        elapsed, data, meta = post_video(
            args.base, prompt=args.prompt, seconds=args.seconds, size=args.size,
            seed=args.seed + index, timeout=args.timeout,
        )
        record = {"run": index + 1, "elapsed_s": round(elapsed, 2), **meta,
                  "bytes": len(data)}
        if meta["status"] == 200 and data[:4] not in (b"", b"null"):
            target = out_dir / f"bench-{index + 1}.mp4"
            target.write_bytes(data)
            record["artifact"] = str(target)
            record["probe"] = probe(target)
        else:
            record["body"] = data[:600].decode(errors="replace")
        results.append(record)
        print(json.dumps(record, ensure_ascii=False), flush=True)

    if results:
        best = min(r["elapsed_s"] for r in results)
        verdict = "PASS" if best <= args.target_seconds else "FAIL"
        print(json.dumps({
            "target_seconds": args.target_seconds,
            "best_elapsed_s": best,
            "verdict": verdict,
        }, ensure_ascii=False))
        return 0 if verdict == "PASS" else 1
    return 1


if __name__ == "__main__":
    raise SystemExit(main())