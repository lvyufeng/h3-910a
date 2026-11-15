#!/usr/bin/env python3
"""Native unfused FL2VA, explicit 49 intervals, complete synchronized step timers.

First/last references and request dimensions/seed match the prior four-step run.
Private aliases remove the locally pinned student schedule; original metadata and
weights remain unchanged. Storage now matches the existing online FP16 cast for
DLO mmap, with FP32 tensors preserved. This is not bit-exact BF16 inference.
"""
from __future__ import annotations
import hashlib
import io
import json
from PIL import Image, ImageDraw
import os
from pathlib import Path
import re
import signal
import subprocess
import time
import urllib.error
import urllib.request
import uuid

# Remote probes copy the parser to /tmp; repository runs reuse the committed
# dependency-free parser next to the previous audit instead of duplicating it.
try:
    from h3_mp4_metadata import parse as parse_mp4
except ModuleNotFoundError:
    import importlib.util
    parser_path = Path(__file__).resolve().parent.parent / "910a-core-audit" / "mp4_metadata.py"
    spec = importlib.util.spec_from_file_location("h3_mp4_metadata", parser_path)
    parser_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(parser_module)
    parse_mp4 = parser_module.parse

ROOT = Path("/home/mseco/h3-910a")
RUN = Path(os.environ.get("H3_AUDIT_RUN_DIR", "/tmp/h3_round54_e2e02_prefix"))
MODEL_PATH = "/home/mseco/models/MiniMax-H3-native-fp16-audit53"
STARTUP_LIMIT_SECONDS = 1800
REQUEST_LIMIT_SECONDS = 5400
PORT = 9198
API = f"http://127.0.0.1:{PORT}"


def get(path, timeout=8):
    with urllib.request.urlopen(API + path, timeout=timeout) as response:
        return json.load(response)


def interrupted(signum, frame):
    raise SystemExit(f"manager interrupted by signal {signum}")


def main():
    if RUN.exists():
        raise RuntimeError("Refuse to overwrite earlier end-to-end evidence")
    try:
        get("/health", 2)
    except Exception:
        pass
    else:
        raise RuntimeError("Port already serves an existing engine; do not interfere")
    RUN.mkdir()
    env = os.environ.copy()
    env.update({"VLLM_OMNI_910A_PERF": "1", "VLLM_OMNI_910A_FLASH_ATTN": "1",
                "TBE_PARALLEL_COMPILER": "1", "VLLM_OMNI_DLO_DP_WAVE_TIMEOUT": "14400",
                "PYTORCH_NPU_ALLOC_CONF": "expandable_segments:True"})
    env.pop("VLLM_OMNI_910A_AUDIT_DIR", None)
    for key in ("VLLM_OMNI_910A_SKIP_ATTN", "VLLM_OMNI_910A_SKIP_MLP", "VLLM_OMNI_910A_A2A_PROBE",
                "VLLM_OMNI_910A_FLASH_SYNC", "VLLM_OMNI_910A_WARMUP"):
        env.pop(key, None)
    cmd = ["/home/mseco/vllm910a/bin/python", "/home/mseco/vllm910a/bin/vllm-omni", "serve",
           MODEL_PATH, "--omni", "--host", "127.0.0.1", "--port", str(PORT),
           "--trust-remote-code", "--task-type", "fl2va", "--num-gpus", "8", "--usp", "8", "--ring", "1",
           "--text-encoder-tp-size", "8", "--vae-parallel-mode", "tile", "--vae-use-tiling",
           "--vae-patch-parallel-size", "8", "--diffusion-attention-backend", "TORCH_SDPA",
           "--stage-init-timeout", "2400", "--init-timeout", "2400", "--dtype", "half",
           "--enable-distributed-layerwise-offload"]
    result = {"command": cmd, "run_dir": str(RUN), "request_limit_seconds": REQUEST_LIMIT_SECONDS,
              "startup_limit_seconds": STARTUP_LIMIT_SECONDS}
    log_path = RUN / "serve.log"
    server = None
    signal.signal(signal.SIGTERM, interrupted)
    signal.signal(signal.SIGINT, interrupted)
    try:
        with log_path.open("xb") as log:
            start = time.perf_counter()
            server = subprocess.Popen(cmd, cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT,
                                      start_new_session=True)
            assert os.getpgid(server.pid) == server.pid
            result["server_pid"] = server.pid
            (RUN / "server.pid").write_text(str(server.pid) + "\n")
            deadline = time.monotonic() + STARTUP_LIMIT_SECONDS
            while time.monotonic() < deadline:
                if server.poll() is not None:
                    raise RuntimeError(f"engine exited during startup, rc={server.returncode}; inspect {log_path}")
                try:
                    if get("/health").get("status") == "healthy":
                        break
                except Exception:
                    pass
                time.sleep(5)
            else:
                raise TimeoutError(f"startup timed out; inspect {log_path}")
            result["startup_seconds"] = time.perf_counter() - start
            print("Engine healthy; startup_s=%.1f" % result["startup_seconds"], flush=True)
            fields = {"prompt": "a red ball bouncing from left to right on the floor", "seconds": "5",
                      "size": "1344x768", "fps": "24", "seed": "1234", "num_inference_steps": "49",
                      "extra_params": json.dumps({"task": "fl2va"})}
            result["request_fields"] = fields
            result["expected_frame_alignment"] = {"requested_frames": 120, "aligned_frames": 124,
                                                    "duration_s": 124 / 24, "rule": "17n+5"}
            boundary = "h3audit" + uuid.uuid4().hex
            chunks = [(f"--{boundary}\r\nContent-Disposition: form-data; name=\"{k}\"\r\n\r\n{v}\r\n").encode()
                      for k, v in fields.items()]
            refs = []
            for position, x in enumerate((320, 1024)):
                image = Image.new("RGB", (1344, 768), (200, 205, 212))
                draw = ImageDraw.Draw(image)
                draw.rectangle((0, 640, 1344, 768), fill=(98, 102, 108))
                draw.ellipse((x - 45, 550, x + 45, 640), fill=(215, 30, 30))
                data = io.BytesIO()
                image.save(data, format="PNG")
                payload = data.getvalue()
                filename = "first.png" if position == 0 else "last.png"
                (RUN / filename).write_bytes(payload)
                refs.append({"filename": filename, "sha256": hashlib.sha256(payload).hexdigest(),
                             "width": 1344, "height": 768})
                chunks.append((f"--{boundary}\r\nContent-Disposition: form-data; name=\"input_references\"; filename=\"{filename}\"\r\n"
                               "Content-Type: image/png\r\n\r\n").encode() + payload + b"\r\n")
            result["reference_images"] = refs
            body = b"".join(chunks) + f"--{boundary}--\r\n".encode()
            start = time.perf_counter()
            req = urllib.request.Request(API + "/v1/videos", data=body,
                                         headers={"Content-Type": f"multipart/form-data; boundary={boundary}"})
            with urllib.request.urlopen(req, timeout=300) as response:
                submitted = json.load(response)
            result["submitted"] = submitted
            vid = submitted["id"]
            print("Submitted:", vid, flush=True)
            deadline = time.monotonic() + REQUEST_LIMIT_SECONDS
            while time.monotonic() < deadline:
                status = get("/v1/videos/" + vid)
                if status.get("status") in ("completed", "failed"):
                    break
                if server.poll() is not None:
                    raise RuntimeError("engine exited during request")
                time.sleep(5)
            else:
                raise TimeoutError("request timed out")
            result["request_wall_seconds"] = time.perf_counter() - start
            result["response"] = status
            if status.get("status") != "completed":
                raise RuntimeError(f"inference failed: {status}")
            video_path = RUN / "video.mp4"
            with urllib.request.urlopen(API + "/v1/videos/" + vid + "/content", timeout=90) as response:
                with video_path.open("xb") as f:
                    written = 0
                    while chunk := response.read(1024 * 1024):
                        written += len(chunk)
                        if written > 32 * 1024**2:
                            raise RuntimeError("output exceeded 32 MiB audit limit")
                        f.write(chunk)
            result["video_bytes"] = video_path.stat().st_size
            result["mp4_metadata"] = parse_mp4(video_path)
            video_tracks = [t for t in result["mp4_metadata"]["tracks"] if t["handler"] == "vide"]
            assert len(video_tracks) == 1, video_tracks
            track = video_tracks[0]
            assert (track["width"], track["height"]) == (1344, 768), track
            assert track["sample_count"] == 124 and abs(track["duration_seconds"] - 124 / 24) < .001, track
            result["verified_768p_5s_frame_alignment"] = True
            try:
                fp = subprocess.run(["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries",
                                     "stream=width,height,duration,nb_frames", "-of", "json", str(video_path)],
                                    capture_output=True, text=True, timeout=20)
                result["ffprobe"] = {"rc": fp.returncode, "result": json.loads(fp.stdout) if fp.returncode == 0 else fp.stderr}
            except FileNotFoundError:
                result["ffprobe"] = {"unavailable": True}
    except BaseException as exc:
        result["failure"] = {"type": type(exc).__name__, "message": str(exc)}
        if isinstance(exc, urllib.error.HTTPError):
            result["failure"]["response_body"] = exc.read().decode(errors="replace")
        raise
    finally:
        if server is not None:
            # Only our own start_new_session process group; never pkill shared workers.
            try:
                os.killpg(server.pid, signal.SIGTERM)
            except ProcessLookupError:
                pass
            try:
                server.wait(timeout=20)
            except subprocess.TimeoutExpired:
                pass
            try:
                os.killpg(server.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            server.wait(timeout=10)
            result["server_exit_after_cleanup"] = server.returncode
        if log_path.exists():
            text = log_path.read_text(errors="replace")
            result["steady_rows"] = [{"step1_s": float(a), "steady_avg_s": float(b)}
                                     for a, b in re.findall(r"denoise_steady step1=([0-9.]+)s steady_avg=([0-9.]+)s", text)]
            result["fa_core_rows"] = [list(map(int, row)) for row in
                re.findall(r"\[FA910A\] core_budget=(\d+) blockDim=(\d+) B=(\d+) H=(\d+) N=(\d+) D=(\d+) Br=(\d+) Bc=(\d+)", text)]
            result["fa_padding_plans"] = [list(map(int, row)) for row in re.findall(
                r"\[FA910A\] plan input_N=(\d+) kernel_N=(\d+) key_limit=(\d+) original_pad=(\d+)", text)]
            result["step_rows"] = [{"step": int(i), "prep_s": float(a), "model_s": float(b), "update_s": float(c)}
                                   for i, a, b, c in re.findall(
                                       r"\[H3PERF\] dstep (\d+) prep=([0-9.]+)s model=([0-9.]+)s update=([0-9.]+)s", text)]
            result["complete_step_rows"] = [{"step": int(i), "total_s": float(a), "callback_and_sync_s": float(b)}
                                             for i, a, b in re.findall(
                                                 r"\[H3PERF\] dstep_complete (\d+) total=([0-9.]+)s callback_and_sync=([0-9.]+)s", text)]
            result["complete_denoise_rows"] = [{"steps": int(i), "first_s": float(a), "after_first_avg_s": float(b),
                                                "observed_total_s": float(c), "last_s": float(d)}
                                               for i, a, b, c, d in re.findall(
                                                 r"\[H3PERF\] denoise_complete steps=(\d+) first=([0-9.]+)s after_first_avg=([0-9.]+)s observed_total=([0-9.]+)s last=([0-9.]+)s", text)]
            result["stage_times"] = {stage: [float(x) for x in re.findall(r"\[H3PERFP\] " + stage + r"=([0-9.]+)s", text)]
                                     for stage in ("encode", "denoise", "decode")}
        (RUN / "result.json").write_text(json.dumps(result, indent=2) + "\n")
        print(json.dumps({k: v for k, v in result.items() if k not in ("command", "submitted")}, indent=2), flush=True)


if __name__ == "__main__":
    main()
