#!/usr/bin/env python3
"""Real HTTP FL2VA warm service decode A/B; one-interval quality NOT accepted."""
import hashlib
import io
import json
import os
from pathlib import Path
import signal
import subprocess
import time
import urllib.request
import uuid
from PIL import Image, ImageDraw
from h3_evidence64 import EvidenceWriter
from h3_mp4_metadata import parse as parse_mp4

ROOT = Path('/tmp/h3_round64_service01')
RUN = ROOT / 'http'
RUN.mkdir(exist_ok=False)
writer = EvidenceWriter(RUN / 'result.json')
MODEL = '/home/mseco/models/MiniMax-H3-native-fp16-audit53'
PYTHON = '/home/mseco/vllm910a/bin/python'
API = 'http://127.0.0.1:9198'


def get(path, timeout=8):
    with urllib.request.urlopen(API + path, timeout=timeout) as response:
        return json.load(response)


def control(label, mode):
    # Exact owned endpoints are checked before atomic config replacement.
    path, temporary = ROOT / 'control.json', ROOT / 'control.pending.json'
    for p in (path, temporary):
        if p.resolve() != p.absolute() or p.parent != ROOT:
            raise RuntimeError('Unexpected control publication path')
    with temporary.open('x') as f:
        json.dump({'label': label, 'mode': mode}, f)
    os.replace(temporary, path)


def form():
    fields = {'prompt': 'a red ball bouncing from left to right on the floor', 'seconds': '5',
              'size': '1344x768', 'fps': '24', 'seed': '1234', 'num_inference_steps': '1',
              'extra_params': json.dumps({'task': 'fl2va'})}
    boundary = 'h3service64' + uuid.uuid4().hex
    chunks = [(f'--{boundary}\r\nContent-Disposition: form-data; name="{key}"\r\n\r\n{value}\r\n').encode()
              for key, value in fields.items()]
    references = []
    for index, x in enumerate((320, 1024)):
        image = Image.new('RGB', (1344, 768), (200, 205, 212))
        draw = ImageDraw.Draw(image)
        draw.rectangle((0, 640, 1344, 768), fill=(98, 102, 108))
        draw.ellipse((x - 45, 550, x + 45, 640), fill=(215, 30, 30))
        stream = io.BytesIO()
        image.save(stream, format='PNG')
        data = stream.getvalue()
        filename = 'first.png' if index == 0 else 'last.png'
        references.append({'filename': filename, 'sha256': hashlib.sha256(data).hexdigest()})
        chunks.append((f'--{boundary}\r\nContent-Disposition: form-data; name="input_references"; filename="{filename}"\r\nContent-Type: image/png\r\n\r\n').encode() + data + b'\r\n')
    return b''.join(chunks) + f'--{boundary}--\r\n'.encode(), boundary, fields, references


def stop(signum, frame):
    raise SystemExit('Manager interrupted: %d' % signum)


server = None
r = {'requests': [], 'scope': 'Actual HTTP FL2VA, 1 denoise interval, 1344x768/5s/24fps/USP8; warm complete service decode times come from rank receipts, not the legacy unsynchronized H3PERFP timer. Not native49-step quality or official T2VA.',
     'source_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
try:
    try:
        get('/health', 2)
    except Exception:
        pass
    else:
        raise RuntimeError('Existing engine on port9198; refuse interference')
    signal.signal(signal.SIGTERM, stop)
    signal.signal(signal.SIGINT, stop)
    env = os.environ.copy()
    env.update(VLLM_OMNI_910A_PERF='1', VLLM_OMNI_910A_FLASH_ATTN='1', VLLM_OMNI_910A_VISION_FP32='0',
               VLLM_OMNI_910A_NO_FUSED_NORM='1', VLLM_OMNI_910A_NO_FUSED_ACT='1', VLLM_OMNI_910A_EAGER_MODULATION='1',
               VLLM_OMNI_910A_VAE_FINITE_SELECT='1', VLLM_OMNI_910A_VAE_QUERY_CHUNK='256',
               VLLM_OMNI_910A_DECODE_TIMING_DIR=str(ROOT), VLLM_OMNI_910A_ATTN_CHUNK='128',
               TBE_PARALLEL_COMPILER='1', VLLM_OMNI_DLO_DP_WAVE_TIMEOUT='14400', PYTORCH_NPU_ALLOC_CONF='expandable_segments:True')
    for key in tuple(env):
        if key.startswith('VLLM_OMNI_910A_') and ('AUDIT_DIR' in key or 'SKIP_' in key or key.endswith(('A2A_PROBE', 'FLASH_SYNC', 'WARMUP'))):
            env.pop(key)
    cmd = [PYTHON, '/home/mseco/vllm910a/bin/vllm-omni', 'serve', MODEL, '--omni', '--host', '127.0.0.1', '--port', '9198',
           '--trust-remote-code', '--task-type', 'fl2va', '--num-gpus', '8', '--usp', '8', '--ring', '1', '--text-encoder-tp-size', '8',
           '--vae-parallel-mode', 'tile', '--vae-use-tiling', '--vae-patch-parallel-size', '8', '--diffusion-attention-backend', 'TORCH_SDPA',
           '--stage-init-timeout', '2400', '--init-timeout', '2400', '--dtype', 'half', '--enable-distributed-layerwise-offload']
    r['command'] = cmd
    r['environment_overrides'] = {key: env[key] for key in env if key.startswith('VLLM_OMNI_910A_')}
    t = time.perf_counter()
    with (RUN / 'serve.log').open('xb') as log:
        server = subprocess.Popen(cmd, cwd='/home/mseco/h3-910a', env=env, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
        assert os.getpgid(server.pid) == server.pid
        r['server_pid'] = server.pid
        (RUN / 'server.pid').write_text(str(server.pid) + '\n')
        deadline = time.monotonic() + 1800
        while time.monotonic() < deadline:
            if server.poll() is not None:
                raise RuntimeError('Engine failed startup: %s' % server.returncode)
            try:
                if get('/health').get('status') == 'healthy':
                    break
            except Exception:
                pass
            time.sleep(5)
        else:
            raise TimeoutError('Startup exceeded1800s')
        r['startup_s'] = time.perf_counter() - t
        print('Healthy startup_s=%.3f' % r['startup_s'], flush=True)
        sequence = [('bWarm', 'baseline'), ('b1', 'baseline'), ('b2', 'baseline'), ('b3', 'baseline'),
                    ('cWarm', 'candidate'), ('c1', 'candidate'), ('c2', 'candidate'), ('c3', 'candidate'), ('paired', 'paired')]
        for label, mode in sequence:
            control(label, mode)
            body, boundary, fields, references = form()
            item = {'label': label, 'mode': mode, 'form_fields': fields, 'references': references,
                    'warmup_excluded': label in ('bWarm', 'cWarm'), 'paired_excluded_from_normal_e2e': mode == 'paired'}
            t = time.perf_counter()
            request = urllib.request.Request(API + '/v1/videos', data=body, headers={'Content-Type': 'multipart/form-data; boundary=' + boundary})
            with urllib.request.urlopen(request, timeout=300) as response:
                submitted = json.load(response)
            item['submitted'] = submitted
            deadline = time.monotonic() + 2400
            while time.monotonic() < deadline:
                status = get('/v1/videos/' + submitted['id'])
                if status['status'] in ('completed', 'failed'):
                    break
                if server.poll() is not None:
                    raise RuntimeError('Engine exited during request')
                time.sleep(2)
            else:
                raise TimeoutError('Request exceeded2400s')
            item['request_wall_s'] = time.perf_counter() - t
            item['response'] = status
            if status['status'] != 'completed':
                raise RuntimeError('Request failed: ' + json.dumps(status))
            path = RUN / (label + '.mp4')
            with urllib.request.urlopen(API + '/v1/videos/' + submitted['id'] + '/content', timeout=90) as response, path.open('xb') as out:
                total = 0
                while data := response.read(1024 * 1024):
                    total += len(data)
                    if total > 32 * 1024**2:
                        raise RuntimeError('MP4 unexpectedly exceeds32MiB')
                    out.write(data)
            item['mp4_sha256'] = hashlib.sha256(path.read_bytes()).hexdigest()
            item['mp4_metadata'] = parse_mp4(path)
            track = next(track for track in item['mp4_metadata']['tracks'] if track['handler'] == 'vide')
            assert (track['width'], track['height'], track['sample_count']) == (1344, 768, 124)
            r['requests'].append(item)
            writer.write(r)
            print(json.dumps({'label': label, 'wall_s': item['request_wall_s'], 'mp4_sha256': item['mp4_sha256']}), flush=True)
    r['completed'] = True
except BaseException as exc:
    r['failure'] = {'type': type(exc).__name__, 'message': str(exc)}
    raise
finally:
    if server is not None:
        for sig, timeout in ((signal.SIGTERM, 20), (signal.SIGKILL, 10)):
            try:
                os.killpg(server.pid, sig)
            except ProcessLookupError:
                pass
            try:
                server.wait(timeout=timeout)
            except subprocess.TimeoutExpired:
                pass
        r['server_exit_after_cleanup'] = server.poll()
    writer.write(r)
    print('Evidence:', RUN / 'result.json', flush=True)
