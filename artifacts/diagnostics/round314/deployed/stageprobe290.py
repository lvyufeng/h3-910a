"""Stage-timing probe for MiniMax-H3 pipeline: encoder / video VAE / audio VAE."""
import time
import json
import atexit

T = {}
STATE = {'installed': False}


def _now():
    import torch
    torch.npu.synchronize()
    return time.perf_counter()


def install():
    if STATE['installed']:
        return
    import torch
    from vllm_omni.diffusion.models.minimax_h3 import pipeline_minimax_h3 as p

    def wrap(cls, name, label):
        orig = getattr(cls, name, None)
        if orig is None:
            return 0
        def wrapper(self, *a, **k):
            t0 = _now()
            try:
                return orig(self, *a, **k)
            finally:
                dt = _now() - t0
                T[label] = T.get(label, 0.0) + dt
                print(f"[STAGE] {label} {dt:.3f}s", flush=True)
        setattr(cls, name, wrapper)
        return 1

    n = 0
    for obj in list(vars(p).values()):
        if not isinstance(obj, type):
            continue
        if 'encode_prompt' in obj.__dict__:
            n += wrap(obj, 'encode_prompt', 'encoder')
        if 'decode_latent' in obj.__dict__:
            n += wrap(obj, 'decode_latent', f'decode_latent:{obj.__name__}')
    STATE['installed'] = True
    print(f"[STAGE] installed wrappers={n}", flush=True)

    def dump():
        try:
            with open('/tmp/h3_r221/stage_times290.json', 'w') as f:
                json.dump(T, f, indent=2)
            print('[STAGE] dumped', T, flush=True)
        except Exception as e:  # noqa: BLE001
            print('[STAGE] dump failed', e, flush=True)
    atexit.register(dump)
