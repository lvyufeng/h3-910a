"""Lazy stage-timing probe: patches the MiniMax-H3 pipeline only when it is imported.

Enabled by H3_STAGE_PROBE=1; H3_SKIP_ENCODER_OFFLOAD=1 makes the encoder resident.
Runs inside every python process on this PYTHONPATH, so it must stay lightweight.
"""
import os
import sys

if os.environ.get('H3_STAGE_PROBE') == '1':
    import atexit
    import importlib.abc
    import json
    import time

    TARGET = 'vllm_omni.diffusion.models.minimax_h3.pipeline_minimax_h3'
    T = {}
    DONE = {'patched': False}

    def _now():
        import torch
        torch.npu.synchronize()
        return time.perf_counter()

    def _dump():
        try:
            path = os.environ.get('H3_STAGE_OUT', '/tmp/h3_r221/stage_times290.json')
            with open(path, 'w') as f:
                json.dump(T, f, indent=2)
            print('[STAGE] dumped', T, flush=True)
        except Exception as exc:  # noqa: BLE001
            print('[STAGE] dump failed', exc, flush=True)

    def _wrap(cls, name, label):
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
                print('[STAGE] %s %.3fs' % (label, dt), flush=True)

        setattr(cls, name, wrapper)
        return 1

    def _patch(module):
        if DONE['patched']:
            return
        DONE['patched'] = True
        n = 0
        for obj in list(vars(module).values()):
            if isinstance(obj, type):
                if 'encode_prompt' in obj.__dict__:
                    n += _wrap(obj, 'encode_prompt', 'encoder')
                if 'decode_latent' in obj.__dict__:
                    n += _wrap(obj, 'decode_latent', 'decode_latent:' + obj.__name__)
        print('[STAGE] pipeline patched wrappers=%d' % n, flush=True)
        if os.environ.get('H3_SKIP_ENCODER_OFFLOAD') == '1':
            from vllm_omni.diffusion.offloader import layerwise_backend as lb
            lb.enable_plan_encoder_layerwise_offload = lambda *a, **k: False
            print('[STAGE] encoder offload disabled (monkeypatch)', flush=True)
        atexit.register(_dump)

    class _Loader(importlib.abc.Loader):
        def __init__(self, inner):
            self.inner = inner

        def create_module(self, spec):
            return self.inner.create_module(spec)

        def exec_module(self, module):
            self.inner.exec_module(module)
            try:
                _patch(module)
            except Exception as exc:  # noqa: BLE001
                print('[STAGE] patch failed', exc, flush=True)

    class _Finder(importlib.abc.MetaPathFinder):
        def find_spec(self, fullname, path=None, target=None):
            if fullname != TARGET:
                return None
            for finder in sys.meta_path:
                if finder is self or finder is _INSTANCE:
                    continue
                spec = finder.find_spec(fullname, path, target)
                if spec is not None:
                    spec.loader = _Loader(spec.loader)
                    return spec
            return None

    _INSTANCE = _Finder()
    sys.meta_path.insert(0, _INSTANCE)
    print('[STAGE] lazy hook armed for %s' % TARGET, flush=True)
