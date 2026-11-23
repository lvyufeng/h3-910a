#!/usr/bin/env python3
"""R155 driver: additive span decomposition of the MLP forward."""
from __future__ import annotations
import json, os, sys
from pathlib import Path
ROOT = Path('/tmp/h3_sync158'); LABEL = os.environ.get('H3_SYNC158_LABEL', 'sync158a')


def main():
    rank = int(os.environ['RANK']); local_rank = int(os.environ['LOCAL_RANK'])
    out = ROOT / LABEL
    assert out.is_dir(), 'launcher must pre-create the output directory'
    sys.path.insert(0, str(ROOT))
    import torch, torch_npu
    import sync_probe158 as probe
    torch.npu.set_device(local_rank)
    import vllm_omni.diffusion.offloader.layerwise_backend as lb
    real_group = lb._install_layerwise_hook_group

    def patched(blocks, device, stream, pin_memory=True):
        hooks = real_group(blocks, device, stream, pin_memory)
        probe.install(torch, blocks)
        return hooks

    lb._install_layerwise_hook_group = patched
    # DIAGNOSTIC ONLY: make the layerwise prefetch synchronous so that no
    # host-to-device copy is in flight while a block computes. This is a control
    # that separates waiting from computing; it is never proposed as an optimisation.
    from vllm_omni.diffusion.offloader import layerwise_backend as _lb
    _orig_prefetch = _lb.LayerwiseOffloadHook.prefetch_layer

    def _sync_prefetch(self, non_blocking=True, *args, **kwargs):
        return _orig_prefetch(self, non_blocking=False, *args, **kwargs)

    _lb.LayerwiseOffloadHook.prefetch_layer = _sync_prefetch

    sys.argv = ['benchmark_actual_attention126.py',
        '--checkpoint', '/home/mseco/models/MiniMax-H3/FL2VA/transformer',
        '--mode', 'baseline', '--label', LABEL, '--tp', '2', '--usp', '4', '--offload', 'native',
        '--audit-sequence', 'strict', '--steps', '25', '--warmup', '2', '--repeats', '1']
    import benchmark_actual_attention126 as bench
    bench.ROOT = ROOT
    bench.main()
    receipt = out / f'candidate_rank{rank}.json'
    if receipt.exists():
        data = json.loads(receipt.read_text())
        data['sync158'] = probe.collect(torch)
        receipt.write_text(json.dumps(data, indent=2) + '\n')


if __name__ == '__main__':
    main()
