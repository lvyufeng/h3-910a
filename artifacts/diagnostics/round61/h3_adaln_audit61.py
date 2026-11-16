"""Default-off bounded diagnostic captures of actual DiT indexed modulation."""
import functools
import json
import os
from pathlib import Path
import sys
import torch

_INSTALLED = False
_COUNTS = {}
_NAMES = ('rms_norm_indexed_scale_shift', 'indexed_gate_rms_norm_scale_shift', 'indexed_gate', 'indexed_scale_shift_')
_ARG_NAMES = {
    'rms_norm_indexed_scale_shift': ('x', 'weight', 'shift', 'scale', 'indices', 'eps'),
    'indexed_gate_rms_norm_scale_shift': ('residual', 'gate', 'branch', 'weight', 'shift', 'scale', 'indices', 'eps'),
    'indexed_gate': ('x', 'gate', 'other', 'indices'),
    'indexed_scale_shift_': ('x', 'shift', 'scale', 'indices'),
}


def _wrapper(original, name, root):
    @functools.wraps(original)
    def capture(*args, **kwargs):
        rows = args[0].shape[0]
        if rows < 4000:
            return original(*args, **kwargs)
        call = _COUNTS.get(name, 0)
        _COUNTS[name] = call + 1
        if call not in (0, 24, 49):
            return original(*args, **kwargs)
        target = root / (name + '_%03d.pt' % call)
        if target.exists():
            return original(*args, **kwargs)
        captured_rows = min(rows, 6144)
        selected = torch.linspace(0, rows-1, captured_rows, dtype=torch.long, device=args[0].device)
        def copy(x):
            if isinstance(x, torch.Tensor):
                if x.ndim >= 1 and x.shape[0] == rows:
                    x = x.index_select(0, selected)
                return x.detach().cpu().contiguous().clone()
            return x
        before = {key: copy(value) for key, value in zip(_ARG_NAMES[name], args)}
        before.update({key: copy(value) for key, value in kwargs.items()})
        result = original(*args, **kwargs)
        output = tuple(copy(x) for x in result) if isinstance(result, tuple) else copy(result)
        tensors = [x for x in before.values() if isinstance(x, torch.Tensor)]
        tensors += list(output) if isinstance(output, tuple) else [output]
        byte_count = sum(x.numel()*x.element_size() for x in tensors)
        if byte_count > 512*1024*1024:
            raise RuntimeError('Bounded AdaLN capture exceeds512MiB')
        metadata = {'name': name, 'call': call, 'source_rows': rows, 'captured_rows': captured_rows,
                    'source_shape': list(args[0].shape), 'row_selection': 'evenly_spaced_independent_rows',
                    'tensor_bytes': byte_count, 'dtypes': {key: str(value.dtype) for key, value in before.items() if isinstance(value, torch.Tensor)},
                    'shapes': {key: list(value.shape) for key, value in before.items() if isinstance(value, torch.Tensor)},
                    'modulation_eager_flag': os.environ.get('VLLM_OMNI_910A_EAGER_MODULATION', '1'),
                    'autocast_enabled': torch.is_autocast_enabled('npu'),
                    'scope': 'Actual first-step DiT rows and indexed tables, instrumentation-only, not clean generation performance'}
        with target.open('xb') as f:
            torch.save({'inputs': before, 'output': output}, f)
        target.with_suffix('.json').write_text(json.dumps(metadata, indent=2)+'\n')
        return result
    return capture


def install():
    global _INSTALLED
    if _INSTALLED:
        return
    _INSTALLED = True
    path = os.environ.get('VLLM_OMNI_910A_ADALN_AUDIT_DIR')
    if not path or not torch.distributed.is_initialized() or torch.distributed.get_rank() != 0:
        return
    root = Path(path)
    root.mkdir(parents=True, exist_ok=True)
    module = sys.modules['vllm_omni.diffusion.models.minimax_h3.minimax_h3_transformer']
    for name in _NAMES:
        setattr(module, name, _wrapper(getattr(module, name), name, root))
