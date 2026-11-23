"""R126 matrix safety audit: preserve all R124 checks, optionally defer host reads.

These checks do not feed values into inference. Deferred mode queues the same
integer IEEE/nonfinite and saturation reductions on the device and reads their
counts once after a complete forward, before any sampler update. It does not
clip, sanitize or narrow activations. Candidate arithmetic lives in a separate
module and must pass a same-input bitwise leaf gate before activation.
"""
from __future__ import annotations
import torch
from npu_half_nan_compat124 import ieee_nonfinite_masks


class MatrixSafetyAudit126:
    def __init__(self, model, mode='strict'):
        assert mode in ('strict', 'deferred')
        self.mode = mode
        self.counts = []
        self.calls = 0
        self.names = []
        self.handles = []
        self.expected_calls = 155
        for name, module in model.named_modules():
            if not name.endswith(('.qkv_proj', '.mlp.fc1', '.adaln_proj.linear')):
                continue
            self.handles.append(module.register_forward_hook(self._hook(name)))

    def _hook(self, name):
        def hook(module, inputs, result):
            value = result[0] if isinstance(result, tuple) else result
            if not isinstance(value, torch.Tensor) or value.dtype != torch.float16:
                return
            self.calls += 1
            nan, posinf, neginf = ieee_nonfinite_masks(value)
            bad = (nan | posinf | neginf).sum(dtype=torch.int32)
            # Preserve R124's exact finite boundary test; do not replace math.
            boundary = (value.abs() == 65504).sum(dtype=torch.int32)
            if self.mode == 'strict':
                nbad, nboundary = int(bad), int(boundary)
                if nbad or nboundary:
                    raise FloatingPointError(f'Half matrix range failure at {name}: {nbad}, {nboundary}')
            else:
                self.counts.append(torch.stack((bad, boundary)))
                self.names.append(name)
        return hook

    def begin(self):
        self.calls = 0
        self.counts.clear()
        self.names.clear()

    def end(self):
        assert self.calls == self.expected_calls, (self.calls, self.expected_calls)
        if self.mode == 'deferred':
            all_counts = torch.stack(self.counts)
            total = all_counts.sum(dim=0, dtype=torch.int32).cpu().tolist()
            if any(total):
                detail = all_counts.cpu().tolist()
                failures = [{'module': name, 'nonfinite': c[0], 'boundary': c[1]}
                            for name, c in zip(self.names, detail) if any(c)]
                raise FloatingPointError(str(failures))
        else:
            total = [0, 0]
        return {'matrix_checks': self.calls, 'nonfinite_count': total[0],
                'boundary_count': total[1], 'mode': self.mode,
                'all_checks_preserved': True, 'checked_before_sampler_update': True}

    def close(self):
        for handle in self.handles:
            handle.remove()
        self.handles.clear()
        self.counts.clear()
