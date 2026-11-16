"""Instance-only vision Linear cast/shadow diagnostics with unchanged GEMM/bias.

No vision math changes, weight mutation, packing/transpose or output cast changes.
Shadow buffers are native weight.float()/bias.float() and are never reused after
mode reset. Offload/reload/.data writes require explicit clear(). This is not a
production offloader integration or independent HF/lossless quality proof.
"""
import types
import torch
import torch_npu
from torch.nn import functional as F
from shadow_cache65 import ShadowWeights
from activation_cast69 import expand_activation


class VisionLinearModes:
    MODES = ('native', 'activation_only', 'shadow_only', 'both', 'restored')
    MAX_BYTES = 2362750400
    MAX_ENTRIES = 232

    def __init__(self, metric):
        self.metric = metric
        self.mode = 'native'
        self.verify = False
        self.checks = []
        self.bound = []
        self.shadow = ShadowWeights(max_bytes=self.MAX_BYTES, max_entries=self.MAX_ENTRIES)

    @staticmethod
    def layout(value):
        return {'shape': list(value.shape), 'stride': list(value.stride()), 'dtype': str(value.dtype),
                'physical_format': torch_npu.get_npu_format(value)}

    def begin_mode(self, mode, verify=False):
        if mode not in self.MODES:
            raise ValueError('Unknown vision conversion experiment mode')
        self.shadow.clear()
        # No cross-mode shadow residency. Counts start from zero for each
        # independent factor, unlike the already-frozen text shadow stack.
        self.shadow = ShadowWeights(max_bytes=self.MAX_BYTES, max_entries=self.MAX_ENTRIES)
        self.mode, self.verify = mode, verify
        self.checks.clear()

    def install(self, vision, compatible_linear_type):
        if self.bound:
            raise RuntimeError('Vision Linear wrappers are already installed')
        modules = [(name, module) for name, module in vision.named_modules() if isinstance(module, compatible_linear_type)]
        if len(modules) != 116 or any(module.bias is None for _, module in modules):
            raise ValueError('Expected116 actual biased vision Linear modules')
        needed = sum((module.weight.numel() + module.bias.numel()) * 4 for _, module in modules)
        if needed != self.MAX_BYTES:
            raise ValueError('Actual vision weight/bias shadow footprint changed')
        descriptions = []
        for name, module in modules:
            original = module.forward
            self.bound.append((module, original))
            def wrapped(current, value, name=name, original=original):
                return self.forward(name, current, original, value)
            module.forward = types.MethodType(wrapped, module)
            descriptions.append({'name': name, 'weight_shape': list(module.weight.shape),
                                 'bias_shape': list(module.bias.shape), 'weight_dtype': str(module.weight.dtype),
                                 'bias_dtype': str(module.bias.dtype)})
        return descriptions

    def forward(self, name, module, original, value):
        if (value.device.type not in ('npu', 'privateuseone') or value.dtype != torch.bfloat16
                or module.weight.dtype != torch.bfloat16 or module.bias.dtype != torch.bfloat16):
            raise ValueError('Vision experiment only supports the measured real NPU BF16 path')
        use_activation = self.mode in ('activation_only', 'both')
        use_shadow = self.mode in ('shadow_only', 'both')
        before = value.detach().cpu().contiguous().clone() if self.verify else None
        with torch.autocast(device_type=value.device.type, enabled=False):
            x32 = expand_activation(value) if use_activation else value.float()
            w32 = self.shadow.get((name, 'weight'), module.weight) if use_shadow else module.weight.float()
            b32 = self.shadow.get((name, 'bias'), module.bias) if use_shadow else module.bias.float()
            # Preserve the original fused-bias F.linear dispatch and every shape;
            # do not pretranspose/pack/cache activations or move bias outside.
            output = F.linear(x32, w32, b32).to(value.dtype)
            if self.verify:
                xref, wref, bref = value.float(), module.weight.float(), module.bias.float()
                reference = original(value)
                row = {'name': name, 'activation_enabled': use_activation, 'shadow_enabled': use_shadow,
                       'input_shape': list(value.shape), 'weight_shape': list(module.weight.shape),
                       'bias_shape': list(module.bias.shape), 'output_shape': list(output.shape),
                       'input_cast': self.metric(x32, xref), 'weight_shadow': self.metric(w32, wref),
                       'bias_shadow': self.metric(b32, bref), 'output': self.metric(output, reference),
                       'input_cast_layout': self.layout(x32), 'native_input_cast_layout': self.layout(xref),
                       'weight_shadow_layout': self.layout(w32), 'native_weight_cast_layout': self.layout(wref),
                       'bias_shadow_layout': self.layout(b32), 'native_bias_cast_layout': self.layout(bref),
                       'output_layout': self.layout(output), 'native_output_layout': self.layout(reference),
                       'input_readonly': torch.equal(value.detach().cpu().contiguous().view(torch.uint8), before.view(torch.uint8))}
                self.checks.append(row)
                proofs = (row['input_cast'], row['weight_shadow'], row['bias_shadow'], row['output'])
                layouts = (('input_cast_layout', 'native_input_cast_layout'),
                           ('weight_shadow_layout', 'native_weight_cast_layout'),
                           ('bias_shadow_layout', 'native_bias_cast_layout'), ('output_layout', 'native_output_layout'))
                if (not all(proof['raw_bits_equal'] and proof['finite'] for proof in proofs)
                        or not row['input_readonly'] or any(row[a] != row[b] for a, b in layouts)):
                    raise AssertionError('Actual vision Linear input/weight/bias/output bits or native layout changed')
        return output

    def clear(self):
        self.shadow.clear()

    def restore(self):
        self.clear()
        for module, original in self.bound:
            module.forward = original
        self.bound.clear()
