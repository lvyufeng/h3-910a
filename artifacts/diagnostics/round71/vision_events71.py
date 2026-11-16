"""Preallocated-event instance-only vision profile; delegates unchanged math.

parent.vision contains all other categories and is NEVER added to their sum.
Events have no per-operator sync; current-stream time may contain dispatch gaps,
dependencies and waits. This is not pure kernel occupancy or an HF quality test.
"""
import torch
from torch import nn
from event_profile66 import EventProfile


class VisionEvents:
    def __init__(self, encoder, candidate):
        self.profile = EventProfile(torch, capacity=512)
        self.bindings = []
        vision = encoder.vision
        def bind(owner, name, category):
            original = getattr(owner, name)
            def wrapped(*args, **kwargs):
                return self.profile.call(category, original, *args, **kwargs)
            setattr(owner, name, wrapped)
            self.bindings.append((owner, name, original))
        bind(vision, 'forward', 'parent.vision')
        bind(vision.patch_embed, 'forward', 'patch_embed')
        bind(vision, 'fast_pos_embed_interpolate', 'pos_interpolate')
        bind(vision, 'rot_pos_emb', 'rot_pos')
        for name, module in vision.named_modules():
            if isinstance(module, candidate._NpuBF16CompatibleLinear):
                bind(module, 'forward', 'linear.' + name)
            elif isinstance(module, candidate._NpuBF16CompatibleLayerNorm):
                bind(module, 'forward', 'norm.' + name)
            elif isinstance(module, nn.GELU):
                bind(module, 'forward', 'gelu.' + name)
        # candidate is an isolated, uninstalled module. These names are vision
        # only; shared text apply-RoPE and causal math are not replaced here.
        bind(candidate, '_apply_rotary_pos_emb_vision', 'apply_rope')
        bind(candidate, '_vision_sdpa_bf16_compatible', 'attention')

    def begin(self):
        self.profile.begin()

    def finish(self):
        return self.profile.finish()

    def restore(self):
        self.profile.active = False
        for owner, name, original in reversed(self.bindings):
            setattr(owner, name, original)
        self.bindings.clear()
