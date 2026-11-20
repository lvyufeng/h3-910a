#!/usr/bin/env python3
"""Attach the verified FP16-residual FP32-output guard only to H3 time projections.

Original FP32 parameter values, sinusoidal inputs, SiLU, TP2 partitions/reduction,
and the rest of the graph remain unchanged. This is an isolated numerical trial,
not a shared-runtime modification or full-model precision/speed acceptance.
"""
import torch
from fp32_patch_residual_guard123 import ResidualFP32Projection123


def install_time_projection_guard123(model):
    affected=[]
    for name in ['proj_in','proj_out']:
        layer=getattr(model.time_embedder,name)
        if layer.weight.dtype!=torch.float32:
            raise TypeError('Time projection parameters must preserve the original FP32 contract')
        layer.quant_method=ResidualFP32Projection123()
        affected.append('time_embedder.'+name)
    return affected
