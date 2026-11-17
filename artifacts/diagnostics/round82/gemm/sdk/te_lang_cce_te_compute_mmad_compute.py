#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Copyright 2019-2020 Huawei Technologies Co., Ltd
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================
"""
mmad_compute
"""
import warnings


def matmul(tensor_a,
           tensor_b,
           trans_a=False,
           trans_b=False,
           format_a="ND",
           format_b="ND",
           alpha_num=1.0,
           beta_num=1.0,
           dst_dtype="float16",
           tensor_bias=None,
           quantize_params=None,
           format_out=None,
           compress_index=None,
           attrs=None,
           kernel_name="MatMul"):
    """
    algorithm: mmad
    calculating  matrix multiplication, C=alpha_num*A*B+beta_num*C

    Parameters:
    tensor_a : the first tensor a

    tensor_b : second tensor b with the same type and shape with a

    If tensor_a/tensor_b is int8/uint8,then L0A must be 16*32,L0B
    must be 32*16.
    If A is transpose , then AShape classification matrix must be
    32*16 in gm/L1,then it is 16*32 in L0A.
    If B is transpose , then BShape classification matrix must be
    16*32 in gm/L1,then it is 32*16 in L0B.

    trans_a : if True, a needs to be transposed

    trans_b : if True, b needs to be transposed

    is_fractal: If type is bool, a and b's format both be fractal or ND, default is ND;
    If type is list, len must be 2, [0] is is_fractal_a, [1] is is_fractal_b

    alpha_num: scalar used for multiplication

    beta_num: scalar used for multiplication

    dst_dtype: output data type,support "float16" "float32", default is "float16"

    tensor_bias :the bias with used to init L0C for tensor c

    quantize_params: quantization parameters, not None means enable quantization, it is dictionary structure

    quantize_alg: quantize mode, support 'NON_OFFSET' 'HALF_OFFSET_A' 'HALF_OFFSET_B' 'ALL_OFFSET'

    scale_mode_a: tensor_a inbound quantization mode, support 'SCALAR' and 'VECTOR'
    scale_mode_b: tensor_b inbound quantization mode, support 'SCALAR' and 'VECTOR'
    scale_mode_out: out tensor quantization mode, support 'SCALAR' and 'VECTOR'

    sqrt_mode_a: tensor_a inbound sqrt mode, support 'NON_SQRT' and 'SQRT'
    sqrt_mode_b: tensor_b inbound sqrt mode, support 'NON_SQRT' and 'SQRT'
    sqrt_mode_out: out tensor sqrt mode, support 'NON_SQRT' and 'SQRT'

    scale_q_a: scale placeholder for tensor_a inbound quantization
    offset_q_a: offset placeholder for tensor_a inbound quantization
    scale_q_b: scale placeholder for tensor_b inbound quantization
    offset_q_b: offset placeholder for tensor_b inbound quantization

    scale_drq: scale placeholder for requantization or dequantization
    offset_drq: scale placeholder for requantization or dequantization
    out_format: output format
    attrs:

    offset_x: the offset for fmap
    offset_w: the offset for w

    compress_index: index for compressed wights, None means not compress wights
    Returns None
    """
    attrs = {} if not attrs else attrs
    warnings.warn("te.lang.cce.te_compute.mmad_compute is expired, "
        "please replace it with the func tbe.dsl.compute.mmad_compute",
        DeprecationWarning)
    from tbe.dsl.compute.mmad_compute import matmul
    return matmul(tensor_a, tensor_b, trans_a, trans_b, format_a, format_b, alpha_num, beta_num, dst_dtype,
                  tensor_bias, quantize_params, format_out, compress_index, attrs, kernel_name)


def get_matmul_performance_format(tensor_a,
                                  tensor_b,
                                  trans_a=False,
                                  trans_b=False,
                                  format_a="ND",
                                  format_b="ND",
                                  alpha_num=1.0,
                                  beta_num=1.0,
                                  dst_dtype="float16",
                                  tensor_bias=None,
                                  quantize_params=None,
                                  format_out=None):
    """
    get matmul performance format

    Parameters:
    tensor_a : the first tensor a

    tensor_b : second tensor b

    trans_a : if True, a needs to be transposed

    trans_b : if True, b needs to be transposed

    is_fractal: format is fractal or ND

    alpha_num: scalar used for multiplication

    beta_num: scalar used for multiplication

    dst_dtype: output data type

    tensor_bias :the bias with used to init L0C for tensor c

    quantize_params: quantization parameters

    out_format: output format

    Returns: tensor a format
    """
    warnings.warn("te.lang.cce.te_compute.mmad_compute is expired, "
        "please replace it with the func tbe.dsl.compute.mmad_compute",
        DeprecationWarning)
    from tbe.dsl.compute.mmad_compute import get_matmul_performance_format
    return get_matmul_performance_format(tensor_a, tensor_b, trans_a, trans_b, format_a, format_b, alpha_num,
                                         beta_num, dst_dtype, tensor_bias, quantize_params, format_out)
