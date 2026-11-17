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
tbe dsl API:
In order to simplify the procedure of writing schedule, TBE provides a set of TensorEngine APIs.
Using those API to develop operators, you can use the "Auto_schedule" create schedule.
"""
from typing import Any
from typing import Dict
from typing import Optional

import traceback
from .compute import cast
from .compute import conv2d_backprop_filter_compute as conv2d_dw_compute
from .compute import conv2d_backprop_input_compute as conv2d_dx_compute
from .compute import conv3d_backprop_filter_compute as conv3d_dw_compute
from .compute import conv3d_backprop_input_compute as conv3d_dx_compute
from .compute import conv3d_compute
from .compute import depthwise_conv2d_compute
from .compute import dilation_compute
from .compute import gemm_compute
from .compute import mmad_compute
from .compute import math
from .compute import nn
from .compute import reduce
from .compute import array
from .compute import inplace
from .compute import pooling2d as pooling2d_compute
from .compute import pooling3d as pooling3d_compute
from .compute import pooling3d_max_grad_grad as pooling3d_max_grad_grad_compute
from .compute import conv_compute
from .unify_schedule import auto_schedule as tbe_auto_schedule
from .unify_schedule.build import build as tbe_build
from .classifier import shape_classifier
from .base import operation


def ceil(raw_tensor, dtype="int32"):
    """
    cast tensor from src_type to dst_dtype with ceiling method

    Parameters
    ----------
    raw_tensor : wrapped_tensor or tvm.tensor
    dtype : string
    dst dtype need to cast to
    Returns
    -------
    wrapped_tensor : casted tensor
    """
    return cast.ceil(raw_tensor, dtype)


def floor(raw_tensor, dtype="int32"):
    """
    cast tensor from src_type to dst_dtype with flooring method

    Parameters
    ----------
    raw_tensor : wrapped_tensor or tvm.tensor
    dtype : string
    dst dtype need to cast to
    Returns
    -------
    wrapped_tensor : casted tensor
    """
    return cast.floor(raw_tensor, dtype)


def round(raw_tensor, dtype="int32", decimals=0):
    """
    cast tensor from src_type to dst_dtype with rounding method

    Parameters
    ----------
    raw_tensor : wrapped_tensor or tvm.tensor
    dtype : string
    dst dtype need to cast to
    Returns
    -------
    wrapped_tensor : casted tensor
    """
    return cast.round(raw_tensor, dtype, decimals)


def trunc(raw_tensor, dtype="int32"):
    """
    cast tensor from src_type to dst_dtype with trunc method

    Parameters
    ----------
    raw_tensor : wrapped_tensor or tvm.tensor

    dtype : string
    dst dtype need to cast to
    Returns
    -------
    wrapped_tensor : casted tensor
    """
    return cast.trunc(raw_tensor, dtype)


def round_half_up(raw_tensor, dtype="int32"):
    """
    cast tensor from src_type to dst_dtype with rounding method

    Parameters
    ----------
    raw_tensor : wrapped_tensor or tvm.tensor
    dtype : string
    dst dtype need to cast to
    Returns
    -------
    wrapped_tensor : casted tensor
    """
    return cast.round_half_up(raw_tensor, dtype)


def cast_to(data, dtype, f1628IntegerFlag=True):
    """
    a wrapped cast operations , cast data to the type of dtype

    Parameters
    ----------
    data : tvm.tensor
    tensors need to change dtype

    dtype : string
    dst dtype need to cast to

    f1628IntegerFlag : bool
    before fp16->int8/uint8, the data is all interger or not. default value
    is False.

    Returns
    -------
    tensor : tvm.tensor
    """
    return cast.cast_to(data, dtype, f1628IntegerFlag)


def vadd(lhs, rhs):
    """
    calculate elewise add

    Parameters
    ----------
    lhs : wrapped_tensor or tvm.tensor
    left hand tensor

    rhs : wrapped_tensor or tvm.tensor
    left hand tensor

    Returns
    -------
    wrapped_tensor : lhs + rhs
    """
    return math.vadd(lhs, rhs)


def vsub(lhs, rhs):
    """
    calculate elewise sub

    Parameters
    ----------
    lhs : wrapped_tensor or tvm.tensor
    left hand tensor

    rhs : wrapped_tensor or tvm.tensor
    left hand tensor

    Returns
    -------
    wrapped_tensor : lhs - rhs
    """
    return math.vsub(lhs, rhs)


def vgcd(lhs, rhs):
    """
    calculate elewise greatest common divisor

    Parameters
    ----------
    lhs : wrapped_tensor or tvm.tensor
    left hand tensor

    rhs : wrapped_tensor or tvm.tensor
    right hand tensor

    Returns
    -------
    wrapped_tensor : vgcd(lhs, rhs)
    """
    return math.vgcd(lhs, rhs)


def vmul(lhs, rhs):
    """
    calculate elewise multiply

    Parameters
    ----------
    lhs : wrapped_tensor or tvm.tensor
    left hand tensor

    rhs : wrapped_tensor or tvm.tensor
    right hand tensor

    Returns
    -------
    wrapped_tensor : lhs*rhs
    """
    return math.vmul(lhs, rhs)


def vdiv(lhs, rhs):
    """
    calculate elewise div

    Parameters
    -----
    lhs: wrapped_tensor or tvm.tensor
    divisor tensor
    rhs: wrapped_tensor or tvm.tensor
    divided tensor

    returns
    -----
    wrapped_tensor: lhs / rhs
    """
    return math.vdiv(lhs, rhs)


def vrec(raw_tensor, impl_mode="high_performance"):
    """
    calculate vrec(raw_tensor)

    Parameters
    ----------
    raw_tensor : wrapped_tensor or tvm.tensor
    impl_mode : only support high_performance and high_precision

    Returns
    -------
    wrapped_tensor : vrec(raw_tensor)
    """
    return math.vrec(raw_tensor, impl_mode)


def vmod(lhs, rhs):
    """
    calculate element-wise remainder of division

    Parameters
    -----
    lhs : wrapped_tensor or tvm.tensor
    left hand tensor

    rhs : wrapped_tensor or tvm.tensor
    right hand tensor

    Returns
    -----
    wrapped_tensor : lhs - floor(lhs/rhs) * rhs
    """
    return math.vmod(lhs, rhs)


def vmax(lhs, rhs):
    """
    calculate elewise compare, return the min one
    Parameters
    ----------
    lhs : wrapped_tensor or tvm.tensor
    left hand tensor
    rhs : wrapped_tensor or tvm.tensor
    left hand tensor
    Returns
    -------
    wrapped_tensor : max(lhs , rhs)
    """
    return math.vmax(lhs, rhs)


def vmin(lhs, rhs):
    """
    calculate elewise compare, return the min one
    Parameters
    ----------
    lhs : wrapped_tensor or tvm.tensor
    left hand tensor
    rhs : wrapped_tensor or tvm.tensor
    left hand tensor
    Returns
    -------
    wrapped_tensor : min(lhs , rhs)
    """
    return math.vmin(lhs, rhs)


def vlog(raw_tensor, impl_mode="high_performance"):
    """
    calculate ln(raw_tensor)

    Parameters
    ----------
    raw_tensor : wrapped_tensor or tvm.tensor
    impl_mode : only support high_performance and high_precision

    Returns
    -------
    wrapped_tensor : log(raw_tensor)
    """
    return math.vlog(raw_tensor, impl_mode)


def vexp(raw_tensor):
    """
    calculate exp(raw_tensor)

    Parameters
    ----------
    raw_tensor : wrapped_tensor or tvm.tensor

    Returns
    -------
    wrapped_tensor : exp(raw_tensor)
    """
    return math.vexp(raw_tensor)


def vabs(raw_tensor):
    """
    calculate abs(raw_tensor)

    Parameters
    ----------
    raw_tensor : wrapped_tensor or tvm.tensor

    Returns
    -------
    wrapped_tensor : abs(raw_tensor)
    """
    return math.vabs(raw_tensor)


def vsqrt(raw_tensor, impl_mode="high_performance"):
    """
    calculate vsqrt(raw_tensor)

    Parameters
    ----------
    raw_tensor : wrapped_tensor or tvm.tensor
    impl_mode : only support high_performance and high_precision

    Returns
    -------
    wrapped_tensor : vsqrt(raw_tensor)
    """
    return math.vsqrt(raw_tensor, impl_mode)


def vrsqrt(raw_tensor, impl_mode="high_performance"):
    """
    calculate vrsqrt(raw_tensor)

    Parameters
    ----------
    raw_tensor : wrapped_tensor or tvm.tensor
    impl_mode : only support high_performance and high_precision

    Returns
    -------
    wrapped_tensor : vrsqrt(raw_tensor)
    """
    return math.vrsqrt(raw_tensor, impl_mode)


def vnot(raw_tensor):
    """
    calculate vnot(raw_tensor)

    Parameters
    ----------
    raw_tensor : wrapped_tensor or tvm.tensor

    Returns
    -------
    wrapped_tensor : vnot(raw_tensor)
    """
    return math.vnot(raw_tensor)


def vor(lhs, rhs):
    """
    calculate bitwise or op, return the or value
    Parameters
    ----------
    lhs : wrapped_tensor or tvm.tensor
    left hand tensor
    rhs : wrapped_tensor or tvm.tensor
    left hand tensor
    Returns
    -------
    wrapped_tensor : or(lhs , rhs)
    """
    return math.vor(lhs, rhs)


def vand(lhs, rhs):
    """
    calculate bitwise and op, return the and value
    Parameters
    ----------
    lhs : wrapped_tensor or tvm.tensor
    left hand tensor
    rhs : wrapped_tensor or tvm.tensor
    left hand tensor
    Returns
    -------
    wrapped_tensor : max(lhs , rhs)
    """
    return math.vand(lhs, rhs)


def vlogic(lhs, rhs=None, operation='logic_and'):
    """
    calculate elewise logic operation

    Parameters
    ----------
    lhs : wrapped_tensor or tvm.tensor
    left hand tensor

    rhs : wrapped_tensor or tvm.tensor
    right hand tensor

    operation : operator type, logic_and, logic_or, logic_not

    Returns
    -------
    wrapped_tensor
    """
    return math.vlogic(lhs, rhs, operation)


def vadds(raw_tensor, scalar):
    """
    add a tensor by a scalar, dtype of raw_tensor and scalar must be the same

    Parameters
    ----------
    raw_tensor : wrapped_tensor or tvm.tensor

    scalar : float, int, or tvm const

    Returns
    -------
    wrapped_tensor : raw_tensor + scalar
    """
    return math.vadds(raw_tensor, scalar)


def vmuls(raw_tensor, scalar):
    """
    multiply a tensor by a scalar, dtype of raw_tensor
    and scalar must be the same

    Parameters
    ----------
    raw_tensor : wrapped_tensor or tvm.tensor

    scalar : float, int, or tvm const

    Returns
    -------
    wrapped_tensor : raw_tensor*scalar
    """
    return math.vmuls(raw_tensor, scalar)


def vmaxs(raw_tensor, scalar):
    """
    Calculate elewise compare, return the max one of scalar or tensor's element,
    dtype of raw_tensor and scalar must be the same

    Parameters
    ----------
    raw_tensor : wrapped_tensor or tvm.tensor

    scalar : float, int, or tvm const

    Returns
    -------
    wrapped_tensor : max(raw_tensor, scalar)
    """
    return math.vmaxs(raw_tensor, scalar)


def vmins(raw_tensor, scalar):
    """
    Calculate elewise compare, return the min one of scalar or tensor's element,
    dtype of raw_tensor and scalar must be the same

    Parameters
    ----------
    raw_tensor : wrapped_tensor or tvm.tensor

    scalar : float, int, or tvm const

    Returns
    -------
    wrapped_tensor : min(raw_tensor, scalar)
    """
    return math.vmins(raw_tensor, scalar)


def vaxpy(lhs, rhs, scalar):
    """
    calculate elewise scalar*lhs + rhs, return the min one
    Parameters
    ----------
    lhs : wrapped_tensor or tvm.tensor
    left hand tensor
    rhs : wrapped_tensor or tvm.tensor
    left hand tensor
    Returns
    -------
    wrapped_tensor : max(lhs , rhs)
    """
    return math.vaxpy(lhs, rhs, scalar)


def vmla(tensor_0, tensor_1, tensor_2):
    """
    calculate x*tensor_1 + tensor_2,  only support float16, float32
    Parameters
    ----------
    x : wrapped_tensor or tvm.tensor
    tensor_1 : wrapped_tensor or tvm.tensor
    tensor_2 : wrapped_tensor or tvm.tensor
    Returns
    -------
    wrapped_tensor : X*tensor_1 + tensor_2
    """
    return math.vmla(tensor_0, tensor_1, tensor_2)


def vmadd(tensor_0, tensor_1, tensor_2):
    """
    calculate tensor_0*tensor_2 + tensor_1,  only support  float16, float32
    Parameters
    ----------
    tensor_0 : wrapped_tensor or tvm.tensor
    tensor_1 : wrapped_tensor or tvm.tensor
    tensor_2 : wrapped_tensor or tvm.tensor
    Returns
    -------
    wrapped_tensor : tensor_0*tensor_2 + tensor_1
    """
    return math.vmadd(tensor_0, tensor_1, tensor_2)


def vcmp(lhs, rhs, operation='lt', mode='bool'):
    """
    calculate elewise compare

    Parameters
    ----------
    lhs : wrapped_tensor or tvm.tensor
    left hand tensor

    rhs : wrapped_tensor or tvm.tensor
    right hand tensor

    operation : operator type, eq, ne, lt, gt, ge, le

    mode : bool, the dtype of return value is bool
    bit, the dtype of return value is uint8

    Returns
    -------
    wrapped_tensor
    """
    return math.vcmp(lhs, rhs, operation, mode)


def vsel(condition, lhs, rhs):
    """
    if condition = ture, the result is lhs,
    select

    Parameters
    ----------
    condition : wrapped_tensor or tvm.tensor, the dtype is bool or uint8

    lhs : wrapped_tensor or tvm.tensor or scalar

    rhs : wrapped_tensor or tvm.tensor or scalar

    Returns
    -------
    wrapped_tensor :
    """
    return math.vsel(condition, lhs, rhs)


def vcmpsel(lhs, rhs=None, operation='lt', slhs=None, srhs=None):
    """
    calculate elewise compare

    Parameters
    ----------
    lhs : wrapped_tensor or tvm.tensor
    compare left hand tensor
    rhs : wrapped_tensor or tvm.tensor or scalar
    compare right hand tensor or scalar
    operation : operator type, eq, ne, lt, gt, ge, le
    slhs : wrapped_tensor or tvm.tensor or scalar
    select left hand tensor or scalar
    srhs : wrapped_tensor or tvm.tensor or scalar
    select right hand tensor or scalar

    Returns
    -------
    wrapped_tensor
    """
    return math.vcmpsel(lhs, rhs, operation, slhs, srhs)


def vmaddrelu(tensor_0, tensor_1, tensor_2):
    """
    calculate relu(tensor_0*tensor_2 + tensor_1), only support  float16, float32
    Parameters
    ----------
    tensor_0 : wrapped_tensor or tvm.tensor
    tensor_1 : wrapped_tensor or tvm.tensor
    tensor_2 : wrapped_tensor or tvm.tensor
    Returns
    -------
    wrapped_tensor : relu(tensor_0*tensor_2 + tensor_1)
    """
    return nn.vmaddrelu(tensor_0, tensor_1, tensor_2)


def vaddrelu(lhs, rhs):
    """
    calculate relu(lhs + rhs)

    Parameters
    ----------
    lhs : wrapped_tensor or tvm.tensor
    left hand tensor

    rhs : wrapped_tensor or tvm.tensor
    left hand tensor

    Returns
    -------
    wrapped_tensor : relu (lhs + rhs)
    """
    return nn.vaddrelu(lhs, rhs)


def vsubrelu(lhs, rhs):
    """
    calculate relu(lhs - rhs)

    Parameters
    ----------
    lhs : wrapped_tensor or tvm.tensor
    left hand tensor

    rhs : wrapped_tensor or tvm.tensor
    left hand tensor

    Returns
    -------
    wrapped_tensor : relu (lhs - rhs)
    """
    return nn.vsubrelu(lhs, rhs)


def vrelu(raw_tensor):
    """
    calculate vrelu(raw_tensor)

    Parameters
    ----------
    raw_tensor : wrapped_tensor or tvm.tensor

    Returns
    -------
    wrapped_tensor : vrelu(raw_tensor)
    """
    return nn.vrelu(raw_tensor)


def vlrelu(raw_tensor, alpha=0):
    """
    calculate leaky_relu

    Parameters
    ----------
    raw_tensor : wrapped_tensor or tvm.tensor

    Returns
    -------
    wrapped_tensor : vlrelu(raw_tensor)
    """
    return nn.vlrelu(raw_tensor, alpha)


def clip(data, max_value, min_value):
    """
    round data to [min_value,max_value]

    Parameters
    ----------
    data : tvm.tensor
    tensors need to change dtype

    max_value/min_value : float
    the range of res

    Returns
    -------
    tensor : tvm.tensor ,elements in tensor is in range [min_value,max_value]
    """
    return nn.clip(data, max_value, min_value)


def broadcast(var_, shape, output_dtype=None):
    """
    broadcast scalar to tensor, only support float16

    Parameters
    ----------
    var_ : can be python instance of int and float, or tvm.const

    shape : tensor shape

    output_dtype : tensor dtype , default : var.dtype

    Returns
    -------
    wrapped_tensor : broadcast tensor
    """
    return nn.broadcast(var_, shape, output_dtype)


def set_value(tensor, condition, value):
    """
    set specified value.
    Parameters
    ----------
    tensor: tvm.tensor

    condition: lambda expr

    value: const, variable or lambda expr
    Returns
    -------
    wrapped_tensor: updated tensor
    """
    return array.set_value(tensor, condition, value)


def transpose(tensor, axes):
    """
    transpose a tensor by permute.
    Parameters
    ----------
    tensor : tvm.tensor
        Original tensor
    axes : list[int]
        Permutes the dimensions according to the value of axes
    Returns
    -------
    tvm.tensor: A transposed tensor
    """
    return array.transpose(tensor, axes)


def transdata(tensor, dst_shape, axes_map, pad_value=0):
    """
    transdata a tensor by axes_map and dst_shape

    Parameters
    ----------
    tensor : tvm.tensor
        Original tensor
    dst_shape : list[int]
        Shape of dst_tensor after transdata
    axes_map : dict
        Permutes the dimensions according to the axes_map
    pad_value : int
        Determine the padding value when padding is required
    Returns
    -------
    tvm.tensor: A transdata tensor that shape is dst_shape
    """
    return array.transdata(tensor, dst_shape, axes_map, pad_value)


def reduce_sum(raw_tensor, axis, keepdims=False):
    """
    calculate reduce_sum of raw_tensor, only support float16
    Parameters
    ----------
    raw_tensor : wrapped_tensor or tvm.tensor
    axis : int or list
    reduce axis (range : [-len(raw_tensor.shape), len(raw_tensor.shape) - 1])
    keepdims : if true, retains reduced dimensions with length 1, default value is None
    Returns
    -------
    res : wrapped_tensor
    """
    return reduce.reduce_sum(raw_tensor, axis, keepdims)


def reduce_mean(raw_tensor, axis, keepdims=False):
    """
    calculate reduce_mean of raw_tensor
    Parameters
    ----------
    raw_tensor : wrapped_tensor or tvm.tensor
    axis : int or list
    reduce axis (range : [-len(raw_tensor.shape), len(raw_tensor.shape) - 1])
    keepdims : if true, retains reduced dimensions with length 1, default value is False
    Returns
    -------
    res : wrapped_tensor
    """
    return reduce.reduce_mean(raw_tensor, axis, keepdims)


def reduce_min(raw_tensor, axis, keepdims=False, impl_mode="high_performance"):
    """
    calculate reduce_min of raw_tensor, only support float16
    Parameters
    ----------
    raw_tensor : wrapped_tensor or tvm.tensor
    axis : int or list
    reduce axis (range : [-len(raw_tensor.shape), len(raw_tensor.shape) - 1])
    keepdims : if true, retains reduced dimensions with length 1, default value is None
    Returns
    -------
    res : wrapped_tensor
    """
    return reduce.reduce_min(raw_tensor, axis, keepdims, impl_mode)


def reduce_max(raw_tensor, axis, keepdims=False, impl_mode="high_performance"):
    """
    calculate reduce_max of raw_tensor, only support float16
    Parameters
    ----------
    raw_tensor : wrapped_tensor or tvm.tensor
    keepdims : if true, retains reduced dimensions with length 1, default value is None
    axis : int or list
    reduce axis (range : [-len(raw_tensor.shape), len(raw_tensor.shape) - 1])
    priority_flag : supported 1(precision) and 0(performance)
    Returns
    -------
    res : wrapped_tensor
    """
    return reduce.reduce_max(raw_tensor, axis, keepdims, impl_mode)


def reduce_all(raw_tensor, axis, keepdims=False):
    """
    calculate reduce_all of raw_tensor
    Parameters
    ----------
    raw_tensor : wrapped_tensor or tvm.tensor
    axis : int or list
    reduce axis (range : [-len(raw_tensor.shape), len(raw_tensor.shape) - 1])
    keepdims : if true, retains reduced dimensions with length 1, default value is False
    Returns
    -------
    res : wrapped_tensor
    """
    return reduce.reduce_all(raw_tensor, axis, keepdims)


def reduce_any(raw_tensor, axis, keepdims=False):
    """
    calculate reduce_any of raw_tensor
    Parameters
    ----------
    raw_tensor : wrapped_tensor or tvm.tensor
    axis : int or list
    reduce axis (range : [-len(raw_tensor.shape), len(raw_tensor.shape) - 1])
    keepdims : if true, retains reduced dimensions with length 1, default value is False
    Returns
    -------
    res : wrapped_tensor
    """
    return reduce.reduce_any(raw_tensor, axis, keepdims)


def reduce_prod(raw_tensor, axis, keepdims=False):
    """
    calculate reduce_prod of raw_tensor, only support float16
    Parameters
    ----------
    raw_tensor : wrapped_tensor or tvm.tensor
    axis : int
    reduce axis (range : [-len(raw_tensor.shape), len(raw_tensor.shape) - 1])
    Returns
    -------
    res : wrapped_tensor
    """
    return reduce.reduce_prod(raw_tensor, axis, keepdims)


def tuple_sum(input_tensors, axis, keepdims=False):
    """
    calculate sum of raw_tensor, only support float16
    Parameters
    ----------
    input_tensors : wrapped_tensor or tvm.tensor list that each tensor has same reduce operation
    axis : int or list
        reduce axis (range : [-len(raw_tensor.shape), len(raw_tensor.shape) - 1])
    keepdims : if true, retains reduced dimensions with length 1, default value is None
    Returns
    -------
    res : wrapped_tensor
    """
    return reduce.tuple_sum(input_tensors, axis, keepdims)


def split(data, split_dim, size_splits):
    """
    Split a tensor into len(size_splits) tensors along one dimension.

    Parameters
    ----------
    data: TVM tensor
    input tensor.
    split_dim: int
    the dimension along which to split.
    size_splits: list or tuple
    a Python list containing the sizes of each output tensor along `split_dim`.

    Returns
    -------
    output_shape_list: list
    the list of output shapes.
    output_tensor_list: list
    the list of output tensors, output tensor type is TVM tensor.
    """
    return array.split(data, split_dim, size_splits)


def concat(raw_tensors, axis):
    """
    concat shapes at axis,  support int8, uint8, int16, int32 float16, float32
    Parameters
    ----------
    raw_tensors : list of tensors
    axis : concat axis
    Returns
    -------
    concat tensor :
    """
    return array.concat(raw_tensors, axis)


def inplace_add(lhs, inplace_ids, rhs):
    """
    calculate inplace add: computes lhs[inplace_ids, :] += rhs; return lhs.

    Parameters
    ----------
    lhs : wrapped_tensor or tvm.tensor
    left hand tensor

    inplace_ids : a vector. Indices into the left-most dimension of lhs.

    rhs : wrapped_tensor or tvm.tensor
    left hand tensor

    Returns
    -------
    wrapped_tensor : computes lhs[inplace_ids, :] += rhs; return lhs.
    """
    return inplace.inplace_add(lhs, inplace_ids, rhs)


def inplace_sub(lhs, inplace_ids, rhs):
    """
    calculate inplace sub: computes lhs[inplace_ids, :] -= rhs; return lhs.

    Parameters
    ----------
    lhs : wrapped_tensor or tvm.tensor
    left hand tensor

    inplace_ids : a vector. Indices into the left-most dimension of lhs.

    rhs : wrapped_tensor or tvm.tensor
    left hand tensor

    Returns
    -------
    wrapped_tensor : computes lhs[inplace_ids, :] -= rhs; return lhs.
    """
    return inplace.inplace_sub(lhs, inplace_ids, rhs)


def inplace_update(lhs, inplace_ids, rhs):
    """
    calculate inplace add: computes lhs[inplace_ids, :] = rhs; return lhs.

    Parameters
    ----------
    lhs : wrapped_tensor or tvm.tensor
    left hand tensor

    inplace_ids : a vector. Indices into the left-most dimension of lhs.

    rhs : wrapped_tensor or tvm.tensor
    left hand tensor

    Returns
    -------
    wrapped_tensor : computes lhs[inplace_ids, :] = rhs; return lhs.
    """
    return inplace.inplace_update(lhs, inplace_ids, rhs)


def pooling2d(tensor_in, window, stride, pooling_mode, padding_mode="SAME",
              pad=(0, 0, 0, 0), dilation=(1, 1), data_mode=1, ceil_mode=0,
              fusion_params=None, impl_mode="high_performance"):
    """
    :params:
    :tensor_in: input tensor
    :window: input window
    :pooling_mode: can be MAX, AVG, GAP, GMP
    :padding_mode: can be SAME, VALID
    :pad: padT, padB, padL, padR
    :dilation: params to be reserved, use default value
    :stride: window move steps in h or w dimension
    :data_mode: can be 0: CAFFE_DATA_MODE, 1: TENSORFLOW_DATA_MODE
    :ceil_mode : caffe round_mode params, 0:CEIL(default), 1:FLOOR
    :return: pooling result
    """
    return pooling2d_compute.pooling2d(tensor_in, window, stride, pooling_mode,
                                       padding_mode, pad, dilation, data_mode,
                                       ceil_mode, fusion_params, impl_mode)


def pooling3d(tensor_in, window, stride, padding_mode="SAME",
              pads=(0, 0, 0, 0, 0, 0),
              pooling_mode="MAX", dilation=(1, 1, 1), ceil_mode=0):
    """
    :params:
    :tensor_in: input tensor
    :window: input window
    :stride: window move steps in d/h/w dimension
    :padding_mode: can be SAME, VALID
    :pads: padFT, padBK,padT,padB,padL,padR, used for caffe,all zero with tf
    :pooling_mode: can be MAX, (AVG, GAP, GMP -- Not support yet)
    :dilation: params to be reserved, use default value
    :ceil_mode : caffe round_mode params, 0:CEIL(default), 1:FLOOR
    :return: pooling result
    """
    return pooling3d_compute.pooling3d(tensor_in, window, stride, padding_mode,
                                       pads, pooling_mode, dilation, ceil_mode)


def max_pooling3d_grad_grad(orig_input, orig_output, grad_grad, assist_tensor,
                            ksize, strides, pads=(0, 0, 0, 0, 0, 0),
                            data_format="NDHWC",
                            padding="SAME"):
    """
    orig_input : dict, shape and dtype of input_data,
    shape is 6 dims, format is NDC1HWC0
    orig_output : dict, result of max_pool3d(orig_input, ksize, ...)
    grad_grad: dict, input grad of grad
    assist_tensor: dict, helper matrix, it's content is 8,7,6,5,4,3,2,1
    if kernel is 2 x 2 x 2
    ksize : list or tuple, the window of max_pool3d,
    only support max_pool3d in D or H or W
    strides : list or tuple, the stride of max_pool3d window,
    only support max_pool3d in D or H or W
    pads : reserved.
    padding : str, the mode of padding, support SAME or VALID
    ceil_mode: reserved
    """
    return pooling3d_max_grad_grad_compute.max_pooling3d_grad_grad(orig_input,
                                                                   orig_output,
                                                                   grad_grad,
                                                                   assist_tensor,
                                                                   ksize,
                                                                   strides,
                                                                   pads,
                                                                   data_format,
                                                                   padding)


def auto_schedule(outs, option=None):
    """
    Entry of auto-Schedule.

    Parameters
    ----------
    outs: Array of Tensor
    The computation graph description of reduce in the format
    of an array of tensors.
    option:
    Returns
    -------
    sch: Schedule
    The computation schedule for the op.
    """
    return tbe_auto_schedule.auto_schedule(outs, option)


def build(sch, config_map=None):
    """
    :param sch:
    :param config_map:
    :return:
    """
    return tbe_build(sch, config_map)


def classify(ins: list, mode: str, extra_params: Optional[Dict[str, Any]] = None):
    """
    classify according to mode
    :param ins:
    :param mode: support elewise, broadcast, reduce
    :param extra_params: must include keepdims when mode is reduce
    :return:
    """
    return shape_classifier.classify(ins, mode, extra_params)


def var(name, bound=None, dtype="int32", addition=None):
    """
    add var for external
    :param name:
    :param bound:
    :param dtype: such as int32, float16...
    :param addition:
    :return:
    """
    return operation.var(name, bound, dtype, addition)


def var_attr(name, bound=None, dtype="int32", addition=None):
    """
    var attribute
    :param name:
    :param bound:
    :param dtype: such as int32, float16, int32[4]
    :param addition:
    :return:
    """
    return operation.var_attr(name, bound, dtype, addition)


def add_build_arg(key, value):
    """
    add build arg
    :param key:
    :param value:
    :return:
    """
    return operation.add_build_arg(key, value)


def add_exclude_bound_var(var_):
    """
    add exclude bound var
    :param var_:
    :return:
    """
    return operation.add_exclude_bound_var(var_)


def compute(_operator=None):
    """
    generate a ComputeContext instance
    :param _operator:
    :return:
    """
    return operation.compute(_operator)


def schedule(_compute=None):
    """
    generate a ScheduleContext instance
    :param _compute:
    :return:
    """
    return operation.schedule(_compute)


def conv2d_backprop_filter(input_x, out_backprop, filter_sizes, para_dict):
    """
    the DSL interface of conv2d backprop filter compute

    Parameters:
    ----------
    x : the featuremap data, tvm.placeholder, 5HD shape

    out_backprop : the grads data, tvm.placeholder, 5HD shape

    filter_sizes : 4-D shape, specifies the filter sizes

    para_dict:

    strides : 2-D shape, specifies in height and width dimension

    padding : 4-D shape, specifies in up/down/left/right dimension

    dilations : 4-D shape, specifies in batch/channel/height/width dimension

    groups : The number of filter's group. Default value is 1.

    res_dtype : the output data type

    Returns
    -------
    result tensor of conv2d_backprop_filter compute
    """
    return conv2d_dw_compute.conv2d_backprop_filter_compute(input_x, out_backprop, filter_sizes, para_dict)


def conv2d_backprop_input(filters, out_backprop, filter_sizes, input_sizes, para_dict):
    """
    DSL interface of conv2d backprop input

    Parameters
    ----------
    filters : weight tensor of fractal shape

    out_backprop : 5D dE/dY tensor

    filter_sizes : shape of weight, [N, C, H, W]

    input_sizes : shape of dE/dX, [N, C, H, W]

    para_dict:

    strides : list of strides, [strideh, stridew]

    padding : list of padding, [pad_up, pad_down, pad_left, pad_right]

    dilations : list of dilations, [dilation_n, dilation_c, dilation_h, dilation_w]

    res_dtype : dE/dX data type, "float16" by default

    offset_x : offset of x

    offset_w : offset of w

    fusion_para: the l1 fuison para

    kernel_name : cce kernel name

    group_dict : The params of group convolution.

    Returns
    ----------
    dx_ddr: dE/dX tensor
    """
    return conv2d_dx_compute.conv2d_backprop_input_compute(filters, out_backprop, filter_sizes, input_sizes,
                                                           para_dict)


def conv3d_backprop_filter(x, out_backprop, filter_size, para_dict):
    """
    DSL interface of conv3d bp dx

    Parameters
    ----------
    x : the featuremap data, tvm.placeholder, 6hd shape

    out_backprop : the grads data, tvm.placeholder, 6hd shape

    filter_size : 5-D shape, specifies the filter sizes

    para_dict : dict of parameters
    strides : 3-D shape, specifies in depth, height and width dimension
    pads : 6-D shape, specifies in up/down/left/right dimension
    dilations : 5-D shape, specifies in batch/channel/depth/height/width dimension
    res_dtype : the output data type
    kernel_name : conv3d_backprop_filter_cce by default
    group_dict : group of parameters

    Returns
    -------
    result tensor of conv3d_backprop_filter compute
    """
    return conv3d_dw_compute.conv3d_dw(x, out_backprop, filter_size, para_dict)


def conv3d_backprop_input(filter, out_backprop, filter_size, input_size, para_dict):
    """
    DSL interface of conv3d bp dx

    Parameters
    ----------
    filter : weight tensor of fractal shape

    out_backprop : 5D dE/dY tensor

    filter_size : shape of weight, [N, C, D, H, W]

    input_size : shape of dE/dX, [N, D, H, W, C]

    para_dict : dict of parameters
    strides : list of strides, [stridebatch, strided, strideh, stridew, stridechannel]
    pads : list of padding, [pad_front, pad_tail, pad_up, pad_down, pad_left, pad_right]
    dilations : [1, 1, 1, 1, 1] by default
    res_dtype : dE/dX data type, "float16" by default
    kernel_name : conv3d_backprop_input_cce by default
    group_dict : group of parameters

    Returns
    ----------
    dx_ddr: dE/dX tensor
    """
    return conv3d_dx_compute.conv3d_dx(filter, out_backprop, filter_size, input_size, para_dict)


def conv3d(x, filter, filter_size, para_dict):
    """
    conv

    Parameters
    ----------
    x: feature map

    weight: filter

    filter_size : filter_size

    para_dict: dict of params

    Returns
    -------
    tensor : res
    """
    return conv3d_compute.conv3d(x, filter, filter_size, para_dict)


def depthwise_conv2d_backprop_filter(fmap,
                                     dout,
                                     kernel_h,
                                     kernel_w,
                                     stride,
                                     pad,
                                     dilations,
                                     w_dtype,
                                     kernel_name="depthwise_conv2d_compute"):
    """
    compute of depthwise conv2d backprop filter
    the interface will be eliminated soon!

    Parameters
    ----------
    fmap : tvm tensor
    feature map tensor in tvm.

    dout : tvm tensor
    dout tensor in tvm.

    kernel_h: int
    height of filter.

    kernel_w: int
    width of filter.

    stride: tuple or list or int
    stride of convolution.

    pad: list
    padding added to each dimension of the input.

    w_dtype: str
    the dtype of dfilter.

    Returns
    -------
    depthwise_dfilter_res: tvm tensor
    the tensor of output.
    """
    return depthwise_conv2d_compute.depthwise_conv2d_backprop_filter_d_compute(
        fmap, dout, kernel_h, kernel_w, stride, pad, dilations, w_dtype, kernel_name)


def depthwise_conv2d_backprop_input(input_shape,
                                    weight,
                                    dout,
                                    weight_sizes,
                                    strides,
                                    pads,
                                    kernel_name="depthwise_conv2d_compute"):
    """
    Computes the gradients of depthwise convolution with respect to the input.

    the interface will be eliminated soon!

    Parameters
    ----------
    input_shape: a list or tuple representing the shape of input,
    6D format [N, C1, 1, H, W, C0]

    weight: a tensor, 5D with shape [C1, Hf*Wf, 1, C0, C0]

    dout: a tensor, 6D format [N, Co1, 1, Ho, Wo, C0]

    weight_sizes: a list or tuple of two ints,
    the height and width of the weight of the convolution

    strides: a list or tuple of two ints, the stride of the sliding window for
    height and width of the input of the convolution

    pads: padding added to each dimension of the input

    Returns
    -------
    dx_res: compute of the gradients of depthwise convolution
    with respect to the input
    """
    return depthwise_conv2d_compute.depthwise_conv2d_backprop_input_d_compute(
        input_shape, weight, dout, weight_sizes, strides, pads, kernel_name)


def dilation(tensor_x, dilations, pads=None, padding_value=0.0):
    """
    dilation_compute
    :param tensor_x: tensor
    :param dilations: list or tuple
    :param pads: list or tuple or None
    :param padding_value: float
    """
    return dilation_compute.dilation_compute(tensor_x, dilations, pads, padding_value)


def gemm(tensor_a, tensor_b, para_dict):
    """
    algorithm: gemm and matmul
    for gemm:
    calculating matrix multiplication, C = alpha_num*A*B+  beta_num*C
    for matmul:
    caculating matrix multiplication with bias, C = A*B + bias

    Parameters:
    tensor_a: the first tensor a

    tensor_b: second tensor b with the same type and shape with a

    If tensor_a/tensor_b is int8/uint8,then L0A must be 16*32,L0B
    must be 32*16.
    If A is transpose , then AShape classification matrix must be
    32*16 in gm/L1,then it is 16*32 in L0A.
    If B is transpose , then BShape classification matrix must be
    16*32 in gm/L1,then it is 32*16 in L0B.

    para_dict:

    Returns result
    """
    return gemm_compute.gemm(tensor_a, tensor_b, para_dict)


def matmul(tensor_a, tensor_b, trans_a=False, trans_b=False, format_a="ND", format_b="ND",
           alpha_num=1.0, beta_num=1.0, dst_dtype="float16", tensor_bias=None,
           quantize_params=None, format_out=None, compress_index=None,
           attrs={}, kernel_name="MatMul"):
    """
    algorithm: mmad
    calculating  matrix multiplication, C=alpha_num*A*B+beta_num*C

    Parameters:
    tensor_a : the first tensor a
        support dtype: float16 and int8
        support format: fractal and ND

    tensor_b : second tensor b with the same type and shape with a
        support dtype: float16 and int8
        support format: fractal and ND

    trans_a : if True, a needs to be transposed

    trans_b : if True, b needs to be transposed

    is_fractal: If type is bool, a and b's format both be fractal or ND,
                default is ND;
                If type is list, len must be 2, [0] is is_fractal_a,
                [1] is is_fractal_b

    alpha_num: scalar used for multiplication, not support now.

    beta_num: scalar used for multiplication, not support now.

    dst_dtype: output data type, support "float16" "float32", default is "float16"

    tensor_bias :the bias with used to init L0C for tensor c

    quantize_params: quantization parameters,
            not None means enable quantization, it is dictionary structure

        quantize_alg: quantize mode,
            support 'NON_OFFSET' 'HALF_OFFSET_A' 'HALF_OFFSET_B' 'ALL_OFFSET'

        scale_mode_a: tensor_a inbound quantization mode, not support now.
        scale_mode_b: tensor_b inbound quantization mode, not support now.
        scale_mode_out: out tensor quantization mode, support 'SCALAR' and 'VECTOR'

        sqrt_mode_a: tensor_a inbound sqrt mode, not support now.
        sqrt_mode_b: tensor_b inbound sqrt mode, not support now.
        sqrt_mode_out: out tensor sqrt mode, support 'NON_SQRT' and 'SQRT'

        scale_q_a: scale placeholder for tensor_a inbound quantization, not support now.
        offset_q_a: offset placeholder for tensor_a inbound quantization, not support now.
        scale_q_b: scale placeholder for tensor_b inbound quantization, not support now.
        offset_q_b: offset placeholder for tensor_b inbound quantization, not support now.

        scale_drq: scale placeholder for requantization or dequantization
        offset_drq: scale placeholder for requantization or dequantization, not support now.
    out_format: output format
    attrs:
        offset_x: the offset for fmap
        offset_w: the offset for w

    compress_index: index for compressed wights, None means not compress wights
    Returns tensor
    """
    offset_x = attrs.get("offset_x", 0)
    offset_w = attrs.get("offset_w")
    para_dict = {
        "trans_a": trans_a,
        "trans_b": trans_b,
        "format_a": format_a,
        "format_b": format_b,
        "tensor_c": tensor_bias,
        "dst_dtype": dst_dtype,
        "format_out": format_out,
        "offset_a": offset_x,
        "offset_b": offset_w,
        "kernel_name": kernel_name
        }

    def find_attr_and_add(attr_name):
        if attr_name in attrs:
            para_dict[attr_name] = attrs.get(attr_name)
    find_attr_and_add("impl_mode")
    find_attr_and_add("batch_shape_a")
    find_attr_and_add("batch_shape_b")
    find_attr_and_add("batch_shape_out")
    find_attr_and_add("quantize_params")
    find_attr_and_add("compress_index")
    result = gemm_compute.gemm(tensor_a=tensor_a, tensor_b=tensor_b, para_dict=para_dict)
    return result


# 'pylint: disable=too-many-arguments
def gather(params, indices, axis=None, batch_dims=0, negative_index_support=False, support_out_of_bound_index=True):
    """
    :param params: The tensor from which to gather values. Must be at least rank axis + 1
    :param indices: The index tensor. Must be one of the following types: int32, int64.
                    The values must be in range [0, params.shape[axis]].
    :param axis: The axis in params to gather indices from. Must be greater than or equal to batch_dims.
                 Defaults to fisrt non-batch dimension.
    :param batch_dims: An integer. The number of batch dimensions. Must be less than or equal to rank(indices).
    :param negative_index_support: Whether to support negative index, An optional bool. Defaults to false.
    :param support_out_of_bound_index: default as True. When the index is out of bound, the corresponding
                                       result will set as 0.
    :return:
    """
    return array.gather(params, indices, axis, batch_dims, negative_index_support, support_out_of_bound_index)


def gather_nd(params, indices, batch_dims=0, negative_index_support=False, support_out_of_bound_index=True):
    """
    :param params: The tensor from which to gather values.
    :param indices: The index tensor. Must be one of the following types: int32, int64.
    :param batch_dims: An integer. The number of batch dimensions.
    :param support_out_of_bound_index: When the index is out of bound, the corresponding result will be
                                       set as 0.
    :return:
    """
    return array.gather_nd(params, indices, batch_dims, negative_index_support, support_out_of_bound_index)


def slice(tensor, begin, end, stride=None):
    """
    :param tensor: The tensor from which to slice values.
    :param begin: The begin indexes to slice for each dimension.
    :param end: The end indexes to slice for each dimension and end point not include.
    :param stride: The stride means slice continues by stride length. now stride is not support.
    :return:
    """
    return array.slice(tensor, begin, end, stride)


def conv(data, weight, para_dict, optim_dict=None, dsl_flag=True):
    """
    conv

    Parameters
    ----------
    data: feature map

    weight: filter

    para_dict: dict of params

    optim_dict: optim flag

    dsl_flag: true if not from topi

    Returns
    -------
    tensor: res
    """
    def cal_weight_ori_shape(weight_frac_z):
        """
        cal nchw weight shape
        """
        cikhkw, co1, co0, ci0 = weight_frac_z
        filter_h, filter_w = para_dict["filter_h"], para_dict["filter_w"]
        cout, cin, kernel_h, kernel_w = co1*co0, (ci0*cikhkw)//(filter_h*filter_w), filter_h, filter_w
        weight_nchw = (cout, cin, kernel_h, kernel_w)
        return weight_nchw

    # get shape info from tensor
    fmap_5hd = list(i.value for i in data.shape)
    weight_frac_z = list(i.value for i in weight.shape)

    # set para_dict for dsl api
    para_dict["a_shape"] = fmap_5hd
    para_dict["weight_ori_shape_nchw"] = cal_weight_ori_shape(weight_frac_z)
    para_dict["group"] = para_dict.get("group", 1)
    para_dict["group_opt"] = para_dict.get("group_opt", 1)
    para_dict["c1_opt"] = para_dict.get("c1_opt", 1)
    para_dict["cout1_opt"] = para_dict.get("cout1_opt", 1)
    para_dict["dilate_h"] = para_dict.get("dilate_h", 1)
    para_dict["dilate_w"] = para_dict.get("dilate_w", 1)
    para_dict["weight_fracz_shape"] = weight_frac_z

    para_dict["fusion_para"] = {
        "fmap_l1_addr_flag": "nothing",
        "fmap_l1_valid_size": -1,
        "slice_offset": (0, 0, 0, 0, 0),
    }

    return conv_compute.conv(data, weight, para_dict, optim_dict, dsl_flag)


def depthwise_conv2d(data, weight, para_dict, optim_dict=None, dsl_flag=True):
    """
    depthwise conv2d

    Parameters
    ----------
    data: feature map

    weight: filter

    para_dict: dict of params

    optim_dict: optim flag

    dsl_flag: true if not from topi

    Returns
    -------
    tensor: res
    """
    return conv(data, weight, para_dict, optim_dict, dsl_flag)


# 'pylint: disable=too-many-arguments,huawei-too-many-arguments
def reduce_window(input_tensor, reduction, window_axes, window_dimensions, window_strides, window_dilations,
                  padding_mode="SAME", padding_dimensions=None, rounding_mode="FLOOR", return_indices=False):
    """
    calculate reduce_window of input

    Parameters
    ----------
    input_tensor : tvm.tensor
        Original tensor
    reduction : str
        Type of reduction, only support "MAX" and "GMP"(AVG, GAP are not supported yet)
    window_axes : list or tuple
        Indices of window, only support
        Range : [-len(input.shape), len(input.shape) - 1]
    window_dimensions : list or tuple
        Dimensions of window, length is equal to window_axes
    window_strides : list or tuple
        Strides of window, length is equal to window_axes
    window_dilations : list or tuple
        Dilations of window, length is equal to window_axes
    padding_mode : str
        Mode of padding, only support "SAME", "VALID" and "CALCULATED"
    padding_dimensions : list[list] or tuple[tuple]
        Dimensions of padding on all window sides and one window side corresponds to two dimensions
        This parameter is valid only when padding_mode is "CALCULATED"
        None means padding dimensions are all zero
    rounding_mode : str
        Mode of rounding in output shape computation, only support "FLOOR"(defalut) and "CEIL"
        This parameter is valid only when padding_mode is "CALCULATED"
    return_indices : bool
        Whether to return the max indices along with the outputs, default value is false
        This parameter is valid only when reduction is "MAX"

    Returns
    -------
    res : wrapped_tensor
    """
    return reduce.reduce_window(input_tensor, reduction, window_axes, window_dimensions, window_strides,
                                window_dilations, padding_mode, padding_dimensions, rounding_mode, return_indices)


def sort(tensor, sort_axis=-1, direction="ascend", return_type="value", indices_dtype=None, need_cast=False):
    """
    :param tensor: tvm.tensor
        The input tensor to sort
    :param sort_axis: int
        The axis along which to sort
    :param direction: str
        The sorting order("ascend" or "descend")
    :param return_type: str
        Return output condition("value", "indices", "both")
    :param indices_dtype: str
        Return indices dtype("int32", "int64")
    """
    return array.sort(tensor, sort_axis, direction, return_type, indices_dtype, need_cast)


def topk(tensor, k, sort_axis=-1, direction="ascend", return_type="value", indices_dtype=None, need_cast=False):
    """
    :param tensor: tvm.tensor
        The input tensor to sort
    :param k: int/Expr
        Number of top elements to look for along the sort axis
    :param sort_axis: int
        The axis along which to sort
    :param direction: str
        The sorting order("ascend" or "descend")
    :param return_type: str
        Return output condition("value", "indices", "both")
    :param indices_dtype: str
        Return indices dtype("int32", "int64")
    """
    return array.topk(tensor, k, sort_axis, direction, return_type, indices_dtype, need_cast)


def scatter(var, indices, update, reduction, scatter_axis=0, is_ref=True, support_out_of_bound_index=True):
    """
    Parameters
    ----------
    var: The tensor for scatter operation.
    indices: Tensor of indices into the specified dimension of var.
    update: A tensor of updated values to modify in var.
    reduction: An str. Specify the scatter operations, such as "add", "sub", etc.
    scatter_axis: An integer. Specify the axis of the scatter operations.
    is_ref: An bool. Specify whether to reuse memory between the output and var.
    support_out_of_bound_index: An bool. Specify whether to check the validity of the data in "indices", the legal
    value range is [0, var[scatter_axis]), matters needing attention, this check will affect the performance.

    Returns
    -------
    tensor: res
    """
    return array.scatter(var, indices, update, reduction, scatter_axis, is_ref, support_out_of_bound_index)


def scatter_nd(var, indices, update, reduction, scatter_axis=0, is_ref=True, support_out_of_bound_index=True):
    """
    Parameters
    ----------
    var: The tensor for scatter operation.
    indices: Tensor of indices into the specified dimension of var.
    update: A tensor of updated values to modify in var.
    reduction: An str. Specify the scatter operations, such as "add", "sub", etc.
    scatter_axis: An integer. Specify the axis of the scatter operations.
    is_ref: An bool. Specify whether to reuse memory between the output and var.
    support_out_of_bound_index: An bool. Specify whether to check the validity of the data in "indices", the legal
    value range is [0, var[scatter_axis]), matters needing attention, this check will affect the performance.

    Returns
    -------
    tensor: res
    """
    return array.scatter_nd(var, indices, update, reduction, scatter_axis, is_ref, support_out_of_bound_index)


def placeholder(x, name="placeholder", attrs=None):
    """
    Parameters
    ----------
    x: The input of the tensor, contains shape and dtype.
    name: The name of the tensor.
    attrs: The attribute of the tensor.

    Returns
    -------
    tensor: The created tensor
    """
    return operation.placeholder(x, name=name, attrs=attrs)
