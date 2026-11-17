#!/usr/bin/env python
# -*- coding: UTF-8 -*-
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
import warnings
from .auto_cast import auto_cast_of_elewise
from .auto_cast import auto_cast_of_reduce
from .auto_cast import auto_cast_of_cast

STACKLEVEL_FOR_DSL_AUTOCAST = 4
STACKLEVEL_FOR_DSL_NO_AUTOCAST = 2


@auto_cast_of_cast
def ceil(raw_tensor):
    """
    cast tensor from src_type to dst_dtype with ceiling method

    Parameters
    ----------
    raw_tensor : wrapped_tensor or tvm.te.tensor

    Returns
    -------
    wrapped_tensor : casted tensor
    """
    warnings.warn("te.lang.cce.ceil is deprecated, please replace it with tbe.dsl.ceil",
                DeprecationWarning, stacklevel=STACKLEVEL_FOR_DSL_AUTOCAST)
    import tbe.dsl
    return tbe.dsl.ceil(raw_tensor)


@auto_cast_of_cast
def floor(raw_tensor):
    """
    cast tensor from src_type to dst_dtype with flooring method

    Parameters
    ----------
    raw_tensor : wrapped_tensor or tvm.te.tensor

    Returns
    -------
    wrapped_tensor : casted tensor
    """
    warnings.warn("te.lang.cce.floor is deprecated, please replace it with tbe.dsl.floor",
                DeprecationWarning, stacklevel=STACKLEVEL_FOR_DSL_AUTOCAST)
    import tbe.dsl
    return tbe.dsl.floor(raw_tensor)


@auto_cast_of_cast
def round(raw_tensor):
    """
    cast tensor from src_type to dst_dtype with rounding method

    Parameters
    ----------
    raw_tensor : wrapped_tensor or tvm.te.tensor

    Returns
    -------
    wrapped_tensor : casted tensor
    """
    warnings.warn("te.lang.cce.round is deprecated, please replace it with tbe.dsl.round",
                DeprecationWarning, stacklevel=STACKLEVEL_FOR_DSL_AUTOCAST)
    import tbe.dsl
    return tbe.dsl.round(raw_tensor)


@auto_cast_of_cast
def trunc(raw_tensor):
    """
    cast tensor from src_type to dst_dtype with trunc method

    Parameters
    ----------
    raw_tensor : wrapped_tensor or tvm.te.tensor

    Returns
    -------
    wrapped_tensor : casted tensor
    """
    warnings.warn("te.lang.cce.trunc is deprecated, please replace it with tbe.dsl.trunc",
                DeprecationWarning, stacklevel=STACKLEVEL_FOR_DSL_AUTOCAST)
    import tbe.dsl
    return tbe.dsl.trunc(raw_tensor)


def round_half_up(raw_tensor):
    """
    cast tensor from src_type to dst_dtype with rounding method

    Parameters
    ----------
    raw_tensor : wrapped_tensor or tvm.te.tensor

    Returns
    -------
    wrapped_tensor : casted tensor
    """
    warnings.warn(
        "te.lang.cce.round_half_up is deprecated, please replace it with tbe.dsl.round_half_up",
         DeprecationWarning, stacklevel=STACKLEVEL_FOR_DSL_NO_AUTOCAST)
    import tbe.dsl
    return tbe.dsl.round_half_up(raw_tensor)


def cast_to(data, dtype, f1628IntegerFlag=True):
    """
    a wrapped cast operations , cast data to the type of dtype

    Parameters
    ----------
    data : tvm.te.tensor
        tensors need to change dtype

    dtype : string
        dst dtype need to cast to

    f1628IntegerFlag : bool
        before fp16->int8/uint8, the data is all interger or not. default value
        is False.

    Returns
    -------
    tensor : tvm.te.tensor
    """
    warnings.warn("te.lang.cce.cast_to is deprecated, please replace it with tbe.dsl.cast_to",
                DeprecationWarning, stacklevel=STACKLEVEL_FOR_DSL_NO_AUTOCAST)
    import tbe.dsl
    return tbe.dsl.cast_to(data, dtype, f1628IntegerFlag)


@auto_cast_of_elewise
def vadd(lhs, rhs):
    """
    calculate elewise add

    Parameters
    ----------
    lhs : wrapped_tensor or tvm.te.tensor
        left hand tensor

    rhs : wrapped_tensor or tvm.te.tensor
        left hand tensor

    Returns
    -------
    wrapped_tensor : lhs + rhs
    """
    warnings.warn("te.lang.cce.vadd is deprecated, please replace it with tbe.dsl.vadd",
                DeprecationWarning, stacklevel=STACKLEVEL_FOR_DSL_AUTOCAST)
    import tbe.dsl
    return tbe.dsl.vadd(lhs, rhs)


@auto_cast_of_elewise
def vsub(lhs, rhs):
    """
    calculate elewise sub

    Parameters
    ----------
    lhs : wrapped_tensor or tvm.te.tensor
        left hand tensor

    rhs : wrapped_tensor or tvm.te.tensor
        left hand tensor

    Returns
    -------
    wrapped_tensor : lhs - rhs
    """
    warnings.warn("te.lang.cce.vsub is deprecated, please replace it with tbe.dsl.vsub",
                DeprecationWarning, stacklevel=STACKLEVEL_FOR_DSL_AUTOCAST)
    import tbe.dsl
    return tbe.dsl.vsub(lhs, rhs)


@auto_cast_of_elewise
def vmul(lhs, rhs):
    """
    calculate elewise multiply

    Parameters
    ----------
    lhs : wrapped_tensor or tvm.te.tensor
        left hand tensor

    rhs : wrapped_tensor or tvm.te.tensor
        right hand tensor

    Returns
    -------
    wrapped_tensor : lhs*rhs
    """
    warnings.warn("te.lang.cce.vmul is deprecated, please replace it with tbe.dsl.vmul",
                DeprecationWarning, stacklevel=STACKLEVEL_FOR_DSL_AUTOCAST)
    import tbe.dsl
    return tbe.dsl.vmul(lhs, rhs)


@auto_cast_of_elewise
def vdiv(lhs, rhs):
    """
    calculate elewise div

    Parameters
    -----
    lhs: wrapped_tensor or tvm.te.tensor
         divisor tensor
    rhs: wrapped_tensor or tvm.te.tensor
         divided tensor

    returns
    -----
    wrapped_tensor: lhs / rhs
    """
    warnings.warn("te.lang.cce.vdiv is deprecated, please replace it with tbe.dsl.vdiv",
                DeprecationWarning, stacklevel=STACKLEVEL_FOR_DSL_AUTOCAST)
    import tbe.dsl
    return tbe.dsl.vdiv(lhs, rhs)


@auto_cast_of_elewise
def vrec(raw_tensor, priority_flag=1):
    """
    calculate vrec(raw_tensor)

    Parameters
    ----------
    raw_tensor : wrapped_tensor or tvm.te.tensor
    priority_flag: priority flag, only support 1(precision), 0(performance)

    Returns
    -------
    wrapped_tensor : vrec(raw_tensor)
    """
    warnings.warn("te.lang.cce.vrec is deprecated, please replace it with tbe.dsl.vrec",
                DeprecationWarning, stacklevel=STACKLEVEL_FOR_DSL_AUTOCAST)
    impl_mode = "high_precision"
    from .te_compute.util import _get_priority_flag_value
    if _get_priority_flag_value(priority_flag) == 0.0:
        impl_mode = "high_performance"
    import tbe.dsl
    return tbe.dsl.vrec(raw_tensor, impl_mode)


def vmod(lhs, rhs):
    """
    calculate element-wise remainder of division

    Parameters
    -----
    lhs : wrapped_tensor or tvm.te.tensor
          left hand tensor

    rhs : wrapped_tensor or tvm.te.tensor
          right hand tensor

    Returns
    -----
    wrapped_tensor : lhs - floor(lhs/rhs) * rhs
    """
    warnings.warn("te.lang.cce.vmod is deprecated, please replace it with tbe.dsl.vmod",
                DeprecationWarning, stacklevel=STACKLEVEL_FOR_DSL_NO_AUTOCAST)
    import tbe.dsl
    return tbe.dsl.vmod(lhs, rhs)


@auto_cast_of_elewise
def vmax(lhs, rhs):
    """
    calculate elewise compare, return the min one
    Parameters
    ----------
    lhs : wrapped_tensor or tvm.te.tensor
        left hand tensor
    rhs : wrapped_tensor or tvm.te.tensor
        left hand tensor
    Returns
    -------
    wrapped_tensor : max(lhs , rhs)
    """
    warnings.warn("te.lang.cce.vmax is deprecated, please replace it with tbe.dsl.vmax",
                DeprecationWarning, stacklevel=STACKLEVEL_FOR_DSL_AUTOCAST)
    import tbe.dsl
    return tbe.dsl.vmax(lhs, rhs)


@auto_cast_of_elewise
def vmin(lhs, rhs):
    """
    calculate elewise compare, return the min one
    Parameters
    ----------
    lhs : wrapped_tensor or tvm.te.tensor
        left hand tensor
    rhs : wrapped_tensor or tvm.te.tensor
        left hand tensor
    Returns
    -------
    wrapped_tensor : min(lhs , rhs)
    """
    warnings.warn("te.lang.cce.vmin is deprecated, please replace it with tbe.dsl.vmin",
                DeprecationWarning, stacklevel=STACKLEVEL_FOR_DSL_AUTOCAST)
    import tbe.dsl
    return tbe.dsl.vmin(lhs, rhs)


@auto_cast_of_elewise
def vlog(raw_tensor, priority_flag=0):
    """
    calculate ln(raw_tensor)

    Parameters
    ----------
    raw_tensor : wrapped_tensor or tvm.te.tensor
    priority_flag : priority flag, only support 1(precision) and 0(performance)

    Returns
    -------
    wrapped_tensor : log(raw_tensor)
    """
    warnings.warn("te.lang.cce.vlog is deprecated, please replace it with tbe.dsl.vlog",
                DeprecationWarning, stacklevel=STACKLEVEL_FOR_DSL_AUTOCAST)
    impl_mode = "high_performance"
    from .te_compute.util import _get_priority_flag_value
    if _get_priority_flag_value(priority_flag) == 1.0:
        impl_mode = "high_precision"
    import tbe.dsl
    return tbe.dsl.vlog(raw_tensor, impl_mode)


@auto_cast_of_elewise
def vexp(raw_tensor):
    """
    calculate exp(raw_tensor)

    Parameters
    ----------
    raw_tensor : wrapped_tensor or tvm.te.tensor

    Returns
    -------
    wrapped_tensor : exp(raw_tensor)
    """
    warnings.warn("te.lang.cce.vexp is deprecated, please replace it with tbe.dsl.vexp",
                DeprecationWarning, stacklevel=STACKLEVEL_FOR_DSL_AUTOCAST)
    import tbe.dsl
    return tbe.dsl.vexp(raw_tensor)


@auto_cast_of_elewise
def vabs(raw_tensor):
    """
    calculate abs(raw_tensor)

    Parameters
    ----------
    raw_tensor : wrapped_tensor or tvm.te.tensor

    Returns
    -------
    wrapped_tensor : abs(raw_tensor)
    """
    warnings.warn("te.lang.cce.vabs is deprecated, please replace it with tbe.dsl.vabs",
                DeprecationWarning, stacklevel=STACKLEVEL_FOR_DSL_AUTOCAST)
    import tbe.dsl
    return tbe.dsl.vabs(raw_tensor)


@auto_cast_of_elewise
def vsqrt(raw_tensor, priority_flag=0):
    """
    calculate vsqrt(raw_tensor)

    Parameters
    ----------
    raw_tensor : wrapped_tensor or tvm.te.tensor

    priority_flag: priority flag, only support 1(precision), 0(performance)

    Returns
    -------
    wrapped_tensor : vsqrt(raw_tensor)
    """
    warnings.warn("te.lang.cce.vsqrt is deprecated, please replace it with tbe.dsl.vsqrt",
                DeprecationWarning, stacklevel=STACKLEVEL_FOR_DSL_AUTOCAST)
    impl_mode = "high_performance"
    from .te_compute.util import _get_priority_flag_value
    if _get_priority_flag_value(priority_flag) == 1.0:
        impl_mode = "high_precision"
    import tbe.dsl
    return tbe.dsl.vsqrt(raw_tensor, impl_mode)


@auto_cast_of_elewise
def vrsqrt(raw_tensor, priority_flag=0):
    """
    calculate vrsqrt(raw_tensor)

    Parameters
    ----------
    raw_tensor : wrapped_tensor or tvm.te.tensor

    Returns
    -------
    wrapped_tensor : vrsqrt(raw_tensor)
    """
    warnings.warn("te.lang.cce.vrsqrt is deprecated, please replace it with tbe.dsl.vrsqrt",
                DeprecationWarning, stacklevel=STACKLEVEL_FOR_DSL_AUTOCAST)
    impl_mode = "high_performance"
    from .te_compute.util import _get_priority_flag_value
    if _get_priority_flag_value(priority_flag) == 1.0:
        impl_mode = "high_precision"
    import tbe.dsl
    return tbe.dsl.vrsqrt(raw_tensor, impl_mode)


@auto_cast_of_elewise
def vnot(raw_tensor):
    """
    calculate vnot(raw_tensor)

    Parameters
    ----------
    raw_tensor : wrapped_tensor or tvm.te.tensor

    Returns
    -------
    wrapped_tensor : vnot(raw_tensor)
    """
    warnings.warn("te.lang.cce.vnot is deprecated, please replace it with tbe.dsl.vnot",
                DeprecationWarning, stacklevel=STACKLEVEL_FOR_DSL_AUTOCAST)
    import tbe.dsl
    return tbe.dsl.vnot(raw_tensor)


@auto_cast_of_elewise
def vor(lhs, rhs):
    """
    calculate bitwise or op, return the or value
    Parameters
    ----------
    lhs : wrapped_tensor or tvm.te.tensor
        left hand tensor
    rhs : wrapped_tensor or tvm.te.tensor
        left hand tensor
    Returns
    -------
    wrapped_tensor : or(lhs , rhs)
    """
    warnings.warn("te.lang.cce.vor is deprecated, please replace it with tbe.dsl.vor",
                DeprecationWarning, stacklevel=STACKLEVEL_FOR_DSL_AUTOCAST)
    import tbe.dsl
    return tbe.dsl.vor(lhs, rhs)


@auto_cast_of_elewise
def vand(lhs, rhs):
    """
    calculate bitwise and op, return the and value
    Parameters
    ----------
    lhs : wrapped_tensor or tvm.te.tensor
        left hand tensor
    rhs : wrapped_tensor or tvm.te.tensor
        left hand tensor
    Returns
    -------
    wrapped_tensor : max(lhs , rhs)
    """
    warnings.warn("te.lang.cce.vand is deprecated, please replace it with tbe.dsl.vand",
                DeprecationWarning, stacklevel=STACKLEVEL_FOR_DSL_AUTOCAST)
    import tbe.dsl
    return tbe.dsl.vand(lhs, rhs)


def vlogic(lhs, rhs=None, operation='logic_and'):
    """
    calculate elewise logic operation

    Parameters
    ----------
    lhs : wrapped_tensor or tvm.te.tensor
        left hand tensor

    rhs : wrapped_tensor or tvm.te.tensor
        right hand tensor

    operation : operator type, logic_and, logic_or, logic_not

    Returns
    -------
    wrapped_tensor
    """
    warnings.warn("te.lang.cce.vlogic is deprecated, please replace it with tbe.dsl.vlogic",
                DeprecationWarning, stacklevel=STACKLEVEL_FOR_DSL_NO_AUTOCAST)
    import tbe.dsl
    return tbe.dsl.vlogic(lhs, rhs, operation)


@auto_cast_of_elewise
def vadds(raw_tensor, scalar):
    """
    add a tensor by a scalar, dtype of raw_tensor and scalar must be the same

    Parameters
    ----------
    raw_tensor : wrapped_tensor or tvm.te.tensor

    scalar : float, int, or tvm const

    Returns
    -------
    wrapped_tensor : raw_tensor + scalar
    """
    warnings.warn("te.lang.cce.vadds is deprecated, please replace it with tbe.dsl.vadds",
                DeprecationWarning, stacklevel=STACKLEVEL_FOR_DSL_AUTOCAST)
    import tbe.dsl
    return tbe.dsl.vadds(raw_tensor, scalar)


@auto_cast_of_elewise
def vmuls(raw_tensor, scalar):
    """
    multiply a tensor by a scalar, dtype of raw_tensor
    and scalar must be the same

    Parameters
    ----------
    raw_tensor : wrapped_tensor or tvm.te.tensor

    scalar : float, int, or tvm const

    Returns
    -------
    wrapped_tensor : raw_tensor*scalar
    """
    warnings.warn("te.lang.cce.vmuls is deprecated, please replace it with tbe.dsl.vmuls",
                DeprecationWarning, stacklevel=STACKLEVEL_FOR_DSL_AUTOCAST)
    import tbe.dsl
    return tbe.dsl.vmuls(raw_tensor, scalar)


@auto_cast_of_elewise
def vmaxs(raw_tensor, scalar):
    """
    Calculate elewise compare, return the max one of scalar or tensor's element,
    dtype of raw_tensor and scalar must be the same

    Parameters
    ----------
    raw_tensor : wrapped_tensor or tvm.te.tensor

    scalar : float, int, or tvm const

    Returns
    -------
    wrapped_tensor : max(raw_tensor, scalar)
    """
    warnings.warn("te.lang.cce.vmaxs is deprecated, please replace it with tbe.dsl.vmaxs",
                DeprecationWarning, stacklevel=STACKLEVEL_FOR_DSL_AUTOCAST)
    import tbe.dsl
    return tbe.dsl.vmaxs(raw_tensor, scalar)


@auto_cast_of_elewise
def vmins(raw_tensor, scalar):
    """
    Calculate elewise compare, return the min one of scalar or tensor's element,
     dtype of raw_tensor and scalar must be the same

    Parameters
    ----------
    raw_tensor : wrapped_tensor or tvm.te.tensor

    scalar : float, int, or tvm const

    Returns
    -------
    wrapped_tensor : min(raw_tensor, scalar)
    """
    warnings.warn("te.lang.cce.vmins is deprecated, please replace it with tbe.dsl.vmins",
                DeprecationWarning, stacklevel=STACKLEVEL_FOR_DSL_AUTOCAST)
    import tbe.dsl
    return tbe.dsl.vmins(raw_tensor, scalar)


@auto_cast_of_elewise
def vaxpy(lhs, rhs, scalar):
    """
    calculate elewise scalar*lhs + rhs, return the min one
    Parameters
    ----------
    lhs : wrapped_tensor or tvm.te.tensor
        left hand tensor
    rhs : wrapped_tensor or tvm.te.tensor
        left hand tensor
    Returns
    -------
    wrapped_tensor : max(lhs , rhs)
    """
    warnings.warn("te.lang.cce.vaxpy is deprecated, please replace it with tbe.dsl.vaxpy",
                DeprecationWarning, stacklevel=STACKLEVEL_FOR_DSL_AUTOCAST)
    import tbe.dsl
    return tbe.dsl.vaxpy(lhs, rhs, scalar)


@auto_cast_of_elewise
def vmla(tensor_0, tensor_1, tensor_2):
    """
    calculate x*tensor_1 + tensor_2,  only support float16, float32
    Parameters
    ----------
    x : wrapped_tensor or tvm.te.tensor
    tensor_1 : wrapped_tensor or tvm.te.tensor
    tensor_2 : wrapped_tensor or tvm.te.tensor
    Returns
    -------
    wrapped_tensor : X*tensor_1 + tensor_2
    """
    warnings.warn("te.lang.cce.vmla is deprecated, please replace it with tbe.dsl.vmla",
                DeprecationWarning, stacklevel=STACKLEVEL_FOR_DSL_AUTOCAST)
    import tbe.dsl
    return tbe.dsl.vmla(tensor_0, tensor_1, tensor_2)


@auto_cast_of_elewise
def vmadd(tensor_0, tensor_1, tensor_2):
    """
    calculate tensor_0*tensor_2 + tensor_1,  only support  float16, float32
    Parameters
    ----------
    tensor_0 : wrapped_tensor or tvm.te.tensor
    tensor_1 : wrapped_tensor or tvm.te.tensor
    tensor_2 : wrapped_tensor or tvm.te.tensor
    Returns
    -------
    wrapped_tensor : tensor_0*tensor_2 + tensor_1
    """
    warnings.warn("te.lang.cce.vmadd is deprecated, please replace it with tbe.dsl.vmadd",
                DeprecationWarning, stacklevel=STACKLEVEL_FOR_DSL_AUTOCAST)
    import tbe.dsl
    return tbe.dsl.vmadd(tensor_0, tensor_1, tensor_2)


def vcmp(lhs, rhs, operation='lt', mode='bool'):
    """
    calculate elewise compare

    Parameters
    ----------
    lhs : wrapped_tensor or tvm.te.tensor
        left hand tensor

    rhs : wrapped_tensor or tvm.te.tensor
        right hand tensor

    operation : operator type, eq, ne, lt, gt, ge, le

    mode : bool, the dtype of return value is bool
           bit, the dtype of return value is uint8

    Returns
    -------
    wrapped_tensor
    """
    warnings.warn("te.lang.cce.vcmp is deprecated, please replace it with tbe.dsl.vcmp",
                DeprecationWarning, stacklevel=STACKLEVEL_FOR_DSL_NO_AUTOCAST)
    import tbe.dsl
    return tbe.dsl.vcmp(lhs, rhs, operation, mode)


def vsel(condition, lhs, rhs):
    """
    if condition = ture, the result is lhs,
        select

    Parameters
    ----------
    condition : wrapped_tensor or tvm.te.tensor, the dtype is bool or uint8

    lhs : wrapped_tensor or tvm.te.tensor or scalar

    rhs : wrapped_tensor or tvm.te.tensor or scalar

    Returns
    -------
    wrapped_tensor :
    """
    warnings.warn("te.lang.cce.vsel is deprecated, please replace it with tbe.dsl.vsel",
                DeprecationWarning, stacklevel=STACKLEVEL_FOR_DSL_NO_AUTOCAST)
    import tbe.dsl
    return tbe.dsl.vsel(condition, lhs, rhs)


def vcmpsel(lhs, rhs=None, operation='lt', slhs=None, srhs=None):
    """
    calculate elewise compare

    Parameters
    ----------
    lhs : wrapped_tensor or tvm.te.tensor
        compare left hand tensor
    rhs : wrapped_tensor or tvm.te.tensor or scalar
        compare right hand tensor or scalar
    operation : operator type, eq, ne, lt, gt, ge, le
    slhs : wrapped_tensor or tvm.te.tensor or scalar
        select left hand tensor or scalar
    srhs : wrapped_tensor or tvm.te.tensor or scalar
        select right hand tensor or scalar

    Returns
    -------
    wrapped_tensor
    """
    warnings.warn("te.lang.cce.vcmpsel is deprecated, please replace it with tbe.dsl.vcmpsel",
                DeprecationWarning, stacklevel=STACKLEVEL_FOR_DSL_NO_AUTOCAST)
    import tbe.dsl
    return tbe.dsl.vcmpsel(lhs, rhs, operation, slhs, srhs)


@auto_cast_of_elewise
def vmaddrelu(tensor_0, tensor_1, tensor_2):
    """
    calculate relu(tensor_0*tensor_2 + tensor_1), only support  float16, float32
    Parameters
    ----------
    tensor_0 : wrapped_tensor or tvm.te.tensor
    tensor_1 : wrapped_tensor or tvm.te.tensor
    tensor_2 : wrapped_tensor or tvm.te.tensor
    Returns
    -------
    wrapped_tensor : relu(tensor_0*tensor_2 + tensor_1)
    """
    warnings.warn("te.lang.cce.vmaddrelu is deprecated, please replace it with tbe.dsl.vmaddrelu",
                DeprecationWarning, stacklevel=STACKLEVEL_FOR_DSL_AUTOCAST)
    import tbe.dsl
    return tbe.dsl.vmaddrelu(tensor_0, tensor_1, tensor_2)


def vaddrelu(lhs, rhs):
    """
    calculate relu(lhs + rhs)

    Parameters
    ----------
    lhs : wrapped_tensor or tvm.te.tensor
        left hand tensor

    rhs : wrapped_tensor or tvm.te.tensor
        left hand tensor

    Returns
    -------
    wrapped_tensor : relu (lhs + rhs)
    """
    warnings.warn("te.lang.cce.vaddrelu is deprecated, please replace it with tbe.dsl.vaddrelu",
                DeprecationWarning, stacklevel=STACKLEVEL_FOR_DSL_NO_AUTOCAST)
    import tbe.dsl
    return tbe.dsl.vaddrelu(lhs, rhs)


def vsubrelu(lhs, rhs):
    """
    calculate relu(lhs - rhs)

    Parameters
    ----------
    lhs : wrapped_tensor or tvm.te.tensor
        left hand tensor

    rhs : wrapped_tensor or tvm.te.tensor
        left hand tensor

    Returns
    -------
    wrapped_tensor : relu (lhs - rhs)
    """
    warnings.warn("te.lang.cce.vsubrelu is deprecated, please replace it with tbe.dsl.vsubrelu",
                DeprecationWarning, stacklevel=STACKLEVEL_FOR_DSL_NO_AUTOCAST)
    import tbe.dsl
    return tbe.dsl.vsubrelu(lhs, rhs)


@auto_cast_of_elewise
def vrelu(raw_tensor):
    """
    calculate vrelu(raw_tensor)

    Parameters
    ----------
    raw_tensor : wrapped_tensor or tvm.te.tensor

    Returns
    -------
    wrapped_tensor : vrelu(raw_tensor)
    """
    warnings.warn("te.lang.cce.vrelu is deprecated, please replace it with tbe.dsl.vrelu",
                DeprecationWarning, stacklevel=STACKLEVEL_FOR_DSL_AUTOCAST)
    import tbe.dsl
    return tbe.dsl.vrelu(raw_tensor)


def vlrelu(raw_tensor, alpha=0):
    """
    calculate leaky_relu

    Parameters
    ----------
    raw_tensor : wrapped_tensor or tvm.te.tensor

    Returns
    -------
    wrapped_tensor : vlrelu(raw_tensor)
    """
    warnings.warn("te.lang.cce.vlrelu is deprecated, please replace it with tbe.dsl.vlrelu",
                DeprecationWarning, stacklevel=STACKLEVEL_FOR_DSL_NO_AUTOCAST)

    dtype = raw_tensor.dtype
    from te.platform import intrinsic_check_support
    import tbe.dsl
    is_current_chip_support = intrinsic_check_support("Intrinsic_vlrelu")
    if not is_current_chip_support:
        if dtype == "int32":
            raw_tensor = cast_to(raw_tensor, "float32")
            res = tbe.dsl.vlrelu(raw_tensor, alpha)
            return cast_to(res, "int32")

    return tbe.dsl.vlrelu(raw_tensor, alpha)


def round_to(data, max_value, min_value):
    """
    round data to [min_value,max_value]

    Parameters
    ----------
    data : tvm.te.tensor
        tensors need to change dtype

    max_value/min_value : float
        the range of res

    Returns
    -------
    tensor : tvm.te.tensor ,elements in tensor is in range [min_value,max_value]
    """
    warnings.warn("te.lang.cce.round_to is deprecated, please replace it with tbe.dsl.round_to",
                DeprecationWarning, stacklevel=STACKLEVEL_FOR_DSL_NO_AUTOCAST)
    import tbe.dsl
    return tbe.dsl.clip(data, max_value, min_value)


def broadcast(var, shape, output_dtype=None):
    """
    broadcast scalar to tensor, only support float16

    Parameters
    ----------
    var : can be python instance of int and float, or tvm.const

    shape : tensor shape

    output_dtype : tensor dtype , default : var.dtype

    Returns
    -------
    wrapped_tensor : broadcast tensor
    """
    warnings.warn("te.lang.cce.broadcast is deprecated, please replace it with tbe.dsl.broadcast",
                DeprecationWarning, stacklevel=STACKLEVEL_FOR_DSL_NO_AUTOCAST)
    import tbe.dsl
    return tbe.dsl.broadcast(var, shape, output_dtype)


@auto_cast_of_reduce
def sum(raw_tensor, axis, keepdims=False):
    """
    calculate reduce_sum of raw_tensor, only support float16
    Parameters
    ----------
    raw_tensor : wrapped_tensor or tvm.te.tensor
    axis : int or list
        reduce axis (range : [-len(raw_tensor.shape), len(raw_tensor.shape) - 1])
    keepdims : if true, retains reduced dimensions with length 1, default value is None
    Returns
    -------
    res : wrapped_tensor
    """
    warnings.warn("te.lang.cce.sum is deprecated, please replace it with tbe.dsl.sum",
                DeprecationWarning, stacklevel=STACKLEVEL_FOR_DSL_AUTOCAST)
    import tbe.dsl
    return tbe.dsl.reduce_sum(raw_tensor, axis, keepdims)


@auto_cast_of_reduce
def reduce_min(raw_tensor, axis, keepdims=False, priority_flag=False):
    """
    calculate reduce_min of raw_tensor, only support float16
    Parameters
    ----------
    raw_tensor : wrapped_tensor or tvm.te.tensor
    axis : int or list
        reduce axis (range : [-len(raw_tensor.shape), len(raw_tensor.shape) - 1])
    keepdims : if true, retains reduced dimensions with length 1, default value is None
    Returns
    -------
    res : wrapped_tensor
    """
    warnings.warn("te.lang.cce.reduce_min is deprecated, please replace it with tbe.dsl.reduce_min",
                DeprecationWarning, stacklevel=STACKLEVEL_FOR_DSL_AUTOCAST)
    impl_mode = "high_performance"
    if priority_flag:
        impl_mode = "high_precision"
    import tbe.dsl
    return tbe.dsl.reduce_min(raw_tensor, axis, keepdims, impl_mode)


@auto_cast_of_reduce
def reduce_max(raw_tensor, axis, keepdims=False, priority_flag=False):
    """
    calculate reduce_max of raw_tensor, only support float16
    Parameters
    ----------
    raw_tensor : wrapped_tensor or tvm.te.tensor
    keepdims : if true, retains reduced dimensions with length 1, default value is None
    axis : int or list
        reduce axis (range : [-len(raw_tensor.shape), len(raw_tensor.shape) - 1])
    priority_flag : supported 1(precision) and 0(performance)
    Returns
    -------
    res : wrapped_tensor
    """
    warnings.warn("te.lang.cce.reduce_max is deprecated, please replace it with tbe.dsl.reduce_max",
                DeprecationWarning, stacklevel=STACKLEVEL_FOR_DSL_AUTOCAST)
    impl_mode = "high_performance"
    if priority_flag:
        impl_mode = "high_precision"
    import tbe.dsl
    return tbe.dsl.reduce_max(raw_tensor, axis, keepdims, impl_mode)


@auto_cast_of_reduce
def reduce_prod(raw_tensor, axis, keepdims=False):
    """
    calculate reduce_prod of raw_tensor, only support float16
    Parameters
    ----------
    raw_tensor : wrapped_tensor or tvm.te.tensor
    axis : int
        reduce axis (range : [-len(raw_tensor.shape), len(raw_tensor.shape) - 1])
    Returns
    -------
    res : wrapped_tensor
    """
    warnings.warn(
        "te.lang.cce.reduce_prod is deprecated, please replace it with tbe.dsl.reduce_prod",
         DeprecationWarning, stacklevel=STACKLEVEL_FOR_DSL_AUTOCAST)
    import tbe.dsl
    return tbe.dsl.reduce_prod(raw_tensor, axis, keepdims)


def split(data, split_dim, size_splits):
    """Split a tensor into len(size_splits) tensors along one dimension.

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
    warnings.warn("te.lang.cce.split is deprecated, please replace it with tbe.dsl.split",
                DeprecationWarning, stacklevel=STACKLEVEL_FOR_DSL_NO_AUTOCAST)
    import tbe.dsl
    return tbe.dsl.split(data, split_dim, size_splits)


def split_compute_com(data, split_dim, size_splits):
    """
    Split a tensor into len(size_splits) tensors along one dimension
    """
    warnings.warn("split_compute_com is deprecated, please replace it with the func split",
                  DeprecationWarning, stacklevel=STACKLEVEL_FOR_DSL_NO_AUTOCAST)
    import tbe.dsl
    return tbe.dsl.split(data, split_dim, size_splits)


def split_schedule_com(data, split_dim, shape_list, tensor_list):
    """Create split schedule.

    Parameters
    ----------
    data: TVM tensor
        input tensor.
    split_dim: int
        the dimension along which to split.
    shape_list: list
        the list of output shapes.
    tensor_list: list
        the list of output tensors, tensor type is TVM tensor.

    Returns
    -------
    sch: Schedule
        The created schedule.
    build_list: list
        the list of input and output tensors, tensor type is TVM tensor.
    """
    warnings.warn("te.lang.cce.split_schedule_com is deprecated",
                DeprecationWarning, stacklevel=STACKLEVEL_FOR_DSL_NO_AUTOCAST)
    from tbe.dsl.static_schedule.split_schedule import split_schedule_com
    return split_schedule_com(data, split_dim, shape_list, tensor_list)


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
    warnings.warn("te.lang.cce.concat is deprecated, please replace it with tbe.dsl.concat",
                DeprecationWarning, stacklevel=STACKLEVEL_FOR_DSL_NO_AUTOCAST)
    import tbe.dsl
    return tbe.dsl.concat(raw_tensors, axis)


def inplace_add(lhs, inplace_ids, rhs):
    """
    calculate inplace add: computes lhs[inplace_ids, :] += rhs; return lhs.

    Parameters
    ----------
    lhs : wrapped_tensor or tvm.te.tensor
        left hand tensor

    inplace_ids : a vector. Indices into the left-most dimension of lhs.

    rhs : wrapped_tensor or tvm.te.tensor
        left hand tensor

    Returns
    -------
    wrapped_tensor : computes lhs[inplace_ids, :] += rhs; return lhs.
    """
    warnings.warn(
        "te.lang.cce.inplace_add is deprecated, please replace it with tbe.dsl.inplace_add",
         DeprecationWarning, stacklevel=STACKLEVEL_FOR_DSL_NO_AUTOCAST)
    import tbe.dsl
    return tbe.dsl.inplace_add(lhs, inplace_ids, rhs)


def inplace_sub(lhs, inplace_ids, rhs):
    """
    calculate inplace sub: computes lhs[inplace_ids, :] -= rhs; return lhs.

    Parameters
    ----------
    lhs : wrapped_tensor or tvm.te.tensor
        left hand tensor

    inplace_ids : a vector. Indices into the left-most dimension of lhs.

    rhs : wrapped_tensor or tvm.te.tensor
        left hand tensor

    Returns
    -------
    wrapped_tensor : computes lhs[inplace_ids, :] -= rhs; return lhs.
    """
    warnings.warn(
        "te.lang.cce.inplace_sub is deprecated, please replace it with tbe.dsl.inplace_sub",
         DeprecationWarning, stacklevel=STACKLEVEL_FOR_DSL_NO_AUTOCAST)
    import tbe.dsl
    return tbe.dsl.inplace_sub(lhs, inplace_ids, rhs)


def inplace_update(lhs, inplace_ids, rhs):
    """
    calculate inplace add: computes lhs[inplace_ids, :] = rhs; return lhs.

    Parameters
    ----------
    lhs : wrapped_tensor or tvm.te.tensor
        left hand tensor

    inplace_ids : a vector. Indices into the left-most dimension of lhs.

    rhs : wrapped_tensor or tvm.te.tensor
        left hand tensor

    Returns
    -------
    wrapped_tensor : computes lhs[inplace_ids, :] = rhs; return lhs.
    """
    warnings.warn(
        "te.lang.cce.inplace_update is deprecated, please replace it with tbe.dsl.inplace_update",
         DeprecationWarning, stacklevel=STACKLEVEL_FOR_DSL_NO_AUTOCAST)
    import tbe.dsl
    return tbe.dsl.inplace_update(lhs, inplace_ids, rhs)


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
    warnings.warn("te.lang.cce.pooling2d is deprecated, please replace it with tbe.dsl.pooling2d",
                DeprecationWarning, stacklevel=STACKLEVEL_FOR_DSL_NO_AUTOCAST)
    import tbe.dsl
    return tbe.dsl.pooling2d(tensor_in, window, stride, pooling_mode,
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
    warnings.warn("te.lang.cce.pooling3d is deprecated, please replace it with tbe.dsl.pooling3d",
                DeprecationWarning, stacklevel=STACKLEVEL_FOR_DSL_NO_AUTOCAST)
    import tbe.dsl
    return tbe.dsl.pooling3d(tensor_in, window, stride, padding_mode,
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
    warnings.warn(
        "te.lang.cce.max_pooling3d_grad_grad is deprecated, please replace it with tbe.dsl.max_pooling3d_grad_grad",
         DeprecationWarning, stacklevel=STACKLEVEL_FOR_DSL_NO_AUTOCAST)
    import tbe.dsl
    return tbe.dsl.max_pooling3d_grad_grad(orig_input,
                                          orig_output,
                                          grad_grad,
                                          assist_tensor,
                                          ksize,
                                          strides,
                                          pads,
                                          data_format,
                                          padding)


def pooling3d_max_grad_grad(orig_input, orig_output, grad_grad, assist_tensor,
                            ksize, strides, pads=(0, 0, 0, 0, 0, 0),
                            data_format="NDHWC",
                            padding="SAME"):
    """
    :params:
    :orig_input : dict, shape and dtype of input_data,
                 shape is 6 dims, format is NDC1HWC0
    :orig_output : dict, result of max_pool3d(orig_input, ksize, ...)
    :grad_grad: dict, input grad of grad
    :assist_tensor: dict, helper matrix, it's content is 8,7,6,5,4,3,2,1
                if kernel is 2 x 2 x 2
    :ksize : list or tuple, the window of max_pool3d,
            only support max_pool3d in D or H or W
    :strides : list or tuple, the stride of max_pool3d window,
              only support max_pool3d in D or H or W
    :pads : reserved.
    :padding : str, the mode of padding, support SAME or VALID
    :return: pooling result
    """
    warnings.warn(
        "pooling3d_max_grad_grad is deprecated, please replace it with max_pooling3d_grad_grad",
        DeprecationWarning, stacklevel=STACKLEVEL_FOR_DSL_NO_AUTOCAST)
    import tbe.dsl
    return tbe.dsl.max_pooling3d_grad_grad(orig_input,
                                          orig_output,
                                          grad_grad,
                                          assist_tensor,
                                          ksize,
                                          strides,
                                          pads,
                                          data_format,
                                          padding)


def auto_schedule(outs, option=None):
    """Entry of auto-Schedule.

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
    warnings.warn(
        "te.lang.cce.auto_schedule is deprecated, please replace it with tbe.dsl.auto_schedule",
         DeprecationWarning, stacklevel=STACKLEVEL_FOR_DSL_NO_AUTOCAST)
    import tbe.dsl
    return tbe.dsl.auto_schedule(outs, option)


def cce_build_code(sch, config_map=None):
    """
    API of building or printing lower code, just can be used when device is CCE

    Parameters
    ----------
    sch : tvm.schedule
        schedule to build or to print lower code

    config_map : dict, default is {} and use default configration

        key_words:

            print_ir : if need print lower IR code, default is True

            need_build : if need build, default is True

            name : kernel name, default is cce_op

    Returns
    -------
    None
    """
    warnings.warn("te.lang.cce.cce_build_code is deprecated, please replace it with tbe.dsl.build",
                DeprecationWarning, stacklevel=STACKLEVEL_FOR_DSL_NO_AUTOCAST)
    import tbe.dsl
    return tbe.dsl.build(sch, config_map)


def tuple_sum(input_tensor_list, axis, keepdims=False):
    """
    calculate sum of raw_tensor, only support float16
    Parameters
    ----------
    input_tensor_list : wrapped_tensor or tvm.te.tensor list that each tensor has same reduce operation
    axis : int or list
        reduce axis (range : [-len(raw_tensor.shape), len(raw_tensor.shape) - 1])
    keepdims : if true, retains reduced dimensions with length 1, default value is None
    Returns
    -------
    res : wrapped_tensor
    """
    from .te_compute.reduction_compute import tuple_sum
    return tuple_sum(input_tensor_list, axis, keepdims)


def unsorted_segment_sum(tensor, segment_ids, num_segments, init_value=0):
    """
    calculate segment sum, return a new tensor which is the sum along segments of a tensor.
    only support float16, int32

    Parameters
    ----------
    tensor : tvm.te.tensor
        input tensor

    segment_ids : list
        index of each segment

    Returns
    -------
    tensor : segment_sum(tensor , segment_ids)
    """
    from .te_compute.segment_compute import unsorted_segment_sum
    return unsorted_segment_sum(tensor, segment_ids, num_segments, init_value)


def unsorted_segment_mean(tensor, segment_ids, num_segments, init_value=0):
    """
    calculate segment mean, return a new tensor which is the mean along segments of a tensor.
    only support float16, int32.

    Parameters
    ----------
    tensor : tvm.te.tensor
        input tensor

    segment_ids : list
        index of each segment

    Returns
    -------
    tensor : segment_mean(tensor , segment_ids)
    """
    from .te_compute.segment_compute import unsorted_segment_mean
    return unsorted_segment_mean(tensor, segment_ids, num_segments, init_value)


def unsorted_segment_prod(tensor, segment_ids, num_segments, init_value=0):
    """
    calculate segment prod, return a new tensor which is the prod along segments of a tensor.
    only support float16, int32.

    Parameters
    ----------
    tensor : tvm.te.tensor
        input tensor

    segment_ids : list
        index of each segment

    Returns
    -------
    tensor : segment_prod(tensor , segment_ids)
    """
    from .te_compute.segment_compute import unsorted_segment_prod
    return unsorted_segment_prod(tensor, segment_ids, num_segments, init_value)


def unsorted_segment_min(tensor, segment_ids, num_segments, init_value=0):
    """
    calculate segment min, return a new tensor which is the min along segments of a tensor.
    only support float16, int32

    Parameters
    ----------
    tensor : tvm.te.tensor
        input tensor

    segment_ids : list
        index of each segment

    Returns
    -------
    tensor : segment_min(tensor , segment_ids)
    """
    from .te_compute.segment_compute import unsorted_segment_min
    return unsorted_segment_min(tensor, segment_ids, num_segments, init_value)


def unsorted_segment_max(tensor, segment_ids, num_segments, init_value=0):
    """
    calculate segment max, return a new tensor which is the max along segments of a tensor.
     only support float16, int32

    Parameters
    ----------
    tensor : tvm.te.tensor
        input tensor

    segment_ids : list
        index of each segment

    Returns
    -------
    tensor : segment_max(tensor , segment_ids)
    """
    from .te_compute.segment_compute import unsorted_segment_max
    return unsorted_segment_max(tensor, segment_ids, num_segments, init_value)
