#!/usr/bin/env python
# -*- coding:utf-8 -*-
"""
Copyright (C) 2021. Huawei Technologies Co., Ltd. All rights reserved.
FILE:     common_params_check_.py
DESC:     params check for tik 1.5
CREATED:  2021-06-24 18:53:42
MODIFIED: 2021-06-24 19:17:00
"""
from collections import namedtuple

from tbe.common.platform import scope_ubuf
from tbe.common.platform import intrinsic_check_support
from tbe.common.utils.log import info
from tbe.tik.tik_lib.tik_check_util import TikCheckUtil
from tbe.tik.tik_lib.tik_expr import Expr
from tbe.tik.common.common_util import is_tensor
from tbe.tik.api.tik_scalar import Scalar
from tbe.tik.common.util import TikUtil
from tbe.tik.tik_lib.tik_params import gen_api_check_statement
from tbe.tik.common.common_util import tik_api_check_support
from tbe.tik.common.common_util import DTYPE_SIZE
from tbe.tik.tik_lib.tik_soc_manager import TikSocManager

CMP_MODE = ["LT", "LE", "GT", "GE", "EQ", "NE"]


def _gen_acc_steps(strides, original_shape):
    """
    get dimension from strides and original_shape
    """
    new_steps = []
    strides_acc = 1
    for stride_i, original_shape_i in zip(strides, original_shape):
        step_i = stride_i // strides_acc
        new_steps.append(Expr(step_i).eval_value())
        strides_acc *= original_shape_i
    return new_steps[::-1]


def _check_tensor_scope(tensor_list, name_list):
    """
    check tensor scope is UB
    """
    for tensor, name in zip(tensor_list, name_list):
        TikCheckUtil.check_equality(is_tensor(tensor), True, "%s should be tensor" % name)
        TikCheckUtil.check_equality(tensor.scope, scope_ubuf, "%s's scope must be UB" % name)


def _check_tensor_dtype(tensor_list, name_list):
    """
    check tensor dtype is same
    """
    dst = tensor_list[0]
    for src, name in zip(tensor_list[1:], name_list[1:]):
        TikCheckUtil.check_equality(dst.dtype, src.dtype,
                                    "%s and %s tensor should be the same dtype" % (name, name_list[0]))


def _check_tensor_step(tensor_list, name_list, context=None):
    """
    check tensor step be same
    """
    dst = tensor_list[0]
    if context is None:
        dst_strides = dst.strides
        dst_original_shape = dst.original_shape
    else:
        dst_strides = context.get_tensor_strides(dst)
        dst_original_shape = context.get_tensor_original_shape(dst)
    dst_step = _gen_acc_steps(dst_strides[::-1], dst_original_shape[::-1])
    if any(not isinstance(s, int) for s in dst_step):
        return
    for src, name in zip(tensor_list[1:], name_list[1:]):
        if context is None:
            src_strides = src.strides
            src_original_shape = src.original_shape
        else:
            src_strides = context.get_tensor_strides(src)
            src_original_shape = context.get_tensor_original_shape(src)
        src_step = _gen_acc_steps(src_strides[::-1], src_original_shape[::-1])
        if any(not isinstance(s, int) for s in src_step):
            return
        TikCheckUtil.check_equality(dst_step, src_step,
                                    "%s and %s tensor should be the same step" % (name, name_list[0]))


def _check_tensor_shape(tensor_list, name_list, context=None):
    """
    check tensor shape be same
    """
    dst = tensor_list[0]
    shape_length_range = [1, 8]
    TikCheckUtil.check_in_range_by_dtype(
        len(dst.shape), msg="length of %s tensor shape should be in the range of [%d, %d]"
        % (name_list[0], shape_length_range[0], shape_length_range[1]), var_range=shape_length_range)
    if context is None:
        dst_shape = dst.shape
    else:
        dst_shape = context.get_tensor_shape(dst)
    if any(not isinstance(s, int) for s in dst_shape):
        return
    for src, name in zip(tensor_list[1:], name_list[1:]):
        if context is None:
            src_shape = src.shape
        else:
            src_shape = context.get_tensor_shape(src)
        if any(not isinstance(s, int) for s in src_shape):
            continue
        TikCheckUtil.check_equality(dst_shape, src_shape,
                                    "%s and %s tensor should be the same shape" % (name, name_list[0]))


def _judge_overlap(tensor_judge_params):
    """
    judge src and dst tensor overlap
    """
    tensor_params = namedtuple(
        "TensorParams", "dst_data dst_strides dst_shape src_data src_strides src_shape dst_name src_name")
    tensor_params_ins = tensor_params(*tensor_judge_params)
    dst_end = tensor_params_ins.dst_data
    for shape, stride in zip(tensor_params_ins.dst_shape, tensor_params_ins.dst_strides):
        dst_end += (shape - 1) * stride
    src_end = tensor_params_ins.src_data
    for shape, stride in zip(tensor_params_ins.src_shape, tensor_params_ins.src_strides):
        src_end += (shape - 1) * stride
    if ((tensor_params_ins.dst_data <= tensor_params_ins.src_data <= dst_end) or
        (tensor_params_ins.src_data <= tensor_params_ins.dst_data <= src_end)) and \
            not (tensor_params_ins.src_data == tensor_params_ins.dst_data and src_end == dst_end):
        TikCheckUtil.raise_error("%s and %s tensor overlap error" %
                                 (tensor_params_ins.src_name, tensor_params_ins.dst_name))


def _check_tensor_overlap(tensor_list, name_list, context=None):
    """
    check tensor overlap
    :param tensor_list: Tensor list
    :param name_list: Tensor name list
    :param context: Debug context
    :return: None
    """
    dst = tensor_list[0]
    if context is None:
        dst_data = dst.data
        dst_strides = dst.strides
        dst_shape = dst.shape
    else:
        dst_data = context.get_tensor_offset(dst)
        dst_strides = context.get_tensor_strides(dst)
        dst_shape = context.get_tensor_shape(dst)
    if any(not isinstance(s, int) for s in [dst_data] + dst_strides + dst_shape):
        return
    for src, name in zip(tensor_list[1:], name_list[1:]):
        if src.buffer is dst.buffer:
            if context is None:
                src_data = src.data
                src_strides = src.strides
                src_shape = src.shape
            else:
                src_data = context.get_tensor_offset(src)
                src_strides = context.get_tensor_strides(src)
                src_shape = context.get_tensor_shape(src)
            if any(not isinstance(s, int) for s in [src_data] + src_strides + src_shape):
                continue
            _judge_overlap((dst_data, dst_strides, dst_shape, src_data, src_strides, src_shape, name_list[0], name))


def _check_tensor(tensor_list, name_list, context=None):
    """
    check tensor input
    """
    if len(tensor_list) == 0:
        return
    # check scope
    _check_tensor_scope(tensor_list, name_list)
    # check dtype
    _check_tensor_dtype(tensor_list, name_list)
    # check step
    _check_tensor_step(tensor_list, name_list, context=context)
    # check shape
    _check_tensor_shape(tensor_list, name_list, context=context)
    # check overlap
    _check_tensor_overlap(tensor_list, name_list, context=context)


def _check_scalar(scalar_list, name_list, dtype):
    """
    check scalar input
    """
    if len(scalar_list) == 0:
        return
    for scalar, name in zip(scalar_list, name_list):
        if isinstance(scalar, (Scalar, Expr)):
            TikCheckUtil.check_equality(scalar.dtype, dtype, "%s should be %s" % (name, dtype))
        elif isinstance(scalar, (int, float)):
            if dtype.startswith(("int", "uint")):
                TikCheckUtil.check_type_match(scalar, int, "%s should be int" % name)
            else:
                TikCheckUtil.check_type_match(scalar, float, "%s should be float" % name)
            TikCheckUtil.check_in_range_by_dtype(scalar, dtype, "%s out of range of %s" % (name, dtype))
        else:
            TikCheckUtil.raise_error("%s should be int, float, scalar or expr" % name)


def check_single_ops_params(check_params, context=None):
    """
    check params for single ops instruction
    :param check_params: contains dst src api_name params
    dst->Tensor or Tensor slice
    src->Tensor or Tensor slice
    api_name->api name
    :param context: Debug context
    :return: None
    """
    _check_tensor([check_params.dst, check_params.src], ["dst", "src"], context=context)
    if check_params.api_name in ("h_sin", "h_cos"):
        dtype_list = ["float16", "float32"]
        if TikSocManager.is_hisi_soc():
            dtype_list = ["float16", ]
        TikCheckUtil.check_var_in_list(
            check_params.src.dtype, dtype_list, gen_api_check_statement(check_params.src.dtype, check_params.api_name))
        info(gen_api_check_statement(check_params.src.dtype, check_params.api_name))
        return
    instr_name = check_params.api_name.replace("h_", "Intrinsic_v")
    TikCheckUtil.check_equality(intrinsic_check_support(instr_name, check_params.dst.dtype), True,
                                gen_api_check_statement(check_params.src.dtype, check_params.api_name))
    if not (TikSocManager.is_v100_soc() or TikSocManager.is_310b_soc()):
        msg = gen_api_check_statement(check_params.src.dtype, check_params.api_name)
        info(msg)


def check_double_ops_params(double_ops_params, context=None):
    """
    check params for double ops instruction
    :param double_ops_params: contains dst src0 src1 api_name params
    dst: Tensor
    src0: Tensor, Tensor slice or Scalar
    src1: Tensor, Tensor slice or Scalar
    api_name: api name
    :param context: Debug context
    :return: None
    """
    if is_tensor(double_ops_params.src0) and is_tensor(double_ops_params.src1):
        _check_tensor([double_ops_params.dst, double_ops_params.src0, double_ops_params.src1],
                      ["dst", "src0", "src1"], context=context)
    elif is_tensor(double_ops_params.src0) and isinstance(double_ops_params.src1, (int, float, Scalar, Expr)):
        _check_tensor([double_ops_params.dst, double_ops_params.src0], ["dst", "src0"], context=context)
        _check_scalar([double_ops_params.src1], ["src1"], double_ops_params.dst.dtype)
    elif isinstance(double_ops_params.src0, (int, float, Scalar, Expr)) and is_tensor(double_ops_params.src1):
        _check_tensor([double_ops_params.dst, double_ops_params.src1], ["dst", "src1"], context=context)
        _check_scalar([double_ops_params.src0], ["src0"], double_ops_params.dst.dtype)
    else:
        TikCheckUtil.raise_error("wrong src data type for Instruction %s" % double_ops_params.api_name)
    instr_name = double_ops_params.api_name.replace("h_", "Intrinsic_v")
    TikCheckUtil.check_equality(intrinsic_check_support(instr_name, double_ops_params.dst.dtype), True,
                                gen_api_check_statement(double_ops_params.dst.dtype, double_ops_params.api_name))
    if not (TikSocManager.is_v100_soc() or TikSocManager.is_310b_soc()):
        msg = gen_api_check_statement(double_ops_params.dst.dtype, double_ops_params.api_name)
        info(msg)

def check_precision_conv_params(dst, src, dtype_str, deqscale=None, context=None):
    """
    check params for precision conv instruction
    :param dst: Tensor
    :param src: Tensor
    :param dtype_str: dtype
    :param deqscale: deqscale
    :param context: Debug context
    :return: None
    """
    _check_tensor_scope([dst, src], ["dst", "src"])
    _check_tensor_step([dst, src], ["dst", "src"], context=context)
    _check_tensor_shape([dst, src], ["dst", "src"], context=context)
    _check_tensor_overlap([dst, src], ["dst", "src"], context=context)
    if dtype_str == "s322f16":
        _check_scalar([deqscale], ["deqscale"], "float16")
    elif dtype_str in ["s162s8", "s162u8"]:
        _check_scalar([deqscale], ["deqscale"], "uint64")
        # check deqscale[46]
        if context is not None:
            deqscale = context.evaluate_expr(deqscale)
        if isinstance(deqscale, int):
            deq_46 = (deqscale >> 46) & 0x1
            if dst.dtype == "int8":
                TikCheckUtil.check_equality(deq_46, 1, "deqscale[46] bit should be 1 "
                                                       "when converting int16 to int8")
            elif dst.dtype == "uint8":
                TikCheckUtil.check_equality(deq_46, 0, "deqscale[46] bit should be 0 "
                                                       "when converting int16 to uint8")


def check_duplicate_params(dup_check_params, context=None):
    """
    check params for duplicate ops instruction
    :param dup_check_params: contains dst src api_name params
    dst: Tensor or Tensor slice
    src: Tensor or Tensor slice
    api_name: api name
    :param context: Debug context
    :return: None
    """
    _check_tensor([dup_check_params.dst], ["dst"], context=context)
    if isinstance(dup_check_params.src, (int, float, Scalar, Expr)):
        _check_scalar([dup_check_params.src], ["src"], dup_check_params.dst.dtype)
        if isinstance(dup_check_params.src, Expr) and dup_check_params.dst.dtype == "float16":
            TikCheckUtil.raise_error(
                "For instruction %s when dtype is float16, src cannot be Expr" % dup_check_params.api_name)
        if isinstance(dup_check_params.src, Expr) and dup_check_params.dst.dtype == "float32" and\
                TikSocManager.is_mini_soc():
            TikCheckUtil.raise_error("For instruction %s when dtype is float32 and "
                                     "chip is Ascend310 AiCore, src cannot be Expr" % dup_check_params.api_name)
    else:
        TikCheckUtil.raise_error("wrong src data type for Instruction %s" % dup_check_params.api_name)
    TikCheckUtil.check_equality(tik_api_check_support("tik.vector_dup", dup_check_params.dst.dtype), True,
                                gen_api_check_statement(dup_check_params.dst.dtype, dup_check_params.api_name))
    if not (TikSocManager.is_v100_soc() or TikSocManager.is_310b_soc()):
        msg = gen_api_check_statement(dup_check_params.dst.dtype, dup_check_params.api_name)
        info(msg)


def _check_tensor_for_cmpv_sel(mask_tensor, mask_tensor_name, tensor_list, name_list, context=None):
    """
    check tensor input of cmpv and sel
    """
    # check for mask tensor
    _check_tensor_scope([mask_tensor], ["mask_tensor"])
    TikCheckUtil.check_equality(mask_tensor.dtype, "bool", "mask_tensor's dtype should be bool")

    # check for others tensor
    _check_tensor_scope(tensor_list, name_list)
    _check_tensor_dtype(tensor_list, name_list)
    _check_tensor_overlap(tensor_list, name_list, context=context)

    # check input tensor's shape and step
    _check_tensor_shape([mask_tensor, *tensor_list], [mask_tensor_name, *name_list], context=context)
    _check_tensor_step([mask_tensor, *tensor_list], [mask_tensor_name, *name_list], context=context)


def check_cmpv_ops_params(cmpv_ops_params, context=None):
    """
    check params for cmpv ops instruction
    :param cmpv_ops_params: contains mask_tensor src0 src1 cmp_mode api_name params
    mask_tensor: mask Tensor
    src0: Tensor or Tensor slice
    src1: Tensor or Tensor slice
    cmp_mode: compare mode
    api_name: api name
    :param context: Debug context
    :return: None
    """
    tensor_dtype = ""
    if is_tensor(cmpv_ops_params.src0) and is_tensor(cmpv_ops_params.src1):
        _check_tensor_for_cmpv_sel(cmpv_ops_params.mask_tensor, "mask_tensor",
                                   [cmpv_ops_params.src0, cmpv_ops_params.src1], ["src0", "src1"], context=context)
        tensor_dtype = cmpv_ops_params.src0.dtype
    elif is_tensor(cmpv_ops_params.src0) and isinstance(cmpv_ops_params.src1, (int, float, Scalar, Expr)):
        _check_tensor_for_cmpv_sel(cmpv_ops_params.mask_tensor,
                                   "mask_tensor", [cmpv_ops_params.src0], ["src0"], context=context)
        _check_scalar([cmpv_ops_params.src1], ["src1"], cmpv_ops_params.src0.dtype)
        tensor_dtype = cmpv_ops_params.src0.dtype
    elif isinstance(cmpv_ops_params.src0, (int, float, Scalar, Expr)) and is_tensor(cmpv_ops_params.src1):
        _check_tensor_for_cmpv_sel(cmpv_ops_params.mask_tensor, "mask_tensor",
                                   [cmpv_ops_params.src1], ["src1"], context=context)
        _check_scalar([cmpv_ops_params.src0], ["src0"], cmpv_ops_params.src1.dtype)
        tensor_dtype = cmpv_ops_params.src1.dtype
    else:
        TikCheckUtil.raise_error("wrong src data type for Instruction %s" % cmpv_ops_params.api_name)

    TikCheckUtil.check_equality(tik_api_check_support("tik.h_cmpv", tensor_dtype), True,
                                gen_api_check_statement(tensor_dtype, cmpv_ops_params.api_name))
    if cmpv_ops_params.cmp_mode not in CMP_MODE:
        TikCheckUtil.raise_error("%s's cmp_mode only be %s, but input %s" %
                                 (cmpv_ops_params.api_name, CMP_MODE, cmpv_ops_params.cmp_mode))
    if not (TikSocManager.is_v100_soc() or TikSocManager.is_310b_soc()):
        msg = gen_api_check_statement(cmpv_ops_params.src0.dtype, cmpv_ops_params.api_name)
        info(msg)


def check_sel_ops_params(sel_check_params, context=None):
    """
    check params for sel ops instruction
    :param sel_check_params: contains dst src0 src1 mask_tensor api_name params
    dst: Tensor
    src0: Tensor
    src1: Tensor
    mask_tensor: mask Tensor
    api_name: api name
    :param context: Debug context
    :return: None
    """
    if is_tensor(sel_check_params.src0) and is_tensor(sel_check_params.src1):
        _check_tensor_for_cmpv_sel(sel_check_params.mask_tensor, "mask_tensor",
                                   [sel_check_params.dst, sel_check_params.src0,
                                    sel_check_params.src1], ["dst", "src0", "src1"], context=context)
    elif is_tensor(sel_check_params.src0) and isinstance(sel_check_params.src1, (int, float, Scalar, Expr)):
        _check_tensor_for_cmpv_sel(sel_check_params.mask_tensor, "mask_tensor",
                                   [sel_check_params.dst, sel_check_params.src0], ["dst", "src0"], context=context)
        _check_scalar([sel_check_params.src1], ["src1"], sel_check_params.src0.dtype)
    elif isinstance(sel_check_params.src0, (int, float, Scalar, Expr)) and\
            is_tensor(sel_check_params.src1):
        _check_tensor_for_cmpv_sel(sel_check_params.mask_tensor,
                                   "mask_tensor",
                                   [sel_check_params.dst, sel_check_params.src1], ["dst", "src1"], context=context)
        _check_scalar([sel_check_params.src0], ["src0"], sel_check_params.src1.dtype)
    else:
        TikCheckUtil.raise_error("wrong src data type for Instruction %s" % sel_check_params.api_name)
    TikCheckUtil.check_equality(tik_api_check_support("tik.h_sel", sel_check_params.dst.dtype), True,
                                gen_api_check_statement(sel_check_params.dst.dtype, sel_check_params.api_name))
    if not (TikSocManager.is_v100_soc() or TikSocManager.is_310b_soc()):
        msg = gen_api_check_statement(sel_check_params.dst.dtype, sel_check_params.api_name)
        info(msg)


def _check_reduce_ops_tensor_shape(src, dst, axis, context=None):
    """
    check the tensor shape according to axis
    """
    _check_tensor_shape([src, ], ["src", ], context=context)
    if isinstance(dst, Scalar):
        return
    _check_tensor_shape([dst, ], ["dst", ], context=context)

    if context is None:
        src_shape = src.shape
        dst_shape = dst.shape
    else:
        src_shape = context.get_tensor_shape(src)
        dst_shape = context.get_tensor_shape(dst)
    if any(not isinstance(s, int) for s in src_shape) or \
            any(not isinstance(s, int) for s in dst_shape):
        return
    dst_shape_inferred = []
    if not isinstance(axis, tuple):
        axis = [axis]
    for i, v in enumerate(src_shape):
        if i not in axis:
            dst_shape_inferred.append(v)
    TikCheckUtil.check_equality(dst_shape_inferred, dst_shape, "dst's shape error.")


def check_reduce_value_ops_params(dst, reduce_ops_params, context=None):
    """
    check params for reduce value ops instruction

    :param reduce_ops_params: contains dst src axis params
    dst->Tensor or Tensor slice
    src->Tensor or Tensor slice
    axis->None or int or tuple of ints
    :param context: Debug context
    :return: None
    """
    _check_tensor_scope([reduce_ops_params.src, ], ["src", ])
    axis = reduce_ops_params.axis
    # check axis
    TikCheckUtil.check_type_match(
        axis, (int, type(None), tuple), "axis should be (int, None, tuple).")
    if axis is None:
        axis = tuple(range(0, len(reduce_ops_params.src.shape)))
    axis_range = [0, len(reduce_ops_params.src.shape) - 1]
    _max_dimension = axis_range[1] + 1
    if isinstance(axis, tuple):

        if len(set(axis)) != len(axis):
            TikCheckUtil.raise_error("duplicate value in 'axis'.")
        for x in axis:
            TikCheckUtil.check_type_match(x, int, "axis's value should be int.")
            TikCheckUtil.check_in_range_by_dtype(x, msg="axis %s is out of bounds for src of dimension %s."
                                                        % (axis, _max_dimension), var_range=axis_range)
    else:
        TikCheckUtil.check_in_range_by_dtype(axis, msg="axis %s is out of bounds for src of dimension %s."
                                                       % (axis, _max_dimension), var_range=axis_range)

    if (isinstance(axis, tuple) and
        len(axis) == len(reduce_ops_params.src.shape)) or \
            (isinstance(axis, int) and len(reduce_ops_params.src.shape) == 1):
        TikCheckUtil.check_type_match(dst, Scalar, "dst should be Scalar.")
        TikCheckUtil.check_equality(dst.dtype, reduce_ops_params.src.dtype,
                                    "src's dtype and dst's dtype should be the same.")
        _check_reduce_ops_tensor_shape(
            reduce_ops_params.src, dst, axis, context=context)
        return

    _check_tensor_scope([dst, reduce_ops_params.src], ["dst", "src"])
    _check_tensor_dtype([dst, reduce_ops_params.src], ["dst", "src"])
    _check_reduce_ops_tensor_shape(
        reduce_ops_params.src, dst, axis, context=context)
    # check overlap
    _check_tensor_overlap([dst, reduce_ops_params.src], ["dst", "src"], context=context)


def _check_reduce_arg_dst_dtype(dst):
    """
    check the dst dtype
    """
    TikCheckUtil.check_equality("int32", dst.dtype, "dst's dtype should be int32.")


def check_reduce_arg_ops_params(dst, reduce_arg_params, context=None):
    """
    check params for reduce arg ops instruction
    :param dst: Tensor or Tensor slice
    :param reduce_arg_params: contains dst src axis params
    src->Tensor or Tensor slice
    axis->None or int
    :param context: Debug context
    :return: None
    """
    _check_tensor_scope([reduce_arg_params.src, ], ["src", ])
    # check axis
    TikCheckUtil.check_type_match(reduce_arg_params.axis, (int, type(None)), "axis should be (int, None).")
    if isinstance(reduce_arg_params.axis, int):
        axis_range = [0, len(reduce_arg_params.src.shape) - 1]
        _max_dimension = axis_range[1] + 1
        TikCheckUtil.check_in_range_by_dtype(reduce_arg_params.axis,
                                             msg="axis %s is out of bounds for src of dimension %s." %
                                             (reduce_arg_params.axis, _max_dimension), var_range=axis_range)

    if reduce_arg_params.axis is None or (isinstance(reduce_arg_params.axis, int) and
                                          len(reduce_arg_params.src.shape) == 1):
        TikCheckUtil.check_type_match(dst, Scalar, "dst should be Scalar.")
        _check_reduce_arg_dst_dtype(dst)
        _check_reduce_ops_tensor_shape(
            reduce_arg_params.src, dst, reduce_arg_params.axis, context=context)
        return

    _check_tensor_scope([dst, reduce_arg_params.src], ["dst", "src"])
    _check_reduce_arg_dst_dtype(dst)
    _check_reduce_ops_tensor_shape(
        reduce_arg_params.src, dst, reduce_arg_params.axis, context=context)
    # check overlap
    _check_tensor_overlap([dst, reduce_arg_params.src], ["dst", "src"], context=context)


def _check_dms_ops_step(dst, src, context=None):
    """
    check dms ops step
    """
    if context is None:
        dst_strides = dst.strides
        dst_original_shape = dst.original_shape
    else:
        dst_strides = context.get_tensor_strides(dst)
        dst_original_shape = context.get_tensor_original_shape(dst)
    dst_step = _gen_acc_steps(dst_strides[::-1], dst_original_shape[::-1])
    if any(not isinstance(s, int) for s in dst_step):
        return

    if context is None:
        src_strides = src.strides
        src_original_shape = src.original_shape
    else:
        src_strides = context.get_tensor_strides(src)
        src_original_shape = context.get_tensor_original_shape(src)
    src_step = _gen_acc_steps(src_strides[::-1], src_original_shape[::-1])
    if any(not isinstance(s, int) for s in src_step):
        return
    for x, y in zip(dst_step, src_step):
        TikCheckUtil.check_equality(x, 1, "the step of dst tensor should be 1.")
        TikCheckUtil.check_equality(y, 1, "the step of src tensor should be 1.")


def _check_tensor_align(tensor, context=None):
    """
    check tensor 32B align
    :param tensor: Tensor
    :param context: Debug context
    :return: bool
    """
    if context is None:
        strides = tensor.strides[::-1]
        shape = tensor.shape[::-1]
    else:
        strides = context.get_tensor_strides(tensor)[::-1]
        shape = context.get_tensor_shape(tensor)[::-1]
    data_len = 1
    n = len(shape)
    curr_dim = -1
    for i in range(n-1):
        # if stride or shape are expr or scalar, skip check.
        if any(Expr(x).eval_value() is None
               for x in (strides[i], strides[i+1], shape[i])):
            return True
        data_len *= shape[i]
        if strides[i] * shape[i] != strides[i+1]:
            break
        curr_dim = i
    # second dimension is not sliced
    if curr_dim == n-2:
        data_len *= shape[n-1]
    data_len_value = Expr(data_len).eval_value()
    if data_len_value is None or (data_len_value * DTYPE_SIZE[tensor.dtype] % 32 == 0):
        return True
    return False


def _check_tensor_offset(tensor_list, name_list, context=None):
    """
    check tensor offset
    :param tensor_list: tensor list
    :param name_list: tensor name list
    :param context: Debug context
    :return: None
    """
    for tensor, name in zip(tensor_list, name_list):
        if context is None:
            offset = tensor.data
        else:
            offset = context.get_tensor_offset(tensor)
        offset_value = Expr(offset).eval_value()
        if offset_value is not None and \
                offset_value * DTYPE_SIZE[tensor.dtype] % 32 != 0:
            TikCheckUtil.raise_error("The start address of %s is not 32B aligned" % name)


def check_dma_ops_params(dma_ops_params, context=None):
    """
    check params for dma ops instruction
    :param dma_ops_params: contains dst src api_name
    dst: Tensor or Tensor slice
    src: Tensor or Tensor slice
    api_name: api name
    :param context: Debug context
    :return: None
    """
    TikCheckUtil.check_equality(is_tensor(dma_ops_params.src), True, "src should be tensor.")
    TikCheckUtil.check_equality(is_tensor(dma_ops_params.dst), True, "dst should be tensor.")
    # check scope
    src_key_str = TikUtil.get_storage_scope(dma_ops_params.src.scope)
    dst_key_str = TikUtil.get_storage_scope(dma_ops_params.dst.scope)
    key = src_key_str + " " + dst_key_str
    if key not in ("UB OUT", "UB UB", "OUT UB"):
        TikCheckUtil.raise_error("doesn't support %s to %s" % (src_key_str, dst_key_str))

    # check dtype
    _check_tensor_dtype([dma_ops_params.dst, dma_ops_params.src], ["dst", "src"])
    # check step
    _check_dms_ops_step(dma_ops_params.dst, dma_ops_params.src, context=context)
    # check shape
    _check_tensor_shape([dma_ops_params.dst, dma_ops_params.src], ["dst", "src"], context=context)
    # check overlap
    _check_tensor_overlap([dma_ops_params.dst, dma_ops_params.src], ["dst", "src"], context=context)

    if dma_ops_params.src.dtype == "bool" or dma_ops_params.dst.dtype == "bool":
        TikCheckUtil.raise_error(gen_api_check_statement("bool", dma_ops_params.api_name))

    _check_tensor_offset([dma_ops_params.src, dma_ops_params.dst], ["src", "dst"], context=context)
    if not _check_tensor_align(dma_ops_params.src, context=context):
        TikCheckUtil.raise_error("The minimum continuous data block of "
                                 "src needs to be a multiple of 32B")
    if not _check_tensor_align(dma_ops_params.dst, context=context):
        TikCheckUtil.raise_error("The minimum continuous data block of "
                                 "dst needs to be a multiple of 32B")
    if not (TikSocManager.is_v100_soc() or TikSocManager.is_310b_soc()):
        info(gen_api_check_statement(dma_ops_params.src.dtype, dma_ops_params.api_name))
