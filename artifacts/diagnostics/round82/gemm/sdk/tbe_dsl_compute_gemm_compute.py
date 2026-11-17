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
gemm_compute
"""
import math

import tbe.common.platform as tbe_platform
import tbe.common.utils as tbe_utils
from tbe.common.platform import platform_info as tbe_platform_info
from tbe.common.utils.errormgr import error_manager_util
from tbe.dsl.base.operation import in_dynamic
from tbe.dsl.compute.mmad_compute import matmul
from tbe.dsl.compute.util import check_input_tensor_shape
from tbe import tvm
from tbe.tvm import Tensor
from tbe.dsl.compute.gemm_integrated_compute import gemm as gemm_integrated
from tbe.dsl.compute.gemm_ub_compute import matmul_ub_compute

BATCH_MATMUL_LENGTH = 5
ND_FORMAT = "ND"
NZ_FORMAT = "FRACTAL_NZ"


def _shape_check(  # pylint: disable=C0301, R0912, R0913, R0914, R0915
    tensor_a,
    tensor_b,
    tensor_bias,
    tensor_alpha,
    tensor_beta,
    trans_a,
    trans_b,
    format_a,
    format_b,
    dst_dtype,
    matmul_flag
):
    """
    Check the given input if legal

    Parameters:
    shape_a: list or tuple
            Shape of the first tensor a with rank > 1
    shape_b:  list or tuple
            Shape of the second tensor b with the same type with a,
            and shape_a, shape_b must be 2 dims
    shape_bias: list or tuple
            Shape of bias, only support the input data format with ND
    trans_a: bool
            If True, shape_a is transposed before multiplication
    trans_b: bool
            If True, shape_b is transposed before multiplication
    is_fractal: bool
            If True, the input data format of a and b must be fractal format

    Returns None
    """

    in_a_dtype = tensor_a.dtype
    in_b_dtype = tensor_b.dtype

    check_input_tensor_shape(tensor_a)
    check_input_tensor_shape(tensor_b)

    shape_a = [_get_value(i) for i in tensor_a.shape]
    shape_b = [_get_value(i) for i in tensor_b.shape]
    shape_bias = ()

    shape_len_a = len(shape_a)
    shape_len_b = len(shape_b)

    is_fractal_a = format_a != "ND"
    is_fractal_b = format_b != "ND"

    if tensor_bias is not None:
        shape_bias = [_get_value(i) for i in tensor_bias.shape]

    if (in_a_dtype in ("uint8", "int8")) and in_b_dtype == "int8":
        k_block_size = tbe_platform.BLOCK_REDUCE_INT8
    else:
        k_block_size = tbe_platform.BLOCK_REDUCE

    if dst_dtype == "int32" and len(shape_bias) == 2:
        for index, value in enumerate(shape_bias):
            if index == 0:
                block = tbe_platform.BLOCK_IN
            else:
                block = tbe_platform.BLOCK_OUT
            shape_bias[index] = ((value + block - 1) // block) * block

    def _check_dtype():
        # check type of tensor_alpha and tensor_beta
        if not matmul_flag:
            if tensor_alpha.dtype != tensor_beta.dtype:
                args_dict = {
                    "errCode": "E60002",
                    "attr_name": "dtype",
                    "param1_name": "alpha",
                    "param1_value": "{}".format(tensor_alpha.dtype),
                    "param2_name": "beta",
                    "param2_value": "{}".format(tensor_beta.dtype)
                }
                raise RuntimeError(
                    args_dict, error_manager_util.get_error_message(args_dict)
                )
            if dst_dtype != tensor_alpha.dtype:
                args_dict = {
                    "errCode": "E60002",
                    "attr_name": "dtype",
                    "param1_name": "y",
                    "param1_value": "{}".format(dst_dtype),
                    "param2_name": "alpha",
                    "param2_value": "{}".format(tensor_alpha.dtype)
                }
                raise RuntimeError(
                    args_dict, error_manager_util.get_error_message(args_dict)
                )
        # ND and fractal support 'float16' and 'b8'
        if not (in_a_dtype == "float16" and in_b_dtype == "float16") and not (
            in_a_dtype in ("uint8", "int8") and (in_b_dtype == "int8")
        ):
            args_dict = {
                "errCode": "E60005",
                "param_name": "in_a_dtype/in_b_dtype",
                "expected_dtype_list": "float16 & float16 and uint8/int8 & int8",
                "dtype": "{}/{}".format(in_a_dtype, in_b_dtype)
            }
            raise RuntimeError(
                args_dict, error_manager_util.get_error_message(args_dict)
            )

        if dst_dtype not in ("float16", "float32", "int32"):
            args_dict = {
                "errCode": "E60005",
                "param_name": "y",
                "expected_dtype_list": "[float16, float32,int32]",
                "dtype": "{}".format(dst_dtype)
            }
            raise RuntimeError(
                args_dict, error_manager_util.get_error_message(args_dict)
            )

    def _check_fractal():
        if format_a not in ("ND", "FRACTAL_NZ", "FRACTAL_Z"):
            args_dict = {
                "errCode": "E60004",
                "param_name": "a",
                "expected_format_list": "[ND, FRACTAL_NZ, FRACTAL_Z]",
                "format": "{}".format(format_a)
            }
            raise RuntimeError(
                args_dict, error_manager_util.get_error_message(args_dict)
            )

        if format_b not in ("ND", "FRACTAL_NZ", "FRACTAL_Z"):
            args_dict = {
                "errCode": "E60004",
                "param_name": "b",
                "expected_format_list": "[ND, FRACTAL_NZ, FRACTAL_Z]",
                "format": "{}".format(format_b)
            }
            raise RuntimeError(
                args_dict, error_manager_util.get_error_message(args_dict)
            )

        # fractal and ND not support
        if is_fractal_a and not is_fractal_b:
            args_dict = {
                "errCode": "E60114",
                "reason": "Not support a is fractal and b is ND!",
                "value": "is_fractal_a = {} and is_fractal_b"
                " = {}".format(is_fractal_a, is_fractal_b)
            }
            raise RuntimeError(
                args_dict, error_manager_util.get_error_message(args_dict)
            )
        if not (GEMMComputeParam.batch_a and not GEMMComputeParam.batch_b):
            if (is_fractal_a == is_fractal_b) and (shape_len_a != shape_len_b):
                args_dict = {
                    "errCode": "E60002",
                    "attr_name": "dim",
                    "param1_name": "a",
                    "param1_value": "{}".format(shape_len_a),
                    "param2_name": "b",
                    "param2_value": "{}".format(shape_len_b)
                }
                raise RuntimeError(
                    args_dict, error_manager_util.get_error_message(args_dict)
                )

    _check_dtype()
    _check_fractal()

    def _check_shape():
        if is_fractal_a:
            if shape_len_a not in (4, 5):
                args_dict = {
                    "errCode": "E60006",
                    "param_name": "tensor a",
                    "expected_length": "[4,5]",
                    "length": "{}".format(shape_len_a)
                }
                raise RuntimeError(
                    args_dict, error_manager_util.get_error_message(args_dict)
                )
        else:
            if shape_len_a not in (2, 3):
                args_dict = {
                    "errCode": "E60006",
                    "param_name": "tensor a",
                    "expected_length": "[2,3]",
                    "length": "{}".format(shape_len_a)
                }
                raise RuntimeError(
                    args_dict, error_manager_util.get_error_message(args_dict)
                )

        if is_fractal_b:
            if shape_len_b not in (4, 5):
                args_dict = {
                    "errCode": "E60006",
                    "param_name": "tensor b",
                    "expected_length": "[4,5]",
                    "length": "{}".format(shape_len_b)
                }
                raise RuntimeError(
                    args_dict, error_manager_util.get_error_message(args_dict)
                )
        else:
            if shape_len_b not in (2, 3):
                args_dict = {
                    "errCode": "E60006",
                    "param_name": "tensor b",
                    "expected_length": "[2,3]",
                    "length": "{}".format(shape_len_b)
                }
                raise RuntimeError(
                    args_dict, error_manager_util.get_error_message(args_dict)
                )

        if shape_len_a in (3, 5) and not (GEMMComputeParam.batch_a and not GEMMComputeParam.batch_b):
            if _get_value(tensor_a.shape[0]) != _get_value(tensor_b.shape[0]):
                args_dict = {
                    "errCode": "E60002",
                    "attr_name": "shape",
                    "param1_name": "tensor a",
                    "param1_value": "{}".format(_get_value(tensor_a.shape[0])),
                    "param2_name": "tensor b",
                    "param2_value": "{}".format(_get_value(tensor_b.shape[0]))
                }
                raise RuntimeError(
                    args_dict, error_manager_util.get_error_message(args_dict)
                )

    _check_shape()

    def _check_a_m_k_n():
        is_vector_a = False
        if not is_fractal_a:
            # shape_len_a is 2 or 3
            if trans_a:
                m_shape = shape_a[shape_len_a - 1]
                km_shape = shape_a[shape_len_a - 2]
            else:
                m_shape = shape_a[shape_len_a - 2]
                km_shape = shape_a[shape_len_a - 1]
        else:
            if trans_a:
                m_shape = shape_a[shape_len_a - 3]
                km_shape = shape_a[shape_len_a - 4]
                a_block_reduce = shape_a[shape_len_a - 1]
                a_block_in = shape_a[shape_len_a - 2]
            else:
                m_shape = shape_a[shape_len_a - 4]
                km_shape = shape_a[shape_len_a - 3]
                a_block_reduce = shape_a[shape_len_a - 1]
                a_block_in = shape_a[shape_len_a - 2]

            if a_block_reduce != k_block_size:
                args_dict = {
                    "errCode": "E60104",
                    "expected_value": "{} or {}".format(
                        tbe_platform.BLOCK_REDUCE_INT8, tbe_platform.BLOCK_REDUCE
                    ),
                    "value": "{}".format(a_block_reduce)
                }
                raise RuntimeError(
                    args_dict, error_manager_util.get_error_message(args_dict)
                )

            if a_block_in not in (tbe_platform.BLOCK_VECTOR, tbe_platform.BLOCK_IN):
                args_dict = {
                    "errCode": "E60103",
                    "expected_value": "{} or {}".format(
                        tbe_platform.BLOCK_VECTOR, tbe_platform.BLOCK_IN
                    ),
                    "value": "{}".format(a_block_in)
                }
                raise RuntimeError(
                    args_dict, error_manager_util.get_error_message(args_dict)
                )
            if a_block_in == tbe_platform.BLOCK_VECTOR:
                is_vector_a = True
                if m_shape != tbe_platform.BLOCK_VECTOR:
                    args_dict = {
                        "errCode": "E60114",
                        "reason": "when block_in of a is {}, m_shape of a should be {}".format(
                            tbe_platform.BLOCK_VECTOR, tbe_platform.BLOCK_VECTOR
                        ),
                        "value": "{}".format(m_shape)
                    }
                    raise RuntimeError(
                        args_dict, error_manager_util.get_error_message(args_dict)
                    )
                if km_shape % (tbe_platform.BLOCK_IN) != 0:
                    args_dict = {
                        "errCode": "E60114",
                        "reason": "k should be multiple of {}".format(
                            tbe_platform.BLOCK_IN * k_block_size
                        ),
                        "value": "k = {}".format(km_shape)
                    }
                    raise RuntimeError(
                        args_dict, error_manager_util.get_error_message(args_dict)
                    )
        return km_shape, is_vector_a

    km_shape, is_vector_a = _check_a_m_k_n()

    def _check_b_m_k_n(is_vector_a):  # pylint: disable=too-many-branches
        is_gemv = False

        def _get_nd_m_k_n():
            # shape_len_b is 2 or 3
            if trans_b:
                kn_shape = shape_b[shape_len_b - 1]
                n_shape = shape_b[shape_len_b - 2]
            else:
                kn_shape = shape_b[shape_len_b - 2]
                n_shape = shape_b[shape_len_b - 1]

            return kn_shape, n_shape

        if not is_fractal_b:
            kn_shape, n_shape = _get_nd_m_k_n()
        else:
            if trans_b:
                kn_shape = shape_b[shape_len_b - 3]
                n_shape = shape_b[shape_len_b - 4]
                b_block_reduce = shape_b[shape_len_b - 2]
                b_block_out = shape_b[shape_len_b - 1]
            else:
                kn_shape = shape_b[shape_len_b - 4]
                n_shape = shape_b[shape_len_b - 3]
                b_block_reduce = shape_b[shape_len_b - 1]
                b_block_out = shape_b[shape_len_b - 2]

            if b_block_reduce != k_block_size:
                args_dict = {
                    "errCode": "E60106",
                    "expected_value": "{} or {}".format(
                        tbe_platform.BLOCK_REDUCE_INT8, tbe_platform.BLOCK_REDUCE
                    ),
                    "value": "{}".format(b_block_reduce)
                }
                raise RuntimeError(
                    args_dict, error_manager_util.get_error_message(args_dict)
                )

            if b_block_out not in (tbe_platform.BLOCK_VECTOR, tbe_platform.BLOCK_IN):
                args_dict = {
                    "errCode": "E60105",
                    "expected_value": "{} or {}".format(
                        tbe_platform.BLOCK_VECTOR, tbe_platform.BLOCK_IN
                    ),
                    "value": "{}".format(b_block_out)
                }
                raise RuntimeError(
                    args_dict, error_manager_util.get_error_message(args_dict)
                )
            if b_block_out == tbe_platform.BLOCK_VECTOR:
                is_gemv = True
                if is_vector_a:
                    args_dict = {
                        "errCode": "E60114",
                        "reason": "input shape M and N can't both be 1",
                        "value": "input shape M and N are both 1"
                    }
                    raise RuntimeError(
                        args_dict, error_manager_util.get_error_message(args_dict)
                    )
                if n_shape != 1:
                    args_dict = {
                        "errCode": "E60114",
                        "reason": "when block_out of b is {}, n_shape of b should be {}".format(
                            tbe_platform.BLOCK_VECTOR, tbe_platform.BLOCK_VECTOR
                        ),
                        "value": "{}".format(n_shape)
                    }
                    raise RuntimeError(
                        args_dict, error_manager_util.get_error_message(args_dict)
                    )
                if kn_shape % (tbe_platform.BLOCK_IN) != 0:
                    args_dict = {
                        "errCode": "E60114",
                        "reason": "k should be multiple of {}".format(
                            tbe_platform.BLOCK_IN * k_block_size
                        ),
                        "value": "k = {}".format(kn_shape)
                    }
                    raise RuntimeError(
                        args_dict, error_manager_util.get_error_message(args_dict)
                    )
                # gemv u8/s8 is transed to gevm(s8/u8), s8/u8 is not support
                # for mad intri
                if in_a_dtype == "uint8" and in_b_dtype == "int8":
                    args_dict = {
                        "errCode": "E60005",
                        "param_name": "in_a_dtype/in_b_dtype",
                        "expected_dtype_list": "int8 & int8",
                        "dtype": "{}/{}".format(in_a_dtype, in_b_dtype)
                    }
                    raise RuntimeError(
                        args_dict, error_manager_util.get_error_message(args_dict)
                    )

        return is_gemv, kn_shape, n_shape

    is_gemv, kn_shape, n_shape = _check_b_m_k_n(is_vector_a)

    def _check_a_between_b():
        if is_fractal_a == is_fractal_b:
            if km_shape != kn_shape:
                args_dict = {
                    "errCode": "E60114",
                    "reason": "reduce axis not same",
                    "value": "reduce axis not same"
                }
                raise RuntimeError(
                    args_dict, error_manager_util.get_error_message(args_dict)
                )

    _check_a_between_b()

    def renew_is_gemv(is_gemv):
        if not is_fractal_a and not is_fractal_b:
            is_gemv = n_shape == 1
        return is_gemv

    is_gemv = renew_is_gemv(is_gemv)

    def _check_bias():
        if shape_bias and not matmul_flag:
            if len(shape_bias) not in (2, 4):
                args_dict = {
                    "errCode": "E60006",
                    "param_name": "c",
                    "expected_length": "2 or 4",
                    "length": "{}".format(len(shape_bias))
                }
                raise RuntimeError(
                    args_dict, error_manager_util.get_error_message(args_dict)
                )
        if shape_bias and matmul_flag:
            if len(shape_bias) != 1:
                args_dict = {
                    "errCode": "E60006",
                    "param_name": "c",
                    "expected_length": "1",
                    "length": "{}".format(len(shape_bias))
                }
                raise RuntimeError(
                    args_dict, error_manager_util.get_error_message(args_dict)
                )

    _check_bias()

    def _check_n_align():
        """
        When the input and output tensors share the memory,
        n not align to block_out is not supported.
        Input: None
        ------------------------
        Return: None
        """
        n_shape = shape_b[0] if trans_b else shape_b[1]
        if format_b == "ND" and (n_shape % tbe_platform.BLOCK_OUT != 0):
            reason = ("When the input format is ND, "
                      "the n direction must be aligned to {}.".format(tbe_platform.BLOCK_OUT))
            args_dict = {
                "errCode": "E60108",
                "op_name": "GEMM",
                "reason": reason
            }
            raise RuntimeError(
                args_dict, error_manager_util.get_error_message(args_dict)
            )

    _check_n_align()


def _get_value(shape_object):
    """
    get the value of shape_object when having attr "value"
    """
    return shape_object.value if hasattr(shape_object, "value") else shape_object


@tbe_utils.para_check.check_input_type(Tensor, Tensor, dict)
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

    kernel_name = para_dict.get("kernel_name", "")
    para_dict_copy = para_dict.copy()
    is_confusion_transpose = para_dict.get("confusion_transpose", False)
    use_old_code = is_confusion_transpose
    if not in_dynamic():
        # GEMMCompute not support input NZ and output FORMAT_ND
        is_nz_in_nd_out = (para_dict.get("format_a") == NZ_FORMAT and para_dict.get("format_b") == NZ_FORMAT and
                           para_dict.get("format_out") == ND_FORMAT)
        use_old_code = use_old_code or (filter_case(tensor_a, tensor_b, kernel_name) and not is_nz_in_nd_out)
        # GEMMCompute not support perm_x1, perm_x2 and perm_y
        use_old_code = use_old_code and all(x not in para_dict for x in ("perm_x1", "perm_x2", "perm_y"))

    use_old_code = use_old_code and not tbe_platform_info.intrinsic_check_support("Intrinsic_fix_pipe_l0c2out")
    support_matmul_ub_to_ub = tbe_platform_info.intrinsic_check_support("Intrinsic_matmul_ub_to_ub")
    if support_matmul_ub_to_ub:
        result = matmul_ub_compute(tensor_a, tensor_b, para_dict)
    elif not use_old_code:
        result = gemm_integrated(tensor_a, tensor_b, para_dict)
    else:
        gemm_compute = GEMMCompute(tensor_a, tensor_b, para_dict)
        result = gemm_compute.calculate()
    setattr(result, "tensor_a", tensor_a)
    setattr(result, "tensor_b", tensor_b)
    setattr(result, "para_dict", para_dict_copy)
    return result


def filter_case(tensor_a, tensor_b, kernel_name):
    # vgg16_faster_rcnn_coco_int8 sppnet vgg16_faster_rcnn_coco vgg_ilsvrc_16
    black_list_compress_fc = [
        "1_12800_400_256_16_32_int8",
        "128_19_16_32_128_6_16_32_int8",
        "128_19_16_32_128_21_16_32_int8"
    ]
    black_list_fc = [
        "304_25088_1568_256_16_16_float16"
    ]

    black_list_1980 = [
        # multi_op
        "64_512_16_16_64_64_16_16_float16",
        "512_4_16_16_16_512_4_16_16_16_float16",
        "512_16_16_16_16_512_4_16_16_16_float16",
        "64_512_16_16_64_512_16_16_float16",
        # bert_nv_1p_512_lamb_bs24
        "64_768_16_16_64_768_16_16_float16",
        "1908_114_16_16_64_1908_16_16_float16",
        "1908_114_16_16_64_114_16_16_float16",
        # relu 710
        "256_1_16_16_256_256_16_16_float16"
    ]

    black_list_910 = {
        "Ascend910A": ["48_2048_16_16_48_48_16_16_float16",
                       # FusionOp_BatchMatMul_Add
                       "3072_4_8_16_16_3072_4_8_16_16_float16",
                       "1_16_16_16_48_1_16_16_float16",
                       "1321_320_16_16_48_1321_16_16_float16",
                       "1321_320_16_16_48_320_16_16_float16",
                       "48_16_16_16_48_16_16_16_float16",
                       "48_16_16_16_48_48_16_16_float16",
                       "48_320_16_16_48_320_16_16_float16",
                       "48_320_16_16_48_48_16_16_float16",
                       "48_2048_16_16_192_2048_16_16_float16",
                       "48_2048_16_16_192_48_16_16_float16",
                       "192_2048_16_16_48_2048_16_16_float16",
                       "3072_8_8_16_16_3072_4_8_16_16_float16",
                       "192_2048_16_16_48_192_16_16_float16",
                       "48_2048_16_16_48_2048_16_16_float16",
                       "48_2048_16_16_48_192_16_16_float16",
                       "192_2048_16_16_192_48_16_16_float16",
                       "48_16_16_16_48_1_16_16_float16",
                       "48_320_16_16_48_1321_16_16_float16",
                       "1_16_16_16_48_16_16_16_float16",
                       "1536_4_2_16_16_1536_4_2_16_16_float16",
                       "256_192_16_16_64_256_16_16_float16",
                       "64_768_16_16_64_768_16_16_float16",
                       "1536_8_8_16_16_1536_4_8_16_16_float16",
                       "64_768_16_16_64_256_16_16_float16",
                       "256_192_16_16_64_192_16_16_float16",
                       "64_192_16_16_64_192_16_16_float16",
                       "1536_2_2_16_16_1536_4_2_16_16_float16",
                       "64_192_16_16_64_256_16_16_float16",
                       "64_96_16_16_64_64_16_16_float16",
                       "1536_4_1_16_16_1536_4_1_16_16_float16",
                       "1536_1_1_16_16_1536_4_1_16_16_float16",
                       "64_96_16_16_64_256_16_16_float16",
                       "256_96_16_16_256_64_16_16_float16",
                       "64_96_16_16_64_2285_16_16_float16",
                       "2285_96_16_16_64_2285_16_16_float16",
                       "2285_96_16_16_64_96_16_16_float16",
                       "64_96_16_16_256_64_16_16_float16",
                       "64_96_16_16_256_96_16_16_float16",
                       "256_96_16_16_64_256_16_16_float16",
                       "256_96_16_16_64_96_16_16_float16",
                       "64_96_16_16_64_96_16_16_float16",
                       "256_192_16_16_256_64_16_16_float16",
                       "64_384_16_16_64_64_16_16_float16",
                       "1536_4_4_16_16_1536_4_4_16_16_float16",
                       "64_384_16_16_64_256_16_16_float16",
                       "256_384_16_16_256_64_16_16_float16",
                       "64_384_16_16_64_2285_16_16_float16",
                       "64_192_16_16_64_2285_16_16_float16",
                       "2285_384_16_16_64_2285_16_16_float16",
                       "2285_384_16_16_64_384_16_16_float16",
                       "64_384_16_16_256_64_16_16_float16",
                       "64_384_16_16_256_384_16_16_float16",
                       "256_384_16_16_64_256_16_16_float16",
                       "256_384_16_16_64_384_16_16_float16",
                       "64_384_16_16_64_384_16_16_float16",
                       "2285_192_16_16_64_2285_16_16_float16",
                       "64_288_16_16_64_64_16_16_float16",
                       "1536_4_3_16_16_1536_4_3_16_16_float16",
                       "1536_3_3_16_16_1536_4_3_16_16_float16",
                       "64_288_16_16_64_256_16_16_float16",
                       "2285_192_16_16_64_192_16_16_float16",
                       "256_288_16_16_256_64_16_16_float16",
                       "64_288_16_16_64_2285_16_16_float16",
                       "2285_288_16_16_64_2285_16_16_float16",
                       "2285_288_16_16_64_288_16_16_float16",
                       "64_288_16_16_256_64_16_16_float16",
                       "64_288_16_16_256_288_16_16_float16",
                       "256_288_16_16_64_256_16_16_float16",
                       "256_288_16_16_64_288_16_16_float16",
                       "64_288_16_16_64_288_16_16_float16",
                       "64_192_16_16_256_64_16_16_float16",
                       "64_192_16_16_256_192_16_16_float16",
                       "64_768_16_16_64_2285_16_16_float16",
                       "2285_768_16_16_64_2285_16_16_float16",
                       "2285_768_16_16_64_768_16_16_float16",
                       "64_768_16_16_256_768_16_16_float16",
                       "195_1000_16_16_64_195_16_16_float16",
                       "64_1000_16_16_32_64_16_16_float16",
                       "1_1000_16_16_1_8_16_16_float16",
                       "16_1000_16_16_8_1000_16_16_float16",
                       "8_1000_16_16_8_16_16_16_float16",
                       "32_1000_16_16_16_1000_16_16_float16",
                       "16_1000_16_16_16_32_16_16_float16",
                       "64_1000_16_16_32_1000_16_16_float16",
                       "32_1000_16_16_32_64_16_16_float16",
                       "195_1000_16_16_64_1000_16_16_float16",
                       "64_1000_16_16_64_195_16_16_float16",
                       "16_1000_16_16_8_16_16_16_float16",
                       "32_1000_16_16_16_32_16_16_float16",
                       "8_1000_16_16_1_1000_16_16_float16"
        ],
        "Ascend910ProA": [
            # FusionOp_MatMul_Mul
            "64_768_16_16_256_768_16_16_float16",
            "256_768_16_16_64_768_16_16_float16",
            # FusionOp_MatMul_AddN
            "48_2048_16_16_48_48_16_16_float16",
            # FusionOp_BatchMatMul_Add
            "3072_4_8_16_16_3072_4_8_16_16_float16",
            "1_16_16_16_48_1_16_16_float16",
            "1321_320_16_16_48_1321_16_16_float16",
            "1321_320_16_16_48_320_16_16_float16",
            "48_16_16_16_48_16_16_16_float16",
            "48_16_16_16_48_48_16_16_float16",
            "48_320_16_16_48_320_16_16_float16",
            "48_320_16_16_48_48_16_16_float16",
            "48_2048_16_16_192_2048_16_16_float16",
            "48_2048_16_16_192_48_16_16_float16",
            "192_2048_16_16_48_2048_16_16_float16",
            "3072_8_8_16_16_3072_4_8_16_16_float16",
            "192_2048_16_16_48_192_16_16_float16",
            "48_2048_16_16_48_2048_16_16_float16",
            "48_2048_16_16_48_192_16_16_float16",
            "192_2048_16_16_192_48_16_16_float16",
            "48_16_16_16_48_1_16_16_float16",
            "48_320_16_16_48_1321_16_16_float16",
            "1_16_16_16_48_16_16_16_float16",
            "1536_4_2_16_16_1536_4_2_16_16_float16",
            "256_192_16_16_64_256_16_16_float16",
            "64_768_16_16_64_768_16_16_float16",
            "1536_8_8_16_16_1536_4_8_16_16_float16",
            "64_768_16_16_64_256_16_16_float16",
            "256_192_16_16_64_192_16_16_float16",
            "64_192_16_16_64_192_16_16_float16",
            "1536_2_2_16_16_1536_4_2_16_16_float16",
            "64_192_16_16_64_256_16_16_float16",
            "64_96_16_16_64_64_16_16_float16",
            "1536_4_1_16_16_1536_4_1_16_16_float16",
            "1536_1_1_16_16_1536_4_1_16_16_float16",
            "64_96_16_16_64_256_16_16_float16",
            "256_96_16_16_256_64_16_16_float16",
            "64_96_16_16_64_2285_16_16_float16",
            "2285_96_16_16_64_2285_16_16_float16",
            "2285_96_16_16_64_96_16_16_float16",
            "64_96_16_16_256_64_16_16_float16",
            "64_96_16_16_256_96_16_16_float16",
            "256_96_16_16_64_256_16_16_float16",
            "256_96_16_16_64_96_16_16_float16",
            "64_96_16_16_64_96_16_16_float16",
            "256_192_16_16_256_64_16_16_float16",
            "64_384_16_16_64_64_16_16_float16",
            "1536_4_4_16_16_1536_4_4_16_16_float16",
            "64_384_16_16_64_256_16_16_float16",
            "256_384_16_16_256_64_16_16_float16",
            "64_384_16_16_64_2285_16_16_float16",
            "64_192_16_16_64_2285_16_16_float16",
            "2285_384_16_16_64_2285_16_16_float16",
            "2285_384_16_16_64_384_16_16_float16",
            "64_384_16_16_256_64_16_16_float16",
            "64_384_16_16_256_384_16_16_float16",
            "256_384_16_16_64_256_16_16_float16",
            "256_384_16_16_64_384_16_16_float16",
            "64_384_16_16_64_384_16_16_float16",
            "2285_192_16_16_64_2285_16_16_float16",
            "64_288_16_16_64_64_16_16_float16",
            "1536_4_3_16_16_1536_4_3_16_16_float16",
            "1536_3_3_16_16_1536_4_3_16_16_float16",
            "64_288_16_16_64_256_16_16_float16",
            "2285_192_16_16_64_192_16_16_float16",
            "256_288_16_16_256_64_16_16_float16",
            "64_288_16_16_64_2285_16_16_float16",
            "2285_288_16_16_64_2285_16_16_float16",
            "2285_288_16_16_64_288_16_16_float16",
            "64_288_16_16_256_64_16_16_float16",
            "64_288_16_16_256_288_16_16_float16",
            "256_288_16_16_64_256_16_16_float16",
            "256_288_16_16_64_288_16_16_float16",
            "64_288_16_16_64_288_16_16_float16",
            "64_192_16_16_256_64_16_16_float16",
            "64_192_16_16_256_192_16_16_float16",
            "256_768_16_16_256_64_16_16_float16",
            "64_768_16_16_64_2285_16_16_float16",
            "2285_768_16_16_64_2285_16_16_float16",
            "2285_768_16_16_64_768_16_16_float16",
            "195_1000_16_16_64_195_16_16_float16",
            "64_1000_16_16_32_64_16_16_float16",
            "1_1000_16_16_1_8_16_16_float16",
            "16_1000_16_16_8_1000_16_16_float16",
            "8_1000_16_16_8_16_16_16_float16",
            "32_1000_16_16_16_1000_16_16_float16",
            "16_1000_16_16_16_32_16_16_float16",
            "64_1000_16_16_32_1000_16_16_float16",
            "32_1000_16_16_32_64_16_16_float16",
            "195_1000_16_16_64_1000_16_16_float16",
            "64_1000_16_16_64_195_16_16_float16",
            "16_1000_16_16_8_16_16_16_float16",
            "32_1000_16_16_16_32_16_16_float16",
            "8_1000_16_16_1_1000_16_16_float16"
        ],
        "Ascend910PremiumA": [
            # FusionOp_MatMul_Mul
            "64_768_16_16_256_768_16_16_float16",
            "256_768_16_16_64_768_16_16_float16",
            # FusionOp_MatMul_AddN
            "48_2048_16_16_48_48_16_16_float16",
            # FusionOp_BatchMatMul_Add
            "3072_4_8_16_16_3072_4_8_16_16_float16",
            "1_16_16_16_48_1_16_16_float16",
            "1321_320_16_16_48_1321_16_16_float16",
            "1321_320_16_16_48_320_16_16_float16",
            "48_16_16_16_48_16_16_16_float16",
            "48_16_16_16_48_48_16_16_float16",
            "48_320_16_16_48_320_16_16_float16",
            "48_320_16_16_48_48_16_16_float16",
            "48_2048_16_16_192_2048_16_16_float16",
            "48_2048_16_16_192_48_16_16_float16",
            "192_2048_16_16_48_2048_16_16_float16",
            "3072_8_8_16_16_3072_4_8_16_16_float16",
            "192_2048_16_16_48_192_16_16_float16",
            "48_2048_16_16_48_2048_16_16_float16",
            "48_2048_16_16_48_192_16_16_float16",
            "192_2048_16_16_192_48_16_16_float16",
            "48_16_16_16_48_1_16_16_float16",
            "48_320_16_16_48_1321_16_16_float16",
            "1_16_16_16_48_16_16_16_float16",
            "1536_4_2_16_16_1536_4_2_16_16_float16",
            "256_192_16_16_64_256_16_16_float16",
            "64_768_16_16_64_768_16_16_float16",
            "1536_8_8_16_16_1536_4_8_16_16_float16",
            "64_768_16_16_64_256_16_16_float16",
            "256_192_16_16_64_192_16_16_float16",
            "64_192_16_16_64_192_16_16_float16",
            "1536_2_2_16_16_1536_4_2_16_16_float16",
            "64_192_16_16_64_256_16_16_float16",
            "64_96_16_16_64_64_16_16_float16",
            "1536_4_1_16_16_1536_4_1_16_16_float16",
            "1536_1_1_16_16_1536_4_1_16_16_float16",
            "64_96_16_16_64_256_16_16_float16",
            "256_96_16_16_256_64_16_16_float16",
            "64_96_16_16_64_2285_16_16_float16",
            "2285_96_16_16_64_2285_16_16_float16",
            "2285_96_16_16_64_96_16_16_float16",
            "64_96_16_16_256_64_16_16_float16",
            "64_96_16_16_256_96_16_16_float16",
            "256_96_16_16_64_256_16_16_float16",
            "256_96_16_16_64_96_16_16_float16",
            "64_96_16_16_64_96_16_16_float16",
            "256_192_16_16_256_64_16_16_float16",
            "64_384_16_16_64_64_16_16_float16",
            "1536_4_4_16_16_1536_4_4_16_16_float16",
            "64_384_16_16_64_256_16_16_float16",
            "256_384_16_16_256_64_16_16_float16",
            "64_384_16_16_64_2285_16_16_float16",
            "64_192_16_16_64_2285_16_16_float16",
            "2285_384_16_16_64_2285_16_16_float16",
            "2285_384_16_16_64_384_16_16_float16",
            "64_384_16_16_256_64_16_16_float16",
            "64_384_16_16_256_384_16_16_float16",
            "256_384_16_16_64_256_16_16_float16",
            "256_384_16_16_64_384_16_16_float16",
            "64_384_16_16_64_384_16_16_float16",
            "2285_192_16_16_64_2285_16_16_float16",
            "64_288_16_16_64_64_16_16_float16",
            "1536_4_3_16_16_1536_4_3_16_16_float16",
            "1536_3_3_16_16_1536_4_3_16_16_float16",
            "64_288_16_16_64_256_16_16_float16",
            "2285_192_16_16_64_192_16_16_float16",
            "256_288_16_16_256_64_16_16_float16",
            "64_288_16_16_64_2285_16_16_float16",
            "2285_288_16_16_64_2285_16_16_float16",
            "2285_288_16_16_64_288_16_16_float16",
            "64_288_16_16_256_64_16_16_float16",
            "64_288_16_16_256_288_16_16_float16",
            "256_288_16_16_64_256_16_16_float16",
            "256_288_16_16_64_288_16_16_float16",
            "64_288_16_16_64_288_16_16_float16",
            "64_192_16_16_256_64_16_16_float16",
            "64_192_16_16_256_192_16_16_float16",
            "256_768_16_16_256_64_16_16_float16",
            "64_768_16_16_64_2285_16_16_float16",
            "2285_768_16_16_64_2285_16_16_float16",
            "2285_768_16_16_64_768_16_16_float16",
            "195_1000_16_16_64_195_16_16_float16",
            "64_1000_16_16_32_64_16_16_float16",
            "1_1000_16_16_1_8_16_16_float16",
            "16_1000_16_16_8_1000_16_16_float16",
            "8_1000_16_16_8_16_16_16_float16",
            "32_1000_16_16_16_1000_16_16_float16",
            "16_1000_16_16_16_32_16_16_float16",
            "64_1000_16_16_32_1000_16_16_float16",
            "32_1000_16_16_32_64_16_16_float16",
            "195_1000_16_16_64_1000_16_16_float16",
            "64_1000_16_16_64_195_16_16_float16",
            "16_1000_16_16_8_16_16_16_float16",
            "32_1000_16_16_16_32_16_16_float16",
            "8_1000_16_16_1_1000_16_16_float16"
        ],
        "Ascend910ProB": [
            # nmt_bs128 matmul_op
            "96_8_16_16_128_96_16_16_float16"
        ],
        "Ascend910B": [
            "192_960_16_16_48_960_16_16_float16",
            "192_960_16_16_192_48_16_16_float16",
            "48_960_16_16_192_960_16_16_float16",
            "48_960_16_16_48_960_16_16_float16"
        ]
    }
    soc_version = tbe_platform.get_soc_spec("FULL_SOC_VERSION")
    if soc_version == "Ascend910":
        soc_version = "Ascend910A"
    shape_a = [str(x.value) for x in tensor_a.shape]
    shape_b = [str(x.value) for x in tensor_b.shape]
    info_list = shape_a + shape_b
    info_list.append(tensor_a.dtype)
    info_str = "_".join(info_list)
    if info_str in black_list_compress_fc and kernel_name.find("compress_fully_connection") != -1:
        return True
    if info_str in black_list_fc and kernel_name.find("fully_connection") != -1:
        return True
    if info_str in black_list_1980 and not (kernel_name.find("fused") != -1 and
                                            tbe_platform_info.get_soc_spec("pipev2_constraints") == "1"):
        return True
    if info_str in black_list_910.get(soc_version, []):
        return True
    # ACL_BERTBASE excute fail
    if kernel_name.find("gelu") != -1 and kernel_name.find("batch_matmul") == -1:
        return True

    return False


@tbe_utils.para_check.check_input_type(Tensor)
def check_batchmatmul_fuse(input_tensor):
    """
    check if fused with batchmatmul

    Parameters:
    input_tensor: the tensor of elem input

    Returns result
    """
    queue = [input_tensor]
    visited = [input_tensor]
    while queue:
        item = queue.pop(0)
        if len(item.shape) == BATCH_MATMUL_LENGTH and ("matmul" in item.op.tag) \
           and item.op.attrs["format"] == "FRACTAL_NZ":
            return True

        for child in item.op.input_tensors:
            if child not in visited:
                queue.append(child)
                visited.append(child)
    return False


@tbe_utils.para_check.check_input_type(Tensor, Tensor, dict, str)
def batchmatmul_elem_nd2nz(batch_matmul, elem_input, para_dict, para_name):
    """
    reshape batchmatmul+elem ubfusion inputs tensors

    Parameters:
    batch_matmul: the tensor of batchmatmul result

    elem_input: the tensor of elem

    para_dict: the dict with batch_shape and format_elem

    para_name: the elemwise name

    Returns result
    """
    batch_shape = para_dict.get("batch_shape", [])
    format_elem = para_dict.get("format_elem", "ND")
    shape_matmul = tbe_utils.shape_util.shape_to_list(batch_matmul.shape)
    shape_elem = tbe_utils.shape_util.shape_to_list(elem_input.shape)
    shape_max = batch_shape + shape_matmul[-4:]

    if format_elem != "FRACTAL_NZ" and shape_elem[-1] != 1:
        elem_ndim = shape_elem[-1]
        shape_elem_batch = [1] * len(batch_shape) if len(shape_elem) == 1 else shape_elem[0:-2]
        shape_elem_nz = shape_elem_batch + [elem_ndim // 16, 1, 1, 16]
        elem_input = tvm.compute(
            shape_elem_nz,
            lambda *indice: (elem_input(indice[-4]*16 + indice[-1]) if len(shape_elem) == 1 \
                            else elem_input(*indice[0:-4], 0, indice[-4]*16 + indice[-1])),
            name="broadcast_nz2nd_" + para_name,
            tag="broadcast_nz2nd")
    return elem_input, shape_max


@tbe_utils.para_check.check_input_type(Tensor, Tensor, list, str)
def batchmatmul_elem_reshape(batch_matmul, elem_input, batch_shape, para_name):
    """
    reshape batchmatmul+elem ubfusion inputs tensors

    Parameters:
    batch_matmul: the tensor of batchmatmul result

    elem_input: the tensor of elem

    batch_shape: the shape of  batch

    para_name: the elemwise name

    Returns result
    """
    shape_matmul = tbe_utils.shape_util.shape_to_list(batch_matmul.shape)

    def _batch_reshape(indices, input_tensor):
        if len(batch_shape) == 1:
            return input_tensor(indices[0], *indices[-4:])
        elif len(batch_shape) == 2:
            return input_tensor(indices[0] // batch_shape[-1],
                                indices[0] % batch_shape[-1],
                                *indices[-4:])
        elif len(batch_shape) == 3:
            return input_tensor(indices[0] // batch_shape[-1] // batch_shape[-2],
                                indices[0] // batch_shape[-1] % batch_shape[-2],
                                indices[0] % batch_shape[-1],
                                *indices[-4:])
        return input_tensor(indices[0] // batch_shape[-1] // batch_shape[-2] // batch_shape[-3],
                            indices[0] // batch_shape[-1] // batch_shape[-2] % batch_shape[-3],
                            indices[0] // batch_shape[-1] % batch_shape[-2],
                            indices[0] % batch_shape[-1],
                            *indices[-4:])

    elem_input = tvm.compute(shape_matmul,
                             lambda *indices: _batch_reshape(indices, elem_input),
                             name="broadcast_reshape_" + para_name,
                             tag="broadcast_reshape")

    return elem_input


def _do_align(tensor_need_align, shape_aligned, name, in_dtype):
    """
    do align for a_martix or b_martix, pad zero along the way.
    input:
        tensor_need_align: tensor, the tensor need align
        shape_aligned: shape, tensor_need_align's aligned shape
        name: str, a or b
        in_dtype: str, input data type
    return:
        aligned tensor
    """
    ax_outer = int(tensor_need_align.shape[0])
    ax_inner = int(tensor_need_align.shape[1])
    tensor_normalize_ub = tvm.compute(
        shape_aligned,
        lambda i, j: tvm.select(
            i < ax_outer,
            tvm.select(
                j < ax_inner,
                tensor_need_align[i, j],
                tvm.convert(0).astype(in_dtype)
            ),
            tvm.convert(0).astype(in_dtype)
        ),
        name="tensor_{}_normalize_ub".format(name)
    )
    return tensor_normalize_ub


def _check_shape_align(shape, factor):
    """
    Check that the shape is aligned
    input:
        shape: the shape for check
        factor: alignment factor
    return:
        is_align
    """
    is_align = True
    if in_dynamic():
        is_align = False
    for item in shape:
        if _get_value(item) % factor != 0:
            is_align = False
            break
    return is_align


def _get_block(dtype):
    """
    Get the number of elements in one block
    1 block = 32 byte

    Input: None
    ---------------------------------
    Return:
        block_reduce
        block_in
        block_out
    """
    if dtype == "float16":
        block_reduce = tbe_platform.BLOCK_REDUCE
    else:
        block_reduce = tbe_platform.BLOCK_REDUCE_INT8

    block_in = tbe_platform.BLOCK_IN
    block_out = tbe_platform.BLOCK_OUT
    return block_reduce, block_in, block_out


def _get_tensor_c_ub(  # pylint: disable=too-many-arguments
    tensor_c,
    out_shape,
    tensor_bias,
    tensor_alpha_ub,
    l0c_support_fp32,
    tensor_beta_bias_ub,
    dst_dtype,
    is_fractal_a,
    matmul_flag
):
    """calculate tensor_c_ub"""
    if not matmul_flag:
        tensor_c_before_mul_ub = tvm.compute(
            out_shape,
            lambda *indices: tensor_c(*indices),  # pylint: disable=W0108
            name="tensor_c_before_mul_ub"
        )
        if tensor_bias is not None:
            tensor_alpha_c_ub = tvm.compute(
                out_shape,
                lambda *indices: tensor_c_before_mul_ub(*indices) * tensor_alpha_ub[0],
                name="tensor_alpha_c_ub"
            )
            if not is_fractal_a:
                tensor_c_ub_temp = tvm.compute(
                    tensor_beta_bias_ub.shape,
                    lambda i, j: tensor_beta_bias_ub[i, j]
                                 + tensor_alpha_c_ub[j // 16, i // 16, i % 16, j % 16],
                    name="tensor_c_ub_temp"
                )
            else:
                tensor_c_ub_temp = tvm.compute(
                    out_shape,
                    lambda *indices: tensor_alpha_c_ub(*indices)
                                     + tensor_beta_bias_ub(*indices),
                    name="tensor_c_ub_temp"
                )
        else:
            tensor_c_ub_temp = tvm.compute(
                out_shape,
                lambda *indices: tensor_c_before_mul_ub(*indices) * tensor_alpha_ub[0],
                name="tensor_c_ub_temp"
            )
    else:
        tensor_c_ub_temp = tensor_c

    if dst_dtype == "float16" and l0c_support_fp32:
        tensor_c_ub = tvm.compute(
            tensor_c_ub_temp.shape,
            lambda *indices: tbe_utils.shape_util.cast(
                tensor_c_ub_temp(*indices),
                dtype="float16"
            ),
            name="tensor_c_ub"
        )
    elif dst_dtype == "float32" and l0c_support_fp32 and not is_fractal_a:
        tensor_c_ub = tensor_c_ub_temp
    elif dst_dtype == "int32" and not is_fractal_a:
        tensor_c_ub = tensor_c_ub_temp
    else:
        tensor_c_ub = tvm.compute(
            tensor_c_ub_temp.shape,
            lambda *indices: tensor_c_ub_temp(*indices),  # pylint: disable=W0108
            name="tensor_c_ub"
        )
    return tensor_c_ub


class GEMMComputeParam:
    """
    be used by gemm_tilingcase
    """
    tiling_info_dict = {}
    batch_a = False
    batch_b = False

    def __init__(self) -> None:
        pass


class GEMMCompute:
    """
    algorithm: General Matrix multiplication
    A' = transpose(A) if transA else A
    B' = transpose(B) if transB else B
    Compute Y = alpha * A' * B' + beta * C

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
    alpha: multiplier for the product of input tensors A * B.

    beta: multiplier for input tensor C.

    trans_a: if True, a needs to be transposed

    trans_b: if True, b needs to be transposed

    format_a: the format of tensor a

    format_b: the format of tensor b

    dst_dtype: output data type,support "float16" "float32", default is "float16"

    tensor_c: the tensor c

    format_out: output format, now support ND,Nz

    kernel_name: kernel name, default is "MatMul"

    quantize_params: quantization parameters,
    not None means enable quantization, it is dictionary structure

    quantize_alg: quantize mode,
    support 'NON_OFFSET' 'HALF_OFFSET_A' 'HALF_OFFSET_B' 'ALL_OFFSET'

    scale_mode_a: tensor_a inbound quantization mode,
    support 'SCALAR' and 'VECTOR'
    scale_mode_b: tensor_b inbound quantization mode,
    support 'SCALAR' and 'VECTOR'
    scale_mode_out: out tensor quantization mode,
    support 'SCALAR' and 'VECTOR'

    sqrt_mode_a: tensor_a inbound sqrt mode, support 'NON_SQRT' and 'SQRT'
    sqrt_mode_b: tensor_b inbound sqrt mode, support 'NON_SQRT' and 'SQRT'
    sqrt_mode_out: out tensor sqrt mode, support 'NON_SQRT' and 'SQRT'

    scale_q_a: scale placeholder for tensor_a inbound quantization
    offset_q_a: offset placeholder for tensor_a inbound quantization
    scale_q_b: scale placeholder for tensor_b inbound quantization
    offset_q_b: offset placeholder for tensor_b inbound quantization

    scale_drq: scale placeholder for requantization or dequantization
    offset_drq: scale placeholder for requantization or dequantization

    offset_a: the offset for tensor a

    offset_b: the offset for tensor b

    compress_index: index for compressed wights, None means not compress wights, now only for matmul

    op_type: op type

    Returns None
    """

    def __init__(self, tensor_a, tensor_b, para_dict):
        self.tensor_alpha = None
        self.tensor_beta = None
        self.in_a_dtype = None
        self.in_b_dtype = None
        self.is_nd_int82fp32 = None
        self.nz_a = None
        self.nz_b = None
        self.kn_shape = None
        self.m_shape = None
        self.m_shape_ori = None
        self.km_shape = None
        self.n_shape = None
        self.n_shape_ori = None
        self.kn_shape_ori = None
        self.matmul_batch = None
        self.is_fractal_a = None
        self.is_fractal_b = None
        self.out_shape = None
        self.tensor_a_length = None
        self.tensor_b_length = None
        self.tensor_a = tensor_a
        self.tensor_b = tensor_b
        self.block_reduce = None
        self.block_in = None
        self.block_out = None
        self.tensor_bias = para_dict.get("tensor_c")
        self.alpha = para_dict.get("alpha")
        self.beta = para_dict.get("beta")
        self.trans_a = para_dict.get("trans_a", False)
        self.trans_b = para_dict.get("trans_b", False)
        self.format_a = para_dict.get("format_a", "ND")
        self.format_b = para_dict.get("format_b", "ND")
        self.dst_dtype = para_dict.get("dst_dtype", "float16")
        self.quantize_params = para_dict.get("quantize_params")
        self.format_out = para_dict.get("format_out")
        self.compress_index = para_dict.get("compress_index")
        self.attrs = self._get_matmul_attrs(para_dict)
        self.kernel_name = para_dict.get("kernel_name", "gemm")
        self.op_type = para_dict.get("op_type", "")
        self._get_matmul_flag()

    def calculate(self):
        """
        Get calculate result of gemm and matmul

        Input: None
        ---------------------------------
        Return: result
        """
        if self.matmul_flag:
            tensor_y = matmul(tensor_a=self.tensor_a,
                              tensor_b=self.tensor_b,
                              trans_a=self.trans_a,
                              trans_b=self.trans_b,
                              format_a=self.format_a,
                              format_b=self.format_b,
                              dst_dtype=self.dst_dtype,
                              tensor_bias=self.tensor_bias,
                              quantize_params=self.quantize_params,
                              format_out=self.format_out,
                              compress_index=self.compress_index,
                              attrs=self.attrs,
                              kernel_name=self.kernel_name,
                              op_type=self.op_type)
        else:
            self._get_tensor_alpha_beta()
            tensor_y = self._gemm_compute()
        return tensor_y


    def _get_matmul_attrs(self, para_dict):
        """
        Get attrs for matmul compute
        Input: para_dict
        ---------------------------------
        Return: attrs
        """
        attrs = dict()
        offset_x = para_dict.get("offset_a")
        offset_w = para_dict.get("offset_b")
        batch_shape_a = para_dict.get("batch_shape_a")
        batch_shape_b = para_dict.get("batch_shape_b")
        batch_shape_out = para_dict.get("batch_shape_out")

        if offset_x:
            attrs["offset_x"] = offset_x
        if offset_w:
            attrs["offset_w"] = offset_w
        if batch_shape_a:
            attrs["batch_shape_a"] = batch_shape_a
        if batch_shape_b:
            attrs["batch_shape_b"] = batch_shape_b
        if batch_shape_out:
            attrs["batch_shape_out"] = batch_shape_out
        return attrs

    def _get_tensor_alpha_beta(self):
        """
        Get tensor of alpha and beta for gemm

        Input: None
        ---------------------------------
        Return: None
        """
        expected_type = (type(None), Tensor, float)
        if type(self.alpha) == type(None):
            self.tensor_alpha = None
        elif isinstance(self.alpha, float):
            self.tensor_alpha = tvm.placeholder([1], name="tensor_alpha", dtype=self.dst_dtype)
        elif isinstance(self.alpha, Tensor):
            self.tensor_alpha = self.alpha
        else:
            args_dict = {
                "errCode": "E60038",
                "desc": "Input parameter alpha type error, expected type is {}, " \
                    "actual input is {}".format(expected_type, type(self.alpha))
            }
            raise RuntimeError(args_dict, error_manager_util.get_error_message(args_dict))

        if type(self.beta) == type(None):
            self.tensor_beta = None
        elif isinstance(self.beta, float):
            self.tensor_beta = tvm.placeholder([1], name="tensor_beta", dtype=self.dst_dtype)
        elif isinstance(self.beta, Tensor):
            self.tensor_beta = self.beta
        else:
            args_dict = {
                "errCode": "E60038",
                "desc": "Input parameter beta type error, expected type is {}, " \
                    "actual input is {}".format(expected_type, type(self.beta))
            }
            raise RuntimeError(args_dict, error_manager_util.get_error_message(args_dict))

    def _get_matmul_flag(self):
        """
        Get flag of matmul compute
        if matmul_flag is True, then use the matmul compute

        Input: None
        ---------------------------------
        Return: None
        """
        self.matmul_flag = False
        if not (isinstance(self.alpha, Tensor) or isinstance(self.beta, Tensor)):
            self.matmul_flag = ((self.alpha is None or self.beta is None) or
                (math.isclose(self.alpha, 1.0) and math.isclose(self.beta, 1.0)))

    def _compute_alpha_beta(self):
        """
        The compute process of tensor alpha and beta in ub

        Input: None
        ---------------------------------
        Return:
            tensor_alpha_ub: tensor alpha in ub
            tensor_beta_ub: tensor beta in ub
        """
        if self.tensor_alpha.dtype == "float16":
            tensor_alpha_temp_ub = tvm.compute(
                self.tensor_alpha.shape,
                lambda *indices: self.tensor_alpha(*indices),  # pylint: disable=W0108
                name="tensor_alpha_temp_ub"
            )

            tensor_beta_temp_ub = tvm.compute(
                self.tensor_beta.shape,
                lambda *indices: self.tensor_beta(*indices),  # pylint: disable=W0108
                name="tensor_beta_temp_ub"
            )

            tensor_alpha_ub = tvm.compute(
                self.tensor_alpha.shape,
                lambda *indices: tbe_utils.shape_util.cast(
                    tensor_alpha_temp_ub(*indices), dtype="float32"
                ),
                name="tensor_alpha_ub"
            )
            tensor_beta_ub = tvm.compute(
                self.tensor_beta.shape,
                lambda *indices: tbe_utils.shape_util.cast(
                    tensor_beta_temp_ub(*indices), dtype="float32"
                ),
                name="tensor_beta_ub"
            )
        else:
            tensor_alpha_ub = tvm.compute(
                self.tensor_alpha.shape,
                lambda *indices: self.tensor_alpha(*indices),  # pylint: disable=W0108
                name="tensor_alpha_ub"
            )
            tensor_beta_ub = tvm.compute(
                self.tensor_beta.shape,
                lambda *indices: self.tensor_beta(*indices),  # pylint: disable=W0108
                name="tensor_beta_ub"
            )
        return tensor_alpha_ub, tensor_beta_ub

    def _get_dtype(self):
        """
        Get the dtype of tensor a , b and int8 to float32 flag

        Input: None
        ---------------------------------
        Return: None
        """
        if (not self.is_fractal_a and not self.is_fractal_b and self.tensor_a.dtype == "int8"
                and self.tensor_bias.dtype == "float32"):
            self.in_a_dtype = "float16"
            self.in_b_dtype = "float16"
            self.is_nd_int82fp32 = True
        else:
            self.in_a_dtype = self.tensor_a.dtype
            self.in_b_dtype = self.tensor_b.dtype
            self.is_nd_int82fp32 = False

    def _get_output_type(self):
        """
        Get the dtype of output

        Input: None
        ---------------------------------
        Return: out_dtype
        """
        l0c_support_fp32 = tbe_platform.intrinsic_check_support("Intrinsic_mmad", "f162f32")

        def _out_dtype():
            if self.in_a_dtype == "float16" and self.in_b_dtype == "float16":
                if self.dst_dtype not in ("float16", "float32"):
                    args_dict = {
                        "errCode": "E60003",
                        "a_dtype": self.in_a_dtype,
                        "expected_dtype_list": "float16, float32",
                        "out_dtype": "{}".format(self.dst_dtype)
                    }
                    raise RuntimeError(
                        args_dict, error_manager_util.get_error_message(args_dict)
                    )
                out_dtype = "float32"
                if not l0c_support_fp32:
                    out_dtype = "float16"
            elif ((self.in_a_dtype == "int8" and self.in_b_dtype == "int8")
                or (self.in_a_dtype == "uint8" and self.in_b_dtype == "int8")):
                out_dtype = "int32"
            else:
                args_dict = {
                    "errCode": "E60114",
                    "reason": "data type of tensor not supported",
                    "value": "in_a_dtype = {},"
                    " in_b_dtype = {}".format(self.in_a_dtype, self.in_b_dtype)
                }
                raise RuntimeError(
                    args_dict, error_manager_util.get_error_message(args_dict)
                )
            if (out_dtype == self.dst_dtype) and (self.quantize_params is not None):
                args_dict = {
                    "errCode": "E60000",
                    "param_name": "quantize_params",
                    "expected_value": "None",
                    "input_value": "{}".format(self.quantize_params)
                }
                raise RuntimeError(
                    args_dict, error_manager_util.get_error_message(args_dict)
                )

            if self.dst_dtype not in (out_dtype, "float16") and not (
                self.dst_dtype == "float32" and out_dtype == "int32"
            ):
                args_dict = {
                    "errCode": "E60114",
                    "reason": "y_dtype should be float16 for a_dtype ="
                    " {} and b_dtype = {}".format(self.in_a_dtype, self.in_b_dtype),
                    "value": self.dst_dtype
                }
                raise RuntimeError(
                    args_dict, error_manager_util.get_error_message(args_dict)
                )
            return out_dtype

        out_dtype = _out_dtype()

        if ((out_dtype not in (self.dst_dtype, "float32")) and (self.quantize_params is None)
                and not (self.dst_dtype == "float32" and out_dtype == "int32")):
            args_dict = {"errCode": "E60001", "param_name": "quantize_params"}
            raise RuntimeError(
                args_dict, error_manager_util.get_error_message(args_dict)
            )
        if (self.quantize_params is not None) and (not isinstance(self.quantize_params, dict)):
            args_dict = {
                "errCode": "E60005",
                "param_name": "quantize_params",
                "expected_dtype_list": "[dict]",
                "dtype": "{}".format(type(self.quantize_params))
            }
            raise RuntimeError(
                args_dict, error_manager_util.get_error_message(args_dict)
            )
        if self.in_a_dtype == "int8" and self.dst_dtype == "float32":
            out_dtype = "float32"
        return out_dtype

    def _get_bias_shape(self):
        """
        Get the shape of bias

        Input: None
        ---------------------------------
        Return:
            bias_shape: the shape of bias for calculation in Davinci
            origin_bias_shape: the origin shape of bias
        """
        if self.tensor_bias.dtype != self.dst_dtype:
            args_dict = {
                "errCode": "E60005",
                "param_name": "c",
                "expected_dtype_list": "[{}]".format(self.dst_dtype),
                "dtype": "{}".format(self.tensor_bias.dtype)
            }
            raise RuntimeError(
                args_dict, error_manager_util.get_error_message(args_dict)
            )
        bias_shape = list(self.tensor_bias.shape)
        if len(bias_shape) == 2:
            origin_bias_shape = bias_shape.copy()
            for index, value in enumerate(bias_shape):
                if index == 0:
                    block = tbe_platform.BLOCK_IN
                else:
                    block = tbe_platform.BLOCK_OUT
                bias_shape[index] = ((value + block - 1) // block) * block
        else:
            origin_bias_shape = None
        return bias_shape, origin_bias_shape

    def _get_a_martix_shape(self, gm_a_shape_normalize):  # pylint: disable=R0912
        """
        Get the martix shape of tensor a

        Input: gm_a_shape_normalize
        ---------------------------------
        Return:
            m_shape
            m_shape_ori
            km_shape
            gm_a_shape_normalize
        """
        if self.trans_a:
            if self.is_fractal_a:
                m_shape = _get_value(self.tensor_a.shape[self.tensor_a_length - 3])
                m_shape_ori = m_shape
                km_shape = _get_value(self.tensor_a.shape[self.tensor_a_length - 4])
                gm_a_shape_normalize = self.tensor_a.shape
            else:
                if self.is_nd_int82fp32:
                    m_shape = math.ceil(_get_value(self.tensor_a.shape[self.tensor_a_length - 1]) / 32) * 32 // 16
                    km_shape = math.ceil(_get_value(self.tensor_a.shape[self.tensor_a_length - 2]) / 32) * 32 // 16
                else:
                    if self.in_a_dtype == "int8":
                        m_shape = math.ceil(_get_value(self.tensor_a.shape[self.tensor_a_length - 1]) / 32) * 32 // 16
                    else:
                        m_shape = math.ceil(_get_value(self.tensor_a.shape[self.tensor_a_length - 1]) / self.block_in)
                    km_shape = math.ceil(_get_value(self.tensor_a.shape[self.tensor_a_length - 2]) / self.block_reduce)
                m_shape_ori = _get_value(self.tensor_a.shape[self.tensor_a_length - 1])
                gm_a_shape_normalize.append(km_shape * self.block_reduce)
                gm_a_shape_normalize.append(m_shape * self.block_in)
        else:
            if self.is_fractal_a:
                m_shape = _get_value(self.tensor_a.shape[self.tensor_a_length - 4])
                m_shape_ori = m_shape
                km_shape = _get_value(self.tensor_a.shape[self.tensor_a_length - 3])
                gm_a_shape_normalize = self.tensor_a.shape
            else:
                if self.is_nd_int82fp32:
                    m_shape = math.ceil(_get_value(self.tensor_a.shape[self.tensor_a_length - 2]) / 32) * 32 // 16
                    km_shape = math.ceil(_get_value(self.tensor_a.shape[self.tensor_a_length - 1]) / 32) * 32 // 16
                else:
                    if self.in_a_dtype == "int8":
                        m_shape = math.ceil(_get_value(self.tensor_a.shape[self.tensor_a_length - 2]) / 32) * 32 // 16
                    else:
                        m_shape = math.ceil(_get_value(self.tensor_a.shape[self.tensor_a_length - 2]) / self.block_in)
                    km_shape = math.ceil(_get_value(self.tensor_a.shape[self.tensor_a_length - 1]) / self.block_reduce)
                m_shape_ori = _get_value(self.tensor_a.shape[self.tensor_a_length - 2])
                gm_a_shape_normalize.append(m_shape * self.block_in)
                gm_a_shape_normalize.append(km_shape * self.block_reduce)

        return m_shape, m_shape_ori, km_shape, gm_a_shape_normalize

    def _get_b_martix_shape(self, gm_b_shape_normalize):
        """
        Get the martix shape of tensor b

        Input: gm_b_shape_normalize
        ---------------------------------
        Return:
            kn_shape
            n_shape
            n_shape_ori
            kn_shape_ori
            gm_b_shape_normalize
        """
        if self.trans_b:
            if self.is_fractal_b:
                kn_shape = _get_value(self.tensor_b.shape[self.tensor_b_length - 3])
                kn_shape_ori = kn_shape
                n_shape = _get_value(self.tensor_b.shape[self.tensor_b_length - 4])
                n_shape_ori = n_shape
                gm_b_shape_normalize = self.tensor_b.shape
            else:
                if self.is_nd_int82fp32:
                    kn_shape = math.ceil(_get_value(self.tensor_b.shape[self.tensor_b_length - 1]) / 32) * 32 // 16
                    n_shape = math.ceil(_get_value(self.tensor_b.shape[self.tensor_b_length - 2]) / 32) * 32 // 16
                else:
                    kn_shape = math.ceil(_get_value(self.tensor_b.shape[self.tensor_b_length - 1]) / self.block_reduce)
                    if self.in_b_dtype == "int8":
                        n_shape = math.ceil(_get_value(self.tensor_b.shape[self.tensor_b_length - 2]) / 32) * 32 // 16
                    else:
                        n_shape = math.ceil(_get_value(self.tensor_b.shape[self.tensor_b_length - 2]) / self.block_out)
                kn_shape_ori = _get_value(self.tensor_b.shape[self.tensor_b_length - 1])
                n_shape_ori = _get_value(self.tensor_b.shape[self.tensor_b_length - 2])
                gm_b_shape_normalize.append(n_shape * self.block_out)
                gm_b_shape_normalize.append(kn_shape * self.block_reduce)
        else:
            if self.is_fractal_b:
                kn_shape = _get_value(self.tensor_b.shape[self.tensor_b_length - 4])
                kn_shape_ori = kn_shape
                n_shape = _get_value(self.tensor_b.shape[self.tensor_b_length - 3])
                n_shape_ori = n_shape
                gm_b_shape_normalize = self.tensor_b.shape
            else:
                if self.is_nd_int82fp32:
                    kn_shape = math.ceil(_get_value(self.tensor_b.shape[self.tensor_b_length - 2]) / 32) * 32 // 16
                    n_shape = math.ceil(_get_value(self.tensor_b.shape[self.tensor_b_length - 1]) / 32) * 32 // 16
                else:
                    kn_shape = math.ceil(_get_value(self.tensor_b.shape[self.tensor_b_length - 2]) / self.block_reduce)
                    if self.in_b_dtype == "int8":
                        n_shape = math.ceil(_get_value(self.tensor_b.shape[self.tensor_b_length - 1]) / 32) * 32 // 16
                    else:
                        n_shape = math.ceil(_get_value(self.tensor_b.shape[self.tensor_b_length - 1]) / self.block_out)
                kn_shape_ori = _get_value(self.tensor_b.shape[self.tensor_b_length - 2])
                n_shape_ori = _get_value(self.tensor_b.shape[self.tensor_b_length - 1])
                gm_b_shape_normalize.append(kn_shape * self.block_reduce)
                gm_b_shape_normalize.append(n_shape * self.block_out)

        return kn_shape, n_shape, n_shape_ori, kn_shape_ori, gm_b_shape_normalize

    def _check_shape(self):
        """
        Check the legitimacy of shape

        Input: None
        ---------------------------------
        Return: None
        """
        if self.km_shape != self.kn_shape:
            args_dict = {
                "errCode": "E60002",
                "attr_name": "shape",
                "param1_name": "km_shape",
                "param1_value": "{}".format(self.km_shape),
                "param2_name": "self.kn_shape",
                "param2_value": "{}".format(self.kn_shape)
            }
            raise RuntimeError(
                args_dict, error_manager_util.get_error_message(args_dict)
            )

        if self.is_fractal_a:
            if self.trans_a:
                if not (
                    _get_value(self.tensor_a.shape[self.tensor_a_length - 1]) == self.block_reduce
                    and _get_value(self.tensor_a.shape[self.tensor_a_length - 2]) == self.block_in
                ):
                    args_dict = {
                        "errCode": "E60108",
                        "reason": "AShape classification matrix is wrong"
                    }
                    raise RuntimeError(
                        args_dict, error_manager_util.get_error_message(args_dict)
                    )
            else:
                if not (
                    _get_value(self.tensor_a.shape[self.tensor_a_length - 2]) == self.block_in
                    and _get_value(self.tensor_a.shape[self.tensor_a_length - 1]) == self.block_reduce
                ):
                    args_dict = {
                        "errCode": "E60108",
                        "reason": "AShape classification matrix is wrong"
                    }
                    raise RuntimeError(
                        args_dict, error_manager_util.get_error_message(args_dict)
                    )
        if self.is_fractal_b:
            if self.trans_b:
                if not (
                    _get_value(self.tensor_b.shape[self.tensor_b_length - 2]) == self.block_reduce
                    and _get_value(self.tensor_b.shape[self.tensor_b_length - 1]) == self.block_out
                ):
                    args_dict = {
                        "errCode": "E60108",
                        "reason": "BShape classification matrix is wrong"
                    }
                    raise RuntimeError(
                        args_dict, error_manager_util.get_error_message(args_dict)
                    )
            else:
                if not (
                    _get_value(self.tensor_b.shape[self.tensor_b_length - 2]) == self.block_out
                    and _get_value(self.tensor_b.shape[self.tensor_b_length - 1]) == self.block_reduce
                ):
                    args_dict = {
                        "errCode": "E60108",
                        "reason": "BShape classification matrix is wrong"
                    }
                    raise RuntimeError(
                        args_dict, error_manager_util.get_error_message(args_dict)
                    )

    def _get_reduce(self):
        """
        Get the kBurstAxis and kPointAxis

        Input: None
        ---------------------------------
        Return:
            reduce_kp: kPointAxis
            reduce_kb: kBurstAxis
        """

        if self.in_a_dtype == "int8" and self.dst_dtype == "float32":
            reduce_kp = tvm.reduce_axis((0, 16), name="kp")
            reduce_kb = tvm.reduce_axis((0, self.km_shape * 2), name="kb")
        else:
            reduce_kp = tvm.reduce_axis((0, self.block_reduce), name="kp")
            reduce_kb = tvm.reduce_axis((0, self.km_shape), name="kb")
        return reduce_kp, reduce_kb

    def _compute_bias(self, bias_shape, origin_bias_shape, tensor_beta_ub):
        """
        The compute process of bias

        Input:
            bias_shape: the shape of bias for calculation in Davinci
            origin_bias_shape: the origin shape of bias
            tensor_beta_ub: the tensor of beta in ub
        ---------------------------------
        Return:
            tensor_beta_bias_ub
        """
        tensor_beta_bias_ub = None
        if not self.matmul_flag:
            if len(bias_shape) == 2:
                if not self.is_fractal_a:
                    bias_m_shape_ori = self.tensor_bias.shape[0]
                    bias_n_shape_ori = self.tensor_bias.shape[1]
                    ub_bias_shape_normalize = [self.m_shape * self.block_in, self.n_shape * self.block_out]
                    tensor_bias_ub = tvm.compute(
                        ub_bias_shape_normalize,
                        lambda i, j: tvm.select(
                            i < bias_m_shape_ori,
                            tvm.select(
                                j < bias_n_shape_ori,
                                self.tensor_bias[i, j],
                                tvm.convert(0).astype(self.tensor_bias.dtype)
                            ),
                            tvm.convert(0).astype(self.tensor_bias.dtype)
                        ),
                        name="tensor_bias_ub"
                    )
                else:
                    tensor_bias_ub = tvm.compute(
                        bias_shape,
                        lambda i, j: tvm.select(
                            j < origin_bias_shape[-1],
                            tvm.select(
                                i < origin_bias_shape[-2],
                                self.tensor_bias[i, j],
                                tvm.convert(0).astype(self.dst_dtype)
                            ),
                            tvm.convert(0).astype(self.dst_dtype)
                        ),
                        name="tensor_bias_ub"
                    )
                    tensor_bias_ub = tvm.compute(
                        self.out_shape,
                        lambda i, j, k, l: tensor_bias_ub[
                            j * self.block_in + k, i * self.block_out + l
                        ]
                        + 0,
                        name="tensor_bias_ub_fract"
                    )
            elif len(bias_shape) == 4:
                tensor_bias_ub = tvm.compute(
                    self.out_shape,
                    lambda *indices: self.tensor_bias(*indices),  # pylint: disable=W0108
                    name="tensor_bias_ub"
                )

            if tensor_beta_ub.dtype == "float32" and tensor_bias_ub.dtype == "float16":
                tensor_float32_bias_ub = tvm.compute(
                    tensor_bias_ub.shape,
                    lambda *indices: tbe_utils.shape_util.cast(
                        tensor_bias_ub(*indices), dtype="float32"
                    ),
                    name="tensor_float32_bias_ub"
                )
                tensor_beta_bias_ub = tvm.compute(
                    tensor_bias_ub.shape,
                    lambda *indices: tensor_beta_ub[0]
                    * tensor_float32_bias_ub(*indices),
                    name="tensor_beta_bias_ub"
                )
            else:
                tensor_beta_bias_ub = tvm.compute(
                    tensor_bias_ub.shape,
                    lambda *indices: tensor_beta_ub[0] * tensor_bias_ub(*indices),
                    name="tensor_beta_bias_ub"
                )
        else:
            if self.tensor_bias is not None:
                tensor_beta_bias_ub = tvm.compute(
                    bias_shape, lambda i:
                    self.tensor_bias[i], name="tensor_bias_ub"
                )

        return tensor_beta_bias_ub

    def _a_nd_part_not_trans(self, gm_a_shape_normalize):
        """
        The compute process of a with ND part and not trans

        Input:
            gm_a_shape_normalize
        ---------------------------------
        Return:
            tensor_a_l0a
        """
        tensor_a_l1_shape = (self.m_shape, self.km_shape, self.block_in, self.block_reduce)
        if self.in_a_dtype == "int8":
            is_a_align = _check_shape_align(self.tensor_a.shape, 32)
            if not is_a_align:
                tensor_a_normalize_ub = _do_align(
                    self.tensor_a, gm_a_shape_normalize, "a", self.in_a_dtype
                )
            else:
                tensor_a_normalize_ub = tvm.compute(
                    gm_a_shape_normalize,
                    lambda i, j: self.tensor_a[i, j],
                    name="tensor_a_normalize_ub"
                )

            tensor_a_fract_k = tvm.compute(
                tensor_a_l1_shape,
                lambda i, j, k, l: tensor_a_normalize_ub[
                    i * self.block_in + k, j * self.block_reduce + l
                ],
                name="a_fract_k"
            )
            tensor_a_l1 = tvm.compute(
                tensor_a_l1_shape,
                lambda *indices: tensor_a_fract_k(*indices),  # pylint: disable=W0108
                name="tensor_a_l1"
            )
            tensor_a_l0a = tvm.compute(
                tensor_a_l1_shape,
                lambda *indices: tensor_a_l1(*indices),  # pylint: disable=W0108
                name="tensor_a_l0a"
            )
        else:
            if self.is_nd_int82fp32:
                is_a_align = _check_shape_align(self.tensor_a.shape, 32)
                if not is_a_align:
                    tensor_a_normalize_ub = _do_align(
                        self.tensor_a, gm_a_shape_normalize, "a", "int8"
                    )
                else:
                    tensor_a_normalize_ub = tvm.compute(
                        gm_a_shape_normalize,
                        lambda i, j: self.tensor_a[i, j],
                        name="tensor_a_normalize_ub"
                    )
                tensor_a_normalize_ub = tvm.compute(
                    gm_a_shape_normalize,
                    lambda *indices: tbe_utils.shape_util.cast(
                        tensor_a_normalize_ub(*indices), "float16"
                    ),
                    name="tensor_a_float16_normalize_ub"
                )
            else:
                is_a_align = _check_shape_align(self.tensor_a.shape, 16)
                if not is_a_align:
                    tensor_a_normalize_ub = _do_align(
                        self.tensor_a, gm_a_shape_normalize, "a", self.in_a_dtype
                    )
                else:
                    tensor_a_normalize_ub = tvm.compute(
                        gm_a_shape_normalize,
                        lambda i, j: self.tensor_a[i, j],
                        name="tensor_a_normalize_ub"
                    )
            tensor_a_fract_k_shape = (self.m_shape, self.km_shape * self.block_reduce, self.block_in)
            tensor_a_fract_k = tvm.compute(
                tensor_a_fract_k_shape,
                lambda i, j, k: tensor_a_normalize_ub[i * self.block_in + k, j],
                name="a_fract_k"
            )
            tensor_a_l1 = tvm.compute(
                tensor_a_fract_k_shape,
                lambda *indices: tensor_a_fract_k(*indices),  # pylint: disable=W0108
                name="tensor_a_l1"
            )
            tensor_a_l0a = tvm.compute(
                tensor_a_l1_shape,
                lambda i, j, k, l: tensor_a_l1[i, j * self.block_reduce + l, k],
                name="tensor_a_l0a"
            )
        return tensor_a_l0a

    def _a_part_not_trans(self, gm_a_shape_normalize):
        """
        The compute process of a with part and not trans

        Input:
            gm_a_shape_normalize
        ---------------------------------
        Return:
            tensor_a_l0a
        """
        if self.is_fractal_a:
            tensor_a_l1 = tvm.compute(
                gm_a_shape_normalize,
                lambda *indices: self.tensor_a(*indices),  # pylint: disable=W0108
                name="tensor_a_l1"
            )
            if GEMMComputeParam.batch_a:
                tensor_a_l0a = tvm.compute(
                    (self.matmul_batch, self.m_shape, self.km_shape, self.block_in, self.block_reduce),
                    lambda b, i, j, k, l: tensor_a_l1[b, i, j, l, k],
                    name="tensor_a_l0a"
                )
            else:
                tensor_a_l0a = tvm.compute(
                    (self.m_shape, self.km_shape, self.block_in, self.block_reduce),
                    lambda i, j, k, l: tensor_a_l1[i, j, l, k],
                    name="tensor_a_l0a"
                )
        else:
            tensor_a_l0a = self._a_nd_part_not_trans(gm_a_shape_normalize)
        return tensor_a_l0a

    def _compute_a_matrix(self, gm_a_shape_normalize):  # pylint: disable=too-many-branches
        """
        The compute process of matrix a

        Input:
            gm_a_shape_normalize
        ---------------------------------
        Input:
            tensor_a_l0a
        """
        if not self.trans_a:
            tensor_a_l0a = self._a_part_not_trans(gm_a_shape_normalize)
        else:
            def _part_trans():
                if self.is_fractal_a:
                    if self.in_a_dtype == "int8" and self.dst_dtype == "float32":
                        tensor_a_ub = tvm.compute(
                            gm_a_shape_normalize,
                            lambda *indices: self.tensor_a(  # pylint: disable=W0108
                                *indices
                            ),
                            name="tensor_a_ub"
                        )
                        tensor_float16_a_ub = tvm.compute(
                            gm_a_shape_normalize,
                            lambda *indices: tbe_utils.shape_util.cast(
                                tensor_a_ub(*indices), "float16"
                            ),
                            name="tensor_float16_a_ub"
                        )
                        new_a_shape = [
                            gm_a_shape_normalize[1],
                            gm_a_shape_normalize[0] * 2,
                            gm_a_shape_normalize[2],
                            gm_a_shape_normalize[3] // 2
                        ]
                        tensor_zz_a_ub = tvm.compute(
                            new_a_shape,
                            lambda i, j, k, l: tensor_float16_a_ub[
                                j // 2, i, k, (j * 16 + l) % 32
                            ],
                            name="tensor_zz_a_ub"
                        )
                        tensor_a_l1 = tvm.compute(
                            new_a_shape,
                            lambda *indices: tensor_zz_a_ub(  # pylint: disable=W0108
                                *indices
                            ),
                            name="tensor_a_l1"
                        )
                        tensor_a_l0a = tvm.compute(
                            new_a_shape,
                            lambda *indices: tensor_a_l1(  # pylint: disable=W0108
                                *indices
                            ),
                            name="tensor_a_l0a"
                        )
                    else:
                        tensor_a_l1 = tvm.compute(
                            gm_a_shape_normalize,
                            lambda *indices: self.tensor_a(  # pylint: disable=W0108
                                *indices
                            ),
                            name="tensor_a_l1"
                        )
                        if GEMMComputeParam.batch_a:
                            tensor_a_l0a = tvm.compute(
                                (self.matmul_batch, self.m_shape, self.km_shape, self.block_in, self.block_reduce),
                                lambda b, i, j, k, l: tensor_a_l1[b, j, i, k, l],
                                name="tensor_a_l0a",
                                attrs={"transpose_a": "true"}
                            )
                        else:
                            tensor_a_l0a = tvm.compute(
                                (self.m_shape, self.km_shape, self.block_in, self.block_reduce),
                                lambda i, j, k, l: tensor_a_l1[j, i, k, l],
                                name="tensor_a_l0a",
                                attrs={"transpose_a": "true"}
                            )
                else:
                    if self.in_a_dtype == "float16":
                        tensor_a_l1_shape = (self.m_shape, self.km_shape, self.block_in, self.block_reduce)
                        if self.is_nd_int82fp32:
                            is_a_align = _check_shape_align(self.tensor_a.shape, 32)
                            if not is_a_align:
                                tensor_a_normalize_ub = _do_align(
                                    self.tensor_a, gm_a_shape_normalize, "a", "int8"
                                )
                            else:
                                tensor_a_normalize_ub = tvm.compute(
                                    gm_a_shape_normalize,
                                    lambda i, j: self.tensor_a[i, j],
                                    name="tensor_a_normalize_ub"
                                )
                            tensor_a_normalize_ub = tvm.compute(
                                gm_a_shape_normalize,
                                lambda *indices: tbe_utils.shape_util.cast(
                                    tensor_a_normalize_ub(*indices), "float16"
                                ),
                                name="tensor_a_float16_normalize_ub"
                            )
                        else:
                            is_a_align = _check_shape_align(self.tensor_a.shape, 16)
                            if not is_a_align:
                                tensor_a_normalize_ub = _do_align(
                                    self.tensor_a, gm_a_shape_normalize, "a", self.in_a_dtype
                                )
                            else:
                                tensor_a_normalize_ub = tvm.compute(
                                    gm_a_shape_normalize,
                                    lambda i, j: self.tensor_a[i, j],
                                    name="tensor_a_normalize_ub"
                                )
                        tensor_a_fract_k_shape = (
                            self.km_shape,
                            self.m_shape * self.block_in,
                            self.block_reduce
                        )
                        tensor_a_fract_k = tvm.compute(
                            tensor_a_fract_k_shape,
                            lambda i, j, k: tensor_a_normalize_ub[
                                i * self.block_reduce + k, j
                            ],
                            name="a_fract_k"
                        )
                        tensor_a_l1 = tvm.compute(
                            tensor_a_fract_k_shape,
                            lambda *indices: tensor_a_fract_k(  # pylint: disable=W0108
                                *indices
                            ),
                            name="tensor_a_l1"
                        )
                        tensor_a_l0a = tvm.compute(
                            tensor_a_l1_shape,
                            lambda i, j, k, l: tensor_a_l1[j, i * self.block_in + k, l],
                            name="tensor_a_l0a",
                            attrs={"transpose_a": "true"}
                        )
                    else:
                        is_a_align = _check_shape_align(self.tensor_a.shape, 32)
                        tensor_a_fract_shape = (
                            self.m_shape,
                            self.km_shape,
                            self.block_in,
                            self.block_reduce
                        )
                        if not is_a_align:
                            tensor_a_normalize_ub = _do_align(
                                self.tensor_a, gm_a_shape_normalize, "a", self.in_a_dtype
                            )
                        else:
                            tensor_a_normalize_ub = tvm.compute(
                                gm_a_shape_normalize,
                                lambda *indices: self.tensor_a(  # pylint: disable=W0108
                                    *indices
                                ),
                                name="tensor_a_normalize_ub"
                            )
                        tensor_a_transpose_shape = (
                            self.m_shape * self.block_in,
                            self.km_shape * self.block_reduce
                        )
                        tensor_a_transpose = tvm.compute(
                            tensor_a_transpose_shape,
                            lambda i, j: tensor_a_normalize_ub[j, i],
                            name="a_transpose"
                        )
                        tensor_a_fract = tvm.compute(
                            tensor_a_fract_shape,
                            lambda i, j, k, l: tensor_a_transpose[
                                i * self.block_in + k, j * self.block_reduce + l
                            ],
                            name="a_fract_k"
                        )
                        tensor_a_l1 = tvm.compute(
                            tensor_a_fract_shape,
                            lambda *indices: tensor_a_fract(  # pylint: disable=W0108
                                *indices
                            ),
                            name="tensor_a_l1"
                        )
                        tensor_a_l0a = tvm.compute(
                            tensor_a_fract_shape,
                            lambda *indices: tensor_a_l1(  # pylint: disable=W0108
                                *indices
                            ),
                            name="tensor_a_l0a",
                            attrs={"transpose_a": "true"}
                        )
                return tensor_a_l0a
            tensor_a_l0a = _part_trans()
        return tensor_a_l0a

    def _b_nd_part_not_trans(self, gm_b_shape_normalize):
        """
        The compute process of b with nd part and not trans

        Input:
            gm_b_shape_normalize
        ---------------------------------
        Input:
            tensor_b_l0b
        """
        if self.in_b_dtype == "int8":
            is_b_align = _check_shape_align(self.tensor_b.shape, 32)
            tensor_b_l1_shape = (self.kn_shape, self.n_shape, self.block_out, self.block_reduce)
            tensor_b_ub_shape = (self.kn_shape * self. block_reduce, self.n_shape * self.block_out)
            if is_b_align is False:
                tensor_b_normalize_ub = _do_align(
                    self.tensor_b, gm_b_shape_normalize, "b", "int8"
                )
            else:
                tensor_b_normalize_ub = tvm.compute(
                    tensor_b_ub_shape,
                    lambda i, j: self.tensor_b[i, j],
                    name="tensor_b_normalize_ub"
                )
            tensor_b_transpose_shape = (self.n_shape * self.block_out, self.kn_shape * self.block_reduce)
            tensor_b_transpose = tvm.compute(
                tensor_b_transpose_shape,
                lambda i, j: tensor_b_normalize_ub[j, i],
                name="b_transpose"
            )
            tensor_b_fract = tvm.compute(
                (self.kn_shape, self.n_shape, self.block_out, self.block_reduce),
                lambda i, j, k, l: tensor_b_transpose[
                    j * self.block_in + k, i * self.block_reduce + l
                ],
                name="b_fract"
            )
            tensor_b_l1 = tvm.compute(
                tensor_b_l1_shape,
                lambda *indices: tensor_b_fract(*indices),  # pylint: disable=W0108
                name="tensor_b_l1"
            )
            tensor_b_l0b = tvm.compute(
                (self.kn_shape, self.n_shape, self.block_out, self.block_reduce),
                lambda i, j, k, l: tensor_b_l1[i, j, k, l],
                name="tensor_b_l0b"
            )
        else:
            if self.is_nd_int82fp32:
                is_b_align = _check_shape_align(self.tensor_b.shape, 32)
                if not is_b_align:
                    tensor_b_normalize_ub = _do_align(
                        self.tensor_b, gm_b_shape_normalize, "b", "int8"
                    )
                else:
                    tensor_b_normalize_ub = tvm.compute(
                        gm_b_shape_normalize,
                        lambda i, j: self.tensor_b[i, j],
                        name="tensor_b_normalize_ub"
                    )
                tensor_b_normalize_ub = tvm.compute(
                    gm_b_shape_normalize,
                    lambda *indices: tbe_utils.shape_util.cast(
                        tensor_b_normalize_ub(*indices), "float16"
                    ),
                    name="tensor_b_float16_normalize_ub"
                )
            else:
                is_b_align = _check_shape_align(self.tensor_b.shape, 16)
                if not is_b_align:
                    tensor_b_normalize_ub = _do_align(
                        self.tensor_b, gm_b_shape_normalize, "b", self.in_b_dtype
                    )
                else:
                    tensor_b_normalize_ub = tvm.compute(
                        gm_b_shape_normalize,
                        lambda i, j: self.tensor_b[i, j],
                        name="tensor_b_normalize_ub"
                    )

            tensor_b_fract_shape = (self.kn_shape, self.n_shape * self.block_out, self.block_reduce)
            tensor_b_fract = tvm.compute(
                tensor_b_fract_shape,
                lambda i, j, k: tensor_b_normalize_ub[i * self.block_reduce + k, j],
                name="b_fract"
            )
            tensor_b_l1 = tvm.compute(
                tensor_b_fract_shape,
                lambda *indices: tensor_b_fract(*indices),  # pylint: disable=W0108
                name="tensor_b_l1"
            )
            tensor_b_l0b = tvm.compute(
                (self.kn_shape, self.n_shape, self.block_out, self.block_reduce),
                lambda i, j, k, l: tensor_b_l1[i, j * self.block_reduce + k, l],
                name="tensor_b_l0b"
            )
        return tensor_b_l0b

    def _b_part_not_trans(self, gm_b_shape_normalize):
        """
        The compute process of b with part and not trans

        Input:
            gm_b_shape_normalize
        ---------------------------------
        Input:
            tensor_b_l0b
        """
        if self.is_fractal_b:
            if self.nz_b:
                tensor_b_l1 = tvm.compute(
                    self.tensor_b.shape,
                    lambda *indices: self.tensor_b(*indices),  # pylint: disable=W0108
                    name="tensor_b_l1"
                )
                tensor_b_l0b = tvm.compute(
                    self.tensor_b.shape,
                    lambda *indices: tensor_b_l1(*indices),  # pylint: disable=W0108
                    name="tensor_b_l0b"
                )
            else:
                if self.in_b_dtype == "int8" and self.dst_dtype == "float32":
                    tensor_b_ub = tvm.compute(
                        self.tensor_b.shape,
                        lambda *indices: self.tensor_b(*indices),  # pylint: disable=W0108
                        name="tensor_b_ub"
                    )
                    tensor_float16_b_ub = tvm.compute(
                        self.tensor_b.shape,
                        lambda *indices: tbe_utils.shape_util.cast(
                            tensor_b_ub(*indices), "float16"
                        ),
                        name="tensor_float16_b_ub"
                    )
                    new_b_shape = [
                        self.tensor_b.shape[0] * 2,
                        self.tensor_b.shape[1],
                        self.tensor_b.shape[2],
                        self.tensor_b.shape[3] // 2
                    ]
                    tensor_zn_b_ub = tvm.compute(
                        new_b_shape,
                        lambda i, j, k, l: tensor_float16_b_ub[
                            i // 2, j, k, (i * 16 + l) % 32
                        ],
                        name="tensor_zn_b_ub"
                    )
                    tensor_b_l1 = tvm.compute(
                        new_b_shape,
                        lambda *indices: tensor_zn_b_ub(  # pylint: disable=W0108
                            *indices
                        ),
                        name="tensor_b_l1"
                    )
                    tensor_b_l0b = tvm.compute(
                        new_b_shape,
                        lambda *indices: tensor_b_l1(*indices),  # pylint: disable=W0108
                        name="tensor_b_l0b"
                    )
                else:
                    tensor_b_l1 = tvm.compute(
                        self.tensor_b.shape,
                        lambda *indices: self.tensor_b(*indices),  # pylint: disable=W0108
                        name="tensor_b_l1"
                    )
                    tensor_b_l0b = tvm.compute(
                        self.tensor_b.shape,
                        lambda *indices: tensor_b_l1(*indices),  # pylint: disable=W0108
                        name="tensor_b_l0b"
                    )
        else:
            tensor_b_l0b = self._b_nd_part_not_trans(gm_b_shape_normalize)
        return tensor_b_l0b

    def _nd_part_trans(self, gm_b_shape_normalize):
        """
        The compute process of b with nd part and trans

        Input:
            gm_b_shape_normalize
        ---------------------------------
        Input:
            tensor_b_l0b
        """
        if self.in_b_dtype == "float16":
            if self.is_nd_int82fp32:
                is_b_align = _check_shape_align(self.tensor_b.shape, 32)
                if not is_b_align:
                    tensor_b_normalize_ub = _do_align(
                        self.tensor_b, gm_b_shape_normalize, "b", "int8"
                    )
                else:
                    tensor_b_normalize_ub = tvm.compute(
                        gm_b_shape_normalize,
                        lambda i, j: self.tensor_b[i, j],
                        name="tensor_b_normalize_ub"
                    )
                if not self.trans_a and self.trans_b:
                    transpose_shape = gm_b_shape_normalize[::-1]
                    tensor_b_transpose_ub = tvm.compute(
                        transpose_shape,
                        lambda i, j: tensor_b_normalize_ub[j, i],
                        name="b_transpose_only"
                    )
                    tensor_b_transpose_zero_ub = tvm.compute(
                        transpose_shape,
                        lambda i, j: tvm.select(
                            i < self.kn_shape_ori,
                            tensor_b_transpose_ub[i, j],
                            tvm.const(0).astype("int8")
                        ),
                        name="b_transpose_zero"
                    )
                    tensor_b_normalize_ub = tvm.compute(
                        gm_b_shape_normalize,
                        lambda i, j: tensor_b_transpose_zero_ub[j, i],
                        name="b_after_process"
                    )
                tensor_b_normalize_ub = tvm.compute(
                    gm_b_shape_normalize,
                    lambda *indices: tbe_utils.shape_util.cast(
                        tensor_b_normalize_ub(*indices), "float16"
                    ),
                    name="tensor_b_float16_normalize_ub"
                )
            else:
                is_b_align = _check_shape_align(self.tensor_b.shape, 16)
                if not is_b_align:
                    tensor_b_normalize_ub = _do_align(
                        self.tensor_b, gm_b_shape_normalize, "b", self.in_b_dtype
                    )
                else:
                    tensor_b_normalize_ub = tvm.compute(
                        gm_b_shape_normalize,
                        lambda i, j: self.tensor_b[i, j],
                        name="tensor_b_normalize_ub"
                    )
            tensor_b_fract_shape = (self.n_shape, self.kn_shape * self.block_reduce, self.block_out)
            tensor_b_fract = tvm.compute(
                tensor_b_fract_shape,
                lambda i, j, k: tensor_b_normalize_ub[i * self.block_out + k, j],
                name="b_fract"
            )
            tensor_b_l1 = tvm.compute(
                tensor_b_fract_shape,
                lambda *indices: tensor_b_fract(*indices),  # pylint: disable=W0108
                name="tensor_b_l1"
            )
            tensor_b_l0b = tvm.compute(
                (self.kn_shape, self.n_shape, self.block_out, self.block_reduce),
                lambda i, j, k, l: tensor_b_l1[j, i * self.block_reduce + l, k],
                name="tensor_b_l0b",
                attrs={"transpose_b": "true"}
            )
        else:
            is_b_align = _check_shape_align(self.tensor_b.shape, 32)
            tensor_b_fract_shape = (
                self.kn_shape,
                self.n_shape,
                self.block_out,
                self.block_reduce
            )
            if not is_b_align:
                tensor_b_normalize_ub = _do_align(
                    self.tensor_b, gm_b_shape_normalize, "b", self.in_b_dtype
                )
            else:
                tensor_b_normalize_ub = tvm.compute(
                    gm_b_shape_normalize,
                    lambda *indices: self.tensor_b(*indices),  # pylint: disable=W0108
                    name="tensor_b_normalize_ub"
                )
            if not self.trans_a and self.trans_b:
                transpose_shape = gm_b_shape_normalize[::-1]
                tensor_b_transpose_ub = tvm.compute(
                    transpose_shape,
                    lambda i, j: tensor_b_normalize_ub[j, i],
                    name="b_transpose_only"
                )
                tensor_b_transpose_zero_ub = tvm.compute(
                    transpose_shape,
                    lambda i, j: tvm.select(
                        i < self.kn_shape_ori,
                        tensor_b_transpose_ub[i, j],
                        tvm.const(0).astype(self.in_b_dtype)
                    ),
                    name="b_transpose_zero"
                )
                tensor_b_normalize_ub = tvm.compute(
                    gm_b_shape_normalize,
                    lambda i, j: tensor_b_transpose_zero_ub[j, i],
                    name="b_after_process"
                )
            tensor_b_fract = tvm.compute(
                tensor_b_fract_shape,
                lambda i, j, k, l: tensor_b_normalize_ub[
                    j * self.block_out + k, i * self.block_reduce + l
                ],
                name="b_fract"
            )
            tensor_b_l1 = tvm.compute(
                tensor_b_fract_shape,
                lambda *indices: tensor_b_fract(*indices),  # pylint: disable=W0108
                name="tensor_b_l1"
            )
            tensor_b_l0b = tvm.compute(
                tensor_b_fract_shape,
                lambda *indices: tensor_b_l1(*indices),  # pylint: disable=W0108
                name="tensor_b_l0b",
                attrs={"transpose_b": "true"}
            )
        return tensor_b_l0b

    def _compute_b_matrix(self, gm_b_shape_normalize):  # pylint: disable=too-many-branches
        """
        The compute process of B martix

        Input:
            gm_b_shape_normalize
        ---------------------------------
        Input:
            tensor_b_l0b
        """
        if not self.trans_b:
            tensor_b_l0b = self._b_part_not_trans(gm_b_shape_normalize)
        else:
            def _part_trans():
                if self.is_fractal_b:
                    if self.nz_b:
                        tensor_b_l1 = tvm.compute(
                            self.tensor_b.shape,
                            lambda *indices: self.tensor_b(  # pylint: disable=W0108
                                *indices
                            ),
                            name="tensor_b_l1"
                        )
                        if GEMMComputeParam.batch_b:
                            tensor_b_l0b = tvm.compute(
                                (self.matmul_batch, self.kn_shape, self.n_shape, self.block_out, self.block_reduce),
                                lambda b, i, j, k, l: tensor_b_l1[b, j, i, l, k],
                                name="tensor_b_l0b",
                                attrs={"transpose_b": "true"}
                            )
                        else:
                            tensor_b_l0b = tvm.compute(
                                (self.kn_shape, self.n_shape, self.block_out, self.block_reduce),
                                lambda i, j, k, l: tensor_b_l1[j, i, l, k],
                                name="tensor_b_l0b",
                                attrs={"transpose_b": "true"}
                            )
                    else:
                        tensor_b_l1 = tvm.compute(
                            self.tensor_b.shape,
                            lambda *indices: self.tensor_b(  # pylint: disable=W0108
                                *indices
                            ),
                            name="tensor_b_l1"
                        )
                        tensor_b_l0b = tvm.compute(
                            (self.kn_shape, self.n_shape, self.block_out, self.block_reduce),
                            lambda i, j, k, l: tensor_b_l1[j, i, l, k],
                            name="tensor_b_l0b"
                        )
                else:
                    tensor_b_l0b = self._nd_part_trans(gm_b_shape_normalize)
                return tensor_b_l0b

            tensor_b_l0b = _part_trans()

        return tensor_b_l0b

    def _compute_c_martix(self, tensor_a_l0a, tensor_b_l0b, reduce_kb, reduce_kp, tensor_alpha_ub,
                          tensor_beta_bias_ub):
        """
        The compute process of C martix

        Input:
            tensor_a_l0a: tensor a in l0a
            tensor_b_l0b: tensor b in l0b
            reduce_kb
            reduce_kp
            tensor_alpha_ub: tensor alpha in ub
            tensor_beta_bias_ub: tensor beta*bias in ub
        ---------------------------------
        Input:
            tensor_c_gm
        """

        l0c_support_fp32 = tbe_platform.intrinsic_check_support("Intrinsic_mmad", "f162f32")

        if self.block_in != tbe_platform.BLOCK_VECTOR:  # gemm
            # define mad compute
            if GEMMComputeParam.batch_a:
                tensor_c = tvm.compute(
                self.out_shape,
                lambda batch, nb, mb, mp, np: tvm.sum(
                    (
                        tensor_a_l0a[batch, mb, reduce_kb, mp, reduce_kp]
                        * (tensor_b_l0b[batch, reduce_kb, nb, np, reduce_kp]
                        if GEMMComputeParam.batch_b else tensor_b_l0b[reduce_kb, nb, np, reduce_kp])
                    ).astype(self.out_dtype),
                    axis=[reduce_kb, reduce_kp]
                ),
                name="tensor_c",
                attrs={"input_order": "positive"}
                )
            else:
                tensor_c = tvm.compute(
                    self.out_shape,
                    lambda nb, mb, mp, np: tvm.sum((
                            tensor_a_l0a[mb, reduce_kb, mp, reduce_kp]
                            * tensor_b_l0b[reduce_kb, nb, np, reduce_kp]
                        ).astype(self.out_dtype),
                        axis=[reduce_kb, reduce_kp]
                    ),
                    name="tensor_c",
                    attrs={"input_order": "positive"}
                )
            if self.matmul_flag and self.tensor_bias is not None:
                if self.tensor_bias.dtype == "float16" and l0c_support_fp32 == 1:
                    if GEMMComputeParam.batch_a:
                        tensor_bias_l0c = tvm.compute(
                            self.out_shape, lambda b, i, j, k, l: tbe_utils.shape_util.cast(
                                tensor_beta_bias_ub[i * self.block_out + l], dtype="float32"),
                                name="tensor_bias_l0c"
                        )
                    else:
                        tensor_bias_l0c = tvm.compute(
                            self.out_shape, lambda i, j, k, l: tbe_utils.shape_util.cast(
                                tensor_beta_bias_ub[i * self.block_out + l], dtype="float32"),
                                name="tensor_bias_l0c"
                        )
                else:
                    if GEMMComputeParam.batch_a:
                        tensor_bias_l0c = tvm.compute(
                            self.out_shape,
                            lambda b, i, j, k, l: tensor_beta_bias_ub[i * self.block_out + l],
                            name="tensor_bias_l0c"
                        )
                    else:
                        tensor_bias_l0c = tvm.compute(
                            self.out_shape,
                            lambda i, j, k, l: tensor_beta_bias_ub[i * self.block_out + l],
                            name="tensor_bias_l0c"
                        )
                tensor_c = tvm.compute(
                    self.out_shape,
                    lambda *indices: tensor_bias_l0c[indices] + tensor_c[indices],
                    name="tensor_c_add_bias"
                )
            tensor_c_ub = _get_tensor_c_ub(
                tensor_c,
                self.out_shape,
                self.tensor_bias,
                tensor_alpha_ub,
                l0c_support_fp32,
                tensor_beta_bias_ub,
                self.dst_dtype,
                self.is_fractal_a,
                self.matmul_flag
            )

            if self.is_fractal_a and self.is_fractal_b:
                tensor_c_gm = tvm.compute(
                    self.out_shape,
                    lambda *indices: tensor_c_ub(*indices),  # pylint: disable=W0108
                    name="tensor_c_gm",
                    tag="gemm",
                    attrs={"kernel_name": self.kernel_name}
                )
            else:
                # ND out shape is dim 2, shape m is original value
                tensor_c_gm = tvm.compute(
                    self.out_shape_ori,
                    lambda i, j: tvm.select(
                        i < self.m_shape_ori,
                        tvm.select(j < self.n_shape_ori, tensor_c_ub[i, j])
                    ),
                    name="tensor_c_gm",
                    tag="gemm",
                    attrs={"kernel_name": self.kernel_name}
                )
        return tensor_c_gm

    def _gemm_compute(self):
        """
        The compute process of gemm and dynamic matmul

        Input:None
        ---------------------------------
        Return: tensor_c_gm: gemm_result
        """
        self.nz_a = False
        if self.format_a == "FRACTAL_NZ":
            self.nz_a = True

        self.nz_b = False
        if self.format_b == "FRACTAL_NZ":
            self.nz_b = True

        self.tensor_a_length = len(self.tensor_a.shape)
        self.tensor_b_length = len(self.tensor_b.shape)

        GEMMComputeParam.batch_a = (self.tensor_a_length == 5)
        GEMMComputeParam.batch_b = (self.tensor_b_length == 5)

        tensor_alpha_ub, tensor_beta_ub = None, None
        if not self.matmul_flag:
            tensor_alpha_ub, tensor_beta_ub = self._compute_alpha_beta()

        _shape_check(self.tensor_a, self.tensor_b, self.tensor_bias, self.tensor_alpha, self.tensor_beta, self.trans_a,
                     self.trans_b, self.format_a, self.format_b, self.dst_dtype, self.matmul_flag)

        self.is_fractal_a = self.format_a != "ND"
        self.is_fractal_b = self.format_b != "ND"

        self._get_dtype()

        if self.tensor_bias is not None:
            bias_shape, origin_bias_shape = self._get_bias_shape()
        else:
            bias_shape, origin_bias_shape = None, None

        self.block_reduce, self.block_in, self.block_out = _get_block(self.in_a_dtype)

        gm_a_shape_normalize = []
        (self.m_shape, self.m_shape_ori, self.km_shape,
         gm_a_shape_normalize) = self._get_a_martix_shape(gm_a_shape_normalize)

        gm_b_shape_normalize = []
        (self.kn_shape, self.n_shape, self.n_shape_ori, self.kn_shape_ori,
         gm_b_shape_normalize) = self._get_b_martix_shape(gm_b_shape_normalize)

        self._check_shape()

        reduce_kp, reduce_kb = self._get_reduce()

        self.out_shape = (_get_value(self.n_shape), _get_value(self.m_shape),
                          _get_value(self.block_in), _get_value(self.block_out))
        self.out_shape_ori = [_get_value(self.m_shape_ori), _get_value(self.n_shape_ori)]

        if GEMMComputeParam.batch_a:
            self.out_shape = list(self.out_shape)
            self.matmul_batch = _get_value(self.tensor_a.shape[0])
            self.out_shape.insert(0, self.matmul_batch)
            self.out_shape_ori.insert(0, self.matmul_batch)

        tensor_beta_bias_ub = self._compute_bias(bias_shape, origin_bias_shape, tensor_beta_ub)

        tensor_a_l0a = self._compute_a_matrix(gm_a_shape_normalize)

        tensor_b_l0b = self._compute_b_matrix(gm_b_shape_normalize)

        self.out_dtype = self._get_output_type()

        tensor_c_gm = self._compute_c_martix(tensor_a_l0a, tensor_b_l0b, reduce_kb, reduce_kp, tensor_alpha_ub,
                                             tensor_beta_bias_ub)

        return tensor_c_gm

