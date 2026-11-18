#!/usr/bin/env python
# -*- coding:utf-8 -*-
"""
Copyright (C) 2019. Huawei Technologies Co., Ltd. All rights reserved.
FILE:     tik_scalar_api.py
DESC:     provide scalar instructions
CREATED:  2019-08-12 18:53:42
MODIFIED: 2020-12-7 19:17:00
"""

from tbe.tvm import call_extern
from tbe.common.platform import intrinsic_check_support
from tbe.common.platform.platform_info import api_check_support
from tbe.tik.api.tik_scalar import Scalar
from tbe.tik.debug.tik_vector_ops_debug.tik_scalar_api_debug import scalar_binary_decorator
from tbe.tik.debug.tik_vector_ops_debug.tik_scalar_api_debug import scalar_single_decorator
from tbe.tik.debug.tik_vector_ops_debug.tik_scalar_api_debug import scalar_conv_decorator
from tbe.tik.tik_lib.tik_util import dtype_convert
from tbe.tik.api.tik_ir_builder import TikIRBuilder
from tbe.tik.tik_lib.tik_check_util import TikCheckUtil
from tbe.tik.tik_lib.tik_params import gen_api_check_statement
from tbe.tik.tik_lib.tik_api_constants import DTYPE_MAP
from tbe.tik.tik_lib.tik_api_constants import ROUND_MODE_MAP
from tbe.tik.tik_lib.tik_source_info import source_info_decorator


class TikScalarApi(TikIRBuilder):
    """
    Scalar Operation Api

    Single operand instruction: abs sqrt countbit0 countbit1 countleading0 conv
    Double operand instruction: max min conv
    """

    def scalar_sqrt(self, dst, src):
        """
        Extracts the square root of a scalar
        Parameters
        ----------
        dst : Destination operand.
        src : Source operand.

        Returns
        -------
        no return
        """
        return self._scalar_single_func('sqrt', dst, src)

    def scalar_abs(self, dst, src):
        """
        Obtains the absolute value of a scalar.
        Parameters
        ----------
        dst : Destination operand.
        src : Source operand.

        Returns
        -------
        no return
        """
        return self._scalar_single_func('abs', dst, src)

    def scalar_countbit0(self, dst, src):
        """
        Counts the number of bits whose values are 0 in the 64-bit binary format of the source operand bitwise.
        Parameters
        ----------
        dst : A scalar of type uint64, for the destination operand
        src : A scalar or an immediate of type uint64, for the source operand

        Returns
        -------
        no return
        """
        return self._scalar_single_func('bcnt0', dst, src)

    def scalar_countbit1(self, dst, src):
        """
        Counts the number of bits whose values are 1 in the 64-bit binary format of the source operand bitwise.
        Parameters
        ----------
        dst : A scalar of type uint64, specifying the destination operand.
        src : A scalar or an immediate of type uint64, specifying the source operand.

        Returns
        -------
        no return
        """
        return self._scalar_single_func('bcnt1', dst, src)

    def scalar_countleading0(self, dst, src):
        """
        Counts the number of consecutive bits whose values are 0 in the 64-bit binary format of the source operand.
        Parameters
        ----------
        dst : A scalar of type uint64, specifying the destination operand.
        src : A scalar or an immediate of type uint64, specifying the source operand.

        Returns
        -------
        no return
        """
        return self._scalar_single_func('clz', dst, src)

    def scalar_max(self, dst, src0, src1):
        """
        Compares two source operands and returns the maximum
        Parameters
        ----------
        dst : A scalar of type int64, specifying the destination operand.
        src0 : A scalar or an immediate of type int64, specifying the source operand 0.
        src1 : A scalar or an immediate of type int64, specifying the source operand 1.

        Returns
        -------
        no return
        """
        return self._scalar_binary_func('max', dst, src0, src1)

    def scalar_min(self, dst, src0, src1):
        """
        Compares two source operands and returns the minimum

        Parameters
        ----------
        dst : A scalar of type int64, specifying the destination operand.
        src0 : A scalar or an immediate of type int64, specifying the source operand 0.
        src1 : A scalar or an immediate of type int64, specifying the source operand 1.

        Returns
        -------
        no return
        """
        return self._scalar_binary_func('min', dst, src0, src1)

    @source_info_decorator()
    @scalar_conv_decorator
    def scalar_conv(self, round_mode, dst, src):
        """
        Converts the scalar precision

        Parameters
        ----------
        dst : A scalar of type float32/float16/int32, for the destination operand
        round_mode : Rounding mode
        src : A scalar of type float32/float16/int32, for the source operand

        Returns
        -------
        no return
        """
        # check Scalar
        TikCheckUtil.check_type_match(dst, Scalar, 'scalar conv dst must be a scalar')
        TikCheckUtil.check_type_match(src, Scalar, 'scalar conv src must be a scalar')
        # check dtype
        dtype_str = DTYPE_MAP.get(src.dtype) + '2' + DTYPE_MAP.get(dst.dtype) + ROUND_MODE_MAP.get(round_mode)
        dtype_str_err = "src " + src.dtype + " dst " + dst.dtype
        TikCheckUtil.check_equality(api_check_support("tik.scalar_conv", dtype_str), True,
                                    gen_api_check_statement(dtype_str_err, "scalar_conv"))
        # code gen
        with self.new_scope():
            # f322s32z: convert f32 to s32, any number out of s32 range
            # will be +/- s32 max number.
            # Example: round mode = Z. round to zero(c language trunc)
            if dtype_str in ('s322f32', 'f322f16', 'f162f32', 'f322s32z'):
                self.emit(call_extern(dst.dtype, "reg_set", dst.get(), dtype_convert(src, dst.dtype)))
            else:
                self.emit(call_extern(dst.dtype, "reg_set", dst.get(),
                                      call_extern(src.dtype, 'conv_' + dtype_str, src.get())))

    @source_info_decorator(depth=2)
    @scalar_single_decorator
    def _scalar_single_func(self, name, dst, src):
        # check Scalar
        TikCheckUtil.check_type_match(dst, Scalar, "dst should be Scalar")
        if isinstance(src, Scalar):
            TikCheckUtil.check_equality(dst.dtype, src.dtype,
                                        "Intrinsic %s's src's dtype should be equal to dst's dtype" % name)
        elif not isinstance(src, (int, float)):
            TikCheckUtil.raise_error("not support this type of src now")

        TikCheckUtil.check_equality(intrinsic_check_support("Intrinsic_%s" % name, dst.dtype), True,
                                    gen_api_check_statement(dst.dtype, name))
        # code gen
        scalar_tmp = dtype_convert(src, dst.dtype)
        with self.new_scope():
            self.emit(call_extern(dst.dtype, "reg_set", dst.get(),
                                  call_extern(dst.dtype, name, scalar_tmp)))

    @source_info_decorator(depth=2)
    @scalar_binary_decorator
    def _scalar_binary_func(self, name, dst, src0, src1):
        # check Scalar
        TikCheckUtil.check_type_match(dst, Scalar, "dst should be Scalar")
        # check dtype
        if isinstance(src0, Scalar):
            TikCheckUtil.check_equality(dst.dtype, src0.dtype,
                                        "Intrinsic %s's src0's dtype should be equal to dst's dtype" % name)
        elif not isinstance(src0, (int, float)):
            TikCheckUtil.raise_error("not support this type of src0 now")
        if isinstance(src1, Scalar):
            TikCheckUtil.check_equality(dst.dtype, src1.dtype,
                                        "Intrinsic %s's src1's dtype should be equal to dst's dtype" % name)
        elif not isinstance(src1, (int, float)):
            TikCheckUtil.raise_error("not support this type of src1 now")
        TikCheckUtil.check_equality(intrinsic_check_support("Intrinsic_%s" % name, dst.dtype), True,
                                    gen_api_check_statement(dst.dtype, name))
        # code gen
        scalar_tmp0 = dtype_convert(src0, dst.dtype)
        scalar_tmp1 = dtype_convert(src1, dst.dtype)
        with self.new_scope():
            self.emit(call_extern(dst.dtype, "reg_set", dst.get(),
                                  call_extern(dst.dtype, name, scalar_tmp0, scalar_tmp1)))

    # @cond

    # @endcond
