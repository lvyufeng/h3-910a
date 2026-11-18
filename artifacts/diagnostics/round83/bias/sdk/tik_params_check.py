#!/usr/bin/env python
# -*- coding:utf-8 -*-
"""
Copyright (C) 2019. Huawei Technologies Co., Ltd. All rights reserved.
FILE:     tik_params_check.py
DESC:     provide params
CREATED:  2019-04-18 18:53:42
MODIFIED: 2021-11-17 17:25:12
"""
from tbe.common.platform import intrinsic_check_support
from tbe.common.platform.platform_info import api_check_support
from tbe.tik.common.util import get_bit_len
from tbe.tik.common.common_util import check_vshl_vshr_scalar
from tbe.tik.common.common_util import int64_support_check
from tbe.tik.common.tik_get_soc_name import get_compatible_rep_size
from tbe.tik.common.tik_get_soc_name import is_compatible_mode
from tbe.tik.tik_lib.tik_params import MAX_BLK_STRIDE_DOUBLE_BYTE
from tbe.tik.tik_lib.tik_params import BLK_NUM_PER_REP
from tbe.tik.tik_lib.tik_params import MAX_REP_STRIDE_12_BITS
from tbe.tik.tik_lib.tik_params import MAX_BLK_STRIDE_SINGLE_BYTE
from tbe.tik.tik_lib.tik_params import MAX_REP_STRIDE_SINGLE_BYTE
from tbe.tik.tik_lib.tik_params import gen_api_check_statement
from tbe.tik.tik_lib.tik_api_constants import DTYPE_MAP
from tbe.tik.tik_lib.tik_api_util import check_repeat_times
from tbe.tik.tik_lib.tik_api_util import check_stride_unit
from tbe.tik.tik_lib.tik_check_util import TikCheckUtil
from tbe.tik.tik_lib.tik_expr import Expr
from tbe.tik.tik_lib.tik_soc_manager import TikSocManager
from tbe.tik.api.tik_scalar import Scalar
from tbe.tik.tik_lib.tik_vector_api.tik_vector_name_map import VCMP_NAME_DICT
from tbe.tik.common.common_util import vec_template_align
from tbe.tik.common.util import DTYPE_SIZE
from tbe.tik.common.util import reduce_mul
from tbe.tik.tik_lib.tik_vector_api.vector_common_util import gen_block_list

_MIN_DST_BLK_STRIDE = 1
MAX_PAD_MODE = 3


class MultiSourceParamsCheck:
    """
    multi Params Check
    """

    def __init__(self, print_name, check_params, name):
        self.print_name = print_name
        self.dst_tensor_op, self.src0_tensor_op, self.src1_tensor_op, self.control_op = check_params
        self.name = name
        self.can_ovelap_instr_name = ["vadd", "vsub", "vmul", "vmax", "vmin", "vor", "vand"]

    def check_repeat_times(self):
        """
        check repeat_times

        Returns
        -------
        None
        """
        check_repeat_times(self.control_op.repeat_times)

    def _check_multis_op_dtype_support(self):
        dtype = self.dst_tensor_op.tensor_obj.dtype
        if dtype == "int64":
            TikCheckUtil.check_equality(int64_support_check("tik." + self.name, dtype), True,
                                        gen_api_check_statement(dtype, self.print_name))
        else:
            TikCheckUtil.check_equality(intrinsic_check_support("Intrinsic_" + self.name, dtype), True,
                                        gen_api_check_statement(dtype, self.print_name))

    def check_multis_op_dtype_support(self):
        """
        check multis_dtype support

        Returns
        -------
        None
        """
        dtype_str = "%s%s%s" % (DTYPE_MAP.get(self.dst_tensor_op.tensor_obj.dtype),
                                DTYPE_MAP.get(self.src0_tensor_op.tensor_obj.dtype),
                                DTYPE_MAP.get(self.src1_tensor_op.tensor_obj.dtype))
        dtype_str_err = "".join(["src0", self.src0_tensor_op.tensor_obj.dtype, "src1",
                                 self.src1_tensor_op.tensor_obj.dtype, "dst", self.dst_tensor_op.tensor_obj.dtype])

        if self.name in ("vmla", "vmadd", "vmaddrelu") and TikSocManager.is_nano_soc():
            TikCheckUtil.check_equality(api_check_support("tik." + self.name, self.dst_tensor_op.tensor_obj.dtype),
                                        True, gen_api_check_statement(dtype_str_err, self.print_name))
        elif self.name in ("vmla", "vmulconv"):
            TikCheckUtil.check_equality(intrinsic_check_support("Intrinsic_" + self.name, dtype_str), True,
                                        gen_api_check_statement(dtype_str_err, self.print_name))
        else:
            TikCheckUtil.check_equality(self.dst_tensor_op.tensor_obj.dtype, self.src0_tensor_op.tensor_obj.dtype,
                                        "Instruction %s's src0's dtype should be equal to dst's dtype"
                                        % self.print_name)
            TikCheckUtil.check_equality(self.dst_tensor_op.tensor_obj.dtype, self.src1_tensor_op.tensor_obj.dtype,
                                        "Instruction %s's src1's dtype should be equal to dst's dtype"
                                        % self.print_name)
            self._check_multis_op_dtype_support()

    def all_check(self, tik_instance):
        """
        all check

        Returns
        -------
        mask_o
        """
        max_blk_stride, max_rep_stride = MAX_BLK_STRIDE_SINGLE_BYTE, MAX_REP_STRIDE_SINGLE_BYTE
        if TikSocManager.is_nano_soc() and is_compatible_mode():
            max_blk_stride, max_rep_stride = max_blk_stride // 2, max_rep_stride // 2
        one_rep_size = get_compatible_rep_size()
        self.dst_tensor_op.set_rep_stride_value()
        self.dst_tensor_op.set_blk_stride_value()
        self.src0_tensor_op.set_rep_stride_value()
        self.src0_tensor_op.set_blk_stride_value()
        self.src1_tensor_op.set_rep_stride_value()
        self.src1_tensor_op.set_blk_stride_value()
        self.dst_tensor_op.check_tensor_op_valid(self.name, max_blk_stride, max_rep_stride)
        self.src0_tensor_op.check_tensor_op_valid(self.name, max_blk_stride, max_rep_stride)
        self.src1_tensor_op.check_tensor_op_valid(self.name, max_blk_stride, max_rep_stride)
        tensor_bit_len = max(get_bit_len(self.dst_tensor_op.tensor_obj.dtype),
                             get_bit_len(self.src0_tensor_op.tensor_obj.dtype),
                             get_bit_len(self.src1_tensor_op.tensor_obj.dtype))
        mask_o = self.control_op.check_and_gen_mask_o(tik_instance, tensor_bit_len)
        self.check_repeat_times()
        check_stride_unit(self.control_op.stride_unit)
        self.check_multis_op_dtype_support()
        block_list_dst = gen_block_list(tensor_bit_len, self.dst_tensor_op.tensor_obj.dtype)
        block_list_src0 = gen_block_list(tensor_bit_len, self.src0_tensor_op.tensor_obj.dtype)
        block_list_src1 = gen_block_list(tensor_bit_len, self.src1_tensor_op.tensor_obj.dtype)
        self.dst_tensor_op.check_tensor_op_overflow(self.print_name, self.control_op, block_list_dst)
        self.src0_tensor_op.check_tensor_op_overflow(self.print_name, self.control_op, block_list_src0)
        self.src1_tensor_op.check_tensor_op_overflow(self.print_name, self.control_op, block_list_src1)
        # overlap
        if self.dst_tensor_op.tensor_obj.buffer == self.src0_tensor_op.tensor_obj.buffer:
            src0_value_range = (self.control_op.repeat_times, self.dst_tensor_op.blk_stride_value,
                                self.dst_tensor_op.rep_stride_value, self.src0_tensor_op.blk_stride,
                                self.src0_tensor_op.rep_stride, self.control_op.stride_unit)
            if all(isinstance(value, int) for value in src0_value_range):
                block_list = [BLK_NUM_PER_REP, one_rep_size //
                              max(get_bit_len(self.dst_tensor_op.tensor_obj.dtype),
                                  get_bit_len(self.src0_tensor_op.tensor_obj.dtype))]
                self.dst_tensor_op.check_address_overlapping(self.print_name, self.control_op, self.src0_tensor_op,
                                                             block_list)
        if self.dst_tensor_op.tensor_obj.buffer == self.src1_tensor_op.tensor_obj.buffer:
            src1_value_range = (self.control_op.repeat_times, self.dst_tensor_op.blk_stride_value,
                                self.dst_tensor_op.rep_stride_value, self.src1_tensor_op.blk_stride,
                                self.src1_tensor_op.rep_stride, self.control_op.stride_unit)
            if all(isinstance(value, int) for value in src1_value_range):
                if self.name not in self.can_ovelap_instr_name or self.dst_tensor_op.tensor_obj.dtype not in \
                        ("float16", "int32", "float32") or self.control_op.repeat_times == 1:
                    block_list = [BLK_NUM_PER_REP, one_rep_size //
                                  max(get_bit_len(self.dst_tensor_op.tensor_obj.dtype),
                                      get_bit_len(self.src1_tensor_op.tensor_obj.dtype))]
                    self.dst_tensor_op.check_address_overlapping(self.print_name, self.control_op, self.src1_tensor_op,
                                                                 block_list)
                else:
                    self.dst_tensor_op.check_dst_src1_overlap_other(self.print_name, self.control_op,
                                                                    self.src0_tensor_op, self.src1_tensor_op)
        return mask_o


class SingleCheckParams:
    """
    Single Params Check
    """

    def __init__(self, print_name, params_list, name):
        self.print_name = print_name
        self.name = name
        self.dst_tensor_op, self.src_tensor_op, self.control_op = params_list

    def check_repeat_times(self):
        """
        check repeat_times

        Returns
        -------
        None
        """
        check_repeat_times(self.control_op.repeat_times)

    def check_single_op_dtype_support(self):
        """
        check single_dtype support

        Returns
        -------
        None
        """
        TikCheckUtil.check_equality(self.dst_tensor_op.tensor_obj.dtype, self.src_tensor_op.tensor_obj.dtype,
                                    "Instruction %s's src's "
                                    "dtype should be equal to dst's dtype" % self.print_name)
        dtype = self.dst_tensor_op.tensor_obj.dtype
        if dtype == "int64":
            TikCheckUtil.check_equality(int64_support_check("tik." + self.name, dtype), True,
                                        gen_api_check_statement(dtype, self.print_name))
        else:
            TikCheckUtil.check_equality(
                intrinsic_check_support("Intrinsic_" + self.name, self.dst_tensor_op.tensor_obj.dtype), True,
                gen_api_check_statement(self.dst_tensor_op.tensor_obj.dtype, self.print_name))

    def check_all(self, tik_instance):
        """
        all check

        Returns
        -------
        mask_o
        """
        # check tensor's tensor, scope, address, rep stride and blk stride
        if TikSocManager.is_nano_soc() and is_compatible_mode():
            max_blk_stride, max_rep_stride = MAX_BLK_STRIDE_DOUBLE_BYTE // 2, MAX_REP_STRIDE_12_BITS // 2
        elif TikSocManager.is_nano_soc() and not is_compatible_mode():
            max_blk_stride, max_rep_stride = MAX_BLK_STRIDE_DOUBLE_BYTE, MAX_REP_STRIDE_12_BITS
        else:
            max_blk_stride, max_rep_stride = MAX_BLK_STRIDE_DOUBLE_BYTE, MAX_REP_STRIDE_SINGLE_BYTE

        self.dst_tensor_op.set_blk_stride_value()
        self.dst_tensor_op.set_rep_stride_value()
        self.src_tensor_op.set_blk_stride_value()
        self.src_tensor_op.set_rep_stride_value()
        self.dst_tensor_op.check_tensor_op_valid(self.name, max_blk_stride, max_rep_stride)
        self.src_tensor_op.check_tensor_op_valid(self.name, max_blk_stride, max_rep_stride)
        self.check_repeat_times()
        check_stride_unit(self.control_op.stride_unit)
        self.check_single_op_dtype_support()
        tensor_bit_len = max(get_bit_len(self.dst_tensor_op.tensor_obj.dtype),
                             get_bit_len(self.src_tensor_op.tensor_obj.dtype))
        mask_o = self.control_op.check_and_gen_mask_o(tik_instance, tensor_bit_len)
        # check tensor overflow

        block_list = [BLK_NUM_PER_REP, get_compatible_rep_size() // tensor_bit_len]
        overflow_value_range = (self.control_op.repeat_times, self.dst_tensor_op.blk_stride,
                                self.dst_tensor_op.rep_stride, self.src_tensor_op.blk_stride,
                                self.src_tensor_op.rep_stride)
        if all(isinstance(value, int) for value in overflow_value_range):
            if self.print_name == "vcbd":
                max_dtype_size = max(DTYPE_SIZE.get(self.dst_tensor_op.tensor_obj.dtype),
                                     DTYPE_SIZE.get(self.src_tensor_op.tensor_obj.dtype))
                self.dst_tensor_op.check_vcbd_overflow(self.control_op, max_dtype_size)
                self.src_tensor_op.check_vcbd_overflow(self.control_op, max_dtype_size)
            else:
                block_list_dst = gen_block_list(tensor_bit_len, self.dst_tensor_op.tensor_obj.dtype)
                block_list_src = gen_block_list(tensor_bit_len, self.src_tensor_op.tensor_obj.dtype)
                self.dst_tensor_op.check_tensor_op_overflow(self.print_name, self.control_op, block_list_dst)
                self.src_tensor_op.check_tensor_op_overflow(self.print_name, self.control_op, block_list_src)
        # check dst src overlapping

        if self.src_tensor_op.tensor_obj.buffer == self.dst_tensor_op.tensor_obj.buffer:
            overlap_value_range = (self.control_op.repeat_times, self.dst_tensor_op.blk_stride,
                                   self.dst_tensor_op.rep_stride, self.src_tensor_op.blk_stride,
                                   self.src_tensor_op.rep_stride, self.control_op.stride_unit)
            if all(isinstance(value, int) for value in overlap_value_range):
                self.dst_tensor_op.check_address_overlapping(self.print_name, self.control_op, self.src_tensor_op,
                                                             block_list)

        return mask_o


class PrecisionConvCheckParams(SingleCheckParams):
    """
    Precision Conv Params Check
    """

    def __init__(self, print_name, params_list, name):
        super(PrecisionConvCheckParams, self).__init__(print_name, params_list, name)
        self.print_name = print_name
        self.name = name
        self.dst_tensor_op, self.src_tensor_op, self.control_op = params_list

    def check_single_op_dtype_support(self):
        """
        check single_dtype support

        Returns
        -------
        None
        """
        src_dst_dtype = DTYPE_MAP.get(self.src_tensor_op.tensor_obj.dtype) + "2" + \
                        DTYPE_MAP.get(self.dst_tensor_op.tensor_obj.dtype)
        dtype_str_err = "src " + self.src_tensor_op.tensor_obj.dtype + " dst " + self.dst_tensor_op.tensor_obj.dtype
        TikCheckUtil.check_equality(intrinsic_check_support("Intrinsic_" + self.name, src_dst_dtype), True,
                                    gen_api_check_statement(dtype_str_err, self.name))


class ScalarMultisCheckParams:
    """
    Scalar Multis Params Check
    """

    def __init__(self, print_name, check_params, round_en, name):
        self.print_name = print_name
        self.round_en = round_en
        self.dst_tensor_op, self.src_tensor_op, self.scalar, self.control_op = check_params
        self.name = name

    def check_repeat_times(self):
        """
        check repeat_times

        Returns
        -------
        None
        """
        check_repeat_times(self.control_op.repeat_times, mask_mode=self.control_op.mask_mode)

    def check_scalar_dtype(self):
        """
        check scalar_dtype

        Returns
        -------
        None
        """
        TikCheckUtil.check_type_match(
            self.scalar, (int, float, Expr, Scalar),
            "scalar should be int, float, Expr or Scalar, "
            "input type of scalar: %s" % type(self.scalar))
        if self.name in ("vshl", "vshr"):
            check_vshl_vshr_scalar(self.src_tensor_op.tensor_obj.dtype, self.scalar)

    def check_round_en_vshr(self):
        """
        check round_en of vshr

        Returns
        -------
        None
        """
        if self.name == "vshr" and self.src_tensor_op.tensor_obj.dtype not in ("int16", "int32", "int64"):
            TikCheckUtil.check_equality(
                self.round_en, 0, "Instruction vshr round_en should be False "
                                  "when src.dtype is %s" % self.src_tensor_op.tensor_obj.dtype)

    def check_scalar_multis_dtype_support(self):
        """
        check scalar_multis_dtype support

        Returns
        -------
        None
        """
        dtype = self.dst_tensor_op.tensor_obj.dtype
        dtype = self.dst_tensor_op.tensor_obj.dtype
        if dtype == "int64":
            TikCheckUtil.check_equality(int64_support_check("tik." + self.name, dtype), True,
                                        gen_api_check_statement(dtype, self.print_name))
        else:
            self._check_scalar_multis_dtype_support()


    def _check_scalar_multis_dtype_support(self):
        if isinstance(self.scalar, Scalar):
            TikCheckUtil.check_equality(self.scalar.dtype, self.src_tensor_op.tensor_obj.dtype,
                                        "Instruction %s's src's dtype should be equal"
                                        " to scalar's dtype" % self.print_name)
            dtype_str = DTYPE_MAP[self.dst_tensor_op.tensor_obj.dtype] + DTYPE_MAP[
                self.src_tensor_op.tensor_obj.dtype] + DTYPE_MAP[self.scalar.dtype]
            dtype_str_err = "src %s scalar %s dst %s" % (self.src_tensor_op.tensor_obj.dtype,
                                                         self.scalar.dtype, self.dst_tensor_op.tensor_obj.dtype)
        else:
            dtype_str = DTYPE_MAP[self.dst_tensor_op.tensor_obj.dtype] + DTYPE_MAP[
                self.src_tensor_op.tensor_obj.dtype] * 2
            dtype_str_err = "src " + self.src_tensor_op.tensor_obj.dtype + " dst " + self.dst_tensor_op.tensor_obj.dtype

        if self.name == "vaxpy" and TikSocManager.is_nano_soc():
            TikCheckUtil.check_equality(api_check_support("tik." + self.name, self.dst_tensor_op.tensor_obj.dtype),
                                        True, gen_api_check_statement(dtype_str_err, self.print_name))
        elif self.name == "vaxpy":
            TikCheckUtil.check_equality(intrinsic_check_support("Intrinsic_" + self.name, dtype_str), True,
                                        gen_api_check_statement(
                                            dtype_str_err, self.print_name))
        else:
            TikCheckUtil.check_equality(self.dst_tensor_op.tensor_obj.dtype, self.src_tensor_op.tensor_obj.dtype,
                                        "Instruction %s's src's dtype should be equal"
                                        " to dst's dtype" % self.print_name)
            TikCheckUtil.check_equality(intrinsic_check_support("Intrinsic_" + self.name,
                                                                self.dst_tensor_op.tensor_obj.dtype),
                                        True, gen_api_check_statement(self.dst_tensor_op.tensor_obj.dtype, self.name))

    def all_check(self, tik_instance):
        """
        all check

        Returns
        -------
        mask_o
        """
        self.dst_tensor_op.set_rep_stride_value()
        self.dst_tensor_op.set_blk_stride_value()
        self.src_tensor_op.set_blk_stride_value()
        self.src_tensor_op.set_rep_stride_value()

        if TikSocManager.is_nano_soc() and is_compatible_mode():
            max_blk_stride, max_rep_stride = MAX_BLK_STRIDE_DOUBLE_BYTE // 2, MAX_REP_STRIDE_12_BITS // 2
        elif TikSocManager.is_nano_soc() and not is_compatible_mode():
            max_blk_stride, max_rep_stride = MAX_BLK_STRIDE_DOUBLE_BYTE, MAX_REP_STRIDE_12_BITS
        else:
            max_blk_stride, max_rep_stride = MAX_BLK_STRIDE_DOUBLE_BYTE, MAX_REP_STRIDE_SINGLE_BYTE

        self.dst_tensor_op.check_tensor_op_valid(self.name, max_blk_stride, max_rep_stride)
        self.src_tensor_op.check_tensor_op_valid(self.name, max_blk_stride, max_rep_stride)
        self.control_op.check_mask_mode()
        tensor_bit_len = max(get_bit_len(self.dst_tensor_op.tensor_obj.dtype),
                             get_bit_len(self.src_tensor_op.tensor_obj.dtype), )
        mask_o = self.control_op.check_and_gen_mask_o(tik_instance, tensor_bit_len)
        self.check_scalar_multis_dtype_support()
        self.check_round_en_vshr()
        self.check_scalar_dtype()
        self.check_repeat_times()
        check_stride_unit(self.control_op.stride_unit)
        # overflow
        block_list_dst = gen_block_list(tensor_bit_len, self.dst_tensor_op.tensor_obj.dtype)
        block_list_src = gen_block_list(tensor_bit_len, self.src_tensor_op.tensor_obj.dtype)
        self.dst_tensor_op.check_tensor_op_overflow(self.print_name, self.control_op, block_list_dst)
        self.src_tensor_op.check_tensor_op_overflow(self.print_name, self.control_op, block_list_src)
        # overlap
        block_list = [BLK_NUM_PER_REP, get_compatible_rep_size() // tensor_bit_len]
        if self.src_tensor_op.tensor_obj.buffer == self.dst_tensor_op.tensor_obj.buffer:
            value_range = (self.control_op.repeat_times, self.dst_tensor_op.blk_stride,
                           self.dst_tensor_op.rep_stride, self.src_tensor_op.blk_stride,
                           self.src_tensor_op.rep_stride, self.control_op.stride_unit)
            if all(isinstance(value, int) for value in value_range):
                self.dst_tensor_op.check_address_overlapping(self.print_name, self.control_op, self.src_tensor_op,
                                                             block_list)

        return mask_o


class VcmpCheckParams:
    """
    Dup Params Check
    """

    def __init__(self, print_name, params_list, name):
        self.print_name = print_name
        self.name = name
        self.src0_tensor_op, self.src1_tensor_op, self.control_op = params_list

    def check_vcmp_op_dtype_support(self):
        """
        check vcmp_xx_dtype support

        Returns
        -------
        None
        """
        TikCheckUtil.check_equality(self.src0_tensor_op.tensor_obj.dtype, self.src1_tensor_op.tensor_obj.dtype,
                                    "Intrinsic %s's src0's dtype should be equal to src1's dtype" % self.print_name)
        dtype = self.src0_tensor_op.tensor_obj.dtype
        if dtype == "int64":
            TikCheckUtil.check_equality(int64_support_check("tik." + "vcmp", dtype), True,
                                        gen_api_check_statement(dtype, self.print_name))
        else:
            TikCheckUtil.check_equality(intrinsic_check_support("Intrinsic_vcmp",
                                                                self.src0_tensor_op.tensor_obj.dtype), True,
                                        gen_api_check_statement(self.src0_tensor_op.tensor_obj.dtype, self.print_name))

    def check_all(self, tik_instance, isdebug=False):
        """
        all check

        Returns
        -------
        None
        """
        # check tensor's tensor, scope
        self.src0_tensor_op.check_tensor_and_scope()
        self.src1_tensor_op.check_tensor_and_scope()
        # check blk_stride
        self.src0_tensor_op.set_blk_stride_value()
        self.src1_tensor_op.set_blk_stride_value()
        self.src0_tensor_op.check_tensor_op_blk_stride(MAX_BLK_STRIDE_DOUBLE_BYTE)
        self.src1_tensor_op.check_tensor_op_blk_stride(MAX_BLK_STRIDE_DOUBLE_BYTE)

        align = vec_template_align(self.src0_tensor_op.tensor_obj.dtype)
        tensor_bit_len = max(get_bit_len(self.src0_tensor_op.tensor_obj.dtype),
                             get_bit_len(self.src1_tensor_op.tensor_obj.dtype))
        # check tensor overflow
        if isdebug is False:
            # check mask
            mask_o = self.control_op.check_and_gen_mask_o(tik_instance, tensor_bit_len)
            # check address_align
            self.src0_tensor_op.check_tensor_op_address_align(align)
            self.src1_tensor_op.check_tensor_op_address_align(align)
            self.check_vcmp_op_dtype_support()
            block_list_src0 = gen_block_list(tensor_bit_len, self.src0_tensor_op.tensor_obj.dtype)
            self.src0_tensor_op.check_tensor_op_overflow(self.print_name, self.control_op, block_list_src0)
            return mask_o
        return align, tensor_bit_len


class VcmpvCheckParams:
    """
    Dup Params Check
    """

    def __init__(self, print_name, params_list):
        self.print_name = print_name
        self.name = VCMP_NAME_DICT.get(self.print_name)
        self.dst_tensor_op, self.src0_tensor_op, self.src1_tensor_op, self.control_op = params_list

    def check_vcmpv_repeat_times(self, isdebug):
        """
        check vcmpv repeat times

        Returns
        -------
        None
        """
        if not isdebug:
            check_repeat_times(self.control_op.repeat_times)

    def check_vcmpv_op_dtype_support(self):
        """
        check vcmpv_xx_dtype support

        Returns
        -------
        None
        """
        dtype = self.src0_tensor_op.tensor_obj.dtype
        if dtype == "int64":
            TikCheckUtil.check_equality(int64_support_check("tik." + self.name, dtype), True,
                                        gen_api_check_statement(dtype, self.print_name))
        else:
            TikCheckUtil.check_var_in_list(
                self.dst_tensor_op.tensor_obj.dtype, ["uint64", "uint32", "uint16", "uint8"],
                "dst should be uint64, uint32, uint16 or uint8")
            TikCheckUtil.check_equality(dtype, self.src1_tensor_op.tensor_obj.dtype,
                                        "Intrinsic %s's src0's dtype should be equal to src1's dtype" % self.print_name)

            TikCheckUtil.check_equality(intrinsic_check_support("Intrinsic_" + self.name, dtype), True,
                                        gen_api_check_statement(dtype, self.print_name))

    def check_all(self, isdebug=False):
        """
        all check

        Returns
        -------
        None
        """
        # check tensor's tensor, scope, address, rep stride and blk stride
        align = vec_template_align(self.dst_tensor_op.tensor_obj.dtype)
        one_rep_size = get_compatible_rep_size()

        self.dst_tensor_op.set_rep_stride_value()
        self.src0_tensor_op.set_rep_stride_value()
        self.src1_tensor_op.set_rep_stride_value()
        self.dst_tensor_op.set_blk_stride_value()
        self.src0_tensor_op.set_blk_stride_value()
        self.src1_tensor_op.set_blk_stride_value()
        self.dst_tensor_op.check_tensor_op_valid(self.name, MAX_BLK_STRIDE_SINGLE_BYTE,
                                                 MAX_REP_STRIDE_SINGLE_BYTE, align)
        self.src0_tensor_op.check_tensor_op_valid(self.name, MAX_BLK_STRIDE_SINGLE_BYTE,
                                                  MAX_REP_STRIDE_SINGLE_BYTE, align)
        self.src1_tensor_op.check_tensor_op_valid(self.name, MAX_BLK_STRIDE_SINGLE_BYTE,
                                                  MAX_REP_STRIDE_SINGLE_BYTE, align)
        self.check_vcmpv_repeat_times(isdebug)
        tensor_bit_len_max1 = max(get_bit_len(self.src0_tensor_op.tensor_obj.dtype),
                                  get_bit_len(self.src1_tensor_op.tensor_obj.dtype))
        tensor_bit_len_max2 = max(get_bit_len(self.dst_tensor_op.tensor_obj.dtype),
                                  get_bit_len(self.src0_tensor_op.tensor_obj.dtype))
        block_list = [BLK_NUM_PER_REP, one_rep_size // tensor_bit_len_max2]

        # check dtype support
        if not isdebug:
            self.check_vcmpv_op_dtype_support()

        # check tensor overlap
        self.control_op.mask = one_rep_size // max(DTYPE_SIZE[self.dst_tensor_op.tensor_obj.dtype],
                                                   DTYPE_SIZE[self.src0_tensor_op.tensor_obj.dtype])
        if self.src0_tensor_op.tensor_obj.buffer == self.dst_tensor_op.tensor_obj.buffer:
            src0_value_range = (self.control_op.repeat_times, self.src0_tensor_op.blk_stride,
                                self.src0_tensor_op.rep_stride)
            if all(isinstance(value, int) for value in src0_value_range) or isdebug:
                self.dst_tensor_op.check_address_overlapping(self.print_name, self.control_op, self.src0_tensor_op,
                                                             block_list)
        if self.src1_tensor_op.tensor_obj.buffer == self.dst_tensor_op.tensor_obj.buffer:
            src1_value_range = (self.control_op.repeat_times, self.src1_tensor_op.blk_stride,
                                self.src1_tensor_op.rep_stride)
            if all(isinstance(value, int) for value in src1_value_range) or isdebug:
                self.dst_tensor_op.check_address_overlapping(self.print_name, self.control_op, self.src1_tensor_op,
                                                             block_list)
        # check tensor overflow
        if not isdebug:
            self.control_op.mask = one_rep_size // DTYPE_SIZE[self.src0_tensor_op.tensor_obj.dtype]
            block_list_src1 = gen_block_list(tensor_bit_len_max1, self.src0_tensor_op.tensor_obj.dtype)
            block_list_src0 = gen_block_list(tensor_bit_len_max1, self.src1_tensor_op.tensor_obj.dtype)
            self.src0_tensor_op.check_tensor_op_overflow(self.print_name, self.control_op, block_list_src0)
            self.src1_tensor_op.check_tensor_op_overflow(self.print_name, self.control_op, block_list_src1)


class VcmpvsCheckParams:
    """
    Dup Params Check
    """

    def __init__(self, print_name, params_list):
        self.one_rep_size = get_compatible_rep_size()
        self.print_name = print_name
        self.name = VCMP_NAME_DICT.get(self.print_name)
        self.dst_tensor_op, self.src_tensor_op, self.scalar_op, self.control_op = params_list

    def check_vcmpvs_op_dtype_support(self, isdebug):
        """
        check vcmpvs_xx_dtype support

        Returns
        -------
        None
        """
        if not isdebug:
            dtype_list = ("uint64", "uint32", "uint16", "uint8")
            TikCheckUtil.check_var_in_list(
                self.dst_tensor_op.tensor_obj.dtype, dtype_list, "dst dtype should be unsigned int, input dst dtype: %s"
                                                                 % self.dst_tensor_op.tensor_obj.dtype)
            if isinstance(self.scalar_op.scalar_obj, Scalar):
                TikCheckUtil.check_equality(self.src_tensor_op.tensor_obj.dtype, self.scalar_op.scalar_obj.dtype,
                                            "Intrinsic %s's src's dtype should be equal to scalar's dtype" % self.name)
            TikCheckUtil.check_equality(intrinsic_check_support("Intrinsic_" +
                                                                self.name, self.src_tensor_op.tensor_obj.dtype), True,
                                        gen_api_check_statement(self.src_tensor_op.tensor_obj.dtype, self.name))

    def check_vcmpvs_repeat_times(self, isdebug):
        """
        check vcmpvs repeat times

        Returns
        -------
        None
        """
        if not isdebug:
            check_repeat_times(self.control_op.repeat_times)

    def check_all(self, isdebug=None):
        """
        all check

        Returns
        -------
        None
        """
        # check tensor's tensor, scope, address, rep stride and blk stride
        align = vec_template_align(self.src_tensor_op.tensor_obj.dtype)
        self.dst_tensor_op.set_rep_stride_value()
        self.dst_tensor_op.set_blk_stride_value()
        self.src_tensor_op.set_blk_stride_value()
        self.src_tensor_op.set_rep_stride_value()
        self.dst_tensor_op.check_tensor_op_valid(self.name, MAX_BLK_STRIDE_DOUBLE_BYTE, MAX_REP_STRIDE_SINGLE_BYTE,
                                                 align)
        self.src_tensor_op.check_tensor_op_valid(self.name, MAX_BLK_STRIDE_DOUBLE_BYTE, MAX_REP_STRIDE_SINGLE_BYTE,
                                                 align)
        # check repeat
        self.check_vcmpvs_op_dtype_support(isdebug)
        # check dtype
        self.check_vcmpvs_repeat_times(isdebug)
        # check tensor overlap
        tensor_bit_len_max = self.one_rep_size // max(get_bit_len(self.dst_tensor_op.tensor_obj.dtype),
                                                      get_bit_len(self.src_tensor_op.tensor_obj.dtype))
        block_list = [BLK_NUM_PER_REP, self.one_rep_size // tensor_bit_len_max]
        if self.dst_tensor_op.tensor_obj.buffer == self.src_tensor_op.tensor_obj.buffer:
            value_range = (self.control_op.repeat_times, self.src_tensor_op.blk_stride, self.src_tensor_op.rep_stride)
            if all(isinstance(value, int) for value in value_range) or isdebug:
                self.control_op.mask = self.one_rep_size // max(DTYPE_SIZE[self.dst_tensor_op.tensor_obj.dtype],
                                                                DTYPE_SIZE[self.src_tensor_op.tensor_obj.dtype])
                self.dst_tensor_op.check_address_overlapping(self.print_name, self.control_op, self.src_tensor_op,
                                                             block_list)
        if not isdebug:
            # check src tensor overflow(static)
            block_list_src1 = gen_block_list(tensor_bit_len_max, self.src_tensor_op.tensor_obj.dtype)
            self.control_op.mask = self.one_rep_size // DTYPE_SIZE[self.src_tensor_op.tensor_obj.dtype]
            self.src_tensor_op.check_tensor_op_overflow(
                self.print_name, self.control_op, block_list_src1)
            # check dst tensor overflow
            if all(Expr(value).eval_value() is not None
                   for value in (self.control_op.repeat_times, self.dst_tensor_op.tensor_obj.offset)):
                self._check_vcmpvs_dst_tensor_overflow()

    def _check_vcmpvs_dst_tensor_overflow(self):
        """
        check vcmpvs dst tensor overflow
        """
        rep_byte_size_work = self.control_op.repeat_times * self.one_rep_size // DTYPE_SIZE[
            self.src_tensor_op.tensor_obj.dtype]

        one_block_len = rep_byte_size_work // get_bit_len(self.dst_tensor_op.tensor_obj.dtype)

        dst_ele_expected = Expr(one_block_len + self.dst_tensor_op.tensor_obj.offset).eval_value()
        dst_ele_actual = reduce_mul(self.dst_tensor_op.tensor_obj.original_shape)
        TikCheckUtil.check_le(dst_ele_expected, dst_ele_actual,
                              "dst tensor overflow, expected dst shape: %s, actual dst shape: %s"
                              % (dst_ele_expected, dst_ele_actual))


class VadddeqreluParamsCheck:
    """
    Vadddeqrelu Params Check
    """
    def __init__(self, print_name, param_list, deqscale):
        self.print_name = print_name
        self.name = print_name
        self.dst_tensor_op, self.src0_tensor_op, self.src1_tensor_op, self.control_op = param_list
        self.deqscale = deqscale

    def check_vadddeqrelu_dtype_support(self):
        """
        check dtype_support

        Returns
        -------
        None
        """
        dtype_str = DTYPE_MAP.get(self.dst_tensor_op.tensor_obj.dtype) + \
                    DTYPE_MAP.get(self.src0_tensor_op.tensor_obj.dtype) + \
                    DTYPE_MAP.get(self.src1_tensor_op.tensor_obj.dtype)
        dtype_str_err = "src0 %s src1 %s dst %s" % (self.src0_tensor_op.tensor_obj.dtype,
                                                    self.src1_tensor_op.tensor_obj.dtype,
                                                    self.dst_tensor_op.tensor_obj.dtype)
        if "s64" in dtype_str:
            TikCheckUtil.check_equality(int64_support_check("tik." + self.name, dtype_str), True,
                                        gen_api_check_statement(dtype_str_err, self.print_name))
        else:
            TikCheckUtil.check_equality(intrinsic_check_support("Intrinsic_" + self.name, dtype_str), True,
                                        gen_api_check_statement(dtype_str_err, self.name))

    def check_deqscale(self):
        """
        check deqscale

        Returns
        -------
        None
        """
        TikCheckUtil.check_type_match(self.deqscale, (int, float, Scalar),
                                      "deqscale only support immediate mode or Scalar")
        if isinstance(self.deqscale, Scalar):
            TikCheckUtil.check_equality(self.deqscale.dtype, "float16", "deqscale scalar must be float16")

    def check_overlap(self, src_tensor_op):
        """
        check overlap

        Parameters
        ----------
        src_tensor_op: src0_tensor_op or src1_tensor_op

        Returns
        -------
        None
        """
        if src_tensor_op.tensor_obj.buffer == self.dst_tensor_op.tensor_obj.buffer:
            value_range = (self.control_op.repeat_times, self.dst_tensor_op.blk_stride,
                           self.dst_tensor_op.rep_stride, src_tensor_op.blk_stride, src_tensor_op.rep_stride)
            if all(isinstance(value, int) for value in value_range):
                block_list = [BLK_NUM_PER_REP, get_compatible_rep_size() //
                              max(get_bit_len(self.dst_tensor_op.tensor_obj.dtype),
                                  get_bit_len(src_tensor_op.tensor_obj.dtype))]
                self.dst_tensor_op.check_address_overlapping(self.print_name, self.control_op, src_tensor_op,
                                                             block_list)

    def all_check(self, tik_instance):
        """
        all check

        Parameters
        ----------
        tik_instance

        Returns
        -------
        None
        """
        check_repeat_times(self.control_op.repeat_times)
        check_stride_unit(self.control_op.stride_unit)
        align = vec_template_align(self.dst_tensor_op.tensor_obj.dtype)
        self.check_vadddeqrelu_dtype_support()
        self.dst_tensor_op.set_rep_stride_value()
        self.dst_tensor_op.set_blk_stride_value()
        self.src0_tensor_op.set_blk_stride_value()
        self.src0_tensor_op.set_rep_stride_value()
        self.src1_tensor_op.set_rep_stride_value()
        self.src1_tensor_op.set_blk_stride_value()
        self.dst_tensor_op.check_tensor_op_valid(self.name, MAX_BLK_STRIDE_SINGLE_BYTE,
                                                 MAX_REP_STRIDE_SINGLE_BYTE, align)
        self.src0_tensor_op.check_tensor_op_valid(self.name, MAX_BLK_STRIDE_SINGLE_BYTE,
                                                  MAX_REP_STRIDE_SINGLE_BYTE, align)
        self.src1_tensor_op.check_tensor_op_valid(self.name, MAX_BLK_STRIDE_SINGLE_BYTE,
                                                  MAX_REP_STRIDE_SINGLE_BYTE, align)
        tensor_bit_len = max(get_bit_len(self.dst_tensor_op.tensor_obj.dtype),
                             get_bit_len(self.src1_tensor_op.tensor_obj.dtype),
                             get_bit_len(self.src0_tensor_op.tensor_obj.dtype))
        mask_o = self.control_op.check_and_gen_mask_o(tik_instance, tensor_bit_len)
        # check_overflow
        block_list_dst = gen_block_list(tensor_bit_len, self.dst_tensor_op.tensor_obj.dtype)
        block_list_src0 = gen_block_list(tensor_bit_len, self.src0_tensor_op.tensor_obj.dtype)
        block_list_src1 = gen_block_list(tensor_bit_len, self.src1_tensor_op.tensor_obj.dtype)
        self.dst_tensor_op.check_tensor_op_overflow(self.print_name, self.control_op, block_list_dst)
        self.src0_tensor_op.check_tensor_op_overflow(self.print_name, self.control_op, block_list_src0)
        self.src1_tensor_op.check_tensor_op_overflow(self.print_name, self.control_op, block_list_src1)
        # check_overlapping
        self.check_overlap(self.src0_tensor_op)
        self.check_overlap(self.src1_tensor_op)

        return mask_o
