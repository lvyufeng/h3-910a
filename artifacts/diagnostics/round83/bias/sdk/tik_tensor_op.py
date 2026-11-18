#!/usr/bin/env python
# -*- coding:utf-8 -*-
"""
Copyright (C) 2019. Huawei Technologies Co., Ltd. All rights reserved.
FILE:     tik_tensor_op.py
DESC:     provide params
CREATED:  2019-04-18 18:53:42
MODIFIED: 2020-12-7 19:17:00
"""

from tbe import tvm
from tbe.common.platform import scope_ubuf
from tbe.tik.api.tik_tensor import Tensor
from tbe.tik.common.util import ceil_div
from tbe.tik.common.util import get_bit_len
from tbe.tik.common.util import DTYPE_SIZE
from tbe.tik.common.util import check_scalar_dtype
from tbe.tik.common.common_util import get_blk_valid_list
from tbe.tik.common.common_util import vec_template_align
from tbe.tik.common.common_util import TikUtil
from tbe.tik.common.common_util import reduce_mul
from tbe.tik.common.common_util import BIT_LEN_32
from tbe.tik.common.common_util import BIT_LEN_8
from tbe.tik.common.common_util import BIT_LEN_16
from tbe.tik.common.common_util import get_8or16bit_dtype_mask_len
from tbe.tik.common.common_util import get_need_offset
from tbe.tik.common.common_check_func import get_32bit_dtype_mask_len
from tbe.tik.common.tik_get_soc_name import get_compatible_rep_size
from tbe.tik.common.tik_get_soc_name import get_compatible_blk_size
from tbe.tik.tik_lib.tik_expr import is_basic_expr
from tbe.tik.tik_lib.tik_expr import Expr
from tbe.tik.tik_lib.tik_expr import BasicExpr
from tbe.tik.tik_lib.tik_check_util import TikCheckUtil
from tbe.tik.tik_lib.tik_params import DEFAULT_STRIDE
from tbe.tik.tik_lib.tik_params import MASK_VALUE_ZERO
from tbe.tik.tik_lib.tik_params import BLK_NUM_PER_REP
from tbe.tik.tik_lib.tik_vector_api.tik_vector_name_map import SINGLE_NAME_DICT
from tbe.tik.tik_lib.tik_vector_api.tik_vector_name_map import FILLS_NAME_DICT
from tbe.tik.debug.util import get_flatten_idx
from tbe.tik.debug.util import get_dtype_size
from tbe.tik.tik_lib.tik_vector_api.vector_common_util import get_new_dst_bs
from tbe.tik.tik_lib.tik_api_constants import DTYPE_BLOCK_NUM
from tbe.tik.tik_lib.tik_api_constants import DTYPE_MAP

_MIN_DST_BLK_STRIDE = 1


class TensorBaseOp:
    """
    Tensor Base ops
    """

    def __init__(self, tensor_obj, blk_stride, rep_stride, tensor_op_name):
        self.tensor_obj = tensor_obj
        self.blk_stride = blk_stride
        self.rep_stride = rep_stride
        self.tensor_op_name = tensor_op_name
        self.context = None
        self.blk_stride_value = self.blk_stride
        self.rep_stride_value = self.rep_stride
        self.offset_value = None
        self.stride_unit_value = 0
        self.mask_value = None
        self.repeat_times_value = None
        self.src_mask_value = None
        self.original_shape_value = None
        self.tensor_offset_value = None

    def set_context(self, context=None):
        """
        debug set context

        Parameters
        ----------
        context

        Returns
        -------
        None
        """
        self.context = context
        self.set_blk_stride_value()
        self.set_rep_stride_value()
        self.eval_offset()

    def eval_offset(self):
        """
        eval offset

        Returns
        -------
        None
        """
        if self.context:
            offset = self.context.evaluate_expr(self.tensor_obj.offset)
        else:
            offset = Expr(self.tensor_obj.offset).eval_value()
        self.offset_value = offset

    def set_stride_unit(self, stride_unit):
        """
        set stride_unit

        Parameters
        ----------
        stride_unit

        Returns
        -------
        None
        """
        if self.context:
            new_parameters = self.context.evaluate_expr(stride_unit)
        else:
            new_parameters = stride_unit
        self.stride_unit_value = new_parameters

    def set_original_shape(self):
        """
        set original shape

        Parameters
        ----------

        Returns
        -------
        None
        """
        if self.context:
            new_parameters = self.context.get_tensor_original_shape(self.tensor_obj)
        else:
            new_parameters = self.tensor_obj.original_shape
        self.original_shape_value = new_parameters

    def set_vreduce_context(self, context=None):
        """
        set vreduce context

        Parameters
        ----------
        context : debug's context

        Returns
        -------
        None
        """
        self.context = context
        self.set_blk_stride_value()
        self.set_rep_stride_value()
        self.get_tensor_offset()
        self.set_original_shape()

    def get_tensor_offset(self):
        """
        set tensor offset

        Returns
        -------
        None
        """
        if self.context:
            new_parameters = self.context.get_tensor_offset(self.tensor_obj)
        else:
            new_parameters = self.tensor_obj.offset
        self.tensor_offset_value = new_parameters

    def set_repeat_times(self, repeat_times):
        """
        set repeat_times

        Parameters
        ----------
        repeat_times

        Returns
        -------
        None
        """
        if self.context:
            new_parameters = self.context.evaluate_expr(repeat_times)
        else:
            new_parameters = repeat_times
        self.repeat_times_value = new_parameters

    def get_mask(self, mask):
        """
        get mask

        Parameters
        ----------
        mask: mask

        Returns
        -------
        None
        """
        if self.context:
            if isinstance(mask, (list, tuple)):
                mask = [self.context.evaluate_expr(value) for value in mask]
            else:
                mask = [self.context.evaluate_expr(mask)]
        else:
            if not isinstance(mask, (list, tuple)):
                mask = [mask]

        self.mask_value = mask

    def set_blk_stride_value(self, parameters=None):
        """
        set blk_stride_value

        Parameters
        ----------
        parameters

        Returns
        -------
        None
        """
        if parameters is None:
            parameters = self.blk_stride
        if self.context:
            new_parameters = self.context.evaluate_expr(parameters)
        else:
            new_parameters = parameters
        self.blk_stride_value = new_parameters

    def set_rep_stride_value(self, rep_stride=None):
        """
        set rep_stride_value

        Returns
        -------
        None
        """
        if rep_stride is None:
            rep_stride = self.rep_stride
        if self.context:
            new_parameters = self.context.evaluate_expr(rep_stride)
        else:
            new_parameters = rep_stride
        self.rep_stride_value = new_parameters

    def set_src_mask(self, src_mask=None):
        """
        get src_mask

        Parameters
        ----------
        src_mask

        Returns
        -------
        None
        """
        self.src_mask_value = src_mask

    def set_src_mask_value(self):
        """

        Returns
        -------
        None
        """
        if self.src_mask_value is not None:
            if not isinstance(self.src_mask_value, (list, tuple)):
                self.src_mask_value = [self.src_mask_value]
        else:
            self.src_mask_value = self.mask_value

    def check_tensor_and_scope(self):
        """
        check tensor and scope

        Returns
        -------
        None
        """
        if not self.context:
            # check tensor
            TikCheckUtil.check_type_match(self.tensor_obj, Tensor, "%s should be Tensor, input type: %s"
                                          % (self.tensor_op_name, type(self.tensor_obj)))
            # check scope
            TikCheckUtil.check_equality(self.tensor_obj.scope, scope_ubuf,
                                        "%s's scope must be UB" % self.tensor_op_name)

    def check_tensor_op_rep_stride(self, max_rep_stride):
        """
        check tensor_op repeat stride

        Parameters
        ----------
        max_rep_stride: max repeat stride
        Returns
        -------
        None
        """
        TikCheckUtil.check_type_match(self.rep_stride_value, (int, BasicExpr),
                                      "%s_rep_stride should be int, Expr or Scalar, input type is %s"
                                      % (self.tensor_op_name, type(self.rep_stride_value)))

        check_scalar_dtype(self.rep_stride_value, "scalar %s_rep_stride should be a scalar of int/uint"
                           % self.tensor_op_name)
        TikCheckUtil.check_in_range_by_dtype(
            self.rep_stride_value, msg="%s_rep_stride should be in the range of [%d, %d], input value is %s"
                                       % (self.tensor_op_name, DEFAULT_STRIDE, max_rep_stride, self.rep_stride_value),
            var_range=[DEFAULT_STRIDE, max_rep_stride])

    def check_tensor_op_blk_stride(self, max_blk_stride):
        """
        check tensor_op block stride

        Parameters
        ----------
        max_blk_stride: max block stride

        Returns
        -------
        None
        """
        TikCheckUtil.check_type_match(self.blk_stride_value, (int, BasicExpr),
                                      "%s_blk_stride should be int, Expr or Scalar, input type is %s"
                                      % (self.tensor_op_name, type(self.blk_stride_value)))

        check_scalar_dtype(self.blk_stride_value, "scalar_%s_blk_stride should be a scalar of int/uint"
                           % self.tensor_op_name)

        TikCheckUtil.check_in_range_by_dtype(
            self.blk_stride_value, msg="%s_blk_stride should be in the range of [%d, %d], input value is %s"
                                       % (self.tensor_op_name, DEFAULT_STRIDE, max_blk_stride, self.blk_stride_value),
            var_range=[DEFAULT_STRIDE, max_blk_stride])

    def check_tensor_op_address_align(self, name, align=None):
        """
        check tensor_op address align

        Returns
        -------
        None
        """
        if not self.context:
            if align is None:
                align = {"vlrelu": get_compatible_blk_size()}.get(name, vec_template_align(self.tensor_obj.dtype))

            tensor_start = Expr(self.tensor_obj.offset).eval_value()
            if tensor_start is not None and tensor_start * DTYPE_SIZE[self.tensor_obj.dtype] % align != 0:
                TikCheckUtil.raise_error("Address align error, %s[%s] is not %s Byte align"
                                         % (self.tensor_op_name, tensor_start, align))


class TensorOp(TensorBaseOp):
    """
    Tensor Ops
    """

    def __init__(self, tensor_obj, blk_stride, rep_stride, tensor_op_name):
        super().__init__(tensor_obj, blk_stride, rep_stride, tensor_op_name)
        self.tensor_obj = tensor_obj
        self.blk_stride = blk_stride
        self.rep_stride = rep_stride
        self.tensor_op_name = tensor_op_name
        self.context = None
        self.blk_stride_value = self.blk_stride
        self.rep_stride_value = self.rep_stride
        self.block_len = None
        self.nblock = None
        self.offset_value = None
        self.stride_unit_value = 0
        self.mask_value = None
        self.repeat_times_value = None
        self.is_910b = False
        self.one_rep_size = get_compatible_rep_size()
        self.one_blk_size = get_compatible_blk_size()

    @staticmethod
    def check_overlapping(src_extend_interval, dst_extend_interval):
        """
        check overlapping
        """
        for interval_dst in dst_extend_interval:
            for interval_src in src_extend_interval:
                if max(interval_src[0], interval_dst[0]) < min(interval_src[1], interval_dst[1]):
                    return False
        return True

    @staticmethod
    def __check_overlap_param(mask, repeat, dst_offset, src_offset):
        """
        get a flag of address overlap check
        """
        is_check = True
        if repeat <= 0:
            is_check = False
        if dst_offset is None or src_offset is None:
            is_check = False
        if is_basic_expr(TikUtil.to_list(mask)):
            is_check = False
        return is_check

    @staticmethod
    def _check_tensor_op_address_overlapping(tensor_op, print_name, control_op, block_list, other_tensor_op):
        try:
            tensor_op.check_address_overlapping(print_name, control_op, other_tensor_op, block_list)
        except(RuntimeError, SystemExit):
            TikCheckUtil.raise_error(
                "when repeat_times>1, %s dst and src1 address overlap is not support any address "
                "overlapping between src0 and src1" % print_name)

    def vector_max_offset_cal(self, mask, control_op, repeat_times, nblock=BLK_NUM_PER_REP):
        """
        get max offset of calculate vector

        Returns
        -------
        max_offset
        """
        self.set_repeat_times(repeat_times)
        if not isinstance(mask, (list, tuple)):
            mask = [mask]
        # mask_len, the last effective element
        mask_len = 0
        bit_len = get_bit_len(self.tensor_obj.dtype)
        if bit_len == BIT_LEN_32:
            mask_len = get_32bit_dtype_mask_len(mask, control_op.mask_mode)
        elif bit_len == BIT_LEN_8 or BIT_LEN_16:
            mask_len = get_8or16bit_dtype_mask_len(mask)

        if control_op.mask_mode == "normal":
            max_offset = self._calculate_vecotor_max_offset(self.repeat_times_value, mask_len)
        # counter mode, offset at penultimate repeat may be larger than offset
        # at last repeat
        else:
            if not self.nblock:
                self.nblock = nblock
            full_mask = self.nblock * self.block_len
            if mask_len == full_mask:
                max_offset = self._calculate_vecotor_max_offset(self.repeat_times_value, mask_len)
            else:
                offset_0 = self._calculate_vecotor_max_offset(self.repeat_times_value - 1, full_mask)
                offset_1 = self._calculate_vecotor_max_offset(self.repeat_times_value, mask_len)
                max_offset = max(offset_0, offset_1)

        return max_offset

    def check_read_mem_out_of_bounds(self, src_buffer_size, control_op, ori_offset=0):
        """
        check if out-of-bounds visit when it's read mode for a tensor

        Parameters
        ----------
        control_op: control_op
        src_buffer_size: tensor's buffer size
        ori_offset: tensor offset

        Return
        ----------
        None
        """
        self.block_len = self.one_rep_size // get_bit_len(self.tensor_obj.dtype)
        one_rep_ele = self.one_rep_size // DTYPE_SIZE[self.tensor_obj.dtype]
        repeat_times = control_op.repeat_times
        if isinstance(control_op.mask, (list, tuple)):
            mask_value = [self.context.evaluate_expr(value) for value in control_op.mask]
        else:
            mask_value = self.context.evaluate_expr(control_op.mask)
        if control_op.mask_mode == "counter":
            repeat_times = ceil_div(control_op.mask, one_rep_ele)
            mask_value = mask_value % one_rep_ele
            if mask_value == MASK_VALUE_ZERO:
                mask_value = one_rep_ele
        extend_offset = self.vector_max_offset_cal(mask_value, control_op, repeat_times)
        offset = get_flatten_idx(self.tensor_obj, self.context)
        expected_size = self.context.evaluate_expr(extend_offset + offset + ori_offset)
        total_size = src_buffer_size // get_dtype_size(self.tensor_obj.dtype)

        if expected_size > total_size:
            TikCheckUtil.raise_error(
                "AccessViolation: tensor %s need read %s elements, "
                "but only %s elements space" % (self.tensor_obj.name, expected_size, total_size))

    def set_blk_stride_value_with_print_name(self, print_name, control_op):
        """
        set block stride value with print_name
        Parameters
        ----------
        print_name: print_name
        control_op: control op

        Returns
        -------
        None
        """
        if print_name in SINGLE_NAME_DICT and isinstance(self.blk_stride, int) and self.blk_stride == 0:
            self.set_blk_stride_value(_MIN_DST_BLK_STRIDE)
        else:
            self.set_blk_stride_value()
        if print_name in FILLS_NAME_DICT:
            new_dst_bs = get_new_dst_bs(self.blk_stride, control_op.stride_unit)
            self.set_blk_stride_value(new_dst_bs)
        else:
            self.set_blk_stride_value()

    def check_tensor_op_overflow(self, print_name, control_op, block_list, context=None):
        """
        check tensor_op overflow

        Parameters
        ----------
        context: debug context
        print_name: print_name
        control_op: control_op
        block_list: block_list

        Returns
        -------
        None
        """
        if context is None:
            mask = control_op.mask
            repeat_times = control_op.repeat_times
        else:
            self.context = context
            control_op.eval_mask(context)
            self.set_repeat_times(control_op.repeat_times)
            mask = control_op.eval_mask_value
            repeat_times = self.repeat_times_value
        self.set_stride_unit(control_op.stride_unit)
        self.set_blk_stride_value_with_print_name(print_name, control_op)
        nblock, block_len = block_list

        self.block_len = block_len
        self.nblock = nblock
        ori_offset = 0
        if is_basic_expr(TikUtil.to_list(mask)) or \
                any(is_basic_expr([value]) for value in [repeat_times, self.blk_stride, self.rep_stride]):
            return
        if repeat_times == 0:
            return
        if not isinstance(self.nblock, int):
            return

        if control_op.mask_mode == "counter":
            mask, repeat_times = self.cal_mask_rep_for_counter_mode(mask, repeat_times)

        offset = self.tensor_obj.offset

        if isinstance(offset, (tvm.tir.IntImm, tvm.tir.FloatImm, tvm.tir.StringImm)):
            offset = offset.value
        total_size = reduce_mul(self.tensor_obj.original_shape)
        extend_offset = self.vector_max_offset_cal(mask, control_op, repeat_times)
        # offset means offset away from tensor head address, it's 16 for tensor[16]
        # entend_offset means valid data offset
        need_offset = get_need_offset(ori_offset, extend_offset, offset)
        if need_offset is not None:
            TikCheckUtil.check_le(
                need_offset, total_size,
                "%s tensor overflow, expected elements nums: %s, actual elements nums: %s"
                % (self.tensor_op_name, need_offset, total_size))

    def cal_mask_rep_for_counter_mode(self, mask, repeat_times):
        """
        call mask, repeat_times for counter mode

        Parameters
        ----------
        mask: mask
        repeat_times: repeat_times

        Returns
        -------

        """
        if not self.is_910b:
            repeat_times = ceil_div(mask, self.nblock * self.block_len)
            mask = mask % (self.nblock * self.block_len)
            if mask == MASK_VALUE_ZERO:
                mask = self.nblock * self.block_len
        else:
            self.nblock = ceil_div(mask * DTYPE_SIZE[self.tensor_obj.dtype], self.one_blk_size)
            mask = mask % (self.nblock * self.block_len)
            if mask == MASK_VALUE_ZERO:
                mask = self.nblock * self.block_len

        return mask, repeat_times

    def check_address_overlapping(self, print_name, control_op, src_tensor_op, block_list):
        """
        check address overlapping

        Parameters
        ----------
        block_list: list of nblock and block_len
        print_name: print_name
        control_op: control_op
        src_tensor_op: src_tensor_op

        Returns
        -------
        None
        """
        if print_name in SINGLE_NAME_DICT and self.blk_stride == 0:
            self.set_blk_stride_value(_MIN_DST_BLK_STRIDE)
        else:
            self.set_blk_stride_value()
        src_tensor_op.set_blk_stride_value()

        self.eval_offset()
        src_tensor_op.eval_offset()
        self.set_stride_unit(control_op.stride_unit)
        src_tensor_op.set_stride_unit(control_op.stride_unit)
        self.set_repeat_times(control_op.repeat_times)
        src_tensor_op.set_repeat_times(control_op.repeat_times)
        nblock, block_len = block_list
        if print_name == "vcbd":
            src_dst_dtype = DTYPE_MAP.get(src_tensor_op.tensor_obj.dtype) + "2" + DTYPE_MAP.get(self.tensor_obj.dtype)
            src_nblock = DTYPE_BLOCK_NUM.get(src_dst_dtype)[0]
            dst_nblock = DTYPE_BLOCK_NUM.get(src_dst_dtype)[1]
        else:
            dst_nblock = BLK_NUM_PER_REP
            src_nblock = BLK_NUM_PER_REP
        if not self.__check_overlap_param(control_op.mask, self.repeat_times_value, self.offset_value,
                                          src_tensor_op.offset_value):
            return

        self.mask_value = control_op.mask
        if control_op.mask_mode == "counter":
            if not self.is_910b:
                self.repeat_times_value = ceil_div(self.mask_value, nblock * block_len)
                mask = self.mask_value % (nblock * block_len)
                if mask == MASK_VALUE_ZERO:
                    mask = nblock * block_len
                self.mask_value = mask
            else:
                # for vcopy, counter mode is special
                dst_nblock = ceil_div(self.mask_value * DTYPE_SIZE[self.tensor_obj.dtype], self.one_blk_size)
                src_nblock = ceil_div(self.mask_value * DTYPE_SIZE[src_tensor_op.tensor_obj.dtype], self.one_blk_size)
                mask = nblock * block_len
                self.mask_value = mask
        self.get_mask(self.mask_value)
        if self.repeat_times_value == 1:
            self.check_overlapping_single_extend(print_name, src_tensor_op)
        else:
            self.check_overlapping_many_extend(print_name, src_tensor_op, dst_nblock=dst_nblock, src_nblock=src_nblock)

    def check_overlapping_many_extend(self, print_name, src_tensor_op, dst_nblock, src_nblock):
        """
        check overlapping double extend
        Parameters
        ----------
        src_nblock
        dst_nblock
        print_name: print_name
        src_tensor_op: src_tensor_op

        Returns
        -------
        None
        """
        dst_block_len = self._get_dst_block_len(print_name)
        src_block_len = self.one_rep_size // get_bit_len(src_tensor_op.tensor_obj.dtype)
        dst_blk_valid_list = get_blk_valid_list(self.mask_value, self.tensor_obj.dtype, dst_block_len)
        self.set_src_mask_value()
        src_blk_valid_list = get_blk_valid_list(self.src_mask_value, src_tensor_op.tensor_obj.dtype, src_block_len)
        dst_block_list = [dst_block_len, dst_nblock]
        src_block_list = [src_block_len, src_nblock]
        for time in range(self.repeat_times_value - 1):
            double_dst_extend_interval = self.get_extend_interval(dst_blk_valid_list, time,
                                                                  dst_block_list)
            double_src_extend_interval = src_tensor_op.get_extend_interval(src_blk_valid_list, time + 1, src_block_list)
            if self.check_overlapping(double_src_extend_interval, double_dst_extend_interval) is False:
                TikCheckUtil.raise_error(
                    "When repeat_times>1, %s %s and %s address overlapping error. It is not support "
                    "iteration N's destination is the source of next iteration"
                    % (print_name, self.tensor_op_name, src_tensor_op.tensor_op_name))

    def check_overlapping_single_extend(self, print_name, src_tensor_op):
        """
        check overlapping single extend
        Parameters
        ----------
        print_name: print_name
        src_tensor_op: src_tensor_op

        Returns
        -------
        None
        """
        dst_block_len = self._get_dst_block_len(print_name)
        src_block_len = self.one_rep_size // get_bit_len(src_tensor_op.tensor_obj.dtype)
        dst_blk_valid_list = get_blk_valid_list(self.mask_value, self.tensor_obj.dtype, dst_block_len)
        self.set_src_mask_value()
        src_blk_valid_list = get_blk_valid_list(self.src_mask_value, src_tensor_op.tensor_obj.dtype, src_block_len)
        nblock = BLK_NUM_PER_REP
        dst_block_list = [dst_block_len, nblock]
        src_block_list = [src_block_len, nblock]
        if not (src_tensor_op.offset_value == self.offset_value and
                src_tensor_op.blk_stride_value == self.blk_stride_value):
            single_dst_extend_interval = self.get_extend_interval(dst_blk_valid_list,
                                                                  self.repeat_times_value - 1, dst_block_list)
            single_src_extend_interval = src_tensor_op.get_extend_interval(src_blk_valid_list,
                                                                           self.repeat_times_value - 1, src_block_list)
            if self.check_overlapping(single_src_extend_interval, single_dst_extend_interval) is False:
                TikCheckUtil.raise_error(
                    "%s address overlapping error. when repeat_times=1, only support %s and %s "
                    "100 percent same, and stride must be same too." % (print_name, self.tensor_op_name,
                                                                        src_tensor_op.tensor_op_name))

    def get_extend_interval(self, blk_valid_list, time, block_list, offset_value=None):
        """
        get tensor each blk interval

        Parameters
        ----------
        offset_value: offset value
        block_list: src_tensor_op
        blk_valid_list: block valid list
        time: repeat_times

        Returns
        -------
        extend_interval
        """
        if offset_value is not None:
            self.offset_value = offset_value
        block_len, nblock = block_list
        extend_interval = []
        for blk_id in blk_valid_list:
            # blk_stride, rep_stride, unit: 32B
            if self.stride_unit_value == 0:
                begin = blk_id * self.blk_stride_value * block_len + time * self.rep_stride_value * block_len + \
                        self.offset_value
            # blk_stride, rep_gap, unit: 32B
            elif self.stride_unit_value == 1:
                begin = blk_id * self.blk_stride_value * block_len + \
                        time * (((nblock - 1) * self.blk_stride_value + 1) * block_len + self.rep_stride_value *
                                block_len) + self.offset_value
            # stride1: blk_gap, stride2: rep_stride, unit: elements
            elif self.stride_unit_value == 2:
                begin = blk_id * (self.blk_stride_value + block_len) + time * self.rep_stride_value + self.offset_value
            # blk_gap, rep_gap, unit: elements
            else:
                begin = blk_id * (block_len + self.blk_stride_value) + \
                        time * (((nblock - 1) * self.blk_stride_value + 1) * block_len + self.rep_stride_value *
                                block_len) + self.offset_value
            end = begin + block_len
            extend_interval.append([begin * DTYPE_SIZE[self.tensor_obj.dtype], end * DTYPE_SIZE[self.tensor_obj.dtype]])
        return extend_interval

    def check_dst_src1_overlap_other(self, print_name, control_op, src0_tensor_op, src1_tensor_op):
        """
        check dst src1 overlap other case
        """
        self.eval_offset()
        src1_tensor_op.eval_offset()
        self.set_rep_stride_value()
        src1_tensor_op.set_rep_stride_value()
        block_list = [BLK_NUM_PER_REP, self.one_rep_size // max(get_bit_len(self.tensor_obj.dtype),
                                                                get_bit_len(src1_tensor_op.tensor_obj.dtype))]
        if self.offset_value is not None and src1_tensor_op.offset_value is not None:
            value_range = (self.offset_value, src1_tensor_op.offset_value, self.blk_stride_value,
                           src1_tensor_op.blk_stride_value, self.rep_stride, src1_tensor_op.rep_stride)
            if all(isinstance(value, int) for value in value_range):
                if self.offset_value != src1_tensor_op.offset_value or \
                        self.blk_stride_value != src1_tensor_op.blk_stride_value \
                        or self.rep_stride != 0 or src1_tensor_op.rep_stride != 0:
                    self.check_address_overlapping(print_name, control_op, src1_tensor_op, block_list)

            if src0_tensor_op.tensor_obj.buffer == src1_tensor_op.tensor_obj.buffer:
                self.__check_address_overlapping_err(print_name, control_op, src0_tensor_op, src1_tensor_op)

    def check_tensor_op_valid(self, name, max_blk_stride, max_rep_stride, align=None):
        """
        check tensor_op valid

        Parameters
        ----------
        align: align
        name: name
        max_blk_stride: max block stride
        max_rep_stride: max repeat stride

        Returns
        -------
        None
        """
        # check tensor and scope
        self.check_tensor_and_scope()
        # check address_align
        self.check_tensor_op_address_align(name, align)
        # check blk_stride
        self.check_tensor_op_blk_stride(max_blk_stride)
        # check rep_stride
        self.check_tensor_op_rep_stride(max_rep_stride)

    def check_tensor_op_stride(self, max_blk_stride, max_rep_stride):
        """
        check tensor_op valid

        Parameters
        ----------
        max_blk_stride: max block stride
        max_rep_stride: max rep stride

        Returns
        -------

        """
        # check blk_stride
        self.check_tensor_op_blk_stride(max_blk_stride)
        # check rep_stride
        self.check_tensor_op_rep_stride(max_rep_stride)

    def get_extent(self, control_op):
        """
        Returns self._extent: current tensor extent
        """
        rep_times = self.repeat_times_value
        if self.repeat_times_value is None:
            rep_times = control_op.repeat_times
        rep_stride = self.rep_stride_value
        blk_stride = self.blk_stride_value
        stride_unit = control_op.stride_unit
        elem_size_in_byte = DTYPE_SIZE.get(self.tensor_obj.dtype)
        blk_len_in_elem = self.one_blk_size // elem_size_in_byte
        # blk_stride: stride, rep_stride: stride, unit: 32B
        if stride_unit == 0:
            repeat_size = (BLK_NUM_PER_REP - 1) * blk_stride + 1
            extent = ((rep_times - 1) * rep_stride + repeat_size) * self.one_blk_size
        # blk_stride: stride, rep_stride: gap, unit: 32B
        elif stride_unit == 1:
            repeat_size = (BLK_NUM_PER_REP - 1) * blk_stride + 1
            extent = ((rep_times - 1) * (rep_stride + repeat_size) + repeat_size) * self.one_blk_size
        # blk_stride: gap, rep_stride: stride, unit: element
        elif stride_unit == 2:
            repeat_size = (BLK_NUM_PER_REP - 1) * (blk_stride + blk_len_in_elem) + blk_len_in_elem
            extent = ((rep_times - 1) * rep_stride + repeat_size) * elem_size_in_byte
        # blk_stride: gap, rep_stride: gap, unit: element
        else:
            repeat_size = (BLK_NUM_PER_REP - 1) * (blk_stride + blk_len_in_elem) + blk_len_in_elem
            extent = ((rep_times - 1) * (rep_stride + repeat_size) + repeat_size) * elem_size_in_byte
        return Expr(extent).get()

    def __check_address_overlapping_err(self, print_name, control_op, src0_tensor_op, src1_tensor_op):
        block_list = [BLK_NUM_PER_REP, self.one_rep_size // max(get_bit_len(src0_tensor_op.tensor_obj.dtype),
                                                                get_bit_len(src1_tensor_op.tensor_obj.dtype))]
        if self.context:
            self._check_tensor_op_address_overlapping(
                src0_tensor_op, print_name, control_op, block_list, src1_tensor_op)
            self._check_tensor_op_address_overlapping(
                src1_tensor_op, print_name, control_op, block_list, src0_tensor_op)
        else:
            if all(isinstance(value, int)
                   for value in (src0_tensor_op.blk_stride_value, src0_tensor_op.rep_stride_value)):
                self._check_tensor_op_address_overlapping(
                    src0_tensor_op, print_name, control_op, block_list, src1_tensor_op)
                self._check_tensor_op_address_overlapping(
                    src1_tensor_op, print_name, control_op, block_list, src0_tensor_op)

    def _get_dst_block_len(self, print_name):
        if print_name in ("vcmin", "vcmax"):
            dst_block_len = 2
        elif print_name == "vcadd":
            dst_block_len = 1
        elif print_name in ("vcgadd", "vcgmax", "vcgmin"):
            dst_block_len = 8
        elif print_name in ("vcpadd", "vec_cpadd"):
            dst_block_len = 64
        else:
            dst_block_len = self.one_rep_size // get_bit_len(self.tensor_obj.dtype)
        return dst_block_len

    def _calculate_vecotor_max_offset(self, repeat_times, mask_len):
        if isinstance(repeat_times, int) and repeat_times == 0:
            return 0
        blk_num_last_rep = ceil_div(mask_len, self.block_len)
        ele_num_last_blk = mask_len % self.block_len if mask_len % self.block_len else self.block_len

        # last rep has multi blocks, blk_stride is zero, last blk num must be set to block len
        if blk_num_last_rep > 0 and self.blk_stride_value == 0:
            ele_num_last_blk = self.block_len
        # blk_stride: stride, rep_stride: stride, unit: 32B
        if self.stride_unit_value == 0:
            max_offset = ((repeat_times - 1) * self.rep_stride_value +
                          (blk_num_last_rep - 1) * self.blk_stride_value) * self.block_len + \
                           ele_num_last_blk
        # blk_stride: stride, rep_stride: gap, unit: 32B
        elif self.stride_unit_value == 1:
            max_offset = \
                ((repeat_times - 1) * (self.rep_stride_value + (self.nblock - 1) * self.blk_stride_value + 1)
                 + (blk_num_last_rep - 1) * self.blk_stride_value) * self.block_len + \
                ele_num_last_blk
        # blk_stride: gap, rep_stride: stride, unit: element
        elif self.stride_unit_value == 2:
            max_offset = (repeat_times - 1) * self.rep_stride_value + \
                         (blk_num_last_rep - 1) * (self.block_len + self.blk_stride_value) + \
                         ele_num_last_blk
        # blk_stride: gap, rep_stride: gap, unit: element
        else:
            max_offset = \
                (repeat_times - 1) * \
                (self.rep_stride_value + (self.nblock - 1) * (self.blk_stride_value + self.block_len) +
                 self.block_len) + \
                (blk_num_last_rep - 1) * (self.blk_stride_value + self.block_len) + \
                ele_num_last_blk

        return max_offset
