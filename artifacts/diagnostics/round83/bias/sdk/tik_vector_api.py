#!/usr/bin/env python
# -*- coding:utf-8 -*-
"""
Copyright (C) 2019. Huawei Technologies Co., Ltd. All rights reserved.
FILE:     tik_vector_api.py
DESC:     provide vector instructions
CREATED:  2020-3-16 21:12:13
MODIFIED: 2021-11-17 16:17:18
"""
from collections import namedtuple

from tbe.tik.common.common_util import is_tensor
from tbe.tik.common.common_nametuple_util import VconvApi
from tbe.tik.common.common_nametuple_util import VecSelApi
from tbe.tik.common.common_nametuple_util import VecSingleApi
from tbe.tik.common.common_nametuple_util import VecScalarMultisApi
from tbe.tik.common.common_nametuple_util import HighPreciParams
from tbe.tik.common.common_nametuple_util import VlnHighPreciTuple
from tbe.tik.tik_lib.tik_vector_api.tik_scalar_multis_api import ScalarMultisApi
from tbe.tik.tik_lib.tik_vector_api.tik_scalar_multis_api_ import ScalarMultisOps
from tbe.tik.tik_lib.tik_vector_api.tik_vector_high_preci import TikHighPrec
from tbe.tik.tik_lib.tik_vector_api.tik_vector_single_api_ import NanoSingleOpApi
from tbe.tik.tik_lib.tik_source_info import source_info_decorator
from tbe.tik.tik_lib.tik_vector_api.tik_vector_multi_api_ import MultiOp
from tbe.tik.tik_lib.tik_vector_api.tik_vector_single_api_ import SingleOpApi
from tbe.tik.tik_lib.tik_vector_api.tik_vector_compare_api_ import VconvOpApi
from tbe.tik.tik_lib.tik_vector_api.tik_vector_compare_api_ import VconvOpApiNano
from tbe.tik.tik_lib.tik_params import MIN_STRIDE
from tbe.tik.tik_lib.tik_params import MIN_STRIDE_UNIT
from tbe.tik.tik_lib.tik_soc_manager import TikSocManager
from tbe.tik.tik_lib.tik_vector_api.tik_vector_multi_api_ import NanoMultiOpApi
from tbe.tik.tik_lib.tik_vector_api.tik_scalar_multis_api_ import NanoScalarMultisOps

# round disable
_ROUND_TO_NEAREST_ENABLE = 0


class TikVectorApiv1(TikHighPrec):
    """
    Vector Api for open
    """
    multi_api = namedtuple('MultiApi', ["name", "mask", "dst", "src0", "src1", "repeat_times", "dst_blk_stride",
                                        "src0_blk_stride", "src1_blk_stride", "dst_rep_stride", "src0_rep_stride",
                                        "src1_rep_stride", "stride_unit"])

    single_api = namedtuple('SingleApi', ["name", "mask", "dst", "src", "repeat_times", "dst_blk_stride",
                                          "src_blk_stride", "dst_rep_stride", "src_rep_stride", "stride_unit"])

    scalar_multis_api = namedtuple('ScalarMultisApi', ["name", "mask", "dst", "src", "scalar", "repeat_times",
                                                       "dst_blk_stride", "src_blk_stride", "dst_rep_stride",
                                                       "src_rep_stride", "stride_unit", "round_en", "mask_mode"])

    def vec_add(self, mask, dst, src0, src1, repeat_times, dst_rep_stride, src0_rep_stride, src1_rep_stride):
        """
        Performs addition element-wise

        Parameter
        ----------
        mask : Effective operation on element, divided into two model: Continuous and bit by bit.
        dst : Destination operand.
        src0 : Source0 operand.
        src1 : Source1 operand.
        repeat_times : Number of iteration repeats
        dst_rep_stride : Block-to-block stride between adjacent iterations of the destination operand.
        src0_rep_stride : Block-to-block stride between adjacent iterations of the source0 operand.
        src1_rep_stride : Block-to-block stride between adjacent iterations of the source1 operand.

        Returns
        ----------
        no return
        """
        default_blk_stride = 1
        multi_api_ins = TikVectorApiv1.multi_api("vec_add", mask, dst, src0, src1, repeat_times, default_blk_stride,
                                                 default_blk_stride, default_blk_stride, dst_rep_stride,
                                                 src0_rep_stride,
                                                 src1_rep_stride, stride_unit=0)
        if TikSocManager.is_nano_soc():
            vec_add_obj = NanoMultiOpApi(self, multi_api_ins)
        else:
            vec_add_obj = MultiOp(self, multi_api_ins)
        vec_add_obj.run_all()

    def vec_sub(self, mask, dst, src0, src1, repeat_times, dst_rep_stride, src0_rep_stride, src1_rep_stride):
        """
        Performs subtraction element-wise

        Parameter
        ----------
        mask : Effective operation on element, divided into two model: Continuous and bit by bit.
        dst : Destination operand.
        src0 : Source0 operand.
        src1 : Source1 operand.
        repeat_times : Number of iteration repeats
        dst_rep_stride : Block-to-block stride between adjacent iterations of the destination operand.
        src0_rep_stride : Block-to-block stride between adjacent iterations of the source0 operand.
        src1_rep_stride : Block-to-block stride between adjacent iterations of the source1 operand.

        Returns
        ----------
        no return
        """
        default_blk_stride = 1
        multi_api_ins = TikVectorApiv1.multi_api("vec_sub", mask, dst, src0, src1, repeat_times, default_blk_stride,
                                                 default_blk_stride, default_blk_stride, dst_rep_stride,
                                                 src0_rep_stride,
                                                 src1_rep_stride, stride_unit=0)
        if TikSocManager.is_nano_soc():
            vec_sub_obj = NanoMultiOpApi(self, multi_api_ins)
        else:
            vec_sub_obj = MultiOp(self, multi_api_ins)
        vec_sub_obj.run_all()

    def vec_max(self, mask, dst, src0, src1, repeat_times, dst_rep_stride, src0_rep_stride, src1_rep_stride):
        """
        Computes the maximum element-wise

        Parameter
        ----------
        mask : Effective operation on element, divided into two model: Continuous and bit by bit.
        dst : Destination operand.
        src0 : Source0 operand.
        src1 : Source1 operand.
        repeat_times : Number of iteration repeats
        dst_rep_stride : Block-to-block stride between adjacent iterations of the destination operand.
        src0_rep_stride : Block-to-block stride between adjacent iterations of the source0 operand.
        src1_rep_stride : Block-to-block stride between adjacent iterations of the source1 operand.

        Returns
        ----------
        no return
        """
        default_blk_stride = 1
        multi_api_ins = TikVectorApiv1.multi_api("vec_max", mask, dst, src0, src1, repeat_times, default_blk_stride,
                                                 default_blk_stride, default_blk_stride, dst_rep_stride,
                                                 src0_rep_stride,
                                                 src1_rep_stride, stride_unit=0)
        if TikSocManager.is_nano_soc():
            vec_max_obj = NanoMultiOpApi(self, multi_api_ins)
        else:
            vec_max_obj = MultiOp(self, multi_api_ins)
        vec_max_obj.run_all()

    def vec_min(self, mask, dst, src0, src1, repeat_times, dst_rep_stride, src0_rep_stride, src1_rep_stride):
        """
        Computes the minimum element-wise

        Parameter
        ----------
        mask : Effective operation on element, divided into two model: Continuous and bit by bit.
        dst : Destination operand.
        src0 : Source0 operand.
        src1 : Source1 operand.
        repeat_times : Number of iteration repeats
        dst_rep_stride : Block-to-block stride between adjacent iterations of the destination operand.
        src0_rep_stride : Block-to-block stride between adjacent iterations of the source0 operand.
        src1_rep_stride : Block-to-block stride between adjacent iterations of the source1 operand.

        Returns
        ----------
        no return
        """
        default_blk_stride = 1
        multi_api_ins = TikVectorApiv1.multi_api("vec_min", mask, dst, src0, src1, repeat_times, default_blk_stride,
                                                 default_blk_stride, default_blk_stride, dst_rep_stride,
                                                 src0_rep_stride,
                                                 src1_rep_stride, stride_unit=0)
        if TikSocManager.is_nano_soc():
            vec_min_obj = NanoMultiOpApi(self, multi_api_ins)
        else:
            vec_min_obj = MultiOp(self, multi_api_ins)
        vec_min_obj.run_all()

    def vec_mul(self, mask, dst, src0, src1, repeat_times, dst_rep_stride, src0_rep_stride, src1_rep_stride):
        """
        Performs multiplication element-wise

        Parameter
        ----------
        mask : Effective operation on element, divided into two model: Continuous and bit by bit.
        dst : Destination operand.
        src0 : Source0 operand.
        src1 : Source1 operand.
        repeat_times : Number of iteration repeats
        dst_rep_stride : Block-to-block stride between adjacent iterations of the destination operand.
        src0_rep_stride : Block-to-block stride between adjacent iterations of the source0 operand.
        src1_rep_stride : Block-to-block stride between adjacent iterations of the source1 operand.

        Returns
        ----------
        no return
        """
        default_blk_stride = 1
        multi_api_ins = TikVectorApiv1.multi_api("vec_mul", mask, dst, src0, src1, repeat_times, default_blk_stride,
                                                 default_blk_stride, default_blk_stride, dst_rep_stride,
                                                 src0_rep_stride,
                                                 src1_rep_stride, stride_unit=0)
        if TikSocManager.is_nano_soc():
            vec_mul_obj = NanoMultiOpApi(self, multi_api_ins)
        else:
            vec_mul_obj = MultiOp(self, multi_api_ins)
        vec_mul_obj.run_all()

    def vec_and(self, mask, dst, src0, src1, repeat_times, dst_rep_stride, src0_rep_stride, src1_rep_stride):
        """
        Performs addition element-wise

        Parameter
        ----------
        mask : Effective operation on element, divided into two model: Continuous and bit by bit.
        dst : Destination operand.
        src0 : Source0 operand.
        src1 : Source1 operand.
        repeat_times : Number of iteration repeats
        dst_rep_stride : Block-to-block stride between adjacent iterations of the destination operand.
        src0_rep_stride : Block-to-block stride between adjacent iterations of the source0 operand.
        src1_rep_stride : Block-to-block stride between adjacent iterations of the source1 operand.

        Returns
        ----------
        no return
        """
        default_blk_stride = 1
        multi_api_ins = TikVectorApiv1.multi_api("vec_and", mask, dst, src0, src1, repeat_times, default_blk_stride,
                                                 default_blk_stride, default_blk_stride, dst_rep_stride,
                                                 src0_rep_stride,
                                                 src1_rep_stride, stride_unit=0)
        if TikSocManager.is_nano_soc():
            vec_and_obj = NanoMultiOpApi(self, multi_api_ins)
        else:
            vec_and_obj = MultiOp(self, multi_api_ins)
        vec_and_obj.run_all()

    def vec_or(self, mask, dst, src0, src1, repeat_times, dst_rep_stride, src0_rep_stride, src1_rep_stride):
        """
        Performs bit-wise OR element-wise

        Parameter
        ----------
        mask : Effective operation on element, divided into two model: Continuous and bit by bit.
        dst : Destination operand.
        src0 : Source0 operand.
        src1 : Source1 operand.
        repeat_times : Number of iteration repeats
        dst_rep_stride : Block-to-block stride between adjacent iterations of the destination operand.
        src0_rep_stride : Block-to-block stride between adjacent iterations of the source0 operand.
        src1_rep_stride : Block-to-block stride between adjacent iterations of the source1 operand.

        Returns
        ----------
        no return
        """
        default_blk_stride = 1
        multi_api_ins = TikVectorApiv1.multi_api("vec_or", mask, dst, src0, src1, repeat_times, default_blk_stride,
                                                 default_blk_stride, default_blk_stride, dst_rep_stride,
                                                 src0_rep_stride,
                                                 src1_rep_stride, stride_unit=0)
        if TikSocManager.is_nano_soc():
            vec_or_obj = NanoMultiOpApi(self, multi_api_ins)
        else:
            vec_or_obj = MultiOp(self, multi_api_ins)
        vec_or_obj.run_all()

    def vec_relu(self, mask, dst, src, repeat_times, dst_rep_stride, src_rep_stride):
        """
        Performs a ReLU operation element-wise

        Parameter
        ----------
        mask : Effective operation on element, divided into two model: Continuous and bit by bit.
        dst : Destination operand.
        src : Source operand.
        repeat_times : Number of iteration repeats
        dst_rep_stride : Block-to-block stride between adjacent iterations of the destination operand
        src_rep_stride : Block-to-block stride between adjacent iterations of the source operand

        Returns
        ----------
        no return
        """
        blk_stride = 1
        single_api = TikVectorApiv1.single_api('vec_relu', mask, dst, src, repeat_times, blk_stride, blk_stride,
                                               dst_rep_stride, src_rep_stride, stride_unit=0)
        if TikSocManager.is_nano_soc():
            vec_relu_obj = NanoSingleOpApi(self, single_api)
        else:
            vec_relu_obj = SingleOpApi(self, single_api)
        vec_relu_obj.run_all()

    def vec_ln(self, mask, dst, src, repeat_times, dst_rep_stride, src_rep_stride):
        """
        Computes the natural logarithm element-wise

        Parameter
        ----------
        mask : Effective operation on element, divided into two model: Continuous and bit by bit.
        dst : Destination operand.
        src : Source operand.
        repeat_times : Number of iteration repeats
        dst_rep_stride : Block-to-block stride between adjacent iterations of the destination operand
        src_rep_stride : Block-to-block stride between adjacent iterations of the source operand

        Returns
        ----------
        no return
        """
        blk_stride = 1
        single_api = TikVectorApiv1.single_api('vec_ln', mask, dst, src, repeat_times, blk_stride, blk_stride,
                                               dst_rep_stride, src_rep_stride, stride_unit=0)
        if TikSocManager.is_nano_soc():
            vec_ln_obj = NanoSingleOpApi(self, single_api)
        else:
            vec_ln_obj = SingleOpApi(self, single_api)
        vec_ln_obj.run_all()

    def vec_exp(self, mask, dst, src, repeat_times, dst_rep_stride, src_rep_stride):
        """
        Computes the natural exponential element-wise

        Parameter
        ----------
        mask : Effective operation on element, divided into two model: Continuous and bit by bit.
        dst : Destination operand.
        src : Source operand.
        repeat_times : Number of iteration repeats
        dst_rep_stride : Block-to-block stride between adjacent iterations of the destination operand
        src_rep_stride : Block-to-block stride between adjacent iterations of the source operand

        Returns
        ----------
        no return
        """
        blk_stride = 1
        single_api = TikVectorApiv1.single_api('vec_exp', mask, dst, src, repeat_times, blk_stride, blk_stride,
                                               dst_rep_stride, src_rep_stride, stride_unit=0)
        if TikSocManager.is_nano_soc():
            vec_exp_obj = NanoSingleOpApi(self, single_api)
        else:
            vec_exp_obj = SingleOpApi(self, single_api)
        vec_exp_obj.run_all()

    def vec_abs(self, mask, dst, src, repeat_times, dst_rep_stride, src_rep_stride):
        """
        Computes the absolute value element-wise

        Parameter
        ----------
        mask : Effective operation on element, divided into two model: Continuous and bit by bit.
        dst : Destination operand.
        src : Source operand.
        repeat_times : Number of iteration repeats
        dst_rep_stride : Block-to-block stride between adjacent iterations of the destination operand
        src_rep_stride : Block-to-block stride between adjacent iterations of the source operand

        Returns
        ----------
        no return
        """
        blk_stride = 1
        single_api = TikVectorApiv1.single_api('vec_abs', mask, dst, src, repeat_times, blk_stride, blk_stride,
                                               dst_rep_stride, src_rep_stride, stride_unit=0)
        if TikSocManager.is_nano_soc():
            vec_abs_obj = NanoSingleOpApi(self, single_api)
        else:
            vec_abs_obj = SingleOpApi(self, single_api)
        vec_abs_obj.run_all()

    @source_info_decorator()
    def vec_expm1_high_preci(self, mask, dst, src, work_tensor, repeat_times, dst_rep_stride, src_rep_stride):
        """
        Computes the natural base element-wise

        Parameter
        ----------
        mask : Effective operation on element, divided into two model: Continuous and bit by bit.
        dst : Destination operand.
        src : Source operand.
        work_tensor: User-defined temporary buffer space for storing intermediate results. The space is limited to
        scope_ubuf and is used for internal calculation.
        repeat_times : Number of iteration repeats
        dst_rep_stride : Block-to-block stride between adjacent iterations of the destination operand
        src_rep_stride : Block-to-block stride between adjacent iterations of the source operand

        Returns
        ----------
        no return
        """
        if TikSocManager.is_hisi_soc():
            default_blk_stride = 1
            stride_unit = 0

            expm1_exp_api = VecSingleApi("vec_expm1_high_preci", mask, dst, src, repeat_times, default_blk_stride,
                                         default_blk_stride, dst_rep_stride, src_rep_stride,
                                         stride_unit, 'vexp')
            vexp_obj = SingleOpApi(self, expm1_exp_api)
            vexp_obj.run_all()
            expm1_adds_api = VecScalarMultisApi("vec_expm1_high_preci", mask, dst, dst, -1, repeat_times,
                                                default_blk_stride, default_blk_stride,
                                                dst_rep_stride, dst_rep_stride, stride_unit,
                                                _ROUND_TO_NEAREST_ENABLE, 'normal', 'vadds')
            vadds_obj = ScalarMultisOps(self, expm1_adds_api)
            vadds_obj.run_all()

        else:
            vexpm_high_params = namedtuple('vexpm_high_api', ["name", "mask", "dst", "src", "work_tensor",
                                           "repeat_times", "dst_rep_stride", "src_rep_stride"])
            vexpm_high_params = vexpm_high_params("vec_expm1_high_preci", mask, dst, src, work_tensor, repeat_times,
                                                  dst_rep_stride, src_rep_stride)
            if TikSocManager.is_310b_610l_soc():
                self.vexpm_high_preci_high_level(vexpm_high_params)
            else:
                self.vexpm_high_preci(vexpm_high_params)
        self.set_high_level_api_state()

    def vec_rec(self, mask, dst, src, repeat_times, dst_rep_stride, src_rep_stride):
        """
        Computes the reciprocal element-wise

        Parameter
        ----------
        mask : Effective operation on element, divided into two model: Continuous and bit by bit.
        dst : Destination operand.
        src : Source operand.
        repeat_times : Number of iteration repeats
        dst_rep_stride : Block-to-block stride between adjacent iterations of the destination operand
        src_rep_stride : Block-to-block stride between adjacent iterations of the source operand

        Returns
        ----------
        no return
        """
        blk_stride = 1
        single_api = TikVectorApiv1.single_api('vec_rec', mask, dst, src, repeat_times, blk_stride, blk_stride,
                                               dst_rep_stride, src_rep_stride, stride_unit=0)
        if TikSocManager.is_nano_soc():
            vec_rec_obj = NanoSingleOpApi(self, single_api)
        else:
            vec_rec_obj = SingleOpApi(self, single_api)
        vec_rec_obj.run_all()

    @source_info_decorator()
    def vec_rec_high_preci(self, mask, dst, src, work_tensor, repeat_times, dst_rep_stride, src_rep_stride):
        """
        Computes the reciprocal element-wise

        Parameter
        ----------
        mask : Effective operation on element, divided into two model: Continuous and bit by bit.
        dst : Destination operand.
        src : Source operand.
        work_tensor: User-defined temporary buffer space for storing intermediate results. The space is limited to
        scope_ubuf and is used for internal calculation.
        repeat_times : Number of iteration repeats
        dst_rep_stride : Block-to-block stride between adjacent iterations of the destination operand
        src_rep_stride : Block-to-block stride between adjacent iterations of the source operand

        Returns
        ----------
        no return
        """
        instr_name = "vec_rec_high_preci"
        is_610_310p = TikSocManager.is_v200_soc() and not TikSocManager.is_hisi_soc()
        if TikSocManager.is_hisi_soc() or (is_tensor(src) and is_610_310p and src.dtype == "float32"):
            vrec_api = VecSingleApi(instr_name, mask, dst, src, repeat_times, MIN_STRIDE, MIN_STRIDE,
                                    dst_rep_stride, src_rep_stride, MIN_STRIDE_UNIT, 'vrec')
            vrec_obj = SingleOpApi(self, vrec_api)
            vrec_obj.run_all()
        else:
            vrec_api = HighPreciParams(dst, src, work_tensor, repeat_times, dst_rep_stride, src_rep_stride)
            if TikSocManager.is_310b_610l_soc():
                self._vrec_high_preci_high_level(mask, instr_name, vrec_api)
            else:
                self._vrec_high_preci(mask, instr_name, vrec_api)
        self.set_high_level_api_state()

    def vec_not(self, mask, dst, src, repeat_times, dst_rep_stride, src_rep_stride):
        """
        Performs bit-wise NOT element-wise

        Parameter
        ----------
        mask : Effective operation on element, divided into two model: Continuous and bit by bit.
        dst : Destination operand.
        src : Source operand.
        repeat_times : Number of iteration repeats
        dst_rep_stride : Block-to-block stride between adjacent iterations of the destination operand
        src_rep_stride : Block-to-block stride between adjacent iterations of the source operand

        Returns
        ----------
        no return
        """
        blk_stride = 1
        single_api = TikVectorApiv1.single_api('vec_not', mask, dst, src, repeat_times, blk_stride, blk_stride,
                                               dst_rep_stride, src_rep_stride, stride_unit=0)
        if TikSocManager.is_nano_soc():
            vec_not_obj = NanoSingleOpApi(self, single_api)
        else:
            vec_not_obj = SingleOpApi(self, single_api)
        vec_not_obj.run_all()

    def vec_rsqrt(self, mask, dst, src, repeat_times, dst_rep_stride, src_rep_stride):
        """
        Computes the reciprocal after extracting the square root element-wise

        Parameter
        ----------
        mask : Effective operation on element, divided into two model: Continuous and bit by bit.
        dst : Destination operand.
        src : Source operand.
        repeat_times : Number of iteration repeats
        dst_rep_stride : Block-to-block stride between adjacent iterations of the destination operand
        src_rep_stride : Block-to-block stride between adjacent iterations of the source operand

        Returns
        ----------
        no return
        """
        blk_stride = 1
        single_api = TikVectorApiv1.single_api('vec_rsqrt', mask, dst, src, repeat_times, blk_stride, blk_stride,
                                               dst_rep_stride, src_rep_stride, stride_unit=0)
        if TikSocManager.is_nano_soc():
            vec_rsqrt_obj = NanoSingleOpApi(self, single_api)
        else:
            vec_rsqrt_obj = SingleOpApi(self, single_api)
        vec_rsqrt_obj.run_all()

    @source_info_decorator()
    def vec_rsqrt_high_preci(self, mask, dst, src, work_tensor, repeat_times, dst_rep_stride, src_rep_stride):
        """
        Computes the reciprocal after extracting the square root element-wise

        Parameter
        ----------
        mask : Effective operation on element, divided into two model: Continuous and bit by bit.
        dst : Destination operand.
        src : Source operand.
        work_tensor: User-defined temporary buffer space for storing intermediate results. The space is limited to
        scope_ubuf and is used for internal calculation.
        repeat_times : Number of iteration repeats
        dst_rep_stride : Block-to-block stride between adjacent iterations of the destination operand
        src_rep_stride : Block-to-block stride between adjacent iterations of the source operand

        Returns
        ----------
        no return
        """
        instr_name = "vec_rsqrt_high_preci"
        if TikSocManager.is_hisi_soc():
            default_blk_stride = 1
            stride_unit = 0
            vrsqrt_obj = VecSingleApi(instr_name, mask, dst, src, repeat_times, default_blk_stride,
                                      default_blk_stride, dst_rep_stride, src_rep_stride, stride_unit, 'vrsqrt')

            vrsqrt_obj = SingleOpApi(self, vrsqrt_obj)
            vrsqrt_obj.run_all()
        else:
            vrsqrt_high_params = namedtuple('vrsqrt_high_api', ["name", "mask", "dst", "src", "work_tensor",
                                            "repeat_times", "dst_rep_stride", "src_rep_stride"])
            vrsqrt_high_params = vrsqrt_high_params(instr_name, mask, dst, src, work_tensor, repeat_times,
                                                    dst_rep_stride, src_rep_stride)
            if TikSocManager.is_310b_610l_soc():
                self._vrsqrt_high_preci_high_level(vrsqrt_high_params)
            else:
                self._vrsqrt_high_preci(vrsqrt_high_params)
        self.set_high_level_api_state()

    @source_info_decorator()
    def vec_axpy(self, mask, dst, src, scalar, repeat_times, dst_rep_stride, src_rep_stride):
        """
        Performs multiplication-accumulation between a vector and a scalar

        Parameter
        ----------
        mask : Effective operation on element, divided into two model: Continuous and bit by bit.
        dst : Destination operand.
        src : Source operand.
        scalar : A scalar or immediate, specifying the scalar source operand
        repeat_times : Number of iteration repeats
        dst_rep_stride : Block-to-block stride between adjacent iterations of the destination operand
        src_rep_stride : Block-to-block stride between adjacent iterations of the source operand

        Returns
        ----------
        no return
        """
        round_en = _ROUND_TO_NEAREST_ENABLE
        default_blk_stride = 1
        stride_unit = 0
        scalar_multis_api = ScalarMultisApi.scalar_multis_api(
            "vec_axpy", mask, dst, src, scalar, repeat_times, default_blk_stride, default_blk_stride, dst_rep_stride,
            src_rep_stride, stride_unit, round_en, mask_mode="normal")
        if TikSocManager.is_nano_soc():
            vec_axpy_obj = NanoScalarMultisOps(self, scalar_multis_api)
        else:
            vec_axpy_obj = ScalarMultisOps(self, scalar_multis_api)
        vec_axpy_obj.run_all()

    @source_info_decorator()
    def vec_adds(self, mask, dst, src, scalar, repeat_times, dst_rep_stride, src_rep_stride, mask_mode="normal"):
        """
        Performs addition between a vector and a scalar element-wise

        Parameter
        ----------
        mask : Effective operation on element, divided into two model: Continuous and bit by bit.
        dst : Destination operand.
        src : Source operand.
        scalar : A scalar or immediate, specifying the scalar source operand
        repeat_times : Number of iteration repeats
        dst_rep_stride : Block-to-block stride between adjacent iterations of the destination operand
        src_rep_stride : Block-to-block stride between adjacent iterations of the source operand
        mask_mode : A string specifying the mask mode.

        Returns
        ----------
        no return
        """
        default_blk_stride = 1
        stride_unit = 0
        round_en = _ROUND_TO_NEAREST_ENABLE
        scalar_multis_api = TikVectorApiv1.scalar_multis_api(
            "vec_adds", mask, dst, src, scalar, repeat_times, default_blk_stride, default_blk_stride, dst_rep_stride,
            src_rep_stride, stride_unit, round_en, mask_mode)
        if TikSocManager.is_nano_soc():
            vec_adds_obj = NanoScalarMultisOps(self, scalar_multis_api)
        else:
            vec_adds_obj = ScalarMultisOps(self, scalar_multis_api)
        vec_adds_obj.run_all()

    @source_info_decorator()
    def vec_muls(self, mask, dst, src, scalar, repeat_times, dst_rep_stride, src_rep_stride, mask_mode="normal"):
        """
        Performs multiplication between a vector and a scalar element-wise

        Parameter
        ----------
        mask : Effective operation on element, divided into two model: Continuous and bit by bit.
        dst : Destination operand.
        src : Source operand.
        repeat_times : Number of iteration repeats
        dst_rep_stride : Block-to-block stride between adjacent iterations of the destination operand
        src_rep_stride : Block-to-block stride between adjacent iterations of the source operand
        mask_mode : A string specifying the mask mode.

        Returns
        ----------
        no return
        """
        default_blk_stride = 1
        stride_unit = 0
        round_en = _ROUND_TO_NEAREST_ENABLE
        scalar_multis_api = TikVectorApiv1.scalar_multis_api(
            "vec_muls", mask, dst, src, scalar, repeat_times, default_blk_stride, default_blk_stride, dst_rep_stride,
            src_rep_stride, stride_unit, round_en, mask_mode)
        if TikSocManager.is_nano_soc():
            vec_muls_obj = NanoScalarMultisOps(self, scalar_multis_api)
        else:
            vec_muls_obj = ScalarMultisOps(self, scalar_multis_api)
        vec_muls_obj.run_all()

    @source_info_decorator()
    def vec_sel(self, mask, mode, dst, sel, src0, src1, repeat_times, dst_rep_stride=0, src0_rep_stride=0,
                src1_rep_stride=0):
        """
        Selects elements bit-wise. 1'b1: selected from src0; other values: selected from src1.

        Parameter
        ----------
        mask : Effective operation on element, divided into two model: Continuous and bit by bit.
        mode : Instruction mode
        dst : Destination operand.
        src0 : Source0 operand.
        src1 : Source1 operand.
        repeat_times : Number of iteration repeats
        dst_rep_stride : Block-to-block stride between adjacent iterations of the destination operand.
        src0_rep_stride : Block-to-block stride between adjacent iterations of the source0 operand.
        src1_rep_stride : Block-to-block stride between adjacent iterations of the source1 operand.

        Returns
        ----------
        no return
        """
        vec_sel_obj = VecSelApi(mask, mode, dst, sel, src0, src1, repeat_times, dst_rep_stride,
                                src0_rep_stride, src1_rep_stride, None, None)
        return self.vec_sel_(vec_sel_obj)

    def vec_conv(self, mask, round_mode, dst, src, repeat_times, dst_rep_stride, src_rep_stride, deqscale=None,
                 ldst_high_half=False):
        """
        Converts the precision based on the data types of the src and dst tensors.

        Parameter
        ----------
        mask : Effective operation on element, divided into two model: Continuous and bit by bit.
        round_mode : Rounding mode.
        dst : Destination operand.
        src : Source operand.
        repeat_times : Number of iteration repeats
        dst_rep_stride : Block-to-block stride between adjacent iterations of the destination operand.
        src_rep_stride : Block-to-block stride between adjacent iterations of the source operand.
        deqscale : Quantization scale, which is an auxiliary conversion parameter. Defaults to None.
        ldst_high_half : A bool specifying whether dst_list or src_list stores or comes from the upper or lower half
        of each block. Defaults to False.True indicates the upper half, and False indicates the lower half.

        Returns
        ----------
        no return
        """

        default_blk_stride = 1
        stride_unit = 0
        vonv_api = VconvApi(mask, round_mode, dst, src, repeat_times, default_blk_stride,
                            default_blk_stride, dst_rep_stride, src_rep_stride, deqscale,
                            ldst_high_half, stride_unit, "vec_conv", None)
        if TikSocManager.is_nano_soc():
            vconv_obj = VconvOpApiNano(self, vonv_api)
        else:
            vconv_obj = VconvOpApi(self, vonv_api)
        vconv_obj.run_all()

    @source_info_decorator()
    def vec_ln_high_preci(self, mask, dst, src, work_tensor, repeat_times, dst_rep_stride, src_rep_stride):
        """
        Computes the natural logarithm element-wise

        Parameters
        ----------
        mask : Effective operation on element, divided into two model: Continuous and bit by bit.
        dst : Destination operand.
        src : Source operand.
        dst_rep_stride : Block-to-block stride between adjacent iterations of the destination operand
        src_rep_stride : Block-to-block stride between adjacent iterations of the source operand

        Returns
        ----------
        no return
        """
        instr_name = "vec_ln_high_preci"
        if not TikSocManager.is_mini_soc():
            default_blk_stride = 1
            stride_unit = 0
            multis_api = VecSingleApi(instr_name, mask, dst, src, repeat_times, default_blk_stride,
                                      default_blk_stride, dst_rep_stride, src_rep_stride, stride_unit, 'vln')
            SingleOpApi(self, multis_api).run_all()

        else:
            op_obj = VlnHighPreciTuple(mask, dst, src, work_tensor, repeat_times, dst_rep_stride, src_rep_stride)
            if TikSocManager.is_310b_610l_soc():
                self.vln_high_preci_high_level(op_obj)
            else:
                self.vln_high_preci(op_obj)
        self.set_high_level_api_state()
