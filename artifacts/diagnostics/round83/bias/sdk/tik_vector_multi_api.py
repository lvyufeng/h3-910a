#!/usr/bin/env python
# -*- coding:utf-8 -*-
"""
Copyright (C) 2019. Huawei Technologies Co., Ltd. All rights reserved.
FILE:     tik_vector_multi_api.py
DESC:
CREATED:  2021-11-03 10:05
MODIFIED: 2021-11-03 10:05
"""

from collections import namedtuple

from tbe.tik.api.tik_ir_builder import TikIRBuilder
from tbe.tik.tik_lib.tik_soc_manager import TikSocManager
from tbe.tik.tik_lib.tik_vector_api.tik_vector_vbi_api_ import VbiOpApi
from tbe.tik.tik_lib.tik_vector_api.tik_vector_multi_api_ import MultiOp
from tbe.tik.tik_lib.tik_vector_api.tik_vector_multi_api_ import NanoMultiOpApi
from tbe.tik.tik_lib.tik_vector_api.tik_vector_multi_api_ import VadddeqreluOp


class MultiOpsApi(TikIRBuilder):
    """
    Multi Vector Api
    """
    multi_api = namedtuple('MultiApi', ["name", "mask", "dst", "src0", "src1", "repeat_times",
                                        "dst_blk_stride", "src0_blk_stride", "src1_blk_stride",
                                        "dst_rep_stride", "src0_rep_stride", "src1_rep_stride", "stride_unit"])
    vadddeqrelu_api = namedtuple('VadddeqreluApi', ["name", "mask", "dst", "deqscale", "src0", "src1",
                                                    "repeat_times", "dst_blk_stride", "src0_blk_stride",
                                                    "src1_blk_stride", "dst_rep_stride", "src0_rep_stride",
                                                    "src1_rep_stride", "stride_unit"])
    vbi_api = namedtuple('VbiApi', ["name", "mask", "dst", "src0", "src1", "src0_offset", "dst_blk_stride",
                                    "vertical_repeat_times", "horizontal_repeat_times", "repeat_mode",
                                    "vertical_repeat_offset", "enable_vbi_double_buffer"])

    def __init__(self):
        super(MultiOpsApi, self).__init__()
        self.enable_vbi_double_buffer = False

    def vbi(self, mask, dst, src0, src1, src0_offset, dst_blk_stride, vertical_repeat_times, horizontal_repeat_times,
            repeat_mode, vertical_repeat_offset):
        """
        vbi instruction, used for bilinear interpolation in ROI alignment

        Parameters
        ----------
        mask: Effective operation on element
        dst: destination tensor
        src0: src0 tensor
        src1: src1 tensor
        src0_offset: src0_offset tensor
        dst_blk_stride: offset of dst operator between different block in one iteration
        vertical_repeat_times: repeat_times in vertical direction
        horizontal_repeat_times: repeat_times in horizontal direction
        repeat_mode: indicate how many elements at src1 are consumed in one iteration
        vertical_repeat_offset: vertical repeat offset between dst address of iterations in the vertical direction

        Returns
        -------
        None
        """
        vbi_api = MultiOpsApi.vbi_api('vbi', mask, dst, src0, src1, src0_offset, dst_blk_stride, vertical_repeat_times,
                                      horizontal_repeat_times, repeat_mode, vertical_repeat_offset,
                                      self.enable_vbi_double_buffer)
        vabs_obj = VbiOpApi(self, vbi_api)
        if TikSocManager.is_910b_soc() or TikSocManager.is_v210_vec_soc() or TikSocManager.is_950_soc():
            vabs_obj.vbi_v220_run_all()
        elif TikSocManager.is_310b_610l_soc():
            vabs_obj.vbi_run_regbase()
        else:
            vabs_obj.vbi_v200_run_all()

    def set_vbi_options(self, enable_double_buffer=False):
        """
        enable double buffer of v220 vbi
        """
        self.enable_vbi_double_buffer = enable_double_buffer

    def vadd(self, mask, dst, src0, src1, repeat_times, dst_blk_stride, src0_blk_stride, src1_blk_stride,
             dst_rep_stride, src0_rep_stride, src1_rep_stride, stride_unit=0):
        """
        Do add by single element.

        Parameters
        ----------
        mask : Effective operation on element, divided into two model: Continuous and bit by bit.
        dst : destination operator
        src0 : source operation
        src1 : source operation
        repeat_times : Repeated iterations times
        dst_blk_stride : offset of dst operator between different block
        src0_blk_stride : offset of src operator between different block
        src1_blk_stride : offset of src operator between different block
        dst_rep_stride : offset of dst operator in the same block between
        src0_rep_stride : offset of src operator in the same block between
        src1_rep_stride : offset of src operator in the same block between
        stride_unit : address and offset unit both affect it. default = 0

        Returns
        -------
        None
        """
        multi_api_ins = MultiOpsApi.multi_api("vadd", mask, dst, src0, src1, repeat_times, dst_blk_stride,
                                              src0_blk_stride, src1_blk_stride, dst_rep_stride, src0_rep_stride,
                                              src1_rep_stride, stride_unit)
        if TikSocManager.is_nano_soc():
            vadd_obj = NanoMultiOpApi(self, multi_api_ins)
        else:
            vadd_obj = MultiOp(self, multi_api_ins)
        vadd_obj.run_all()

    def vsub(self, mask, dst, src0, src1, repeat_times, dst_blk_stride, src0_blk_stride, src1_blk_stride,
             dst_rep_stride, src0_rep_stride, src1_rep_stride, stride_unit=0):
        """
        Do minus by single elements.

        Parameters
        ----------
        mask : Effective operation on element, divided into two model: Continuous and bit by bit.
        dst : destination operator
        src0 : source operation
        src1 : source operation
        repeat_times : Repeated iterations times
        dst_blk_stride : offset of dst operator between different block
        src0_blk_stride : offset of src operator between different block
        src1_blk_stride : offset of src operator between different block
        dst_rep_stride : offset of dst operator in the same block between
        src0_rep_stride : offset of src operator in the same block between
        src1_rep_stride : offset of src operator in the same block between
        stride_unit : address and offset unit both affect it. default = 0

        Returns
        -------
        None
        """
        multi_api_ins = MultiOpsApi.multi_api("vsub", mask, dst, src0, src1, repeat_times, dst_blk_stride,
                                              src0_blk_stride, src1_blk_stride, dst_rep_stride, src0_rep_stride,
                                              src1_rep_stride, stride_unit)
        if TikSocManager.is_nano_soc():
            vsub_obj = NanoMultiOpApi(self, multi_api_ins)
        else:
            vsub_obj = MultiOp(self, multi_api_ins)
        vsub_obj.run_all()

    def vmul(self, mask, dst, src0, src1, repeat_times, dst_blk_stride, src0_blk_stride, src1_blk_stride,
             dst_rep_stride, src0_rep_stride, src1_rep_stride, stride_unit=0):
        """
        Get product by single element.

        Parameters
        ----------
        mask : Effective operation on element, divided into two model: Continuous and bit by bit.
        dst : destination operator
        src0 : source operation
        src1 : source operation
        repeat_times : Repeated iterations times
        dst_blk_stride : offset of dst operator between different block
        src0_blk_stride : offset of src operator between different block
        src1_blk_stride : offset of src operator between different block
        dst_rep_stride : offset of dst operator in the same block between
        src0_rep_stride : offset of src operator in the same block between
        src1_rep_stride : offset of src operator in the same block between
        stride_unit : address and offset unit both affect it. default = 0

        Returns
        -------
        None
        """
        multi_api_ins = MultiOpsApi.multi_api("vmul", mask, dst, src0, src1, repeat_times, dst_blk_stride,
                                              src0_blk_stride, src1_blk_stride, dst_rep_stride, src0_rep_stride,
                                              src1_rep_stride, stride_unit)
        if TikSocManager.is_nano_soc():
            vmul_obj = NanoMultiOpApi(self, multi_api_ins)
        else:
            vmul_obj = MultiOp(self, multi_api_ins)
        vmul_obj.run_all()

    def vdiv(self, mask, dst, src0, src1, repeat_times, dst_blk_stride, src0_blk_stride, src1_blk_stride,
             dst_rep_stride, src0_rep_stride, src1_rep_stride, stride_unit=0):
        """
        Get division by single element.

        Parameters
        ----------
        mask : Effective operation on element, divided into two model: Continuous and bit by bit.
        dst : destination operator
        src0 : source operation
        src1 : source operation
        repeat_times : Repeated iterations times
        dst_blk_stride : offset of dst operator between different block
        src0_blk_stride : offset of src operator between different block
        src1_blk_stride : offset of src operator between different block
        dst_rep_stride : offset of dst operator in the same block between
        src0_rep_stride : offset of src operator in the same block between
        src1_rep_stride : offset of src operator in the same block between
        stride_unit : address and offset unit both affect it. default = 0

        Returns
        -------
        None
        """
        multi_api_ins = MultiOpsApi.multi_api("vdiv", mask, dst, src0, src1, repeat_times, dst_blk_stride,
                                              src0_blk_stride, src1_blk_stride, dst_rep_stride, src0_rep_stride,
                                              src1_rep_stride, stride_unit)
        if TikSocManager.is_nano_soc():
            vdiv_obj = NanoMultiOpApi(self, multi_api_ins)
        else:
            vdiv_obj = MultiOp(self, multi_api_ins)
        vdiv_obj.run_all()

    def vmax(self, mask, dst, src0, src1, repeat_times, dst_blk_stride, src0_blk_stride, src1_blk_stride,
             dst_rep_stride, src0_rep_stride, src1_rep_stride, stride_unit=0):
        """
        Get the max value in all elements.

        Parameters
        ----------
        mask : Effective operation on element, divided into two model: Continuous and bit by bit.
        dst : destination operator
        src0 : source operation
        src1 : source operation
        repeat_times : Repeated iterations times
        dst_blk_stride : offset of dst operator between different block
        src0_blk_stride : offset of src operator between different block
        src1_blk_stride : offset of src operator between different block
        dst_rep_stride : offset of dst operator in the same block between
        src0_rep_stride : offset of src operator in the same block between
        src1_rep_stride : offset of src operator in the same block between
        stride_unit : address and offset unit both affect it. default = 0

        Returns
        -------
        None
        """
        multi_api_ins = MultiOpsApi.multi_api("vmax", mask, dst, src0, src1, repeat_times, dst_blk_stride,
                                              src0_blk_stride, src1_blk_stride, dst_rep_stride, src0_rep_stride,
                                              src1_rep_stride, stride_unit)
        if TikSocManager.is_nano_soc():
            vmax_obj = NanoMultiOpApi(self, multi_api_ins)
        else:
            vmax_obj = MultiOp(self, multi_api_ins)
        vmax_obj.run_all()

    def vmin(self, mask, dst, src0, src1, repeat_times, dst_blk_stride, src0_blk_stride, src1_blk_stride,
             dst_rep_stride, src0_rep_stride, src1_rep_stride, stride_unit=0):
        """
        Get the min value by single element.

        Parameters
        ----------
        mask : Effective operation on element, divided into two model: Continuous and bit by bit.
        dst : destination operator
        src0 : source operation
        src1 : source operation
        repeat_times : Repeated iterations times
        dst_blk_stride : offset of dst operator between different block
        src0_blk_stride : offset of src operator between different block
        src1_blk_stride : offset of src operator between different block
        dst_rep_stride : offset of dst operator in the same block between
        src0_rep_stride : offset of src operator in the same block between
        src1_rep_stride : offset of src operator in the same block between
        stride_unit : address and offset unit both affect it. default = 0

        Returns
        -------
        None
        """
        multi_api_ins = MultiOpsApi.multi_api("vmin", mask, dst, src0, src1, repeat_times, dst_blk_stride,
                                              src0_blk_stride, src1_blk_stride, dst_rep_stride, src0_rep_stride,
                                              src1_rep_stride, stride_unit)
        if TikSocManager.is_nano_soc():
            vmin_obj = NanoMultiOpApi(self, multi_api_ins)
        else:
            vmin_obj = MultiOp(self, multi_api_ins)
        vmin_obj.run_all()

    def vand(self, mask, dst, src0, src1, repeat_times, dst_blk_stride, src0_blk_stride, src1_blk_stride,
             dst_rep_stride, src0_rep_stride, src1_rep_stride, stride_unit=0):
        """
        Do bitwise and by single element.

        Parameters
        ----------
        mask : Effective operation on element, divided into two model: Continuous and bit by bit.
        dst : destination operator
        src0 : source operation
        src1 : source operation
        repeat_times : Repeated iterations times
        dst_blk_stride : offset of dst operator between different block
        src0_blk_stride : offset of src operator between different block
        src1_blk_stride : offset of src operator between different block
        dst_rep_stride : offset of dst operator in the same block between
        src0_rep_stride : offset of src operator in the same block between
        src1_rep_stride : offset of src operator in the same block between
        stride_unit : address and offset unit both affect it. default = 0

        Returns
        -------
        None
        """
        multi_api_ins = MultiOpsApi.multi_api("vand", mask, dst, src0, src1, repeat_times, dst_blk_stride,
                                              src0_blk_stride, src1_blk_stride, dst_rep_stride, src0_rep_stride,
                                              src1_rep_stride, stride_unit)
        if TikSocManager.is_nano_soc():
            vand_obj = NanoMultiOpApi(self, multi_api_ins)
        else:
            vand_obj = MultiOp(self, multi_api_ins)
        vand_obj.run_all()

    def vor(self, mask, dst, src0, src1, repeat_times, dst_blk_stride, src0_blk_stride, src1_blk_stride,
            dst_rep_stride, src0_rep_stride, src1_rep_stride, stride_unit=0):
        """
        Do bitwise or by single element.

        Parameters
        ----------
        mask : Effective operation on element, divided into two model: Continuous and bit by bit.
        dst : destination operator
        src0 : source operation
        src1 : source operation
        repeat_times : Repeated iterations times
        dst_blk_stride : offset of dst operator between different block
        src0_blk_stride : offset of src operator between different block
        src1_blk_stride : offset of src operator between different block
        dst_rep_stride : offset of dst operator in the same block between
        src0_rep_stride : offset of src operator in the same block between
        src1_rep_stride : offset of src operator in the same block between
        stride_unit : address and offset unit both affect it. default = 0

        Returns
        -------
        None
        """
        multi_api_ins = MultiOpsApi.multi_api("vor", mask, dst, src0, src1, repeat_times, dst_blk_stride,
                                              src0_blk_stride, src1_blk_stride, dst_rep_stride, src0_rep_stride,
                                              src1_rep_stride, stride_unit)
        if TikSocManager.is_nano_soc():
            vor_obj = NanoMultiOpApi(self, multi_api_ins)
        else:
            vor_obj = MultiOp(self, multi_api_ins)
        vor_obj.run_all()

    def vmla(self, mask, dst, src0, src1, repeat_times, dst_blk_stride, src0_blk_stride, src1_blk_stride,
             dst_rep_stride, src0_rep_stride, src1_rep_stride, stride_unit=0):
        """
        Multiply by element and accumulate. dst = src0*src1 + dst

        Parameters
        ----------
        mask : Effective operation on element, divided into two model: Continuous and bit by bit.
        dst : destination operator
        src0 : source operation
        src1 : source operation
        repeat_times : Repeated iterations times
        dst_blk_stride : offset of dst operator between different block
        src0_blk_stride : offset of src operator between different block
        src1_blk_stride : offset of src operator between different block
        dst_rep_stride : offset of dst operator in the same block between
        src0_rep_stride : offset of src operator in the same block between
        src1_rep_stride : offset of src operator in the same block between
        stride_unit : address and offset unit both affect it. default = 0

        Returns
        -------
        None
        """
        multi_api_ins = MultiOpsApi.multi_api('vmla', mask, dst, src0, src1, repeat_times, dst_blk_stride,
                                              src0_blk_stride, src1_blk_stride, dst_rep_stride,
                                              src0_rep_stride, src1_rep_stride, stride_unit)
        if TikSocManager.is_nano_soc():
            vmla_obj = NanoMultiOpApi(self, multi_api_ins)
        else:
            vmla_obj = MultiOp(self, multi_api_ins)
        vmla_obj.run_all()

    def vmulconv(self, mask, dst, src0, src1, repeat_times, dst_blk_stride, src0_blk_stride, src1_blk_stride,
             dst_rep_stride, src0_rep_stride, src1_rep_stride, stride_unit=0):
        """
        Multiply by element and then convert precision

        Parameters
        ----------
        mask : Effective operation on element, divided into two model: Continuous and bit by bit.
        dst : destination operator
        src0 : source operation
        src1 : source operation
        repeat_times : Repeated iterations times
        dst_blk_stride : offset of dst operator between different block
        src0_blk_stride : offset of src operator between different block
        src1_blk_stride : offset of src operator between different block
        dst_rep_stride : offset of dst operator in the same block between
        src0_rep_stride : offset of src operator in the same block between
        src1_rep_stride : offset of src operator in the same block between
        stride_unit : address and offset unit both affect it. default = 0

        Returns
        -------
        None
        """
        multi_api_ins = MultiOpsApi.multi_api('vmulconv', mask, dst, src0, src1, repeat_times, dst_blk_stride,
                                              src0_blk_stride, src1_blk_stride, dst_rep_stride,
                                              src0_rep_stride, src1_rep_stride, stride_unit)
        vabs_obj = MultiOp(self, multi_api_ins)
        vabs_obj.run_all()

    def vmadd(self, mask, dst, src0, src1, repeat_times, dst_blk_stride, src0_blk_stride, src1_blk_stride,
              dst_rep_stride, src0_rep_stride, src1_rep_stride, stride_unit=0):
        """
        Multiply by element and accumulate. dst = src0*dst + src1

        Parameters
        ----------
        mask : Effective operation on element, divided into two model: Continuous and bit by bit.
        dst : destination operator
        src0 : source operation
        src1 : source operation
        repeat_times : Repeated iterations times
        dst_blk_stride : offset of dst operator between different block
        src0_blk_stride : offset of src operator between different block
        src1_blk_stride : offset of src operator between different block
        dst_rep_stride : offset of dst operator in the same block between
        src0_rep_stride : offset of src operator in the same block between
        src1_rep_stride : offset of src operator in the same block between
        stride_unit : address and offset unit both affect it. default = 0

        Returns
        -------
        None
        """
        multi_api_ins = MultiOpsApi.multi_api('vmadd', mask, dst, src0, src1, repeat_times, dst_blk_stride,
                                              src0_blk_stride, src1_blk_stride, dst_rep_stride, src0_rep_stride,
                                              src1_rep_stride, stride_unit)
        if TikSocManager.is_nano_soc():
            vmadd_obj = NanoMultiOpApi(self, multi_api_ins)
        else:
            vmadd_obj = MultiOp(self, multi_api_ins)
        vmadd_obj.run_all()

    def vmaddrelu(self, mask, dst, src0, src1, repeat_times, dst_blk_stride, src0_blk_stride, src1_blk_stride,
                  dst_rep_stride, src0_rep_stride, src1_rep_stride, stride_unit=0):
        """
        Multiply by element and accumulate and then Linear rectification

        Parameters
        ----------
        mask : Effective operation on element, divided into two model: Continuous and bit by bit.
        dst : destination operator
        src0 : source operation
        src1 : source operation
        repeat_times : Repeated iterations times
        dst_blk_stride : offset of dst operator between different block
        src0_blk_stride : offset of src operator between different block
        src1_blk_stride : offset of src operator between different block
        dst_rep_stride : offset of dst operator in the same block between
        src0_rep_stride : offset of src operator in the same block between
        src1_rep_stride : offset of src operator in the same block between
        stride_unit : address and offset unit both affect it. default = 0

        Returns
        -------
        None
        """
        multi_api_ins = MultiOpsApi.multi_api('vmaddrelu', mask, dst, src0, src1, repeat_times, dst_blk_stride,
                                              src0_blk_stride, src1_blk_stride, dst_rep_stride, src0_rep_stride,
                                              src1_rep_stride, stride_unit)
        if TikSocManager.is_nano_soc():
            vmaddrelu_obj = NanoMultiOpApi(self, multi_api_ins)
        else:
            vmaddrelu_obj = MultiOp(self, multi_api_ins)
        vmaddrelu_obj.run_all()

    def vadddeqrelu(self, mask, dst, deqscale, src0, src1, repeat_times, dst_blk_stride, src0_blk_stride,
                    src1_blk_stride, dst_rep_stride, src0_rep_stride, src1_rep_stride, stride_unit=0):
        """
        vadd+vconv+vrelu

        Parameters
        ----------
        mask : Effective operation on element, divided into two model: Continuous and bit by bit.
        dst : destination operator
        deqscale : dst_i=relu(dequant*(dequant(src0_i+src1_i)*2^17))
        src0 : source operation
        src1 : source operation
        repeat_times: Repeated iterations times
        dst_blk_stride: offset of dst operator between different block
        src0_blk_stride: offset of src operator between different block
        src1_blk_stride: offset of src operator between different block
        dst_rep_stride: offset of dst operator in the same block between two repeats
        src0_rep_stride: offset of src operator in the same block between two repeats
        src1_rep_stride: offset of src operator in the same block between two repeats
        stride_unit: address and offset unit both affect it. default = 0

        Returns
        -------
        None
        """
        vadddeqrelu_api = MultiOpsApi.vadddeqrelu_api("vadddeqrelu", mask, dst, deqscale, src0, src1, repeat_times,
                                                      dst_blk_stride, src0_blk_stride, src1_blk_stride, dst_rep_stride,
                                                      src0_rep_stride, src1_rep_stride, stride_unit)
        vadddeqrelu_obj = VadddeqreluOp(self, vadddeqrelu_api)
        vadddeqrelu_obj.run_all()
