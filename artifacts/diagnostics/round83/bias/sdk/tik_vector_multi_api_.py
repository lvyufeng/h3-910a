#!/usr/bin/env python
# -*- coding:utf-8 -*-
"""
Copyright (C) 2019. Huawei Technologies Co., Ltd. All rights reserved.
FILE:     tik_vector_multi_api_.py
DESC:     multi op api
CREATED:  2021-10-25 18:53:42
MODIFIED: 2021-11-1 19:17:00
"""
from tbe import tvm
from tbe.common.platform import scope_ubuf
from tbe.tik.tik_lib.tik_expr import Expr
from tbe.tik.common.tik_get_soc_name import is_compatible_mode
from tbe.tik.common.util import reassign_mask
from tbe.tik.common.util import check_mask1_mask2
from tbe.tik.common.util import DTYPE_SIZE
from tbe.tik.common.util import get_bit_len
from tbe.tik.common.tik_get_soc_name import get_block_size
from tbe.tik.debug.tik_vector_ops_debug.tik_vector_debug import vec_multi_elewise_func_dec_new
from tbe.tik.debug.tik_vector_ops_debug.tik_vector_debug import vadddeqrelu_decorator
from tbe.tik.tik_lib.tik_vector_api.tik_params_check import MultiSourceParamsCheck
from tbe.tik.tik_lib.tik_vector_api.tik_params_check import VadddeqreluParamsCheck
from tbe.tik.tik_lib.tik_vector_api.vector_common_util import gen_b64_mask_mode
from tbe.tik.tik_lib.tik_soc_manager import TikSocManager
from tbe.tik.tik_lib.tik_util import type_convert
from tbe.tik.tik_lib.tik_api_constants import DTYPE_MAP
from tbe.tik.tik_lib.tik_params import PIPE_V
from tbe.tik.tik_lib.tik_params import ONE_BLK_SIZE
from tbe.tik.tik_lib.tik_params import BLK_NUM_PER_REP
from tbe.tik.tik_lib.tik_vector_api.tik_compute_control import ControlOp
from tbe.tik.tik_lib.tik_vector_api.tik_tensor_op import TensorOp
from tbe.tik.tik_lib.tik_source_info import source_info_decorator
from tbe.tik.tik_lib.tik_vector_api.tik_vector_name_map import MULTI_NAME_DICT
from tbe.tik.tik_lib.tik_mask_concat_ import mask_concat


class MultiOp:
    """
    multi Vector Ops
    """
    def __init__(self, tik_instance, multi_source_api, mask_mode="normal", mask_o=None):
        super().__init__()
        self.tik_instance = tik_instance
        self.name = MULTI_NAME_DICT.get(multi_source_api.name)
        self.print_name = multi_source_api.name
        if hasattr(multi_source_api, "api_name"):
            self.name = multi_source_api.api_name
        self.control_op = ControlOp(
            multi_source_api.mask, multi_source_api.repeat_times, multi_source_api.stride_unit, mask_mode)
        self.dst_tensor_op = TensorOp(
            multi_source_api.dst, multi_source_api.dst_blk_stride, multi_source_api.dst_rep_stride, "dst")
        self.src0_tensor_op = TensorOp(
            multi_source_api.src0, multi_source_api.src0_blk_stride, multi_source_api.src0_rep_stride, "src0")
        self.src1_tensor_op = TensorOp(
            multi_source_api.src1, multi_source_api.src1_blk_stride, multi_source_api.src1_rep_stride, "src1")
        # debug check instance
        self.check_params = (self.dst_tensor_op, self.src0_tensor_op, self.src1_tensor_op, self.control_op)
        self.mask_o = mask_o
        self.multi_check_obj = MultiSourceParamsCheck(self.print_name, self.check_params, self.name)

    @vec_multi_elewise_func_dec_new
    def code_gen(self):
        """
        code gen

        Returns
        -------
        None
        """
        # code gen
        config = [
            self.control_op.repeat_times, self.dst_tensor_op.blk_stride, self.src0_tensor_op.blk_stride,
            self.src1_tensor_op.blk_stride,
            self.dst_tensor_op.rep_stride, self.src0_tensor_op.rep_stride, self.src1_tensor_op.rep_stride
        ]
        if TikSocManager.is_v200_soc() or (TikSocManager.is_v210_soc() and TikSocManager.is_aicore_core()):
            config.append(self.control_op.stride_unit & 0b01)
            config.append((self.control_op.stride_unit & 0b10) >> 1)
        if self.dst_tensor_op.tensor_obj.dtype == "int64":
            mask_mode = gen_b64_mask_mode(self.control_op.mask)
            config.extend(mask_mode)
        if self.name == "vmulconv":
            self.name = self.name + "_f162" + DTYPE_MAP.get(self.dst_tensor_op.tensor_obj.dtype)
        if self.name in ("vmadd", "vmaddrelu", "vmla"):
            dst_acc = "rw"
        else:
            dst_acc = "w"
        dst_extent = self.dst_tensor_op.get_extent(self.control_op)
        src0_extent = self.src0_tensor_op.get_extent(self.control_op)
        src1_extent = self.src1_tensor_op.get_extent(self.control_op)
        with self.tik_instance.new_scope():
            if TikSocManager.is_310b_610l_soc():
                self.tik_instance.add_source_id()
            instr = tvm.call_extern(self.dst_tensor_op.tensor_obj.dtype, self.name,
                                    self.dst_tensor_op.tensor_obj.access_ptr(dst_acc, extent=dst_extent),
                                    self.src0_tensor_op.tensor_obj.access_ptr("r", extent=src0_extent),
                                    self.src1_tensor_op.tensor_obj.access_ptr("r", extent=src1_extent),
                                    *type_convert(config))
            self.tik_instance.emit(tvm.call_extern("int64", "set_vector_mask", *self.mask_o))
            self.tik_instance.scope_attr(tvm.thread_axis("cce"), "coproc_scope", PIPE_V)
            self.tik_instance.emit(instr)

    @source_info_decorator(depth=2)
    def run_all(self):
        """
        run all_check and code_gen

        Returns
        -------
        None
        """
        mask_o = self.multi_check_obj.all_check(self.tik_instance)
        if self.mask_o is None:
            self.mask_o = mask_o
        self.code_gen()


class NanoMultiOpApi(MultiOp):
    """
    Single Vector Ops
    """
    def blklen_16_gen_code(self, params_list, api_name):
        """
        not compatible, code gen

        Parameter
        ----------
        params_list: list of params

        Returns
        -------
        None
        """
        mask_o, dst, src0, src1, dst_blk_stide, src0_blk_stride, src1_blk_stride, \
            dst_rep_stride, src0_rep_stride, src1_rep_stride = params_list

        config = [self.control_op.repeat_times, dst_blk_stide, src0_blk_stride,
                  src1_blk_stride, dst_rep_stride, src0_rep_stride, src1_rep_stride]

        dst_extent = self.dst_tensor_op.get_extent(self.control_op)
        src0_extent = self.src0_tensor_op.get_extent(self.control_op)
        src1_extent = self.src1_tensor_op.get_extent(self.control_op)
        with self.tik_instance.new_scope():
            instr = tvm.call_extern(dst.dtype, api_name,
                                    dst.access_ptr("w", extent=dst_extent),
                                    src0.access_ptr("r", extent=src0_extent),
                                    src1.access_ptr("r", extent=src1_extent),
                                    *type_convert(config))
            self.tik_instance.emit(tvm.call_extern("int64", "set_vector_mask", *mask_o))
            self.tik_instance.scope_attr(tvm.thread_axis("cce"), "coproc_scope", PIPE_V)
            self.tik_instance.emit(instr)

    def double_incompatible_mode(self):
        """
        double op non compatible mode code gen
        Returns
        -------

        """
        mask_o = self.multi_check_obj.all_check(self.tik_instance)
        self.blklen_16_gen_code((mask_o, self.dst_tensor_op.tensor_obj, self.src0_tensor_op.tensor_obj,
                                 self.src1_tensor_op.tensor_obj, self.dst_tensor_op.blk_stride,
                                 self.src0_tensor_op.blk_stride, self.src1_tensor_op.blk_stride,
                                 self.dst_tensor_op.rep_stride, self.src0_tensor_op.rep_stride,
                                 self.src1_tensor_op.rep_stride), self.name)

    def run_all_double(self):
        """
        for double api, run all_check and code_gen

        Returns
        -------
        None
        """
        if is_compatible_mode():
            self.multi_check_obj.all_check(self.tik_instance)
            tensor_bit_len = get_bit_len(self.dst_tensor_op.tensor_obj.dtype)

            one_block_elements = get_block_size() // DTYPE_SIZE[self.dst_tensor_op.tensor_obj.dtype]
            mask1, mask2 = reassign_mask(self.tik_instance, self.control_op.mask, one_block_elements)
            enable_mask1, enable_mask2 = check_mask1_mask2(mask1, mask2)
            if enable_mask1:
                mask_o1 = mask_concat(self.tik_instance, mask1, self.control_op.mask_mode, tensor_bit_len)
                self.blklen_16_gen_code((mask_o1, self.dst_tensor_op.tensor_obj, self.src0_tensor_op.tensor_obj,
                                         self.src1_tensor_op.tensor_obj, 2 * self.dst_tensor_op.blk_stride,
                                         2 * self.src0_tensor_op.blk_stride, 2 * self.src1_tensor_op.blk_stride,
                                         2 * self.dst_tensor_op.rep_stride, 2 * self.src0_tensor_op.rep_stride,
                                         2 * self.src1_tensor_op.rep_stride), self.name)
            if enable_mask2:
                mask_o2 = mask_concat(self.tik_instance, mask2, self.control_op.mask_mode, tensor_bit_len)
                self.blklen_16_gen_code((mask_o2, self.dst_tensor_op.tensor_obj[one_block_elements:],
                                         self.src0_tensor_op.tensor_obj[one_block_elements:],
                                         self.src1_tensor_op.tensor_obj[one_block_elements:],
                                         2 * self.dst_tensor_op.blk_stride, 2 * self.src0_tensor_op.blk_stride,
                                         2 * self.src1_tensor_op.blk_stride, 2 * self.dst_tensor_op.rep_stride,
                                         2 * self.src0_tensor_op.rep_stride, 2 * self.src1_tensor_op.rep_stride),
                                        self.name)
        else:
            self.double_incompatible_mode()

    def run_triple_incompatible(self, dummy_dst):
        """
        not compatible triple code gen

        Parameter
        ----------
        dummy_dst: a temporary dst

        Returns
        -------
        None
        """
        mask_o = self.multi_check_obj.all_check(self.tik_instance)
        if self.name == "vmla":
            self.blklen_16_gen_code((mask_o, dummy_dst, self.src0_tensor_op.tensor_obj,
                                     self.src1_tensor_op.tensor_obj, self.dst_tensor_op.blk_stride,
                                     self.src0_tensor_op.blk_stride, self.src1_tensor_op.blk_stride,
                                     self.dst_tensor_op.rep_stride, self.src0_tensor_op.rep_stride,
                                     self.src1_tensor_op.rep_stride), "vmul")
            self.blklen_16_gen_code((mask_o, self.dst_tensor_op.tensor_obj, self.dst_tensor_op.tensor_obj,
                                     dummy_dst, self.dst_tensor_op.blk_stride,
                                     self.dst_tensor_op.blk_stride, self.dst_tensor_op.blk_stride,
                                     self.dst_tensor_op.rep_stride, self.dst_tensor_op.rep_stride,
                                     self.dst_tensor_op.rep_stride), "vadd")
        if self.name in ("vmadd", "vmaddrelu"):
            self.blklen_16_gen_code((mask_o, dummy_dst, self.src0_tensor_op.tensor_obj,
                                     self.dst_tensor_op.tensor_obj, self.dst_tensor_op.blk_stride,
                                     self.src0_tensor_op.blk_stride, self.dst_tensor_op.blk_stride,
                                     self.dst_tensor_op.rep_stride, self.src0_tensor_op.rep_stride,
                                     self.dst_tensor_op.rep_stride), "vmul")
            self.blklen_16_gen_code((mask_o, self.dst_tensor_op.tensor_obj, dummy_dst,
                                     self.src1_tensor_op.tensor_obj, self.dst_tensor_op.blk_stride,
                                     self.dst_tensor_op.blk_stride, self.src1_tensor_op.blk_stride,
                                     self.dst_tensor_op.rep_stride, self.dst_tensor_op.rep_stride,
                                     self.src1_tensor_op.rep_stride), "vadd")
        if self.name == "vmaddrelu":
            self.tik_instance.vrelu(self.control_op.mask, self.dst_tensor_op.tensor_obj,
                                    self.dst_tensor_op.tensor_obj, self.control_op.repeat_times,
                                    self.dst_tensor_op.blk_stride, self.dst_tensor_op.blk_stride,
                                    self.dst_tensor_op.rep_stride, self.dst_tensor_op.rep_stride)
            self.tik_instance.set_high_level_api_state()

    def run_triple_compatible_vmla_left(self, mask_o1, dummy_dst):
        """
        The left half of each block code gen about vmla

        Parameter
        ----------
        mask_o1: left half mask
        dummy_dst: a temporary dst

        Returns
        -------
        None
        """
        self.blklen_16_gen_code((mask_o1, dummy_dst, self.src0_tensor_op.tensor_obj,
                                 self.src1_tensor_op.tensor_obj, 2 * self.dst_tensor_op.blk_stride,
                                 2 * self.src0_tensor_op.blk_stride, 2 * self.src1_tensor_op.blk_stride,
                                 2 * self.dst_tensor_op.rep_stride, 2 * self.src0_tensor_op.rep_stride,
                                 2 * self.src1_tensor_op.rep_stride), "vmul")
        self.blklen_16_gen_code((mask_o1, self.dst_tensor_op.tensor_obj, self.dst_tensor_op.tensor_obj,
                                 dummy_dst, 2 * self.dst_tensor_op.blk_stride,
                                 2 * self.dst_tensor_op.blk_stride, 2 * self.dst_tensor_op.blk_stride,
                                 2 * self.dst_tensor_op.rep_stride, 2 * self.dst_tensor_op.rep_stride,
                                 2 * self.dst_tensor_op.rep_stride), "vadd")

    def run_triple_compatible_vmla_right(self, mask_o2, dummy_dst, one_block_elements):
        """
        The right half of each block code gen about vmla

        Parameter
        ----------
        mask_o2: right half mask
        dummy_dst: a temporary dst
        one_block_elements: one block offset

        Returns
        -------
        None
        """
        self.blklen_16_gen_code((mask_o2, dummy_dst[one_block_elements:],
                                 self.src0_tensor_op.tensor_obj[one_block_elements:],
                                 self.src1_tensor_op.tensor_obj[one_block_elements:],
                                 2 * self.dst_tensor_op.blk_stride,
                                 2 * self.src0_tensor_op.blk_stride, 2 * self.src1_tensor_op.blk_stride,
                                 2 * self.dst_tensor_op.rep_stride, 2 * self.src0_tensor_op.rep_stride,
                                 2 * self.src1_tensor_op.rep_stride), "vmul")
        self.blklen_16_gen_code((mask_o2, self.dst_tensor_op.tensor_obj[one_block_elements:],
                                 self.dst_tensor_op.tensor_obj[one_block_elements:],
                                 dummy_dst[one_block_elements:], 2 * self.dst_tensor_op.blk_stride,
                                 2 * self.dst_tensor_op.blk_stride, 2 * self.dst_tensor_op.blk_stride,
                                 2 * self.dst_tensor_op.rep_stride, 2 * self.dst_tensor_op.rep_stride,
                                 2 * self.dst_tensor_op.rep_stride), "vadd")

    def run_triple_compatible_vmadd_left(self, mask_o1, dummy_dst):
        """
        The left half of each block code gen about vmadd or vmaddrelu

        Parameter
        ----------
        mask_o1: left half mask
        dummy_dst: a temporary dst

        Returns
        -------
        None
        """
        self.blklen_16_gen_code((mask_o1, dummy_dst, self.src0_tensor_op.tensor_obj,
                                 self.dst_tensor_op.tensor_obj, 2 * self.dst_tensor_op.blk_stride,
                                 2 * self.src0_tensor_op.blk_stride, 2 * self.dst_tensor_op.blk_stride,
                                 2 * self.dst_tensor_op.rep_stride, 2 * self.src0_tensor_op.rep_stride,
                                 2 * self.dst_tensor_op.rep_stride), "vmul")
        self.blklen_16_gen_code((mask_o1, self.dst_tensor_op.tensor_obj, dummy_dst,
                                 self.src1_tensor_op.tensor_obj, 2 * self.dst_tensor_op.blk_stride,
                                 2 * self.dst_tensor_op.blk_stride, 2 * self.src1_tensor_op.blk_stride,
                                 2 * self.dst_tensor_op.rep_stride, 2 * self.dst_tensor_op.rep_stride,
                                 2 * self.src1_tensor_op.rep_stride), "vadd")

    def run_triple_compatible_vmadd_right(self, mask_o2, dummy_dst, one_block_elements):
        """
        The right half of each block code gen about vmadd or vmaddrelu

        Parameter
        ----------
        mask_o2: right half mask
        dummy_dst: a temporary dst
        one_block_elements: one block offset

        Returns
        -------
        None
        """
        self.blklen_16_gen_code(
            (mask_o2, dummy_dst[one_block_elements:], self.src0_tensor_op.tensor_obj[one_block_elements:],
             self.dst_tensor_op.tensor_obj[one_block_elements:], 2 * self.dst_tensor_op.blk_stride,
             2 * self.src0_tensor_op.blk_stride, 2 * self.dst_tensor_op.blk_stride,
             2 * self.dst_tensor_op.rep_stride, 2 * self.src0_tensor_op.rep_stride,
             2 * self.dst_tensor_op.rep_stride), "vmul")
        self.blklen_16_gen_code(
            (mask_o2, self.dst_tensor_op.tensor_obj[one_block_elements:], dummy_dst[one_block_elements:],
             self.src1_tensor_op.tensor_obj[one_block_elements:], 2 * self.dst_tensor_op.blk_stride,
             2 * self.dst_tensor_op.blk_stride, 2 * self.src1_tensor_op.blk_stride,
             2 * self.dst_tensor_op.rep_stride, 2 * self.dst_tensor_op.rep_stride,
             2 * self.src1_tensor_op.rep_stride), "vadd")

    def run_all_triple(self, dummy_dst):
        """
        run triple code gen

        Parameter
        ----------
        dummy_dst: a temporary dst

        Returns
        -------
        None
        """
        if is_compatible_mode():
            self.multi_check_obj.all_check(self.tik_instance)
            tensor_bit_len = get_bit_len(self.dst_tensor_op.tensor_obj.dtype)
            one_block_elements = get_block_size() // DTYPE_SIZE[self.dst_tensor_op.tensor_obj.dtype]
            mask1, mask2 = reassign_mask(self.tik_instance, self.control_op.mask, one_block_elements)
            enable_mask1, enable_mask2 = check_mask1_mask2(mask1, mask2)

            if self.name == "vmla":
                if enable_mask1:
                    mask_o1 = mask_concat(self.tik_instance, mask1, self.control_op.mask_mode, tensor_bit_len)
                    self.run_triple_compatible_vmla_left(mask_o1, dummy_dst)
                if enable_mask2:
                    mask_o2 = mask_concat(self.tik_instance, mask2, self.control_op.mask_mode, tensor_bit_len)
                    self.run_triple_compatible_vmla_right(mask_o2, dummy_dst, one_block_elements)
            if self.name in ("vmadd", "vmaddrelu"):
                if enable_mask1:
                    mask_o1 = mask_concat(self.tik_instance, mask1, self.control_op.mask_mode, tensor_bit_len)
                    self.run_triple_compatible_vmadd_left(mask_o1, dummy_dst)
                if enable_mask2:
                    mask_o2 = mask_concat(self.tik_instance, mask2, self.control_op.mask_mode, tensor_bit_len)
                    self.run_triple_compatible_vmadd_right(mask_o2, dummy_dst, one_block_elements)
            if self.name == "vmaddrelu":
                self.tik_instance.vrelu(self.control_op.mask, self.dst_tensor_op.tensor_obj,
                                        self.dst_tensor_op.tensor_obj, self.control_op.repeat_times,
                                        self.dst_tensor_op.blk_stride, self.dst_tensor_op.blk_stride,
                                        self.dst_tensor_op.rep_stride, self.dst_tensor_op.rep_stride)
        else:
            self.run_triple_incompatible(dummy_dst)

    @source_info_decorator(depth=2)
    def run_all(self):
        """
        run all_check and code_gen

        Returns
        -------
        None
        """

        # block_size considered as 32
        if self.name in ("vmla", "vmadd", "vmaddrelu"):
            dummy_dst = self.tik_instance.Tensor(self.dst_tensor_op.tensor_obj.dtype,
                                                 self.dst_tensor_op.tensor_obj.original_shape,
                                                 name="dummy_dst",  scope=scope_ubuf)

            dummy_dst = dummy_dst[self.dst_tensor_op.tensor_obj.offset:]
            self.run_all_triple(dummy_dst)
        else:
            self.run_all_double()

    @source_info_decorator(depth=2)
    def run_all_incompatible_mode(self):
        self.double_incompatible_mode()


class VadddeqreluOp:
    """
    Vadddeqrelu Vector Ops
    """
    def __init__(self, tik_instance, vadddeqrelu_api):
        super().__init__()
        self.tik_instance = tik_instance
        self.name = vadddeqrelu_api.name
        self.print_name = vadddeqrelu_api.name
        self.control_op = ControlOp(vadddeqrelu_api.mask, vadddeqrelu_api.repeat_times, vadddeqrelu_api.stride_unit)
        self.dst_tensor_op = TensorOp(vadddeqrelu_api.dst, vadddeqrelu_api.dst_blk_stride,
                                      vadddeqrelu_api.dst_rep_stride, "dst")
        self.src0_tensor_op = TensorOp(vadddeqrelu_api.src0, vadddeqrelu_api.src0_blk_stride,
                                       vadddeqrelu_api.src0_rep_stride, "src0")
        self.src1_tensor_op = TensorOp(vadddeqrelu_api.src1, vadddeqrelu_api.src1_blk_stride,
                                       vadddeqrelu_api.src1_rep_stride, "src1")
        self.mask_o = None
        self.deqscale = vadddeqrelu_api.deqscale
        self.check_params = (self.dst_tensor_op, self.src0_tensor_op, self.src1_tensor_op,
                             self.control_op)
        self.vadddeqrelu_check_obj = VadddeqreluParamsCheck(self.print_name, self.check_params, self.deqscale)

    @vadddeqrelu_decorator
    def code_gen(self):
        """
        code gen

        Returns
        -------
        None
        """
        config = [
            self.control_op.repeat_times, self.dst_tensor_op.blk_stride, self.src0_tensor_op.blk_stride,
            self.src1_tensor_op.blk_stride, self.dst_tensor_op.rep_stride, self.src0_tensor_op.rep_stride,
            self.src1_tensor_op.rep_stride
        ]
        if self.src0_tensor_op.tensor_obj.dtype == "int64":
            mask_mode = gen_b64_mask_mode(self.control_op.mask)
            config.extend(mask_mode)
        res_args = Expr(self.deqscale, dtype="float16")
        with self.tik_instance.new_scope():
            self.tik_instance.scope_attr(tvm.thread_axis("cce"), "coproc_scope", PIPE_V)
            self.tik_instance.emit(tvm.call_extern("float16", "set_deqscale", type_convert(res_args)))
        with self.tik_instance.new_scope():
            mem_access_param = type_convert(config)
            instr = tvm.call_extern(self.dst_tensor_op.tensor_obj.dtype, "vadddeqrelu",
                                    self.dst_tensor_op.tensor_obj.access_ptr(
                                        "w", extent=self._calculate_extent(self.dst_tensor_op.rep_stride,
                                                                           self.dst_tensor_op.blk_stride)),
                                    self.src0_tensor_op.tensor_obj.access_ptr(
                                        "r", extent=self._calculate_extent(self.src0_tensor_op.rep_stride,
                                                                           self.src0_tensor_op.blk_stride)),
                                    self.src1_tensor_op.tensor_obj.access_ptr(
                                        "r", extent=self._calculate_extent(self.src1_tensor_op.rep_stride,
                                                                           self.src1_tensor_op.blk_stride)),
                                    *mem_access_param)
            self.tik_instance.emit(tvm.call_extern("int64", "set_vector_mask", *self.mask_o))
            self.tik_instance.scope_attr(tvm.thread_axis("cce"), "coproc_scope", PIPE_V)
            self.tik_instance.emit(instr)

    @source_info_decorator(depth=2)
    def run_all(self):
        """
        run all_check and code_gen

        Returns
        -------
        None
        """
        self.mask_o = self.vadddeqrelu_check_obj.all_check(self.tik_instance)
        self.code_gen()

    def _calculate_extent(self, rep_stride, blk_stride):
        """
        calculate extent

        Parameters
        ----------

        Returns
        -------
        extent
        """
        repeat_time = self.control_op.repeat_times - 1
        blk_stride = (BLK_NUM_PER_REP - 1) * blk_stride
        extent = Expr((repeat_time * rep_stride + blk_stride + 1) * ONE_BLK_SIZE)
        return extent.get()
