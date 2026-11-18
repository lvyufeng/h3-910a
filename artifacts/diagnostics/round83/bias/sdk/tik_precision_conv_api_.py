#!/usr/bin/env python
# -*- coding:utf-8 -*-
"""
Copyright (C) 2021. Huawei Technologies Co., Ltd. All rights reserved.
FILE:     tik_precision_conv_api_.py
DESC:     precision convert operations vector instructions for tik 1.5
CREATED:  2021-06-24 18:53:42
MODIFIED: 2021-06-24 19:17:00
"""

from collections import namedtuple
from tbe import tvm
from tbe.common.platform import intrinsic_check_support
from tbe.common.utils.log import info
from tbe.tik.tik_lib.tik_h_vec_api.common_params_check_ import check_precision_conv_params
from tbe.tik.tik_lib.tik_h_vec_api.common_util_ import set_tik_version_1_5
from tbe.tik.tik_lib.tik_check_util import TikCheckUtil
from tbe.tik.tik_lib.tik_source_info import source_info_decorator
from tbe.tik.tik_lib.tik_params import gen_api_check_statement
from tbe.tik.tik_lib.tik_expr import Expr
from tbe.tik.tik_lib.tik_api_constants import DTYPE_MAP
from tbe.tik.tik_lib.tik_api_constants import ROUND_MODE_MAP
from tbe.tik.tik_lib.tik_h_vec_api.tik_h_vec_api_base_ import TikHVecApiBase
from tbe.tik.tik_lib.tik_soc_manager import TikSocManager
from tbe.tik.debug.decorators_high_vec import cast_conv_debug_for_high_api
from tbe.tik.debug.decorators_high_vec import quant_debug_for_high_api


class TikPrecisionConvApi(TikHVecApiBase):
    """
    Precision Convert Operation Api
    """

    h_cast_api = namedtuple("Cast", "dst src round_mode api_name dtype_str")
    h_quant_api = namedtuple("Quant", "dst src deqscale api_name dtype_str")

    def __init__(self):
        super(TikPrecisionConvApi, self).__init__()
        self.core_arch = None
        self.core_version = None
        self.ir_generator = self

    @source_info_decorator()
    def h_conv(self, dst, src, round_mode):
        """
        vector conv for tik 1.5
        Parameters
        ----------
        dst: Tensor or Tensor slice
        src: Tensor or Tensor slice
        round_mode: round mode for precision conv

        Returns
        -------
        None
        """

        return self.h_cast(dst, src, round_mode)

    @source_info_decorator()
    def h_cast(self, dst, src, round_mode):
        """
        vector cast for tik 1.5
        Parameters
        ----------
        dst: Tensor or Tensor slice
        src: Tensor or Tensor slice
        round_mode: string

        Returns
        -------
        None
        """
        api_name = "h_cast"

        TikCheckUtil.check_var_in_list(round_mode, ROUND_MODE_MAP,
                                       "round_mode: %s is not supported" % round_mode)
        dtype_str = "%s2%s%s" % (DTYPE_MAP[src.dtype], DTYPE_MAP[dst.dtype], ROUND_MODE_MAP[round_mode])
        dtype_str_error = "src %s dst %s round_mode '%s'" % (src.dtype, dst.dtype, round_mode)
        is_supported_dtype = intrinsic_check_support("Intrinsic_vconv", dtype_str) and \
                             dtype_str not in ["s322f16", "s162s8", "s162u8"]
        TikCheckUtil.check_equality(is_supported_dtype, True,
                                    gen_api_check_statement(dtype_str_error, api_name))
        if not (TikSocManager.is_v100_soc() or TikSocManager.is_310b_soc()):
            info(gen_api_check_statement(dtype_str_error, api_name))
        h_cast_api_ins = TikPrecisionConvApi.h_cast_api(dst, src, round_mode, api_name, dtype_str)
        check_precision_conv_params(dst, src, dtype_str)
        self._cast_ops_code_make(h_cast_api_ins)

    @source_info_decorator()
    def h_quant(self, dst, src, deqscale):
        """
        vector quant for tik 1.5
        Parameters
        ----------
        dst: Tensor or Tensor slice
        src: Tensor or Tensor slice
        deqscale: int/float/Scalar/Expr

        Returns
        -------
        None
        """
        api_name = "h_quant"

        dtype_str = DTYPE_MAP.get(src.dtype) + "2" + DTYPE_MAP.get(dst.dtype)
        dtype_str_error = "src " + src.dtype + " dst " + dst.dtype
        is_supported_dtype = \
            intrinsic_check_support("Intrinsic_vconv", dtype_str) and dtype_str in ["s322f16", "s162s8", "s162u8"]

        TikCheckUtil.check_equality(is_supported_dtype, True,
                                    gen_api_check_statement(dtype_str_error, api_name))
        if not (TikSocManager.is_v100_soc() or TikSocManager.is_310b_soc()):
            info(gen_api_check_statement(dtype_str_error, api_name))

        check_precision_conv_params(dst, src, dtype_str, deqscale)
        if dtype_str == "s322f16":
            deqscale = Expr(deqscale, "float16")
        else:
            deqscale = Expr(deqscale, "uint64")
        h_quant_api_ins = TikPrecisionConvApi.h_quant_api(dst, src, deqscale, api_name, dtype_str)
        self._quant_ops_code_make(h_quant_api_ins)

    @set_tik_version_1_5()
    @cast_conv_debug_for_high_api
    def _cast_ops_code_make(self, cast_params):
        """
        code maker for cast, conv ops instruction
        """
        with self.new_scope():
            self.add_source_id()
            self.ir_generator.emit(
                tvm.tir.Evaluate(
                    tvm.call_cce_intrin(
                        "int32", cast_params.api_name,
                        cast_params.dst.info_node, cast_params.src.info_node, cast_params.round_mode)))

    @set_tik_version_1_5()
    @quant_debug_for_high_api
    def _quant_ops_code_make(self, quant_params):
        """
        code maker for quant ops instruction
        """
        with self.new_scope():
            self.add_source_id()
            self.ir_generator.emit(
                tvm.tir.Evaluate(
                    tvm.call_cce_intrin(
                        "int32", quant_params.api_name,
                        quant_params.dst.info_node, quant_params.src.info_node, quant_params.deqscale.get())))
