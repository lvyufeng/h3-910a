#!/usr/bin/env python
# -*- coding:utf-8 -*-
"""
Copyright (C) 2019. Huawei Technologies Co., Ltd. All rights reserved.
FILE:     tik_cube_api_.py
DESC:     provide cube calculation related instructions
CREATED:  2019-08-12 18:53:42
MODIFIED: 2020-12-7 19:17:00
"""
from collections import namedtuple
from tbe.tik.api.tik_ir_builder import TikIRBuilder
from tbe.tik.tik_lib.tik_check_util import TikCheckUtil
from tbe.tik.api.tik_tensor import Tensor
from tbe.tik.tik_lib.tik_source_info import source_info_decorator
from tbe.tik.debug.decorators import high_level_api_debug_decorator
from tbe.tik.tik_lib.tik_params import C1_C04
from tbe.tik.api.cube.fixpipe import FixpipeImpl
from tbe.tik.tik_lib.tik_soc_manager import TikSocManager
from tbe.tik.api.cube.conv2d import Conv2dImpl
from tbe.tik.api.cube.conv2d_c04 import Conv2dC04Impl
from tbe.tik.api.cube.fixpipe_v2 import FixPipeImplV2
from tbe.tik.api.cube.matmul import MatMulImpl

MatmulApiCls = namedtuple("MatmulApi", ["mat_a", "mat_b", "m_num", "k_num", "n_num", "l1_bias"])
Conv2dApiCls = namedtuple("Conv2dApi", ["dst", "feature_map", "weight", "fm_shape", "kernel_shape",
                                        "stride", "pad", "dilation", "pad_value", "init_l1out", "bias"])
FixpipeApiCls = namedtuple("FixpipeApi", ["dst", "src", "cburst_num", "burst_len", "dst_stride",
                                          "src_stride", "extend_params"])


def _checkout_bias_tensor_dim(tensor):
    """
    check dim of bias tensor
    Parameters
    ----------
    tensor: Tensor

    Returns
    ----------
    None
    """
    TikCheckUtil.check_type_match(tensor, Tensor, "input must be Tensor")
    TikCheckUtil.check_equality(len(tensor.shape), 4,
                                "tensor shape must be 4 dimension")
    TikCheckUtil.check_equality(
        tensor.shape[-1], 16,
        "the bias input Tensor 3th dimension must be 16")


class TikCubeOpenApi(TikIRBuilder):
    """
    Cube Api
    """

    def __init__(self):
        super(TikCubeOpenApi, self).__init__()
        self.core_arch = None
        self.core_version = None

    @source_info_decorator()
    @high_level_api_debug_decorator
    def conv2d(self, dst, feature_map, weight, fm_shape, kernel_shape, stride, pad, dilation, pad_value=0,
               init_l1out=True, bias=None):
        """
        Performs 2D convolution on an input tensor and a weight tensor and
        outputs a result tensor.
        Parameters
        ----------
        dst: Start element of the destination operand.
        feature_map: Start element of the input tensor operand.
        weight: Start element of the weight tensor operand.
        fm_shape: Shape of the input tensor, in the format of [C1, H, W, C0].C1 * C0
        -         indicates the number of input channels.
        kernel_shape: Shape of each convolution kernel tensor, in the format of [C1, Kh, Kw, Cout, C0].
        stride: Convolution stride, in the format of [stride_h, stride_w].
        pad: Padding factors, in the format of [pad_left, pad_right, pad_top, pad_bottom].
        dilation: Convolution dilation factors, in the format of [dilation_h, dilation_w]
        pad_value: Padding value, an immediate of int or float Defaults to 0.
        init_l1out: A bool specifying whether to initialize dst. Defaults to True.
        bias: Start element of the Bias oprand, in the format of [cout, ], it only support for ascend910b
        Returns
        -------
        no return
        """
        conv2d_api = Conv2dApiCls(dst, feature_map, weight, fm_shape, kernel_shape,
                                  stride, pad, dilation, pad_value, init_l1out, bias)
        if fm_shape[-1] == C1_C04 or TikSocManager.is_910b_soc() or TikSocManager.is_310b_610l_soc():
            conv2d_op = Conv2dC04Impl(self)
        else:
            conv2d_op = Conv2dImpl(self)
        conv2d_op.execute(conv2d_api)
        self.set_high_level_api_state()

    @source_info_decorator()
    @high_level_api_debug_decorator
    def fixpipe(self, dst, src, cburst_num, burst_len, dst_stride, src_stride, extend_params=None):
        """
        Processes the matrix computation result
        Parameters
        ----------
        dst: A tensor of type float16, float32, or int32, for the start element of the destination operand.
        src: tensor of type float32 or int32, for the start element of the source operand.
        cburst_num: An immediate of type int specifying the number of bursts. The value range is [1, 4095].
        burst_len: Burst length, in the unit of 32 bytes. The value is an even number within the range [2, 65535].
        dst_stride: Tail-to-header stride between adjacent bursts of the dst operand tensor,
        -           in the unit of 32 bytes.
        src_stride: Tail-to-header stride between adjacent bursts of the src operand tensor,
        -           in the unit of 256 elements.
        extend_params: A dictionary of extended parameters. Defaults to None.

        Returns
        -------
        no return
        """
        fixpipe_api = FixpipeApiCls(dst, src, cburst_num, burst_len, dst_stride, src_stride, extend_params)
        if TikSocManager.is_910b_soc() or TikSocManager.is_310b_610l_soc():
            fixpipe_op = FixPipeImplV2(self)
        else:
            fixpipe_op = FixpipeImpl(self)
        fixpipe_config = fixpipe_op.check_all(fixpipe_api)
        fixpipe_op.execute(fixpipe_api, fixpipe_config)

    @source_info_decorator()
    @high_level_api_debug_decorator
    def matmul(self, dst, a, b, m, k, n, init_l1out=True, bias=None):
        """
        Multiplies matrix a by matrix b and outputs a result tensor.
        Parameters
        ----------
        dst: Start element of the destination operand. For details about the data type restrictions,
        -    see Table Data type combination of a, b, and dst.
        a: Source operand, matrix tensor a.
        b: Source operand, matrix tensor b.
        m: An immediate of type int specifying the valid height of matrix a. The value range is [1, 4096].
        k: An immediate of type int specifying the valid width of matrix a and the valid height of matrix b.
        n: An immediate of type int specifying the valid width of matrix b. The value range is [1, 4096].
        init_l1out: A bool specifying whether to initialize dst. Defaults to True.

        Returns
        -------
        no return
        """

        matmul_api = MatmulApiCls(a, b, m, k, n, bias)
        matmul_op = MatMulImpl(init_l1out, dst, self)
        matmul_op.execute(matmul_api)
        self.set_high_level_api_state()
