#!/usr/bin/env python
# -*- coding:utf-8 -*-
"""
Copyright (C) 2019. Huawei Technologies Co., Ltd. All rights reserved.
FILE:     matmul.py
DESC:     matmul function
CREATED:  2020-4-23 21:12:13
MODIFIED: 2020-12-7 19:17:00
"""

import math
import itertools
from collections import namedtuple
from tbe.common.platform import scope_ca
from tbe.common.platform import scope_cbuf
from tbe.common.platform.platform_info import api_check_support
from tbe.tik.tik_lib.tik_params import scope_cbuf_out
from tbe.tik.tik_lib.tik_api_constants import DTYPE_MAP
from tbe.tik.tik_lib.tik_params import gen_api_check_statement
from tbe.tik.tik_lib.tik_params import BYTE_PER_C0
from tbe.tik.tik_lib.tik_expr import Expr
from tbe.tik.api.cube.reindex import ReIndexProxy
from tbe.tik.common.util import ceil_div
from tbe.tik.common.util import reduce_mul
from tbe.tik.common.util import DTYPE_SIZE
from tbe.tik.common.util import TikCheckUtil
from tbe.tik.tik_lib.tik_soc_manager import TikSocManager
from tbe.tik.tik_lib.tik_params import COUT_B16
from tbe.tik.api.tik_tensor import Tensor
from tbe.tik.api.cube.cube_common import CubeTileCycles
from tbe.tik.api.cube.cube_common import CubeTilingMake
from tbe.tik.api.cube.cube_common import CubeModeTiling
from tbe.tik.common.common_util import get_l0c_align
from tbe.tik.common.common_util import check_address_align
from tbe.tik.common.tik_get_soc_name import get_cube_size

MatmulTileInfoStruct = namedtuple("MatmulTileInfoV1", ["m", "n", "k", "c0", "m_tile_block",
                                                       "m_thread_num", "n_tile_block",
                                                       "n_thread_num", "k_tile_block",
                                                       "k_thread_num", "l0b_mode", "loop_mode"])
MatmulValidInfo = namedtuple("MatMulValidTilingInfoV1", ["m_num", "k_num", "n_num", "c_0", "block_size",
                                                         "fm_dtype", "filter_dtype", "tik_instance"])
MatmulParamClass = namedtuple(
    "MatMulParam", ["feature_map", "weight", "tik_instance", "init_l1out", "l1_bias"])


class MatmulTileInfo(CubeTileCycles):
    """
    matmul tile info class
    """

    def __init__(self, matmul_tile, cube_size, l0c_dtype):
        """
        init the matmul tile info.

        Parameter
        ---------
        m:   the number of m.
        k:   the number of k.
        n:   the number of n.
        c0:  the number of c0.
        m_tile_block:  the block  number of m.
        m_thread_num:  the thread number of m.
        n_tile_block:  the block  number of n.
        n_thread_num:  the thread number of n.
        k_tile_block:  the block number of k.
        k_thread_num:  the thread number of k.
        loop_mode:   the mode of loop.
        """
        super(MatmulTileInfo, self).__init__(cube_size)
        self.is_matmul = True
        self.m_num = matmul_tile.m
        self.n_num = matmul_tile.n
        self.k_num = matmul_tile.k
        self.c_0 = matmul_tile.c0
        self.k_thread_num = matmul_tile.k_thread_num
        self.k_tile_block = matmul_tile.k_tile_block
        self.m_tile_block = matmul_tile.m_tile_block
        self.m_thread_num = matmul_tile.m_thread_num
        self.n_tile_block = matmul_tile.n_tile_block
        self.n_thread_num = matmul_tile.n_thread_num
        self.loop_mode = matmul_tile.loop_mode
        self.block_size = 16
        if l0c_dtype == "float16":
            self.is_output_fp16 = True
        self.m_iter_num = math.ceil(self.m_num / (self.m_tile_block * self.block_size))
        self.n_iter_num = math.ceil(self.n_num / (self.n_tile_block * self.block_size))
        self.k_iter_num = math.ceil(self.k_num / (self.k_tile_block * self.c_0))

        self.m_block_num = ceil_div(matmul_tile.m, self.block_size)
        self.k_block_num = ceil_div(matmul_tile.k, self.c_0)
        self.n_block_num = ceil_div(matmul_tile.n, self.block_size)
        self.m_tile_nums = self.m_tile_block * self.block_size

        self.m_tail_block = self.m_block_num - (self.m_iter_num - 1) * matmul_tile.m_tile_block
        self.k_tail_block = self.k_block_num - (self.k_iter_num - 1) * matmul_tile.k_tile_block
        self.n_tail_block = self.n_block_num - (self.n_iter_num - 1) * matmul_tile.n_tile_block

        self.k_has_tail = self.k_tail_block != self.k_tile_block
        self.n_has_tail = self.n_tail_block != self.n_tile_block

        if self.m_num != self.m_iter_num * self.m_tile_block * self.block_size:
            self.m_has_tail = 1
        else:
            self.m_has_tail = 0

        self.m_tail_nums = 0
        if self.m_has_tail:
            self.m_tail_nums = self.m_num - (self.m_iter_num - 1) * self.m_tile_block * self.block_size

        self.total_cycles = 0
        self.l0b_mode = matmul_tile.l0b_mode
        self.cal_cycles()

    def __str__(self):
        return "MatmulTileInfo\n" \
               "m iter {} block {} thread {} has_tail {}\n" \
               "n iter {} block {} thread {} has_tail {}\n" \
               "k iter {} block {} thread {} has_tail {}\n" \
               "loop mode {} \n" \
               "total cycles: {}".format(
            self.m_iter_num, self.m_tile_block, self.m_thread_num, self.m_has_tail,
            self.n_iter_num, self.n_tile_block, self.n_thread_num, self.n_has_tail,
            self.k_iter_num, self.k_tile_block, self.k_thread_num, self.k_has_tail,
            self.loop_mode, self.total_cycles)

    def __lt__(self, other):
        """sort by total_cycles with ascending order"""
        return self.total_cycles < other.total_cycles

    @staticmethod
    def get_load_l0a_cycles(m_actual_block, k_actual_block, l0b_load_cycles):
        """
        cal load l0a cycles of tiling
        """
        # get_load_l0a_cycles is method of father class and called by father
        # so l0b_load_cycles must be added, but child is not used
        # cost between two load_l0a instructions
        between_cost = 10
        if k_actual_block == 1:
            # Load one column at a time
            return m_actual_block
        # Load one row
        return (k_actual_block + between_cost) * m_actual_block - between_cost

    def cal_cycles(self):
        """
        cal cycles of diff loop mode
        """
        # method 1: All conditions are covered
        if self.loop_mode == 'nm':
            self.total_cycles = self.cal_nm_loop_cycle()
        elif self.loop_mode == 'mn':
            self.total_cycles = self.cal_mn_loop_cycles()
        else:
            TikCheckUtil.raise_error("matmul contains error loop_mode")


class MatMulValidTilingInfo(CubeTilingMake):
    """
    valid tiling info
    """

    def __init__(self, matmul_valid, cube_size, l0c_dtype):
        super(MatMulValidTilingInfo, self).__init__()
        self.m_num = matmul_valid.m_num
        self.k_num = matmul_valid.k_num
        self.n_num = matmul_valid.n_num
        self.m_block_num = math.ceil(matmul_valid.m_num / matmul_valid.block_size)
        self.n_block_num = math.ceil(matmul_valid.n_num / matmul_valid.block_size)
        self.k_block_num = math.ceil(matmul_valid.k_num / matmul_valid.c_0)
        self.block_size = matmul_valid.block_size
        self.c_0 = matmul_valid.c_0
        self.tik_instance = matmul_valid.tik_instance

        self.fm_dtype_size = DTYPE_SIZE.get(matmul_valid.fm_dtype)
        self.filter_dtype_size = DTYPE_SIZE.get(matmul_valid.filter_dtype)
        self.cube_size = cube_size
        self.l0c_dtype = l0c_dtype
        self.valid_solution = []

    @staticmethod
    def get_l0b_mode(tile_block):
        """
        get l0b mode
        """
        if tile_block == 1:
            return 1
        return 0

    def gen_valid_tiling(self, bias=None):
        """
        create all tiling
        """
        loop_modes = self.get_loop_modes(self.m_block_num, self.n_block_num,
                                         self.k_block_num, ["nm", "mn"])

        def _is_valid_tiling_with_bias(n_tile_block):
            is_valid = True
            if bias and (n_tile_block > 16 or (n_thread_num > 1 and n_tile_block > 8)):
                is_valid = False
            if bias and loop_mode == 'nm' and k_thread_num > 1 and n_tile_block > 8:
                is_valid = False
            if bias and loop_mode == 'mn' and (k_thread_num > 1 or m_thread_num > 1) and n_tile_block > 8:
                is_valid = False
            return is_valid

        def _valid_tiling(a_nums, k_tile_block_z):
            for n_tile_block in range(1, self.n_block_num + 1):
                n_iter_num = math.ceil(self.n_block_num / n_tile_block)
                # if cut m then we must load l0b one column at a time
                if (m_iter_num > 1 and n_tile_block > 1) or n_thread_num > n_iter_num or \
                        self.check_tensor_size_by_mode(a_nums, m_tile_block, k_tile_block_z,
                                                       n_tile_block) is False:
                    continue
                if not _is_valid_tiling_with_bias(n_tile_block):
                    continue

                l0b_mode = self.get_l0b_mode(n_tile_block)
                tile_info = MatmulTileInfo(MatmulTileInfoStruct(self.m_num, self.n_num, self.k_num, self.c_0,
                                                                m_tile_block, m_thread_num, n_tile_block,
                                                                n_thread_num, k_tile_block_z, k_thread_num,
                                                                l0b_mode, loop_mode), self.cube_size, self.l0c_dtype)
                self.valid_solution.append(tile_info)

        for nums, k_tile_block in itertools.product(loop_modes, range(1, self.k_block_num + 1)):
            m_thread_num, n_thread_num, k_thread_num, loop_mode = nums
            k_iter_num = math.ceil(self.k_block_num / k_tile_block)
            if k_thread_num > k_iter_num:
                continue
            for m_tile_block in range(1, self.m_block_num + 1):
                m_iter_num = math.ceil(self.m_block_num / m_tile_block)
                if m_thread_num > m_iter_num:
                    continue
                _valid_tiling(nums, k_tile_block)


def gen_best_tiling(best_tiling, l1_bias=None):
    """
    generator the best of tiling.
    src0's shape : [m, k], src1's shape : [k, n]

    Parameters
    ----------

    Returns
    ----------
    best_tiling: the best tiling
    """
    cube_size = get_cube_size()
    if best_tiling.a_dtype == "int8":
        cube_size[2] = cube_size[2] * 2

    valid_tiling = MatMulValidTilingInfo(MatmulValidInfo(best_tiling.round_m, best_tiling.round_k,
                                                         best_tiling.round_n, best_tiling.c_0, best_tiling.block_size,
                                                         best_tiling.a_dtype, best_tiling.b_dtype,
                                                         best_tiling.tik_instance), cube_size, best_tiling.c_dtype)
    valid_tiling.gen_valid_tiling(l1_bias)
    valid_tiling.valid_solution.sort()
    return valid_tiling.valid_solution[0]


class MatMulImpl(CubeModeTiling):
    """
    the implement of matmul
    """

    def __init__(self, init_l1out, l1_out_dst, tik_instance):
        """
        init of matmul.

        Parameters
        ----------
        tik_instance: the instance of tik.
        l1_out_dst: the result tensor of mmad.
        init_l1out: mark whether to init on l1out tensor mat_c if False, will accumulate on l1out tensor mat_c
        """
        super(MatMulImpl, self).__init__()
        self.init_l1out = init_l1out
        self.l1_out_dst = l1_out_dst
        self.tik_instance = tik_instance
        self.block_size = 16
        self.round_m = 0
        self.round_k = 0
        self.round_n = 0
        self.a_dtype = None
        self.b_dtype = None
        self.c_dtype = None
        self.mat_a = None
        self.mat_b = None
        self.m_num = 0
        self.k_num = 0
        self.c_0 = 0
        self.n_num = 0
        self.k_has_tail_ele = False
        self.k_tail_ele = 0

    @staticmethod
    def get_make_code():
        """
        Obtains the namedtuple object.
        """
        load_l0a_cls = namedtuple("LoadL0A", ["k_actual", "k_idx", "m_actual_size", "m_idx", "tiling", "param"])
        load_l0b_cls = namedtuple("LoadL0B", ["cin_actual", "cout_actual", "cout_i", "cin_i", "tiling", "param"])
        mmad_cls = namedtuple("Mmad", ["data_l0a", "data_l0b", "k_actual", "m_actual_size", "n_actual", "k_idx",
                                       "m_idx", "n_idx", "tiling", "param"])
        return load_l0a_cls, load_l0b_cls, mmad_cls

    def execute(self, matmul_api):
        """
        execute of matmul.
        Parameters
        ----------
        matmul_api.mat_a:  the a tensor
        matmul_api.mat_b:  the b tensor.
        matmul_api.m_num:      the number of m.
        matmul_api.k_num:      the number of k.
        matmul_api.n_num:      the number of n.

        Returns
        -------
        no return
        """
        self.m_num = matmul_api.m_num
        self.k_num = matmul_api.k_num
        self.n_num = matmul_api.n_num
        self._check_params(matmul_api.mat_a, matmul_api.mat_b, matmul_api.l1_bias)

        self.mat_a = ReIndexProxy(matmul_api.mat_a, (matmul_api.m_num, matmul_api.k_num))
        self.mat_b = ReIndexProxy(matmul_api.mat_b, (matmul_api.k_num, matmul_api.n_num))
        self.a_dtype = self.mat_a.dtype
        self.b_dtype = self.mat_b.dtype
        self.c_dtype = self.l1_out_dst.dtype

        # means fp16:16*16 or int8:16*32 or fp32:8*16
        self.c_0 = BYTE_PER_C0 // DTYPE_SIZE.get(self.a_dtype)

        # shape cut
        self.round_m = math.ceil(self.m_num / self.block_size) * self.block_size
        self.round_k = math.ceil(self.k_num / self.c_0) * self.c_0
        self.round_n = math.ceil(self.n_num / self.block_size) * self.block_size
        self._check_overflow(matmul_api.mat_a, matmul_api.mat_b)
        self._check_operator_align(matmul_api.mat_a, matmul_api.mat_b)

        self.l1_out_dst = ReIndexProxy(
            self.l1_out_dst, (math.ceil(self.n_num / self.block_size), self.round_m, self.block_size))
        tiling = gen_best_tiling(self, matmul_api.l1_bias)
        self.k_has_tail_ele = self.round_k != self.k_num
        self.k_tail_ele = self.k_num % (tiling.k_tile_block * self.c_0)

        param = MatmulParamClass(
            self.mat_a, self.mat_b, self.tik_instance, self.init_l1out, matmul_api.l1_bias)

        if tiling.loop_mode == "nm":
            self.do_nm_tiling(tiling, param)
        else:
            self.do_mn_tiling(tiling, param)
        return tiling

    def make_load_l0a_code(self, param_cls):
        """
        make load l0a code
        """
        inst = self.tik_instance
        repeat_times = ceil_div(param_cls.m_actual_size, self.block_size)
        data_l0a = inst.Tensor(
            self.a_dtype, (repeat_times * self.block_size * param_cls.k_actual * self.c_0,),
            name=param_cls.param.feature_map.tensor.buffer.name + "L0A", scope=scope_ca)
        if TikSocManager.is_610l_soc():
            inst.load2dv3(data_l0a, self.mat_a.flat_access(0),
                          param_cls.m_idx * param_cls.tiling.m_tile_block,  # mStartPosition
                          param_cls.k_idx * param_cls.tiling.k_tile_block,  # kStartPosition
                          param_cls.tiling.m_block_num, repeat_times,  # srcStride, dstStride
                          repeat_times, param_cls.k_actual, 0)  # mStep, kStep
            return data_l0a
        if param_cls.k_actual == 1:
            # Load one column
            l1a_b = param_cls.k_idx * param_cls.tiling.k_tile_block * param_cls.tiling.m_block_num * \
                    self.block_size * self.c_0
            l1a_offset = param_cls.m_idx * param_cls.tiling.m_tile_block * self.block_size * self.c_0 + l1a_b
            if TikSocManager.is_v100_soc():
                inst.load2dv1(data_l0a, self.mat_a.flat_access(l1a_offset), 0, repeat_times, 1, 0)
            else:
                inst.load2dv2(data_l0a, self.mat_a.flat_access(l1a_offset), 0, repeat_times, 0, 1, 0)
        else:
            # Load row by row
            with inst.for_range(0, repeat_times) as m_tile_idx:
                l0a_offset = m_tile_idx * param_cls.k_actual * self.block_size * self.c_0
                l1a_offset = \
                    (param_cls.m_idx * param_cls.tiling.m_tile_block + m_tile_idx) * \
                    self.block_size * self.c_0 + param_cls.k_idx * \
                    param_cls.tiling.k_tile_block * param_cls.tiling.m_block_num * \
                    self.block_size * self.c_0
                if TikSocManager.is_v100_soc():
                    inst.load2dv1(data_l0a[l0a_offset], self.mat_a.flat_access(l1a_offset),
                                  0, param_cls.k_actual, param_cls.tiling.m_block_num, 0)
                else:
                    inst.load2dv2(data_l0a[l0a_offset], self.mat_a.flat_access(l1a_offset),
                                  0, param_cls.k_actual, 0, param_cls.tiling.m_block_num, 0)
        return data_l0a

    def make_mmad_code(self, param_cls):
        """
        make mmad code
        Parameters
        ----------
        :param_cls: input params for matmul
        """
        l1_offset = param_cls.tiling.n_tile_block * param_cls.n_idx * COUT_B16
        l1_bias, init_l0c = self.get_bais_info(param_cls.param.l1_bias, l1_offset, param_cls.param.init_l1out)

        inst = self.tik_instance
        loc_block = param_cls.n_idx * param_cls.tiling.n_tile_block * param_cls.tiling.m_block_num * \
                    self.block_size * self.block_size
        l0c_offset = loc_block + param_cls.m_idx * param_cls.tiling.m_tile_block * self.block_size * self.block_size
        if param_cls.tiling.k_iter_num == 1:
            inst.mmad(self.l1_out_dst.flat_access(l0c_offset),
                      param_cls.data_l0a, param_cls.data_l0b, param_cls.m_actual_size,
                      self.k_num, param_cls.n_actual * self.block_size, init_l0c,
                      bias_tensor=l1_bias)
        elif self.init_l1out and self.k_has_tail_ele:
            with inst.if_scope(param_cls.k_idx == 0):
                inst.mmad(self.l1_out_dst.flat_access(l0c_offset),
                          param_cls.data_l0a, param_cls.data_l0b, param_cls.m_actual_size,
                          param_cls.k_actual * self.c_0, param_cls.n_actual * self.block_size, init_l0c,
                          bias_tensor=l1_bias)
            with inst.else_scope():
                self._mmad_has_k_tail(param_cls, l0c_offset)

        elif not self.init_l1out and self.k_has_tail_ele:
            self._mmad_has_k_tail(param_cls, l0c_offset)
        elif self.init_l1out and not self.k_has_tail_ele:
            with inst.if_scope(param_cls.k_idx == 0):
                inst.mmad(self.l1_out_dst.flat_access(l0c_offset),
                          param_cls.data_l0a, param_cls.data_l0b, param_cls.m_actual_size,
                          param_cls.k_actual * self.c_0, param_cls.n_actual * self.block_size, init_l0c,
                          bias_tensor=l1_bias)
            with inst.else_scope():
                inst.mmad(self.l1_out_dst.flat_access(l0c_offset),
                          param_cls.data_l0a, param_cls.data_l0b, param_cls.m_actual_size,
                          param_cls.k_actual * self.c_0, param_cls.n_actual * self.block_size, 1)
        else:
            inst.mmad(self.l1_out_dst.flat_access(l0c_offset),
                      param_cls.data_l0a, param_cls.data_l0b, param_cls.m_actual_size,
                      param_cls.k_actual * self.c_0, param_cls.n_actual * self.block_size, 1)

    def _check_params(self, mat_a, mat_b, bias):
        TikCheckUtil.check_type_match(self.l1_out_dst, Tensor,
                                      "dst should be Tensor, but dst's type is %s" % type(self.l1_out_dst))
        TikCheckUtil.check_type_match(mat_a, Tensor,
                                      "a should be Tensor, but a's type is %s" % type(mat_a))
        TikCheckUtil.check_type_match(mat_b, Tensor,
                                      "b should be Tensor, but b's type is %s" % type(mat_b))
        TikCheckUtil.check_equality(self.l1_out_dst.scope, scope_cbuf_out,
                                    "dst's scope should be L1_OUT, but dst's scope is %s" % self.l1_out_dst.scope)
        dst_align = get_l0c_align(self.l1_out_dst)
        check_address_align((self.l1_out_dst,), ("dst",), dst_align)
        TikCheckUtil.check_equality(mat_a.scope, scope_cbuf,
                                    "a's scope should be L1, but a's scope is %s" % mat_a.scope)
        TikCheckUtil.check_equality(mat_b.scope, scope_cbuf,
                                    "b's scope should be L1, but b's scope is %s" % mat_b.scope)
        l1_align = 512
        check_address_align((mat_a, mat_b), ("a", "b"), l1_align)
        # check dtype
        dtype_str = DTYPE_MAP.get(mat_a.dtype) + DTYPE_MAP.get(mat_b.dtype) + DTYPE_MAP.get(self.l1_out_dst.dtype)
        TikCheckUtil.check_equality(api_check_support("tik.matmul", dtype_str),
                                    True, gen_api_check_statement(dtype_str, "matmul"))
        TikCheckUtil.check_type_match(self.m_num, int,
                                      "m should be int, but m's type is %s" % type(self.m_num))
        TikCheckUtil.check_type_match(self.k_num, int,
                                      "k should be int, but k's type is %s" % type(self.k_num))
        TikCheckUtil.check_type_match(self.n_num, int,
                                      "n should be int, but n's type is %s" % type(self.n_num))
        # check m.k.n
        m_range = [1, 4096]
        # fp32 max k is 8192, fp16 max k is 16384, int8 max k is 32768
        k_range = [1, 32768 // DTYPE_SIZE.get(mat_a.dtype)]
        n_range = m_range
        TikCheckUtil.check_in_range_by_dtype(self.m_num, msg="m should be in range of [%d, %d], but m is %s"
                                                             % (m_range[0], m_range[1], self.m_num), var_range=m_range)
        TikCheckUtil.check_in_range_by_dtype(self.k_num, msg="k should be in range of [%d, %d], but k is %s"
                                                             % (k_range[0], k_range[1], self.k_num), var_range=k_range)
        TikCheckUtil.check_in_range_by_dtype(self.n_num, msg="n should be in range of [%d, %d], but n is %s"
                                                             % (n_range[0], n_range[1], self.n_num), var_range=n_range)
        # check init_l1out
        TikCheckUtil.check_type_match(self.init_l1out, bool, "init_l1out should be bool type.")
        n1 = ceil_div(self.n_num, self.block_size)
        b_shape = [n1, n1 * self.block_size, self.block_size]
        self.check_params_bias(bias, self.l1_out_dst, b_shape, self.init_l1out, "matmul")

    def _check_overflow(self, mat_a, mat_b):
        # check dst
        need_element = self.round_m * self.round_n
        if self.l1_out_dst.is_single_point():
            total_element = 1
        else:
            total_element = Expr(
                reduce_mul(self.l1_out_dst.original_shape) - self.l1_out_dst.offset).eval_value()
        if total_element is not None:
            TikCheckUtil.check_ge(total_element, need_element,
                                  "dst tensor overflow, expected elements: %s, actual elements: %s"
                                  % (need_element, total_element))
        # check a
        need_element = self.round_m * self.round_k
        if mat_a.is_single_point():
            total_element = 1
        else:
            total_element = Expr(reduce_mul(mat_a.original_shape) - mat_a.offset).eval_value()
        if total_element is not None:
            TikCheckUtil.check_ge(total_element, need_element,
                                  "a tensor overflow, expected elements: %s, actual elements: %s"
                                  % (need_element, total_element))
        # check b
        need_element = self.round_k * self.round_n
        if mat_b.is_single_point():
            total_element = 1
        else:
            total_element = Expr(reduce_mul(mat_b.original_shape) - mat_b.offset).eval_value()
        if total_element is not None:
            TikCheckUtil.check_ge(total_element, need_element,
                                  "b tensor overflow, expected elements: %s, actual elements: %s"
                                  % (need_element, total_element))

    def _check_operator_align(self, mat_a, mat_b):
        tensor_offset = Expr(self.l1_out_dst.offset).eval_value()
        if tensor_offset is not None:
            TikCheckUtil.check_equality(tensor_offset * DTYPE_SIZE.get(self.l1_out_dst.dtype) % 1024, 0,
                                        "dst address should be 1024B aligned")
        if TikSocManager.is_v100_soc():
            align = 512
        else:
            align = 32
        for cur_tensor in (mat_a, mat_b):
            tensor_offset = Expr(cur_tensor.offset).eval_value()
            if tensor_offset is not None:
                TikCheckUtil.check_equality(tensor_offset * DTYPE_SIZE.get(cur_tensor.dtype) % align, 0,
                                            "source a and b address should be %sB aligned" % align)

    def _mmad_has_k_tail(self, param_cls, l0c_offset):
        inst = self.tik_instance
        with inst.if_scope(param_cls.k_idx == param_cls.tiling.k_iter_num - 1):
            inst.mmad(self.l1_out_dst.flat_access(l0c_offset),
                      param_cls.data_l0a, param_cls.data_l0b, param_cls.m_actual_size,
                      self.k_tail_ele, param_cls.n_actual * self.block_size, 1)
        with inst.else_scope():
            inst.mmad(self.l1_out_dst.flat_access(l0c_offset),
                      param_cls.data_l0a, param_cls.data_l0b, param_cls.m_actual_size,
                      param_cls.k_actual * self.c_0, param_cls.n_actual * self.block_size, 1)
