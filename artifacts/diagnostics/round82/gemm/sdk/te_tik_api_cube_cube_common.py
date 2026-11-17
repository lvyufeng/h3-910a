#!/usr/bin/env python
# -*- coding:utf-8 -*-
"""
Copyright (C) 2019. Huawei Technologies Co., Ltd. All rights reserved.
FILE:     cube_common.py
DESC:     provide tiling method
CREATED:  2020-6-5 17:35:36
MODIFIED: 2020-12-7 19:17:00
"""
import itertools
from collections import namedtuple
from tbe import tvm
from tbe.common.platform import scope_ca
from tbe.common.platform import scope_cb
from tbe.common.platform import scope_cbuf
from tbe.tik.tik_lib.tik_check_util import TikCheckUtil
from tbe.tik.api.tik_logic_compute import all as tik_all
from tbe.tik.api.tik_tensor import Tensor
from tbe.tik.common.tik_get_soc_name import get_soc_name
from tbe.tik.tik_lib.tik_soc_manager import TikSocManager
from tbe.tik.tik_lib.tik_params import COUT_B16
from tbe.tik.common.tik_get_soc_name import get_l0a_buf_size, get_l0b_buf_size

TilingLoopInfo = namedtuple("TilingLoopInfo", ["iter_num", "mte1_for", "mte1_tail", "cube_for", "cube_tail",
                                               "thread_num", "origin_thread_num", "has_tail_blk", "inner_iter_num"])
LoadL0aCls = namedtuple("LoadL0A", ["cin_actual", "cin_i", "hw_actual_size", "hw_i", "tiling", "param"])
LoadL0bCls = namedtuple("LoadL0B", ["cin_actual", "cout_actual", "cout_i", "cin_i", "tiling", "param"])
MmadCls = namedtuple("Mmad", ["data_l0a", "data_l0b", "cin_actual", "hw_actual_size", "cout_actual", "cin_i",
                              "hw_i", "cout_i", "tiling", "param"])
TilingLoopInfo.__new__.__defaults__ = (0, 0, 0, 0, 0, 1, 1, False, 0)


class CubeTileCycles:
    """
    class for conv2d and matmul to calculate cycles of different tiling
    """

    def __init__(self, cube_size):
        self.m_tile_block = 0
        self.k_tile_block = 0
        self.n_tile_block = 0
        self.m_has_tail = False
        self.k_has_tail = False
        self.n_has_tail = False
        self.m_tail_block = 0
        self.k_tail_block = 0
        self.n_tail_block = 0
        self.m_thread_num = 1
        self.k_thread_num = 1
        self.n_thread_num = 1
        self.l0b_mode = 0
        self.n_iter_num = 0
        self.m_iter_num = 0
        self.loop_mode = None
        self.is_conv2d = False
        self.is_conv2d_c04 = False
        self.is_matmul = False
        self.block_size = 16
        self.is_output_fp16 = False
        _, self.cube_k_size, self.cube_n_size = cube_size

    @staticmethod
    def _cal_kn_nk_mn_nm_cycles(mte1_cycles, cube_cycles, iter_nums, head_cycles, inner_iter_num=0):
        """
        cal kn nk mn nm loop modes double buffer cycles, only two for loop.
        """
        if mte1_cycles >= cube_cycles:
            if inner_iter_num % 2 == 0:
                tail_cycles = cube_cycles + sum(head_cycles) - mte1_cycles
                total_cycles = mte1_cycles * iter_nums + tail_cycles
                total_mte1_cycles = mte1_cycles * iter_nums
                total_cube_cycles = total_cycles - sum(head_cycles)
            else:
                tail_cycles = cube_cycles + sum(head_cycles) - mte1_cycles
                total_cycles = mte1_cycles + (mte1_cycles + tail_cycles - head_cycles[0]) * (
                        iter_nums - 1) + tail_cycles
                total_mte1_cycles = mte1_cycles * iter_nums
                total_cube_cycles = total_cycles - sum(head_cycles)
        else:
            if len(head_cycles) == 2:
                tail_cycles = cube_cycles + sum(head_cycles) - mte1_cycles
            else:
                tail_cycles = 2 * cube_cycles - mte1_cycles

            total_cycles = cube_cycles * iter_nums + sum(head_cycles)
            total_mte1_cycles = total_cycles - tail_cycles
            total_cube_cycles = cube_cycles * iter_nums
        return total_cycles, total_mte1_cycles, total_cube_cycles

    def get_cube_cycles(self, m_tile_block, k_tile_block, n_tile_block):
        """
        cal mmad cycles
        """
        # M/K/N is block nums
        # theroy cycles: float16: M*16*K*16*N*16/16/16/16 -> M*K*N
        #                int8:    M*16*K*32*N*16/16/32/16 -> M*K*N
        cube_cycles = m_tile_block * k_tile_block * n_tile_block * \
                      self.c_0 * self.block_size // self.cube_k_size // self.cube_n_size
        # M or N is odd, cube conflict cycles is (K - 2)
        # only mini has this problem
        if TikSocManager.is_v100_soc() or (TikSocManager.is_hisi_sd_cs() and self.is_output_fp16):
            if k_tile_block > 1 and (n_tile_block % 2 != 0 or m_tile_block % 2 != 0):
                cube_cycles += k_tile_block - 2
        # load head-tail cost, instruction lines is 12,
        # so head-tail cost is 12 - 1
        # but camodel log result is 15
        head_cost = 15
        return cube_cycles + head_cost

    def is_hisi_l0b_cycles(self, k_tile_block, n_tile_block):
        head_cost = 22
        if self.l0b_mode == 0 and self.n_iter_num != 1:
            # l0b mode is 0
            if k_tile_block < n_tile_block:
                load_l0b_cycles = k_tile_block * (n_tile_block * 4 + head_cost)
            else:
                load_l0b_cycles = n_tile_block * (k_tile_block * 4 + head_cost)
        else:
            # l0b mode is 1 or (l0b mode is 0 and n_iter_num is 1)load_l0b_cycles
            load_l0b_cycles = k_tile_block * n_tile_block * 4 + head_cost
        loop_mode_m_thread = self.loop_mode == "mn" and self.m_thread_num > 1
        n_iter_thread = self.n_iter_num > 1 and self.n_thread_num == 1
        k_iter_thread = k_tile_block == 1 and self.k_thread_num > 1
        if loop_mode_m_thread and n_iter_thread and k_iter_thread:
            load_l0b_cycles -= 14
        return load_l0b_cycles

    def get_load_l0b_cycles(self, k_tile_block, n_tile_block):
        """
        cal load2d cycles
        """
        if self.is_output_fp16 and TikSocManager.is_hisi_sd_cs():
            load_l0b_cycles = self.is_hisi_l0b_cycles(k_tile_block, n_tile_block)
        else:
            # DMA latency is 10 ~ 11
            head_cost = 10
            if self.l0b_mode == 0 and self.n_iter_num != 1:
                # l0b mode is 0
                load_l0b_cycles = k_tile_block * (n_tile_block * 2 + head_cost)
            else:
                # l0b mode is 1 or (l0b mode is 0 and n_iter_num is 1)
                load_l0b_cycles = k_tile_block * n_tile_block * 2 + head_cost
            return load_l0b_cycles
        return load_l0b_cycles

    def deal_with_double_buffer_cycles(self, do_loop_info, head_cycles=None, axis=None):
        """
        cal double buffer of multi for-loop cycles
        """
        head_cycles = [do_loop_info.mte1_for] if head_cycles is None else head_cycles
        # if loop mode is knm, kmn
        if axis == "k":
            total_cycles, total_mte1_cycles, total_cube_cycles = \
                self._cal_knm_kmn_cycles(do_loop_info, head_cycles)
        else:
            total_cycles, total_mte1_cycles, total_cube_cycles = \
                self._cal_kn_nk_mn_nm_cycles(do_loop_info.mte1_for, do_loop_info.cube_for, do_loop_info.iter_num,
                                             head_cycles)
        return total_cycles, total_mte1_cycles, total_cube_cycles

    def deal_with_for_range(self, do_loop_info, head_cycles=None, axis=None):
        """
        deal with for loop body to cal cycles
        """
        head_cycles = [do_loop_info.mte1_for] if head_cycles is None else head_cycles
        tail_cycles = do_loop_info.cube_for + sum(head_cycles) - do_loop_info.mte1_for
        if do_loop_info.thread_num == 1:
            total_cycles = (sum(head_cycles) + do_loop_info.cube_for) * do_loop_info.iter_num
            return total_cycles, total_cycles - tail_cycles, total_cycles - sum(head_cycles)
        return self.deal_with_double_buffer_cycles(do_loop_info, head_cycles, axis)

    def do_loop(self, do_loop_info, head_cycles_for=None, head_cycles_tail=None, axis=None):
        """
        deal with for loop to cal cycles
        """
        loop_cycles = 0
        loop_mte1_cycles = 0
        loop_cube_cycles = 0
        loop_head = 0
        if do_loop_info.iter_num > 0:
            head_cycles_for = [do_loop_info.mte1_for] if head_cycles_for is None else head_cycles_for
            loop_cycles, loop_mte1_cycles, loop_cube_cycles = \
                self.deal_with_for_range(do_loop_info, head_cycles_for, axis)
            loop_head = head_cycles_for
        if do_loop_info.has_tail_blk:
            head_cycles_tail = [do_loop_info.mte1_tail] if head_cycles_tail is None else head_cycles_tail
            tail_cycles_tail = do_loop_info.cube_tail + sum(head_cycles_tail) - do_loop_info.mte1_tail
            for_cycles_tail = loop_cycles - loop_mte1_cycles
            # for loop is always partially parallel to tail process
            if do_loop_info.origin_thread_num == 2:
                # in this case, for loop is partially parallel to tail process
                loop_cycles += sum(head_cycles_tail) + do_loop_info.cube_tail - min(head_cycles_tail[0],
                                                                                    for_cycles_tail)
            else:
                loop_cycles += sum(head_cycles_tail) + do_loop_info.cube_tail
            loop_head = head_cycles_tail if loop_head == 0 else loop_head
            loop_mte1_cycles = loop_cycles - tail_cycles_tail
            loop_cube_cycles = loop_cycles - sum(loop_head)
        return [loop_cycles, loop_mte1_cycles, loop_cube_cycles, loop_head]

    def cal_mn_loop_cycles(self):
        """
        cal mn loop mode cycles
        """
        k_iter_num, k_thread_num = get_iter_thread_num(self, 'cin')
        m_iter_num, m_thread_num = get_iter_thread_num(self, 'hw', k_thread_num)
        n_nums = get_iter_thread_num(self, 'cout', max(k_thread_num, m_thread_num))
        _, mn_loop_mte1_cycles_for, mn_loop_cube_cycles_for, mn_loop_head_for = self._do_mn_loop_cycles(
            self.k_tile_block, m_iter_num, m_thread_num, n_nums)
        mn_loop_mte1_cycles_tail = 0
        mn_loop_cube_cycles_tail = 0
        mn_loop_head_tail = 0
        if self.k_has_tail:
            _, mn_loop_mte1_cycles_tail, mn_loop_cube_cycles_tail, mn_loop_head_tail = self._do_mn_loop_cycles(
                self.k_tail_block, m_iter_num, m_thread_num, n_nums)
        if self.is_conv2d_c04 is True:
            tiling_loop_info = TilingLoopInfo(
                k_iter_num, mn_loop_mte1_cycles_for, mn_loop_mte1_cycles_tail, mn_loop_cube_cycles_for,
                mn_loop_cube_cycles_tail, k_thread_num, self.k_thread_num, self.k_has_tail, n_nums[0])
        else:
            tiling_loop_info = TilingLoopInfo(
                k_iter_num, mn_loop_mte1_cycles_for, mn_loop_mte1_cycles_tail, mn_loop_cube_cycles_for,
                mn_loop_cube_cycles_tail, k_thread_num, self.k_thread_num, self.k_has_tail)
        kmn_loop_cycles, _, _, _ = self.do_loop(
            tiling_loop_info, head_cycles_for=mn_loop_head_for, head_cycles_tail=mn_loop_head_tail, axis="k")
        return kmn_loop_cycles

    def cal_nm_loop_cycle(self):
        """
        cal knm loop mode cycles
        """
        k_iter_num, k_thread_num = get_iter_thread_num(self, 'cin')
        n_iter_num, n_thread_num = get_iter_thread_num(self, 'cout', k_thread_num)
        m_muns = get_iter_thread_num(self, 'hw', max(k_thread_num, n_thread_num))
        _, nm_loop_mte1_cycles_for, nm_loop_cube_cycles_for, nm_loop_head_for = \
            self._do_nm_loop_cycle(self.k_tile_block, n_iter_num, n_thread_num, m_muns)

        nm_loop_mte1_cycles_tail = 0
        nm_loop_cube_cycles_tail = 0
        nm_loop_head_tail = 0
        if self.k_has_tail:
            _, nm_loop_mte1_cycles_tail, nm_loop_cube_cycles_tail, nm_loop_head_tail = \
                self._do_nm_loop_cycle(self.k_tail_block, n_iter_num, n_thread_num, m_muns)

        if self.is_conv2d_c04 is True:
            tiling_loop_info = TilingLoopInfo(
                k_iter_num, nm_loop_mte1_cycles_for, nm_loop_mte1_cycles_tail, nm_loop_cube_cycles_for,
                nm_loop_cube_cycles_tail, k_thread_num, self.k_thread_num, self.k_has_tail, m_muns[0])
        else:
            tiling_loop_info = TilingLoopInfo(
                k_iter_num, nm_loop_mte1_cycles_for, nm_loop_mte1_cycles_tail, nm_loop_cube_cycles_for,
                nm_loop_cube_cycles_tail, k_thread_num, self.k_thread_num, self.k_has_tail)

        knm_loop_cycles, _, _, _ = self.do_loop(
            tiling_loop_info, head_cycles_for=nm_loop_head_for, head_cycles_tail=nm_loop_head_tail, axis="k")
        return knm_loop_cycles

    def _do_nm_loop_cycle(self, k_actual_blk, n_iter_num, n_thread_num, m_nums):
        """
        cal do nm loop
        """
        l0b_cycles_for = self.get_load_l0b_cycles(k_actual_blk, self.n_tile_block)
        _, m_loop_mte1_cycles_for, m_loop_cube_cycles_for, m_loop_head_for = \
            self._do_m_loop_cycle(self.n_tile_block, k_actual_blk, l0b_cycles_for, m_nums)
        l0b_cycles_tail = 0
        m_loop_mte1_cycles_tail = 0
        m_loop_cube_cycles_tail = 0
        m_loop_head_tail = [0]
        if self.n_has_tail:
            l0b_cycles_tail = self.get_load_l0b_cycles(k_actual_blk, self.n_tail_block)
            _, m_loop_mte1_cycles_tail, m_loop_cube_cycles_tail, m_loop_head_tail = \
                self._do_m_loop_cycle(self.n_tail_block, k_actual_blk, l0b_cycles_tail, m_nums)

        if self.is_conv2d_c04 is True:
            tiling_loop_info = TilingLoopInfo(
                n_iter_num, l0b_cycles_for + m_loop_mte1_cycles_for, l0b_cycles_tail + m_loop_mte1_cycles_tail,
                m_loop_cube_cycles_for, m_loop_cube_cycles_tail, n_thread_num,
                max(self.k_thread_num, self.n_thread_num), self.n_has_tail, m_nums[0])
        else:
            tiling_loop_info = TilingLoopInfo(
                n_iter_num, l0b_cycles_for + m_loop_mte1_cycles_for, l0b_cycles_tail + m_loop_mte1_cycles_tail,
                m_loop_cube_cycles_for, m_loop_cube_cycles_tail, n_thread_num,
                max(self.k_thread_num, self.n_thread_num), self.n_has_tail)

        return self.do_loop(tiling_loop_info, head_cycles_for=[l0b_cycles_for, m_loop_head_for[0]],
                            head_cycles_tail=[l0b_cycles_tail, m_loop_head_tail[0]])

    def _do_m_loop_cycle(self, n_actual_blk, k_actual_blk, load_l0b_cycles, m_nums):
        """cal do m loop """
        m_iter_num, m_thread_num = m_nums
        l0a_cycles_for = self.get_load_l0a_cycles(self.m_tile_block, k_actual_blk, load_l0b_cycles)
        cube_cycles_for = self.get_cube_cycles(self.m_tile_block, k_actual_blk, n_actual_blk)
        l0a_cycles_tail = 0
        cube_cycles_tail = 0
        if self.m_has_tail:
            l0a_cycles_tail = self.get_load_l0a_cycles(self.m_tail_block, k_actual_blk, load_l0b_cycles)
            cube_cycles_tail = self.get_cube_cycles(self.m_tail_block, k_actual_blk, n_actual_blk)
        return self.do_loop(TilingLoopInfo(
            m_iter_num, l0a_cycles_for, l0a_cycles_tail, cube_cycles_for, cube_cycles_tail, m_thread_num,
            max(self.k_thread_num, self.n_thread_num, self.m_thread_num), self.m_has_tail))

    def _do_mn_loop_cycles(self, k_actual_blk, m_iter_num, m_thread_num, n_nums):
        """
        cal do mn loop
        """
        l0b_cycles_for = self.get_load_l0b_cycles(k_actual_blk, self.n_tile_block)
        _, n_loop_mte1_cycles_for, n_loop_cube_cycles_for, n_loop_head_for = \
            self._do_n_loop_cycles(self.m_tile_block, k_actual_blk, l0b_cycles_for, n_nums)
        n_loop_mte1_cycles_tail = 0
        n_loop_cube_cycles_tail = 0
        n_loop_head_tail = [0]
        l0a_cycles_tail = 0
        if self.m_has_tail:
            l0a_cycles_tail = self.get_load_l0a_cycles(self.m_tail_block, k_actual_blk, l0b_cycles_for)
            _, n_loop_mte1_cycles_tail, n_loop_cube_cycles_tail, n_loop_head_tail = \
                self._do_n_loop_cycles(self.m_tail_block, k_actual_blk, l0b_cycles_for, n_nums)

        load_l0a_cycles = self.get_load_l0a_cycles(self.m_tile_block, k_actual_blk, l0b_cycles_for)
        if self.is_conv2d_c04 is True:
            tiling_loop_info = TilingLoopInfo(
                m_iter_num, load_l0a_cycles + n_loop_mte1_cycles_for, l0a_cycles_tail + n_loop_mte1_cycles_tail,
                n_loop_cube_cycles_for, n_loop_cube_cycles_tail, m_thread_num,
                max(self.k_thread_num, self.m_thread_num), self.m_has_tail, n_nums[0])
        else:
            tiling_loop_info = TilingLoopInfo(
                m_iter_num, load_l0a_cycles + n_loop_mte1_cycles_for, l0a_cycles_tail + n_loop_mte1_cycles_tail,
                n_loop_cube_cycles_for, n_loop_cube_cycles_tail, m_thread_num,
                max(self.k_thread_num, self.m_thread_num), self.m_has_tail)

        return self.do_loop(tiling_loop_info, head_cycles_for=[load_l0a_cycles, n_loop_head_for[0]],
                            head_cycles_tail=[l0a_cycles_tail, n_loop_head_tail[0]])

    def _do_n_loop_cycles(self, m_actual_blk, k_actual_blk, l0b_cycles_for, n_nums):
        """
        cal do n loop
        """
        n_iter_num, n_thread_num = n_nums
        cube_cycles_for = self.get_cube_cycles(m_actual_blk, k_actual_blk, self.n_tile_block)
        l0b_cycles_tail = 0
        cube_cycles_tail = 0
        if self.n_has_tail:
            l0b_cycles_tail = self.get_load_l0b_cycles(k_actual_blk, self.n_tail_block)
            cube_cycles_tail = self.get_cube_cycles(m_actual_blk, k_actual_blk, self.n_tail_block)
        return self.do_loop(TilingLoopInfo(
            n_iter_num, l0b_cycles_for, l0b_cycles_tail, cube_cycles_for, cube_cycles_tail, n_thread_num,
            max(self.k_thread_num, self.m_thread_num, self.n_thread_num), self.n_has_tail))

    def _cal_knm_kmn_cycles(self, do_loop_info, head_cycles):
        """
        cal loop mode knm, kmn double buffer cycles, three for loop,
        """
        # for knm kmn, if n or m loop times is odd,
        n_iter_num_bool = self.n_iter_num > 1 and self.n_iter_num & 1
        m_iter_num_bool = self.m_iter_num > 1 and self.m_iter_num & 1
        if (self.loop_mode == "nm" and n_iter_num_bool) or (self.loop_mode == "mn" and m_iter_num_bool):
            tail_cycles = do_loop_info.cube_for + sum(head_cycles) - do_loop_info.mte1_for
            # completely serial
            total_cycles = (sum(head_cycles) + do_loop_info.cube_for) * do_loop_info.iter_num
            total_mte1_cycles = total_cycles - tail_cycles
            total_cube_cycles = total_cycles - sum(head_cycles)
        # for knm kmn, if inner loop m or n is odd, ping last inner L0B or L0A
        # addr is same with pong first L0B or L0A, so may be raise interval
        elif (self.loop_mode == "nm" and m_iter_num_bool) or (self.loop_mode == "mn" and n_iter_num_bool):
            tail_cycles = do_loop_info.cube_for + sum(head_cycles) - do_loop_info.mte1_for
            # partially parallel
            total_cycles = (sum(head_cycles) + do_loop_info.cube_for) * do_loop_info.iter_num - \
                           (min(head_cycles[0], tail_cycles)) * (do_loop_info.iter_num - 1)
            total_mte1_cycles = total_cycles - tail_cycles
            total_cube_cycles = total_cycles - sum(head_cycles)
        else:
            total_cycles, total_mte1_cycles, total_cube_cycles = \
                self._cal_kn_nk_mn_nm_cycles(do_loop_info.mte1_for, do_loop_info.cube_for,
                                             do_loop_info.iter_num, head_cycles, do_loop_info.inner_iter_num)
        return total_cycles, total_mte1_cycles, total_cube_cycles


class CubeCodeMake():
    """
    class for conv2d or matmul to make code for load3d, load2d, mmad
    """

    @staticmethod
    def get_make_code():
        """
        Obtains the namedtuple object.
        """

        return LoadL0aCls, LoadL0bCls, MmadCls

    @staticmethod
    def make_load_l0a_code(param_cls):
        """
        create code of load3d, load data from L1 to L0A
        Parameters
        ----------
        param_cls.cin_actual: k tile block num
        param_cls.cin_i: index of k tile block
        param_cls.hw_actual: m tile block num
        param_cls.hw_i: index of m tile block
        param_cls.tiling : data struct save tiling info
        param_cls.param: others params, input fm shape, kernel info, dilations
        Returns
        -------
        data_l0a: tensor of load3d result
        """
        inst = param_cls.param.tik_instance
        data_l0a = inst.Tensor(param_cls.param.fm_desc.input_dtype, (param_cls.tiling.m_tile_block,
                                                                     param_cls.tiling.k_tile_block,
                                                                     param_cls.tiling.block_size, param_cls.tiling.c_0),
                               name=param_cls.param.feature_map.tensor.buffer.name + "L0A", scope=scope_ca)
        cin_pos = param_cls.cin_i * param_cls.tiling.k_tile_block
        # load by row
        if param_cls.tiling.l0a_mode == 1:
            with inst.for_range(0, param_cls.cin_actual) as k:
                ho_wo_pos = param_cls.hw_i * param_cls.tiling.m_tile_block * param_cls.tiling.block_size
                ho_idx = ho_wo_pos // param_cls.param.fm_desc.w_o
                wo_idx = ho_wo_pos % param_cls.param.fm_desc.w_o
                hi_idx = ho_idx * param_cls.param.fm_desc.stride_h
                wi_idx = wo_idx * param_cls.param.fm_desc.stride_w
                c1_idx = (cin_pos + k) // (param_cls.param.k_h * param_cls.param.k_w)
                k_hw_idx = (cin_pos + k) % (param_cls.param.k_h * param_cls.param.k_w)
                l0a_idx = k * param_cls.tiling.block_size * param_cls.tiling.c_0
                disable_c1 = 0  # cin_i * Cin_tiling + k
                c1_offset = c1_idx * param_cls.tiling.c_0 * param_cls.param.fm_desc.h_i * param_cls.param.fm_desc.w_i
                inst.load3dv1(data_l0a[l0a_idx:], param_cls.param.feature_map.flat_access(c1_offset),
                              param_cls.param.fm_desc.pad_list, param_cls.param.fm_desc.h_i,
                              param_cls.param.fm_desc.w_i, disable_c1, k_hw_idx % param_cls.param.k_w,
                              k_hw_idx // param_cls.param.k_w, wi_idx - param_cls.param.fm_desc.pad_list[0],
                              hi_idx - param_cls.param.fm_desc.pad_list[2], param_cls.param.fm_desc.stride_w,
                              param_cls.param.fm_desc.stride_h, param_cls.param.k_w,
                              param_cls.param.k_h, param_cls.param.dilation_w, param_cls.param.dilation_h,
                              param_cls.cin_actual, 1, param_cls.tiling.m_tile_block,
                              pad_value=param_cls.param.pad_value)
        # load by column
        elif param_cls.tiling.l0a_mode == 0:
            with inst.for_range(0, param_cls.tiling.m_tile_block) as k:
                ho_wo_pos = (param_cls.hw_i * param_cls.tiling.m_tile_block + k) * param_cls.tiling.block_size
                ho_idx = ho_wo_pos // param_cls.param.fm_desc.w_o
                wo_idx = ho_wo_pos % param_cls.param.fm_desc.w_o
                hi_idx = ho_idx * param_cls.param.fm_desc.stride_h
                wi_idx = wo_idx * param_cls.param.fm_desc.stride_w
                # we load the whole row in 1 load3d
                c1_idx = cin_pos // (param_cls.param.k_h * param_cls.param.k_w)
                k_hw_idx = cin_pos % (param_cls.param.k_h * param_cls.param.k_w)
                l0a_idx = k * param_cls.cin_actual * param_cls.tiling.block_size * param_cls.tiling.c_0
                disable_c1 = 0  # Cin_i * Cin_tiling + k
                c1_offset = c1_idx * param_cls.tiling.c_0 * param_cls.param.fm_desc.h_i * param_cls.param.fm_desc.w_i
                inst.load3dv1(data_l0a[l0a_idx:], param_cls.param.feature_map.flat_access(c1_offset),
                              param_cls.param.fm_desc.pad_list, param_cls.param.fm_desc.h_i,
                              param_cls.param.fm_desc.w_i, disable_c1, k_hw_idx % param_cls.param.k_w,
                              k_hw_idx // param_cls.param.k_w, wi_idx - param_cls.param.fm_desc.pad_list[0],
                              hi_idx - param_cls.param.fm_desc.pad_list[2], param_cls.param.fm_desc.stride_w,
                              param_cls.param.fm_desc.stride_h, param_cls.param.k_w, param_cls.param.k_h,
                              param_cls.param.dilation_w, param_cls.param.dilation_h, 1, 0, param_cls.cin_actual,
                              pad_value=param_cls.param.pad_value)
        return data_l0a

    @staticmethod
    def make_load_l0b_code(param_cls):
        """
        create code of load2d, load data from L1 to L0B
        Parameters
        ----------
        param_cls.cin_actual: k tile block num
        param_cls.cout_actual: n tile block num
        param_cls.cout_i: index of n tile block
        param_cls.cin_i: index of k tile block
        param_cls.tiling : data struct save tiling info
        param_cls.param: others params, input fm shape, kernel info, dilations
        Returns
        -------
        data_l0b: tensor of load2d result
        """
        inst = param_cls.param.tik_instance
        size = param_cls.tiling.n_tile_block * param_cls.tiling.k_tile_block * \
               param_cls.tiling.block_size * param_cls.tiling.c_0
        data_l0b = inst.Tensor(param_cls.param.weight.dtype, (size,),
                               name=param_cls.param.weight.tensor.buffer.name + "L0B", scope=scope_cb)

        if TikSocManager.is_610l_soc():
            inst.load2dv3(data_l0b, param_cls.param.weight.flat_access(0),
                          param_cls.cout_i * param_cls.tiling.n_tile_block,
                          param_cls.cin_i * param_cls.tiling.k_tile_block,
                          param_cls.tiling.n_block_num, param_cls.cout_actual,
                          param_cls.cout_actual, param_cls.cin_actual, 0)
            return data_l0b
        if param_cls.tiling.l0b_mode == 0:
            if param_cls.tiling.n_iter_num == 1:
                # load one column at once
                w_size = param_cls.tiling.block_size * param_cls.tiling.c_0
                w_idx = ((param_cls.cin_i * param_cls.tiling.k_tile_block) * param_cls.tiling.n_block_num + \
                         param_cls.cout_i * param_cls.tiling.n_tile_block) * w_size

                l0b_idx = 0
                if TikSocManager.is_v100_soc():
                    inst.load2dv1(data_l0b[l0b_idx:], param_cls.param.weight.flat_access(w_idx), 0,
                                  param_cls.cout_actual * param_cls.cin_actual, 1, 0)
                else:
                    inst.load2dv2(data_l0b[l0b_idx:], param_cls.param.weight.flat_access(w_idx), 0,
                                  param_cls.cout_actual * param_cls.cin_actual, 0, 1, 0)
            else:
                # load row by row
                with inst.for_range(0, param_cls.cin_actual) as sub_cin_idx:
                    w_size = param_cls.cout_i * param_cls.tiling.n_tile_block * param_cls.tiling.block_size * \
                             param_cls.tiling.c_0
                    w_idx = (param_cls.cin_i * param_cls.tiling.k_tile_block + sub_cin_idx) * \
                            param_cls.tiling.n_block_num * param_cls.tiling.block_size * param_cls.tiling.c_0 + w_size

                    l0b_idx = sub_cin_idx * param_cls.cout_actual * param_cls.tiling.block_size * param_cls.tiling.c_0
                    if TikSocManager.is_v100_soc():
                        inst.load2dv1(data_l0b[l0b_idx:], param_cls.param.weight.flat_access(w_idx), 0,
                                      param_cls.cout_actual, 1, 0)
                    else:
                        inst.load2dv2(data_l0b[l0b_idx:], param_cls.param.weight.flat_access(w_idx), 0,
                                      param_cls.cout_actual, 0, 1, 0)
        elif param_cls.tiling.l0b_mode == 1:
            # load one column at once
            w_idx = param_cls.cin_i * param_cls.tiling.k_tile_block * param_cls.tiling.n_block_num * \
                    param_cls.tiling.block_size * param_cls.tiling.c_0 + \
                    param_cls.cout_i * param_cls.tiling.n_tile_block * \
                    param_cls.tiling.block_size * param_cls.tiling.c_0
            if TikSocManager.is_v100_soc():
                inst.load2dv1(data_l0b, param_cls.param.weight.flat_access(w_idx), 0,
                              param_cls.cin_actual, param_cls.tiling.n_block_num, 0)
            else:
                inst.load2dv2(data_l0b, param_cls.param.weight.flat_access(w_idx), 0,
                              param_cls.cin_actual, 0, param_cls.tiling.n_block_num, 0)
        return data_l0b

    @staticmethod
    def get_bais_info(l1_bias, l1_offset, init_l1out):
        """
        get bias info: bias_tensor and init_l0c flag
        Parameters
        ----------
        :l1_bias: bias tensor on l1 scope
        :l1_offset: start offset of l1 tensor
        :init_l1out: init_l1out flag from api
        Returns
        -------
        tmp_bias: bias on l1, init_l0c: is_bias flag for mmad
        """
        tmp_bias = None
        if l1_bias is not None:
            # first move bias L1 to BT, data size of BT according the split N size
            tmp_bias = l1_bias[l1_offset:]
        # init_l1out is true: means init L0C with 0 or bias if given
        # is_bias for mmad is true: means adding L0C value to result (C=A*B+C)
        # l1_bias     init_l1out   l0c-value     mmad-is_bias
        #   None       True           0             False
        # not None     True          bias           True
        #   None       False         l0c            True
        # not None     False   ----> api check will make sure init_l1out is True when l1-bias is not None
        init_l0c = init_l1out ^ (l1_bias is None)
        return tmp_bias, init_l0c

    @staticmethod
    def make_mmad_code(param_cls):
        """
        create code of mmad
        Parameters
        ----------
        param_cls.data_l0a: load3d result tensor, data in L0A
        param_cls.data_l0b: load2d result tensor, data in L0B
        param_cls.cin_actual: k tile block num
        param_cls.hw_actual_size: m tile size
        param_cls.cout_actual: n tile block num
        param_cls.cin_i: index of k tile block
        param_cls.hw_i: index of m tile block
        param_cls.cout_i: index of n tile block
        param_cls.tiling : data struct save tiling info
        param_cls.param: others params, input fm shape, kernel info, dilations
        Returns
        -------
        None
        """
        l1_offset = param_cls.tiling.n_tile_block * param_cls.cout_i * COUT_B16
        l1_bias, init_l0c = CubeCodeMake.get_bais_info(param_cls.param.l1_bias, l1_offset, param_cls.param.init_l1out)

        # only care data in K dim
        b_size = param_cls.tiling.block_size * param_cls.tiling.block_size
        dst_flatten_idx = (param_cls.cout_i * param_cls.tiling.m_block_num * param_cls.tiling.n_tile_block +
                           param_cls.hw_i * param_cls.tiling.m_tile_block) * b_size
        inst = param_cls.param.tik_instance
        hw_actual_size = param_cls.hw_actual_size

        # for m_extension is 1, mmad is GEMV mode, GEMV mode must L0A shape M is 1
        # but current L0A shape M is not 1, set hw_actual_size to 2, mmad work in GEMM mode,
        # set to 2 won't inspect mmad result
        if hw_actual_size == 1:
            hw_actual_size = 2
        with inst.if_scope(tik_all(param_cls.cin_i == 0, param_cls.param.init_l1out)):
            inst.mmad(
                param_cls.param.dst[dst_flatten_idx:], param_cls.data_l0a, param_cls.data_l0b, hw_actual_size,
                param_cls.cin_actual * param_cls.tiling.c_0, param_cls.cout_actual * param_cls.tiling.block_size,
                init_l0c, bias_tensor=l1_bias)

        with inst.else_scope():
            inst.mmad(
                param_cls.param.dst[dst_flatten_idx:], param_cls.data_l0a, param_cls.data_l0b, hw_actual_size,
                param_cls.cin_actual * param_cls.tiling.c_0, param_cls.cout_actual * param_cls.tiling.block_size, 1)


class CubeTilingMake:
    """
    class for conv2d or matmul to creat tiling
    """

    def __init__(self):
        self.tik_instance = None
        self.block_size = 0
        self.c_0 = 0
        self.m_block_num = 0
        self.k_block_num = 0
        self.n_block_num = 0
        self.fm_dtype_size = 0
        self.filter_dtype_size = 0

    @staticmethod
    def _get_m_n_thread_nums(m_block_num, n_block_num):
        # if block num > 1, thread num can be 1,2, else can only be 1
        if m_block_num > 1:
            m_thread_nums = [2, 1]
        else:
            m_thread_nums = [1]

        if n_block_num > 1:
            n_thread_nums = [2, 1]
        else:
            n_thread_nums = [1]
        return m_thread_nums, n_thread_nums

    def get_loop_modes(self, m_block_num, n_block_num, k_block_num, modes):
        """
        according the block num and modes to create loop modes
        """
        m_thread_nums, n_thread_nums = self._get_m_n_thread_nums(m_block_num, n_block_num)
        if k_block_num >= 4:
            k_thread_nums = [2]
        elif k_block_num > 1:
            k_thread_nums = [2, 1]
        else:
            k_thread_nums = [1]

        loop_modes = []
        for num in itertools.product(m_thread_nums, n_thread_nums, k_thread_nums, modes):
            loop_modes.append(num)
        return loop_modes

    def check_tensor_size(self, buf_scope, high_tile_block, width_tile_block, thread_num):
        """
        according the high and width to calculate the needed tensor size
        """
        if buf_scope == "l0a":
            buffer_size = get_l0a_buf_size()
            dtype_size = self.fm_dtype_size
        else:
            buffer_size = get_l0b_buf_size()
            dtype_size = self.filter_dtype_size
        needed_size = high_tile_block * width_tile_block * self.block_size * self.c_0 * dtype_size * thread_num
        if needed_size > buffer_size:
            return False
        return True

    def check_tensor_size_by_mode(self, nums, m_tile_block, k_tile_block, n_tile_block):
        """
        according diff loop mode to check tensor size of diff tiling
        """
        m_thread_num, n_thread_num, k_thread_num, loop_mode = nums
        if loop_mode == "mn":
            if self.check_tensor_size(
                    "l0a", m_tile_block, k_tile_block,
                    max(k_thread_num, m_thread_num)) is False:
                return False
            if self.check_tensor_size(
                    "l0b", k_tile_block, n_tile_block,
                    max(k_thread_num, n_thread_num,
                        m_thread_num)) is False:
                return False
        else:
            if self.check_tensor_size(
                    "l0a", m_tile_block, k_tile_block,
                    max(k_thread_num, n_thread_num,
                        m_thread_num)) is False:
                return False
            if self.check_tensor_size(
                    "l0b", k_tile_block, n_tile_block,
                    max(k_thread_num, n_thread_num)) is False:
                return False
        return True


def get_iter_thread_num(tiling, mode, outer_thread_num=1):
    """
    according input iter num, thread num and tail to
    calculate new iter num and thread num

    Parameters
    ----------
    tiling : data struct save tiling info
    mode: cout, cin, hw, loop axis
    outer_thread_num: the thread num of outer loop, if outer_thread_num > 1, inner loop may be can enable double buffer

    Returns
    -------
    iter_num, thread_num
    """
    has_tail = False
    iter_num = 0
    thread_num = 0
    if mode == "cout":
        iter_num = tiling.n_iter_num
        # get the max thread num as current loop's thread num
        thread_num = max(tiling.n_thread_num, outer_thread_num)
        has_tail = tiling.n_has_tail
    elif mode == "cin":
        iter_num = tiling.k_iter_num
        thread_num = max(tiling.k_thread_num, outer_thread_num)
        has_tail = tiling.k_has_tail
    elif mode == 'hw':
        iter_num = tiling.m_iter_num
        thread_num = max(tiling.m_thread_num, outer_thread_num)
        has_tail = tiling.m_has_tail
    else:
        TikCheckUtil.raise_error("Input mode(%s) not support" % mode)

    if has_tail:
        iter_num -= 1
        if iter_num < thread_num:
            thread_num = 1
    elif iter_num < thread_num:
        thread_num = 1
    return iter_num, thread_num


class CubeModeTiling(CubeCodeMake):
    """
    class for conv2d and matmul for deal with mn, nm, kn, nk tiling mode
    """

    def __init__(self):
        super(CubeModeTiling, self).__init__()
        self.tik_instance = None
        self.load_l0a_cls, self.load_l0b_cls, self.mmad_cls = self.get_make_code()

    @staticmethod
    def check_params_bias(bias, dst, kernel_shape, init_1out, api_name):
        if bias is not None:
            TikCheckUtil.check_equality((TikSocManager.is_910b_soc() or TikSocManager.is_310b_610l_soc()) and
                                        TikSocManager.is_aicore_core(), True,
                                        "%s param bias, not support %s." % (api_name, get_soc_name()))
            TikCheckUtil.check_equality(init_1out, True, "if %s with bias, init_l1out should be True." % api_name)
            TikCheckUtil.check_type_match(bias, Tensor, "bias should be tensor, input type is: %s" % type(bias))
            TikCheckUtil.check_type_match(bias.shape, (list, tuple), "bias shape should be list or tuple")
            TikCheckUtil.check_equality(len(bias.shape), 1, "bias shape should be equal to [%s]" % (kernel_shape[-2]))
            if isinstance(bias.shape[0], int):
                TikCheckUtil.check_equality(bias.shape[0], kernel_shape[-2],
                                            "bias shape should be equal to [%s],"
                                            " input is: %s" % (kernel_shape[-2], bias.shape))
            TikCheckUtil.check_equality(bias.scope, scope_cbuf,
                                        "bias scope should be l1, input is: %s" % bias.scope)
            TikCheckUtil.check_equality(bias.dtype, dst.dtype,
                                        "bias dtype should be %s, equal to dst dtype,"
                                        " input type is: %s" % (dst.dtype, bias.dtype))

    @staticmethod
    def _check_num(tiling, has_tail, iter_num, thread_num):
        """
        check the iter num, whether is odd, if iter num is odd,
        """
        return (tiling.is_conv2d and has_tail or tiling.is_matmul) and iter_num > 2 and thread_num > 1 and iter_num & 1

    def do_mn_tiling(self, tiling, param):
        """
        create code of according the tiling and params for mode mn
        Parameters
        ----------
        tiling: the tiling info
        param: others needed parameters
        Returns
        -------
        None
        """

        def _mn_cout_code_gen(cin_actual, cin_i, hw_actual, hw_i, data_l0a):
            with inst.for_range(0, n_iter_num - 1, thread_num=n_thread_num) as cout_i:
                data_l0b = self.make_load_l0b_code(self.load_l0b_cls(
                    cin_actual, tiling.n_tile_block,
                    cout_i, cin_i, tiling, param))
                self.make_mmad_code(self.mmad_cls(
                    data_l0a, data_l0b, cin_actual, hw_actual,
                    tiling.n_tile_block, cin_i, hw_i, cout_i,
                    tiling, param))
            with inst.new_stmt_scope():
                data_l0b = self.make_load_l0b_code(self.load_l0b_cls(
                    cin_actual, tiling.n_tile_block,
                    n_iter_num - 1, cin_i, tiling, param))
                self.make_mmad_code(self.mmad_cls(
                    data_l0a, data_l0b, cin_actual, hw_actual,
                    tiling.n_tile_block, cin_i, hw_i, n_iter_num - 1,
                    tiling, param))

        def _mn_cout_loop(cin_actual, cin_i, hw_actual, hw_i, data_l0a):
            # check the iter num, whether is odd, if iter num is odd,
            # change iter num to even and then do double buffer
            # cannot delete, delete will loss performance
            # matmul is different with conv2d, it's needn't to check
            # whether has tail, matmul may have enough space,
            # so it can get another space for tail
            if self._check_num(tiling, tiling.n_has_tail, n_iter_num, n_thread_num):
                _mn_cout_code_gen(cin_actual, cin_i, hw_actual, hw_i, data_l0a)
            else:
                with inst.for_range(0, n_iter_num, thread_num=n_thread_num) as cout_i:
                    data_l0b = self.make_load_l0b_code(self.load_l0b_cls(
                        cin_actual, tiling.n_tile_block,
                        cout_i, cin_i, tiling, param))
                    self.make_mmad_code(self.mmad_cls(
                        data_l0a, data_l0b, cin_actual, hw_actual,
                        tiling.n_tile_block, cin_i, hw_i, cout_i,
                        tiling, param))
            if tiling.n_has_tail:
                data_l0b = self.make_load_l0b_code(self.load_l0b_cls(
                    cin_actual, tiling.n_tail_block,
                    n_iter_num, cin_i, tiling, param))
                self.make_mmad_code(self.mmad_cls(
                    data_l0a, data_l0b, cin_actual, hw_actual,
                    tiling.n_tail_block, cin_i, hw_i, n_iter_num,
                    tiling, param))

        def _mn_hw_code_gen(cin_actual, cin_i):
            with inst.for_range(0, m_iter_num - 1, thread_num=m_thread_num) as hw_i:
                data_l0a = self.make_load_l0a_code(self.load_l0a_cls(cin_actual, cin_i,
                                                                     tiling.m_tile_nums,
                                                                     hw_i, tiling, param))
                _mn_cout_loop(cin_actual, cin_i, tiling.m_tile_nums, hw_i, data_l0a)
            with inst.new_stmt_scope():
                data_l0a = self.make_load_l0a_code(self.load_l0a_cls(cin_actual, cin_i,
                                                                     tiling.m_tile_nums, m_iter_num - 1, tiling, param))
                _mn_cout_loop(cin_actual, cin_i, tiling.m_tile_nums, m_iter_num - 1, data_l0a)

        def _mn_hw_loop(cin_actual, cin_i):
            if self._check_num(tiling, tiling.m_has_tail, m_iter_num, m_thread_num):
                _mn_hw_code_gen(cin_actual, cin_i)
            else:
                with inst.for_range(0, m_iter_num, thread_num=m_thread_num) as hw_i:
                    data_l0a = self.make_load_l0a_code(self.load_l0a_cls(cin_actual, cin_i,
                                                                         tiling.m_tile_nums, hw_i, tiling, param))
                    _mn_cout_loop(cin_actual, cin_i, tiling.m_tile_nums, hw_i, data_l0a)
            if tiling.m_has_tail:
                data_l0a = self.make_load_l0a_code(self.load_l0a_cls(cin_actual, cin_i,
                                                                     tiling.m_tail_nums, m_iter_num, tiling, param))
                _mn_cout_loop(cin_actual, cin_i, tiling.m_tail_nums, m_iter_num, data_l0a)

        inst = self.tik_instance
        k_iter_num, k_thread_num = get_iter_thread_num(tiling, "cin")
        m_iter_num, m_thread_num = get_iter_thread_num(tiling, "hw", tiling.k_thread_num)
        n_iter_num, n_thread_num = get_iter_thread_num(
            tiling, "cout", max(tiling.k_thread_num, tiling.m_thread_num))

        if k_iter_num > 2 and k_thread_num > 1 and k_iter_num & 1:
            with inst.for_range(0, k_iter_num - 1, thread_num=k_thread_num) as k_i:
                _mn_hw_loop(tiling.k_tile_block, k_i)

            # here must with new_stmt_scope, else may be error by no space
            with inst.new_stmt_scope():
                _mn_hw_loop(tiling.k_tile_block, k_iter_num - 1)
        else:
            with inst.for_range(0, k_iter_num, thread_num=k_thread_num) as k_i:
                _mn_hw_loop(tiling.k_tile_block, k_i)
        if tiling.k_has_tail:
            _mn_hw_loop(tiling.k_tail_block, k_iter_num)

    def do_nm_tiling(self, tiling, param):
        """
        create code of according the tiling and params for mode nm
        Parameters
        ----------
        tiling: the tiling info
        param: others needed parameters
        Returns
        -------
        None
        """

        def _nm_hw_code_gen(cin_actual, cin_i, cout_actual, cout_i, data_l0b):
            with inst.for_range(0, m_iter_num - 1,
                                thread_num=m_thread_num) as hw_i:
                data_l0a = self.make_load_l0a_code(self.load_l0a_cls(cin_actual, cin_i,
                                                                     tiling.m_tile_nums, hw_i, tiling, param))
                self.make_mmad_code(self.mmad_cls(data_l0a, data_l0b, cin_actual,
                                                  tiling.m_tile_nums, cout_actual, cin_i, hw_i, cout_i, tiling, param))
            with inst.new_stmt_scope():
                data_l0a = self.make_load_l0a_code(self.load_l0a_cls(cin_actual, cin_i,
                                                                     tiling.m_tile_nums, m_iter_num - 1, tiling, param))
                self.make_mmad_code(self.mmad_cls(data_l0a, data_l0b, cin_actual, tiling.m_tile_nums,
                                                  cout_actual, cin_i, m_iter_num - 1, cout_i, tiling, param))

        def _nm_hw_loop(cin_actual, cin_i, cout_actual, cout_i, data_l0b):
            # matmul is different with conv2d, it's needn't to check
            # whether has tail, matmul may have enough space,
            # so it can get another space for tail
            if self._check_num(tiling, tiling.m_has_tail, m_iter_num, m_thread_num):
                _nm_hw_code_gen(cin_actual, cin_i, cout_actual, cout_i, data_l0b)
            else:
                with inst.for_range(0, m_iter_num, thread_num=m_thread_num) as hw_i:
                    data_l0a = self.make_load_l0a_code(self.load_l0a_cls(cin_actual, cin_i, tiling.m_tile_nums, hw_i,
                                                                         tiling, param))
                    self.make_mmad_code(self.mmad_cls(data_l0a, data_l0b, cin_actual, tiling.m_tile_nums,
                                                      cout_actual, cin_i, hw_i, cout_i, tiling, param))
            if tiling.m_has_tail:
                data_l0a = self.make_load_l0a_code(self.load_l0a_cls(cin_actual, cin_i, tiling.m_tail_nums, m_iter_num,
                                                                     tiling, param))
                self.make_mmad_code(self.mmad_cls(data_l0a, data_l0b, cin_actual, tiling.m_tail_nums,
                                                  cout_actual, cin_i, m_iter_num, cout_i, tiling, param))

        def _nm_cout_code_gen(cin_actual, cin_i):
            with inst.for_range(0, n_iter_num - 1, thread_num=n_thread_num) as cout_i:
                data_l0b = self.make_load_l0b_code(self.load_l0b_cls(cin_actual, tiling.n_tile_block, cout_i, cin_i,
                                                                     tiling, param))
                _nm_hw_loop(cin_actual, cin_i, tiling.n_tile_block,
                            cout_i, data_l0b)
            with inst.new_stmt_scope():
                data_l0b = self.make_load_l0b_code(self.load_l0b_cls(cin_actual, tiling.n_tile_block,
                                                                     n_iter_num - 1, cin_i, tiling, param))
                _nm_hw_loop(cin_actual, cin_i, tiling.n_tile_block, n_iter_num - 1, data_l0b)

        def _nm_cout_loop(cin_actual, cin_i):
            if self._check_num(tiling, tiling.n_has_tail, n_iter_num, n_thread_num):
                _nm_cout_code_gen(cin_actual, cin_i)
            else:
                with inst.for_range(0, n_iter_num, thread_num=n_thread_num) as cout_i:
                    data_l0b = self.make_load_l0b_code(self.load_l0b_cls(cin_actual, tiling.n_tile_block, cout_i, cin_i,
                                                                         tiling, param))
                    _nm_hw_loop(cin_actual, cin_i, tiling.n_tile_block, cout_i, data_l0b)
            if tiling.n_has_tail:
                data_l0b = self.make_load_l0b_code(self.load_l0b_cls(cin_actual, tiling.n_tail_block, n_iter_num, cin_i,
                                                                     tiling, param))
                _nm_hw_loop(cin_actual, cin_i, tiling.n_tail_block, n_iter_num, data_l0b)

        inst = self.tik_instance
        k_iter_num, k_thread_num = get_iter_thread_num(tiling, "cin")
        n_iter_num, n_thread_num = get_iter_thread_num(tiling, "cout", tiling.k_thread_num)
        m_iter_num, m_thread_num = get_iter_thread_num(tiling, "hw", max(tiling.k_thread_num, tiling.n_thread_num))
        if k_iter_num > 2 and k_thread_num > 1 and k_iter_num & 1:
            with inst.for_range(0, k_iter_num - 1, thread_num=k_thread_num) as k_i:
                _nm_cout_loop(tiling.k_tile_block, k_i)
            # here must with new_stmt_scope, else may be error by no space
            with inst.new_stmt_scope():
                _nm_cout_loop(tiling.k_tile_block, k_iter_num - 1)
        else:
            with inst.for_range(0, k_iter_num, thread_num=k_thread_num) as k_i:
                _nm_cout_loop(tiling.k_tile_block, k_i)
        if tiling.k_has_tail:
            _nm_cout_loop(tiling.k_tail_block, k_iter_num)

    def do_nk_tiling(self, tiling, param):
        """
        create code of according the tiling and params for mode nk
        Parameters
        ----------
        tiling: the tiling info
        param: others needed parameters
        Returns
        -------
        None
        """

        def _nk_cin_loop(cout_actual, cout_i):
            # nk mode, no split M, so if m has tail, m block nums is m_tail_nums
            m_block_nums = tiling.m_tile_nums
            if tiling.m_has_tail:
                m_block_nums = tiling.m_tail_nums

            # check the iter num, whether is odd, if iter num is odd,
            # change iter num to even and then do double buffer
            # cannot delete, delete will loss performance
            if tiling.k_has_tail and k_iter_num > 2 and k_thread_num > 1 and k_iter_num & 1:
                with inst.for_range(0, k_iter_num - 1, thread_num=k_thread_num) as cin_i:
                    # load l0a must be before load l0b,
                    # it can improve performace
                    data_l0a = self.make_load_l0a_code(self.load_l0a_cls(
                        tiling.k_tile_block, cin_i, m_block_nums, 0, tiling, param))
                    data_l0b = self.make_load_l0b_code(self.load_l0b_cls(
                        tiling.k_tile_block, cout_actual, cout_i, cin_i, tiling, param))
                    self.make_mmad_code(self.mmad_cls(data_l0a, data_l0b, tiling.k_tile_block, m_block_nums,
                                                      cout_actual, cin_i, 0, cout_i, tiling, param))
                # here must with new_stmt_scope, else may be error by no space
                with inst.new_stmt_scope():
                    data_l0a = self.make_load_l0a_code(self.load_l0a_cls(
                        tiling.k_tile_block, k_iter_num - 1, m_block_nums, 0, tiling, param))
                    data_l0b = self.make_load_l0b_code(self.load_l0b_cls(
                        tiling.k_tile_block, cout_actual, cout_i, k_iter_num - 1, tiling, param))
                    self.make_mmad_code(self.mmad_cls(data_l0a, data_l0b, tiling.k_tile_block, m_block_nums,
                                                      cout_actual, k_iter_num - 1, 0, cout_i, tiling, param))
            else:
                with inst.for_range(0, k_iter_num, thread_num=k_thread_num) as cin_i:
                    data_l0a = self.make_load_l0a_code(self.load_l0a_cls(
                        tiling.k_tile_block, cin_i, m_block_nums, 0, tiling, param))
                    data_l0b = self.make_load_l0b_code(self.load_l0b_cls(tiling.k_tile_block, cout_actual, cout_i,
                                                                         cin_i, tiling, param))
                    self.make_mmad_code(self.mmad_cls(data_l0a, data_l0b, tiling.k_tile_block, m_block_nums,
                                                      cout_actual, cin_i, 0, cout_i, tiling, param))
            if tiling.k_has_tail:
                data_l0a = self.make_load_l0a_code(self.load_l0a_cls(
                    tiling.k_tail_block, k_iter_num, m_block_nums, 0, tiling, param))
                data_l0b = self.make_load_l0b_code(self.load_l0b_cls(
                    tiling.k_tail_block, cout_actual, cout_i, k_iter_num, tiling, param))
                self.make_mmad_code(self.mmad_cls(data_l0a, data_l0b, tiling.k_tail_block, m_block_nums,
                                                  cout_actual, k_iter_num, 0, cout_i, tiling, param))

        inst = self.tik_instance
        n_iter_num, n_thread_num = get_iter_thread_num(tiling, "cout")
        k_iter_num, k_thread_num = get_iter_thread_num(tiling, "cin", tiling.n_thread_num)
        with inst.for_range(0, n_iter_num, thread_num=n_thread_num) as cout_i:
            _nk_cin_loop(tiling.n_tile_block, cout_i)
        if tiling.n_has_tail:
            _nk_cin_loop(tiling.n_tail_block, n_iter_num)

    def do_kn_tiling(self, tiling, param):
        """
        create code of according the tiling and params for mode kn
        Parameters
        ----------
        tiling: the tiling info
        param: others needed parameters
        Returns
        -------
        None
        """

        def _kn_cout_loop(cin_actual, cin_i):
            # nk mode, no split M, so if m has tail, m block nums is m_tail_nums
            m_block_nums = tiling.m_tile_nums
            if tiling.m_has_tail:
                m_block_nums = tiling.m_tail_nums

            data_l0a = self.make_load_l0a_code(self.load_l0a_cls(cin_actual, cin_i,
                                                                 m_block_nums, 0, tiling, param))
            if tiling.n_has_tail and n_iter_num > 2 and n_thread_num > 1 and n_iter_num & 1:
                with inst.for_range(0, n_iter_num - 1, thread_num=n_thread_num) as cout_i:
                    data_l0b = self.make_load_l0b_code(self.load_l0b_cls(
                        cin_actual, tiling.n_tile_block, cout_i, cin_i, tiling, param))
                    self.make_mmad_code(self.mmad_cls(data_l0a, data_l0b, cin_actual, m_block_nums,
                                                      tiling.n_tile_block, cin_i, 0, cout_i, tiling, param))
                with inst.new_stmt_scope():
                    data_l0b = self.make_load_l0b_code(self.load_l0b_cls(
                        cin_actual, tiling.n_tile_block, n_iter_num - 1, cin_i, tiling, param))
                    self.make_mmad_code(self.mmad_cls(
                        data_l0a, data_l0b, cin_actual, m_block_nums,
                        tiling.n_tile_block, cin_i, 0, n_iter_num - 1, tiling, param))
            else:
                with inst.for_range(0, n_iter_num, thread_num=n_thread_num) as cout_i:
                    data_l0b = self.make_load_l0b_code(self.load_l0b_cls(
                        cin_actual, tiling.n_tile_block, cout_i, cin_i, tiling, param))
                    self.make_mmad_code(self.mmad_cls(
                        data_l0a, data_l0b, cin_actual, m_block_nums,
                        tiling.n_tile_block, cin_i, 0, cout_i, tiling, param))
            if tiling.n_has_tail:
                data_l0b = self.make_load_l0b_code(self.load_l0b_cls(
                    cin_actual, tiling.n_tail_block, n_iter_num, cin_i, tiling, param))
                self.make_mmad_code(self.mmad_cls(
                    data_l0a, data_l0b, cin_actual, m_block_nums, tiling.n_tail_block,
                    cin_i, 0, n_iter_num, tiling, param))

        inst = self.tik_instance
        k_iter_num, k_thread_num = get_iter_thread_num(tiling, "cin")
        n_iter_num, n_thread_num = get_iter_thread_num(tiling, "cout", tiling.k_thread_num)
        if k_iter_num > 2 and k_thread_num > 1 and k_iter_num & 1:
            with inst.for_range(0, k_iter_num - 1, thread_num=k_thread_num) as cin_i:
                _kn_cout_loop(tiling.k_tile_block, cin_i)

            # here must with new_stmt_scope, else may be error by no space
            with inst.new_stmt_scope():
                _kn_cout_loop(tiling.k_tile_block, k_iter_num - 1)
        else:
            with inst.for_range(0, k_iter_num, thread_num=k_thread_num) as cin_i:
                _kn_cout_loop(tiling.k_tile_block, cin_i)
        if tiling.k_has_tail:
            _kn_cout_loop(tiling.k_tail_block, k_iter_num)
