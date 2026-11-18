#!/usr/bin/env python
# -*- coding:utf-8 -*-
"""
Copyright (C) 2019. Huawei Technologies Co., Ltd. All rights reserved.
FILE:     tik_compute_control.py
DESC:     provide params
CREATED:  2019-04-18 18:53:42
MODIFIED: 2020-12-7 19:17:00
"""
from tbe.tik.tik_lib.tik_soc_manager import TikSocManager
from tbe.tik.tik_lib.tik_check_util import TikCheckUtil
from tbe.tik.tik_lib.tik_mask_concat_ import mask_concat


_BIT_LEN_16 = 16
_MIN_MASK = 1
_MAX_MASK_N = 129
_MAX_MASK_64 = 65


class ControlOp:
    """
    Control Ops
    """
    def __init__(self, mask=None, repeat_times=1, stride_unit=0, mask_mode="normal"):
        self.mask = mask
        self.mask_mode = mask_mode
        self.repeat_times = repeat_times
        self.stride_unit = stride_unit
        self.mask_o = None
        self.eval_mask_value = None

    def eval_mask(self, context=None):
        """
        evaluate mask

        Parameters
        ----------
        context : the stack context

        Returns
        -------
        mask_value
        """
        if context and (self.mask is not None):
            if isinstance(self.mask, (list, tuple)):
                mask_value = [context.evaluate_expr(value) for value in self.mask]
            else:
                mask_value = context.evaluate_expr(self.mask)
        else:
            mask_value = self.mask

        self.eval_mask_value = mask_value

    def check_and_gen_mask_o(self, tik_instance, tensor_bit_len_max, mask_o=None):
        """
        check and gen mask_o

        Parameters
        ----------
        mask_o: mask
        tik_instance: Tik, tik_instance
        tensor_bit_len_max: max tensor bit len

        Returns
        -------
        mask_o

        """
        if mask_o is None:
            self.mask_o = mask_concat(tik_instance, self.mask, mask_mode=self.mask_mode,
                                      tensor_bit_len=tensor_bit_len_max)
        else:
            self.mask_o = mask_o
        return self.mask_o

    def check_mask_mode(self):
        """
        check mask_mode

        Returns
        -----
        None
        """
        TikCheckUtil.check_type_match(
            self.mask_mode, str, "mask_mode should be str, input type: %s" % type(self.mask_mode))
        # v100 not support counter_mode
        if TikSocManager.is_v100_soc():
            TikCheckUtil.check_var_in_list(
                self.mask_mode, ["normal", ], "mask_mode should be 'normal', input mask_mode: %s" % self.mask_mode)
        else:
            TikCheckUtil.check_var_in_list(
                self.mask_mode, ["normal", "counter"], "mask_mode should be 'normal' or 'counter', "
                                                       "input mask_mode: %s" % self.mask_mode)
