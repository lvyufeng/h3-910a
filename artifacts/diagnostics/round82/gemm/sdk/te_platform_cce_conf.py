#!/usr/bin/env python
# coding: utf-8
# Copyright 2019-2020 Huawei Technologies Co., Ltd
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================
"""
parser the config params
"""
from __future__ import absolute_import as _abs

import os
import stat

from tbe.tvm._ffi.registry import _init_api
from tbe.tvm import get_global_func
from tbe.tvm._ffi import register_func
from tvm.error_mgr.log import warn
from tvm.error_mgr import raise_tbe_python_err, TBE_DEFAULT_PYTHON_ERROR_CODE


# product version
# This is used for DSL/AutoSchedule ONLY!
# For other components, use te.platform.get_soc_spec("SHORT_SOC_VERSION")!
VERSION_CLOUD = "1980"
VERSION_MINI = "1910"
VERSION_MINI_L = "Ascend310B"
VERSION_MINI_T = "Ascend031"
VERSION_MINI_LITE = "Ascend610Lite"
VERSION_MINI_NGM510 = "MC62CM12AA"
AS31XM1X = "AS31XM1X"
ASCEND_310B1 = "Ascend310B1"
VERSION_MINI_NG1 = "ng1dc"
VERSION_MINI_NG1M = "ng1mdc"
VERSION_MINI_NG1PG2 = "ng1pg2"
VERSION_SHISI = "smallhisi"
VERSION_KIRINX90 = "KirinX90"
VERSION_KIRIN9030 = "Kirin9030"
AIC = "AiCore"
AIV = "VectorCore"
SOC_VERSION_MAP = {
    "Ascend310": {
        AIC: {"AICoreNum": 2, "ProductVersion": VERSION_MINI},
    },
    "Ascend910": {
        AIC: {"AICoreNum": 32, "ProductVersion": VERSION_CLOUD},
    },
    "Ascend910Pro": {
        AIC: {"AICoreNum": 32, "ProductVersion": VERSION_CLOUD},
    },
    "Ascend910Lite": {
        AIC: {"AICoreNum": 30, "ProductVersion": VERSION_CLOUD},
    },
    "Ascend910A": {
        AIC: {"AICoreNum": 32, "ProductVersion": VERSION_CLOUD},
    },
    "Ascend910ProA": {
        AIC: {"AICoreNum": 32, "ProductVersion": VERSION_CLOUD},
    },
    "Ascend910PremiumA": {
        AIC: {"AICoreNum": 32, "ProductVersion": VERSION_CLOUD},
    },
    "Ascend910B": {
        AIC: {"AICoreNum": 30, "ProductVersion": VERSION_CLOUD},
    },
    "Ascend910B1": {
        AIC: {"ProductVersion": VERSION_CLOUD},
        AIV: {"ProductVersion": VERSION_CLOUD},
    },
    "MC62CM12AA": {
        AIC: {"ProductVersion": VERSION_MINI_NGM510},
        AIV: {"ProductVersion": VERSION_MINI_NGM510},   
    },
    "MC62DM22AA": {
        AIC: {"ProductVersion": VERSION_MINI_NGM510},
        AIV: {"ProductVersion": VERSION_MINI_NGM510},   
    },
    "MC62CM13AA": {
        AIC: {"ProductVersion": VERSION_MINI_NGM510},
        AIV: {"ProductVersion": VERSION_MINI_NGM510},   
    },
    "MC62DM23AA": {
        AIC: {"ProductVersion": VERSION_MINI_NGM510},
        AIV: {"ProductVersion": VERSION_MINI_NGM510},   
    },
    "MC62CM13AB": {
        AIC: {"ProductVersion": VERSION_MINI_NGM510},
        AIV: {"ProductVersion": VERSION_MINI_NGM510},   
    },
    "MC62DM23AB": {
        AIC: {"ProductVersion": VERSION_MINI_NGM510},
        AIV: {"ProductVersion": VERSION_MINI_NGM510},   
    },
    "MC62CM12AC": {
        AIC: {"ProductVersion": VERSION_MINI_NGM510},
        AIV: {"ProductVersion": VERSION_MINI_NGM510},   
    },
    "MC62DM22AC": {
        AIC: {"ProductVersion": VERSION_MINI_NGM510},
        AIV: {"ProductVersion": VERSION_MINI_NGM510},   
    },
    "MC62CM12AD": {
        AIC: {"ProductVersion": VERSION_MINI_NGM510},
        AIV: {"ProductVersion": VERSION_MINI_NGM510},   
    },
    "MC62DM22AD": {
        AIC: {"ProductVersion": VERSION_MINI_NGM510},
        AIV: {"ProductVersion": VERSION_MINI_NGM510},   
    },
    "MC62CM12AE": {
        AIC: {"ProductVersion": VERSION_MINI_NGM510},
        AIV: {"ProductVersion": VERSION_MINI_NGM510},   
    },
    "MC62DM22AE": {
        AIC: {"ProductVersion": VERSION_MINI_NGM510},
        AIV: {"ProductVersion": VERSION_MINI_NGM510},   
    },
    "MC62CM12AF": {
        AIC: {"ProductVersion": VERSION_MINI_NGM510},
        AIV: {"ProductVersion": VERSION_MINI_NGM510},   
    },
    "MC62DM22AF": {
        AIC: {"ProductVersion": VERSION_MINI_NGM510},
        AIV: {"ProductVersion": VERSION_MINI_NGM510},   
    },
    "MC62CM12AB": {
        AIC: {"ProductVersion": VERSION_MINI_NGM510},
        AIV: {"ProductVersion": VERSION_MINI_NGM510},   
    },
    "MC62DM22AB": {
        AIC: {"ProductVersion": VERSION_MINI_NGM510},
        AIV: {"ProductVersion": VERSION_MINI_NGM510},   
    },
    "Ascend910B2": {
        AIC: {"ProductVersion": VERSION_CLOUD},
        AIV: {"ProductVersion": VERSION_CLOUD},
    },
    "Ascend910B2C": {
        AIC: {"ProductVersion": VERSION_CLOUD},
        AIV: {"ProductVersion": VERSION_CLOUD},
    },
    "Ascend910B3": {
        AIC: {"ProductVersion": VERSION_CLOUD},
        AIV: {"ProductVersion": VERSION_CLOUD},
    },
    "Ascend910B4": {
        AIC: {"ProductVersion": VERSION_CLOUD},
        AIV: {"ProductVersion": VERSION_CLOUD},
    },
    "Ascend910B4-1": {
        AIC: {"ProductVersion": VERSION_CLOUD},
        AIV: {"ProductVersion": VERSION_CLOUD},
    },
    "Ascend910_9391": {
        AIC: {"ProductVersion": VERSION_CLOUD},
        AIV: {"ProductVersion": VERSION_CLOUD},
    },
    "Ascend910_9392": {
        AIC: {"ProductVersion": VERSION_CLOUD},
        AIV: {"ProductVersion": VERSION_CLOUD},
    },
    "Ascend910_9381": {
        AIC: {"ProductVersion": VERSION_CLOUD},
        AIV: {"ProductVersion": VERSION_CLOUD},
    },
    "Ascend910_9382": {
        AIC: {"ProductVersion": VERSION_CLOUD},
        AIV: {"ProductVersion": VERSION_CLOUD},
    },
    "Ascend910_9372": {
        AIC: {"ProductVersion": VERSION_CLOUD},
        AIV: {"ProductVersion": VERSION_CLOUD},
    },
    "Ascend910_9362": {
        AIC: {"ProductVersion": VERSION_CLOUD},
        AIV: {"ProductVersion": VERSION_CLOUD},
    },
    "Ascend910_9363": {
        AIC: {"ProductVersion": VERSION_CLOUD},
        AIV: {"ProductVersion": VERSION_CLOUD},
    },
    "Ascend910ProB": {
        AIC: {"AICoreNum": 30, "ProductVersion": VERSION_CLOUD},
    },
    "Ascend310P1": {
        AIC: {"ProductVersion": VERSION_MINI_NG1},
        AIV: {"ProductVersion": VERSION_MINI_NG1},
    },
    "Ascend310P2": {
        AIC: {"ProductVersion": VERSION_MINI_NG1},
        AIV: {"ProductVersion": VERSION_MINI_NG1},
    },
    "Ascend310P3": {
        AIC: {"ProductVersion": VERSION_MINI_NG1},
        AIV: {"ProductVersion": VERSION_MINI_NG1},
    },
    "Ascend310P5": {
        AIC: {"ProductVersion": VERSION_MINI_NG1},
        AIV: {"ProductVersion": VERSION_MINI_NG1},
    },
    "Ascend310P7": {
        AIC: {"ProductVersion": VERSION_MINI_NG1},
        AIV: {"ProductVersion": VERSION_MINI_NG1},
    },
    "Ascend310P4": {
        AIC: {"ProductVersion": VERSION_MINI_NG1},
        AIV: {"ProductVersion": VERSION_MINI_NG1},
    },
    "Ascend610": {
        AIC: {"AICoreNum": 10, "ProductVersion": VERSION_MINI_NG1M},
        AIV: {"AICoreNum": 8, "ProductVersion": VERSION_MINI_NG1M},
    },
    "BS9SX1A": {
        AIC: {"AICoreNum": 10, "ProductVersion": VERSION_MINI_NG1PG2},
        AIV: {"AICoreNum": 8, "ProductVersion": VERSION_MINI_NG1PG2 + AIV},
    },
    "BS9SX1AA": {
        AIC: {"ProductVersion": VERSION_MINI_NG1PG2},
        AIV: {"ProductVersion": VERSION_MINI_NG1PG2 + AIV},
    },
    "BS9SX1AB": {
        AIC: {"ProductVersion": VERSION_MINI_NG1PG2},
        AIV: {"ProductVersion": VERSION_MINI_NG1PG2 + AIV},
    },
    "BS9SX1AC": {
        AIC: {"ProductVersion": VERSION_MINI_NG1PG2},
        AIV: {"ProductVersion": VERSION_MINI_NG1PG2 + AIV},
    },
    "Ascend610B": {
        AIC: {"ProductVersion": VERSION_MINI_NG1PG2},
        AIV: {"ProductVersion": VERSION_MINI_NG1PG2 + AIV},
    },
    "Ascend610B1": {
        AIC: {"ProductVersion": VERSION_MINI_NG1PG2},
        AIV: {"ProductVersion": VERSION_MINI_NG1PG2 + AIV},
    },
    "Ascend610B2": {
        AIC: {"ProductVersion": VERSION_MINI_NG1PG2},
        AIV: {"ProductVersion": VERSION_MINI_NG1PG2 + AIV},
    },
    "Hi3796CV300ES": {
        AIC: {"AICoreNum": 1, "ProductVersion": VERSION_SHISI},
    },
    "Hi3796CV300CS": {
        AIC: {"AICoreNum": 1, "ProductVersion": VERSION_SHISI},
    },
    "SD3403": {
        AIC: {"AICoreNum": 1, "ProductVersion": VERSION_SHISI},
    },
    "Hi3519AV200": {
        AIC: {"AICoreNum": 1, "ProductVersion": None},
    },
    "Ascend610Lite": {
        AIC: {"ProductVersion": VERSION_MINI_LITE},
        AIV: {"ProductVersion": VERSION_MINI_LITE},
    },
    "BS9SX2A": {
        AIC: {"ProductVersion": VERSION_MINI_LITE},
    },
    "BS9SX2AA": {
        AIC: {"AICoreNum": 4, "ProductVersion": VERSION_MINI_LITE},
    },
    "BS9SX2AB": {
        AIC: {"AICoreNum": 3, "ProductVersion": VERSION_MINI_LITE},
    },
    "MC61AM21A": {
        AIC: {"ProductVersion": VERSION_MINI_LITE},
    },
    "MC61AM21AA": {
        AIC: {"AICoreNum": 4, "ProductVersion": VERSION_MINI_LITE},
    },
    "MC61AM21AB": {
        AIC: {"AICoreNum": 3, "ProductVersion": VERSION_MINI_LITE},
    },
    ASCEND_310B1: {
        AIC: {"ProductVersion": VERSION_MINI_L},
        AIV: {"ProductVersion": VERSION_MINI_L},
    },
    "Ascend310B2": {
        AIC: {"ProductVersion": VERSION_MINI_L},
        AIV: {"ProductVersion": VERSION_MINI_L},
    },
    "Ascend310B3": {
        AIC: {"ProductVersion": VERSION_MINI_L},
        AIV: {"ProductVersion": VERSION_MINI_L},
    },
    "Ascend310B4": {
        AIC: {"ProductVersion": VERSION_MINI_L},
        AIV: {"ProductVersion": VERSION_MINI_L},
    },
    AS31XM1X: {
        AIC: {"ProductVersion": VERSION_MINI_L},
        AIV: {"ProductVersion": VERSION_MINI_L},
    },
    "Ascend031": {
        AIC: {"AICoreNum": 1, "ProductVersion": VERSION_MINI_T},
        AIV: {"AICoreNum": 1, "ProductVersion": VERSION_MINI_T},
    },
    "Ascend035": {
        AIC: {"ProductVersion": VERSION_CLOUD},
        AIV: {"ProductVersion": VERSION_CLOUD},
    },
    "Ascend035A": {
        AIC: {"ProductVersion": VERSION_CLOUD},
        AIV: {"ProductVersion": VERSION_CLOUD},
    },
    "Ascend035B": {
        AIC: {"ProductVersion": VERSION_CLOUD},
        AIV: {"ProductVersion": VERSION_CLOUD},
    },
    "Ascend950PR_9599" : {
        AIC: {"ProductVersion": VERSION_CLOUD},
        AIV: {"ProductVersion": VERSION_CLOUD},
    },
    "Ascend950PR_958a" : {
        AIC: {"ProductVersion": VERSION_CLOUD},
        AIV: {"ProductVersion": VERSION_CLOUD},
    },
    "Ascend950PR_9589" : {
        AIC: {"ProductVersion": VERSION_CLOUD},
        AIV: {"ProductVersion": VERSION_CLOUD},
    },
    "Ascend950PR_958b" : {
        AIC: {"ProductVersion": VERSION_CLOUD},
        AIV: {"ProductVersion": VERSION_CLOUD},
    },
    "Ascend950PR_9579" : {
        AIC: {"ProductVersion": VERSION_CLOUD},
        AIV: {"ProductVersion": VERSION_CLOUD},
    },
    "Ascend950PR_957b" : {
        AIC: {"ProductVersion": VERSION_CLOUD},
        AIV: {"ProductVersion": VERSION_CLOUD},
    },
    "Ascend950PR_957bx" : {
        AIC: {"ProductVersion": VERSION_CLOUD},
        AIV: {"ProductVersion": VERSION_CLOUD},
    },
    "Ascend950PR_957c" : {
        AIC: {"ProductVersion": VERSION_CLOUD},
        AIV: {"ProductVersion": VERSION_CLOUD},
    },
    "Ascend950PR_957d" : {
        AIC: {"ProductVersion": VERSION_CLOUD},
        AIV: {"ProductVersion": VERSION_CLOUD},
    },
    "Ascend950PR_950z" : {
        AIC: {"ProductVersion": VERSION_CLOUD},
        AIV: {"ProductVersion": VERSION_CLOUD},
    },
    "Ascend950DT_950x" : {
        AIC: {"ProductVersion": VERSION_CLOUD},
        AIV: {"ProductVersion": VERSION_CLOUD},
    },
    "Ascend950DT_950y" : {
        AIC: {"ProductVersion": VERSION_CLOUD},
        AIV: {"ProductVersion": VERSION_CLOUD},
    },
    "Ascend950DT_95A1" : {
        AIC: {"ProductVersion": VERSION_CLOUD},
        AIV: {"ProductVersion": VERSION_CLOUD},
    },
    "Ascend950DT_95A2" : {
        AIC: {"ProductVersion": VERSION_CLOUD},
        AIV: {"ProductVersion": VERSION_CLOUD},
    },
    "Ascend950DT_9591" : {
        AIC: {"ProductVersion": VERSION_CLOUD},
        AIV: {"ProductVersion": VERSION_CLOUD},
    },
    "Ascend950DT_9592" : {
        AIC: {"ProductVersion": VERSION_CLOUD},
        AIV: {"ProductVersion": VERSION_CLOUD},
    },
    "Ascend950DT_9595" : {
        AIC: {"ProductVersion": VERSION_CLOUD},
        AIV: {"ProductVersion": VERSION_CLOUD},
    },
    "Ascend910_9690": {
        AIC: {"ProductVersion": VERSION_CLOUD},
        AIV: {"ProductVersion": VERSION_CLOUD},
    },
    "Ascend910_9691": {
        AIC: {"ProductVersion": VERSION_CLOUD},
        AIV: {"ProductVersion": VERSION_CLOUD},
    },
    "Ascend910_9692": {
        AIC: {"ProductVersion": VERSION_CLOUD},
        AIV: {"ProductVersion": VERSION_CLOUD},
    },
    "Ascend910_9693": {
        AIC: {"ProductVersion": VERSION_CLOUD},
        AIV: {"ProductVersion": VERSION_CLOUD},
    },
    "Ascend910_9694": {
        AIC: {"ProductVersion": VERSION_CLOUD},
        AIV: {"ProductVersion": VERSION_CLOUD},
    },
    "Ascend910_9695": {
        AIC: {"ProductVersion": VERSION_CLOUD},
        AIV: {"ProductVersion": VERSION_CLOUD},
    },
    "Ascend910_9698": {
        AIC: {"ProductVersion": VERSION_CLOUD},
        AIV: {"ProductVersion": VERSION_CLOUD},
    },
    "Ascend910_9699": {
        AIC: {"ProductVersion": VERSION_CLOUD},
        AIV: {"ProductVersion": VERSION_CLOUD},
    },
    "Ascend910_9685": {
        AIC: {"ProductVersion": VERSION_CLOUD},
        AIV: {"ProductVersion": VERSION_CLOUD},
    },
    "Ascend910_9689": {
        AIC: {"ProductVersion": VERSION_CLOUD},
        AIV: {"ProductVersion": VERSION_CLOUD},
    },
    "Ascend960PR_969A1": {
        AIC: {"ProductVersion": VERSION_CLOUD},
        AIV: {"ProductVersion": VERSION_CLOUD},
    },
    "Ascend960PR_964A1": {
        AIC: {"ProductVersion": VERSION_CLOUD},
        AIV: {"ProductVersion": VERSION_CLOUD},
    },
    "Ascend960_9689": {
        AIC: {"ProductVersion": VERSION_CLOUD},
        AIV: {"ProductVersion": VERSION_CLOUD},
    },
    "Ascend350_354f": {
        AIC: {"ProductVersion": VERSION_CLOUD},
        AIV: {"ProductVersion": VERSION_CLOUD},
    },
    "Ascend350_355e": {
        AIC: {"ProductVersion": VERSION_CLOUD},
        AIV: {"ProductVersion": VERSION_CLOUD},
    },
    "Ascend950DT_9596" : {
        AIC: {"ProductVersion": VERSION_CLOUD},
        AIV: {"ProductVersion": VERSION_CLOUD},
    },
    "Ascend950DT_9581" : {
        AIC: {"ProductVersion": VERSION_CLOUD},
        AIV: {"ProductVersion": VERSION_CLOUD},
    },
    "Ascend950DT_9581x" : {
        AIC: {"ProductVersion": VERSION_CLOUD},
        AIV: {"ProductVersion": VERSION_CLOUD},
    },
    "Ascend950DT_9582" : {
        AIC: {"ProductVersion": VERSION_CLOUD},
        AIV: {"ProductVersion": VERSION_CLOUD},
    },
    "Ascend950DT_9582x" : {
        AIC: {"ProductVersion": VERSION_CLOUD},
        AIV: {"ProductVersion": VERSION_CLOUD},
    },
    "Ascend950DT_9583" : {
        AIC: {"ProductVersion": VERSION_CLOUD},
        AIV: {"ProductVersion": VERSION_CLOUD},
    },
    "Ascend950DT_9584" : {
        AIC: {"ProductVersion": VERSION_CLOUD},
        AIV: {"ProductVersion": VERSION_CLOUD},
    },
    "Ascend950DT_9585" : {
        AIC: {"ProductVersion": VERSION_CLOUD},
        AIV: {"ProductVersion": VERSION_CLOUD},
    },
    "Ascend950DT_9586" : {
        AIC: {"ProductVersion": VERSION_CLOUD},
        AIV: {"ProductVersion": VERSION_CLOUD},
    },
    "Ascend950DT_9587" : {
        AIC: {"ProductVersion": VERSION_CLOUD},
        AIV: {"ProductVersion": VERSION_CLOUD},
    },
    "Ascend950DT_9588" : {
        AIC: {"ProductVersion": VERSION_CLOUD},
        AIV: {"ProductVersion": VERSION_CLOUD},
    },
    "Ascend950DT_9571" : {
        AIC: {"ProductVersion": VERSION_CLOUD},
        AIV: {"ProductVersion": VERSION_CLOUD},
    },
    "Ascend950DT_9572" : {
        AIC: {"ProductVersion": VERSION_CLOUD},
        AIV: {"ProductVersion": VERSION_CLOUD},
    },
    "Ascend950DT_9573" : {
        AIC: {"ProductVersion": VERSION_CLOUD},
        AIV: {"ProductVersion": VERSION_CLOUD},
    },
    "Ascend950DT_9574" : {
        AIC: {"ProductVersion": VERSION_CLOUD},
        AIV: {"ProductVersion": VERSION_CLOUD},
    },
    "Ascend950DT_9575" : {
        AIC: {"ProductVersion": VERSION_CLOUD},
        AIV: {"ProductVersion": VERSION_CLOUD},
    },
    "Ascend950DT_9576" : {
        AIC: {"ProductVersion": VERSION_CLOUD},
        AIV: {"ProductVersion": VERSION_CLOUD},
    },
    "Ascend950DT_9577" : {
        AIC: {"ProductVersion": VERSION_CLOUD},
        AIV: {"ProductVersion": VERSION_CLOUD},
    },
    "Ascend950DT_9578" : {
        AIC: {"ProductVersion": VERSION_CLOUD},
        AIV: {"ProductVersion": VERSION_CLOUD},
    },
    "KirinX90": {
    AIC: {"AICoreNum": 1, "ProductVersion": VERSION_KIRINX90},
    AIV: {"AICoreNum": 1, "ProductVersion": VERSION_KIRINX90},
    },
    "Kirin9030": {
    AIC: {"AICoreNum": 1, "ProductVersion": VERSION_KIRIN9030},
    AIV: {"AICoreNum": 1, "ProductVersion": VERSION_KIRIN9030},
    },
}


# Example: pylint: disable=invalid-name
def getValue(key):
    """
    call global func to get product value

    Parameters
        ----------
        key : str
            key
    """
    if "Buffer" in key:
        func = get_global_func("cce.product_conf_buffer")
        value = func(key)
        if value == -1:
            raise_tbe_python_err(TBE_DEFAULT_PYTHON_ERROR_CODE, "Unsupported buffer name: %s" %
                                 key.split("_Buffer")[0])

        return value

    if "Compiler" in key:
        func = get_global_func("cce.product_conf_compiler")
        value = func(key)
        if value == "":
            raise_tbe_python_err(TBE_DEFAULT_PYTHON_ERROR_CODE, "Unsupported compiler param: %s" %
                                 key.split("Compiler_")[1])
        return value

    if "Intrinsic" in key:
        func = get_global_func("cce.product_conf_intrinsic")
        value = func(key)
        if value == "":
            raise_tbe_python_err(TBE_DEFAULT_PYTHON_ERROR_CODE, "Unsupported intrinsic: %s" %
                                 key.split("Intrinsic_")[1])
        return value

    if "Sid" in key:
        func = get_global_func("cce.product_conf_sid")
        value = func(key)
        if value == "":
            raise_tbe_python_err(TBE_DEFAULT_PYTHON_ERROR_CODE, "Unsupported sid param: %s" %
                                 key.split("Sid_")[1])
        return value

    if "Device" in key:
        func = get_global_func("cce.product_conf_device")
        value = func(key)
        if value == -1:
            raise_tbe_python_err(TBE_DEFAULT_PYTHON_ERROR_CODE, "Unsupported device param: %s" %
                                 key.split("Device_")[1])
        return value

    return None


# Example: pylint: disable=useless-object-inheritance, bad-classmethod-argument, invalid-name
class CceProductParams(object):
    """
    define Cce Product Params class
    """
    cce_product = None
    build_options = {}

    def __init__(self):
        pass

    # singletom pattern
    def __new__(cls, *args, **kwargs):
        _ = args
        _ = kwargs
        if not hasattr(cls, "_instance"):
            cls._instance = object.__new__(cls)
        return cls._instance

    def getParams(self, key):
        """
        get Cce Product Params info
        """
        if self.cce_product is None:
            raise_tbe_python_err(TBE_DEFAULT_PYTHON_ERROR_CODE, "not set product info")

        value = getValue(key)

        # if product supports os
        if key == "Compiler_aicpu_support_os":
            # string to bool
            value = (value == "true")

        return value

    # This is used for DSL/AutoSchedule ONLY!
    # For other components, use te.platform.get_soc_spec("SHORT_SOC_VERSION")!
    def get_product_version(self):
        """
        get product version
        ----------

        Returns
        -------
        cloud: cloud product
        mini: mini product
        """
        return self.cce_product

    # This is used for DSL/AutoSchedule ONLY!
    # For other components, use te.platform.get_soc_spec("SHORT_SOC_VERSION")!
    def is_mini_version(self):
        """
        check if mini version
        -------

        Returns
        -------
        True: mini version
        False: Other version
        """
        if self.cce_product == VERSION_MINI:
            return True
        return False

    # This is used for DSL/AutoSchedule ONLY!
    # For other components, use te.platform.get_soc_spec("SHORT_SOC_VERSION")!
    def is_cloud_version(self):
        """
        check if cloud-ASCEND_910 version
        ----------

        Returns
        -------
        True: cloud version
        False: Other version
        """
        if self.cce_product == VERSION_CLOUD:
            return True
        return False

    # This is used for DSL/AutoSchedule ONLY!
    # For other components, use te.platform.get_soc_spec("SHORT_SOC_VERSION")!
    def is_lhisi_version(self):
        """
        check if 3796ES version
        -------

        Returns
        -------
        True: 3796ES version
        False: Other version
        """
        if self.cce_product == VERSION_SHISI:
            return True
        return False

    # This is used for DSL/AutoSchedule ONLY!
    # For other components, use te.platform.get_soc_spec("SHORT_SOC_VERSION")!
    def is_ng1_version(self):
        """
        check if ng1/ng1m version
        ----------

        Returns
        -------
        True: ng1 version
        False: Other version
        """
        if self.cce_product in (VERSION_MINI_NG1, VERSION_MINI_NG1M, VERSION_MINI_NG1PG2):
            return True
        return False

    def set_build_option(self, key, value):
        """set build option"""
        self.build_options[key] = value

    def get_build_option(self, key):
        """get build option"""
        if key not in self.build_options:
            return None
        return self.build_options.get(key)

    cce_product = None
    build_options = {}


def _check_soc_version(soc_version, core_type):
    # check Soc_Vesion
    if not isinstance(soc_version, str):
        raise_tbe_python_err(TBE_DEFAULT_PYTHON_ERROR_CODE, "Soc_Vesion type should be 'str', it is [%s]"
                             % type(soc_version))
    if soc_version not in SOC_VERSION_MAP:
        raise_tbe_python_err(TBE_DEFAULT_PYTHON_ERROR_CODE, "Unsupported Soc_Vesion: %s" % soc_version)

    # check Core_Type
    if not isinstance(core_type, str):
        raise_tbe_python_err(TBE_DEFAULT_PYTHON_ERROR_CODE, "Core_Type type should be 'str', it is [%s]"
                             % type(core_type))
    if core_type not in SOC_VERSION_MAP.get(soc_version):
        raise_tbe_python_err(TBE_DEFAULT_PYTHON_ERROR_CODE, "%s Unsupported Core_Type: %s"
                             % (soc_version, core_type))


def _check_and_get_aicore_num(aicore_num):
    # check AICore_Num
    if aicore_num in [None, "0", 0, ""]:
        aicore_num = ""
    elif isinstance(aicore_num, int):
        if not 0 < aicore_num:
            raise_tbe_python_err(TBE_DEFAULT_PYTHON_ERROR_CODE, "Unsupported AICore_Num: %s" % aicore_num)
        aicore_num = str(aicore_num)
    elif isinstance(aicore_num, str):
        try:
            check_num = int(aicore_num)
        except Exception:
            raise_tbe_python_err(TBE_DEFAULT_PYTHON_ERROR_CODE, "Unsupported AICore_Num: %s" % aicore_num)
        finally:
            if not 0 < check_num:
                raise_tbe_python_err(TBE_DEFAULT_PYTHON_ERROR_CODE, "Unsupported AICore_Num: %s" % aicore_num)
    else:
        raise_tbe_python_err(TBE_DEFAULT_PYTHON_ERROR_CODE, "Unsupported AICore_Num: %s" % aicore_num)

    return aicore_num


def _check_and_get_l1_fusion(l1_fusion):
    # check l1_fusion
    if l1_fusion is None:
        l1_fusion = ""
    elif l1_fusion is True:
        l1_fusion = "true"
    elif l1_fusion is False:
        l1_fusion = "false"
    elif l1_fusion in ("True", "False", "TRUE", "FALSE", "true", "false", ""):
        l1_fusion = l1_fusion.lower()
    else:
        raise_tbe_python_err(TBE_DEFAULT_PYTHON_ERROR_CODE, "Unsupported l1_fusion: %s" % l1_fusion)

    return l1_fusion


def set_kernel_meta_parent_dir(kwargs):
    warn("'from te.platform.cce_conf import set_kernel_meta_parent_dir' is expired, "
         "please replace it with "
         "'from tbe.common.buildcfg import build_config/GlobalInfoContainer'.")
    if isinstance(kwargs, dict):
        op_debug_dir_str = kwargs.get('op_debug_dir', '.')
        if not (op_debug_dir_str is (None or "")):
            kernel_meta_dir = op_debug_dir_str + "/kernel_meta"
            if not os.path.exists(kernel_meta_dir):
                try:
                    os.makedirs(kernel_meta_dir, stat.S_IRWXU +
                                stat.S_IRGRP + stat.S_IXGRP)
                except FileExistsError:
                    pass
                finally:
                    pass
            from tbe.common.buildcfg import set_current_build_config
            set_current_build_config(
                "kernel_meta_parent_dir", op_debug_dir_str)


class KernelName(object):
    """store kernel name, to across python > c++ > python
    """
    kernel_name = ''

    @classmethod
    def set_kernel_name(cls, kernel_name):
        cls.kernel_name = kernel_name

    @classmethod
    def get_kernel_name(cls):
        return cls.kernel_name


def get_kernel_meta_dir():
    """get kernel_meta parent dir

    """
    warn("'from te.platform.cce_conf import get_kernel_meta_dir' is expired, "
         "please replace it with "
         "'from tbe.common.buildcfg import get_current_build_config or GlobalInfoContainer' "
         "instead.")
    from tbe.common.buildcfg import get_current_build_config
    from tbe.common.buildcfg import GlobalInfoContainer
    from tbe.common.buildcfg.ascend import AscendPassContext
    if AscendPassContext.get_scope_size() > 0:
        if get_current_build_config("build_fatbin"):
            return os.path.normpath(os.path.join(get_current_build_config("kernel_meta_parent_dir"),
                                                 'kernel_meta', KernelName.get_kernel_name()))
        return os.path.normpath(os.path.join(get_current_build_config("kernel_meta_parent_dir"),
                                             'kernel_meta'))
    return GlobalInfoContainer.global_info["kernel_meta_parent_dir"] + "/kernel_meta/"


def get_kernel_meta_parent_dir():
    """get kernel_meta parent dir

    """
    warn("'from te.platform.cce_conf import get_kernel_meta_parent_dir' is expired, "
         "please replace it with "
         "'from tbe.common.buildcfg import get_current_build_config or GlobalInfoContainer' "
         "instead.")
    from tbe.common.buildcfg import get_current_build_config
    from tbe.common.buildcfg import GlobalInfoContainer
    from tbe.common.buildcfg.ascend import AscendPassContext
    if AscendPassContext.get_scope_size() > 0:
        return os.path.normpath(get_current_build_config("kernel_meta_parent_dir"))
    return GlobalInfoContainer.global_info["kernel_meta_parent_dir"]


def set_mdl_bank_path(kwargs):
    """set model bank path
    """
    if not isinstance(kwargs, dict):
        return

    mdl_bank_path_str = kwargs.get('mdl_bank_path')
    if mdl_bank_path_str is (None or ""):
        return

    CceProductParams().set_build_option("mdl_bank_path", mdl_bank_path_str)


def set_op_bank_path(kwargs):
    """set model bank path
    """
    if not isinstance(kwargs, dict):
        return

    op_bank_path_str = kwargs.get('op_bank_path')
    if op_bank_path_str is (None or ""):
        return

    CceProductParams().set_build_option("op_bank_path", op_bank_path_str)


# Example: pylint: disable=too-many-arguments
# 'pylint:disable=unused-argument
def te_set_version(soc_version, core_type="AiCore",
                   aicore_num=None, l1_fusion=None,
                   l2_mode="0", l2_fusion=None, kwargs=None):
    """set version info

    Parameters
    ----------
    soc_version : str
        "Ascend310"/"Ascend910"/"Ascend310P1~P4"/"Ascend610" ...
    core_type : str
        "AiCore" or "VectorCore"
    aicore_num: int
        example: 32
    l1_fusion: bool
        example: True/False

    Returns
    -------
    errmsg : str
        error message, 'success' for OK.
    """
    warn("'from te.platform import te_set_version' is expired, "
         "please replace it with 'from tbe.common.platform import set_current_compile_soc_info'.")

    l1_fusion = _check_and_get_l1_fusion(l1_fusion)
    if core_type in (None, ""):
        core_type = "AiCore"
    _check_soc_version(soc_version, core_type)
    aicore_num = _check_and_get_aicore_num(aicore_num)

    func = get_global_func("cce.product_init")
    # 'pylint:disable=unused-variable
    value = func(soc_version, core_type, aicore_num, l1_fusion)

    from tbe.common.platform import set_current_compile_soc_info
    set_current_compile_soc_info(soc_version, core_type=core_type,
                                 aicore_num=aicore_num, l1_fusion=l1_fusion)

    from tbe.common.buildcfg import compatible_interface

    if l1_fusion in ("true", True):
        compatible_interface.set_L1_info("L1_fusion_enabled", True)
    elif l1_fusion in ("false", False):
        compatible_interface.set_L1_info("L1_fusion_enabled", False)

    if l2_fusion in ("true", True):
        compatible_interface.set_L1_info("L2_fusion_enabled", True)
    else:
        compatible_interface.set_L1_info("L2_fusion_enabled", False)

    te_set_op_impl_mode(kwargs)
    set_mdl_bank_path(kwargs)
    set_op_bank_path(kwargs)

    cur_cce_product_params = CceProductParams()
    cur_cce_product_params.cce_product = \
        SOC_VERSION_MAP.get(soc_version).get(core_type).get("ProductVersion")
    return cur_cce_product_params


def te_set_op_impl_mode(kwargs):
    """
    set op impl mode
    """
    if not isinstance(kwargs, dict):
        return

    from te_fusion import fusion_util
    fusion_util.OpImplPolicy.op_impl_mode_dict = kwargs.get(
        'op_impl_mode_dict', {})


def te_set_l2_mode(l2_mode):
    """set l2 mode

    Parameters
    ----------
    l2_mode : int

    Returns
    -------
    success_flag : boolean
    """
    warn("'from te.platform.cce_conf import te_set_l2_mode' is expired, "
         "please replace it with "
         "'from tbe.common.buildcfg import build_config(l2_mode=value)/GlobalInfoContainer.' "
         "instead.")
    from tbe.common.buildcfg import set_current_build_config
    set_current_build_config("l2_mode", int(l2_mode))


SHORT_SOC_VERSION = "SHORT_SOC_VERSION"
FULL_SOC_VERSION = "FULL_SOC_VERSION"
SHORT_SOC_VERSION = "SHORT_SOC_VERSION"
AICORE_TYPE = "AICORE_TYPE"
CORE_NUM = "CORE_NUM"
MAX_CORE_NUM = "MAX_CORE_NUM"
UB_SIZE = "UB_SIZE"
L2_SIZE = "L2_SIZE"
L1_SIZE = "L1_SIZE"
FB_SIZE = "FB_SIZE"
FB0_SIZE = "FB0_SIZE"
FB1_SIZE = "FB1_SIZE"
FB2_SIZE = "FB2_SIZE"
FB3_SIZE = "FB3_SIZE"
BT_SIZE = "BT_SIZE"
CUBE_SIZE = "CUBE_SIZE"
L0A_SIZE = "L0A_SIZE"
L0B_SIZE = "L0B_SIZE"
L0C_SIZE = "L0C_SIZE"
SMASK_SIZE = "SMASK_SIZE"
UNZIP = "UNZIP"
VREG_SIZE = "VREG_SIZE"
AREG_SIZE = "AREG_SIZE"
PREG_SIZE = "PREG_SIZE"
UREG_SIZE = "UREG_SIZE"
WREG_SIZE = "WREG_SIZE"
WIDE_REG_WIDTH = "WIDE_REG_WIDTH"
CUBE_VECTOR_SPLIT = "CUBE_VECTOR_SPLIT"


def get_soc_spec(key):
    """
    call global func to get soc spec

    Parameters
        ----------
        key : str
            key
    """
    warn("'from te.platform import get_soc_spec' is expired, "
         "please replace it with 'from tbe.common.platform import get_soc_spec'.")

    # 'pylint: disable=redefined-outer-name
    from tbe.common.platform import get_soc_spec
    return get_soc_spec(key)


@register_func("te.cce.get_product")
def get_product():
    """
    get product c++ code.

    Parameters
    ----------

    Returns
    -------
    value: device product.
        end of execution
    """

    return CceProductParams().cce_product


def api_check_support(intrinsic, dtype=""):
    """
    check if current chip support this api.

    Parameters
    ----------
    intrinsic : str, the intrinsic need to check
    dtype: str, optional args, if not empty, will check the dtype.
    Returns
    -------
    value: bool, True if chip contains such api, else return False

    """
    # Example: pylint: disable=import-outside-toplevel
    warn("'from te.platform import api_check_support' is expired, "
         "please replace it with 'from tbe.common.platform import api_check_support'.")

    # 'pylint: disable=redefined-outer-name
    from tbe.common.platform import api_check_support
    return api_check_support(intrinsic, dtype=dtype)


def intrinsic_check_support(intrinsic, dtype=""):
    """
    check if current chip support this intrinsic.

    Parameters
    ----------
    intrinsic : str, the intrinsic need to check
    dtype: str, optional args, if not empty, will check the dtype.
    Returns
    -------
    value: bool, True if chip contains such api, else return False

    """
    warn("'from te.platform import intrinsic_check_support' is expired, "
         "please replace it with 'from tbe.common.platform import intrinsic_check_support'.")

    # 'pylint: disable=redefined-outer-name
    from tbe.common.platform import intrinsic_check_support
    return intrinsic_check_support(intrinsic, dtype=dtype)


def is_v200_version_new():
    """
    check if Ascend610/Ascend310P/Hi3796CV300CS version
    ----------

    Returns
    -------
    True:  Ascend610/BS9SX1A/Ascend610B/Ascend310P/Hi3796CV300CS version
    False: Other version
    """
    short_soc_version = get_soc_spec("SHORT_SOC_VERSION")
    if short_soc_version in ("Ascend310P", "Ascend610", "BS9SX1A", "Ascend610B",
                             "Hi3796CV300CS", "SD3403"):
        return True
    return False


def get_product_version():
    """
    get product version
    ----------

    Returns
    -------
    cloud: cloud product
    mini: mini product
    """
    short_soc_version = get_soc_spec("SHORT_SOC_VERSION")
    if short_soc_version in ["Ascend310"]:
        version = VERSION_MINI
    elif short_soc_version in ["Ascend910"]:
        version = VERSION_CLOUD
    elif short_soc_version in ["Hi3796CV300ES", "Hi3796CV300CS", "SD3403"]:
        version = VERSION_SHISI
    elif short_soc_version in ["Ascend310P"]:
        version = VERSION_MINI_NG1
    elif short_soc_version in ["Ascend610"]:
        version = VERSION_MINI_NG1M
    elif short_soc_version in ["BS9SX1A"] or short_soc_version in ["Ascend610B"]:
        version = VERSION_MINI_NG1PG2
    elif short_soc_version in ["Ascend310B"]:
        version = VERSION_MINI_L
    else:
        raise_tbe_python_err(TBE_DEFAULT_PYTHON_ERROR_CODE, "Current product version is wrong!Please check!!")

    return version


def is_lhisi_version():
    """
    check if 3796ES version
    -------

    Returns
    -------
    True: 3796ES version
    False: Other version
    """
    version = get_product_version()
    if version == VERSION_SHISI:
        return True
    return False


def is_v200_version():
    """
    check if ng1/ng1m version
    ----------

    Returns
    -------
    True: ng1 version
    False: Other version
    """
    version = get_product_version()
    if version in (VERSION_MINI_NG1, VERSION_MINI_NG1M):
        return True
    return False


def is_lhisi_cs_version():
    """
    check if 3796CS version
    -------

    Returns
    -------
    True: 3796CS version
    False: Other version
    """
    short_soc_version = get_soc_spec("SHORT_SOC_VERSION")
    if short_soc_version in ["Hi3796CV300CS"]:
        return True
    return False


def is_cloud_version():
    """
    check if cloud-ASCEND_910 version
    ----------

    Returns
    -------
    True: cloud version
    False: Other version
    """
    version = get_product_version()
    if version == VERSION_CLOUD:
        return True
    return False


def is_mini_version():
    """
    check if mini version
    -------

    Returns
    -------
    True: mini version
    False: Other version
    """
    version = get_product_version()
    if version == VERSION_MINI:
        return True
    return False


def is_mini_or_lhisi_version():
    """
    check if mini or lhisi version
    -------

    Returns
    -------
    True: mini version or lhisi
    False: Other version
    """
    version = get_product_version()
    if version in (VERSION_MINI, VERSION_SHISI):
        return True
    return False


def is_mini_l_version():
    """
    check if mini_l_version
    -------

    Returns
    -------
    True: mini_l_version
    False: Other version
    """
    version = get_product_version()
    if version in (VERSION_MINI_L,):
        return True
    return False


def spport_hw_sync():
    short_soc_version = get_soc_spec("SHORT_SOC_VERSION")
    support_list = ["Ascend910B", "Ascend910_93"]
    return short_soc_version in support_list


_init_api("te.platform.cce_conf")
