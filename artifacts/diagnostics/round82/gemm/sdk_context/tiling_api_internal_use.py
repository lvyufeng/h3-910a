#!/usr/bin/env python
# -*- coding:utf-8 -*-
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
Get the tiling
"""
import os
import ctypes
import json
from ctypes import cdll
from ctypes import cast
from tbe.tvm import deepcopy
from tbe.common.tiling.auto_tiling_log import AUTOTILINGLOG
from tbe.common.context import get_context
from tbe.common.context import op_context
from tbe.common.utils.errormgr.error_manager_cube import raise_err_message_cube
from tbe.common.tiling.tiling_global_var_internal_use import MODEL_TYPE
from tbe.common.tiling.tiling_global_var_internal_use import TRANSFORMER_OP
from tbe.common.platform.platform_info import intrinsic_check_support

# define the op list which is supported in new Tiling Path
SUPPORT_NEW_TILING_OP = [
    "conv2d", "conv2d_backprop_input", "conv2d_backprop_filter", "matmul", "convolution_3d", "conv3d_backprop_input",
    "conv3d_backprop_filter"
]


def split_env_var(env_var: str, split: str) -> list:
    """
    split env var

    Parameters
    ----------
    env_var: str
        env
    split: str
        split mark

    Returns
    -------
    path : list
        return path_list
    """
    if os.environ.get(env_var, None):
        return [path.strip() for path in os.environ[env_var].split(split)]
    return []


def get_lib_path() -> str:
    """
    get lib path

    Parameters
    ----------

    Returns
    -------
    lib_found : str
        return lib path
    """
    so_path = []
    so_path.extend(split_env_var('LD_LIBRARY_PATH', ':'))
    lib_path = [os.path.join(path, 'libauto_tiling.so') for path in so_path]
    lib_found = [path for path in lib_path if os.path.exists(path) and os.path.isfile(path)]
    if not lib_found:
        raise_err_message_cube(
            "the environment path {} is not set or invalid, expect {}.".format(
                "LD_LIBRARY_PATH", "the path of libauto_tiling.so"))
    AUTOTILINGLOG.debug("[auto_tiling]libauto_tiling.so path:{}".format(lib_found[0]))
    return lib_found[0]


def get_op_bank_path() -> str:
    """
    Get the op repository path from te_fusion

    Parameters
    ----------

    Returns
    -------
    op_bank_path : str
        The result op_bank_path.
    """
    op_bank_path = get_context().get_addition("op_bank_path")
    if op_bank_path is None:
        op_bank_path = ""
    if not isinstance(op_bank_path, str):
        raise_err_message_cube(
            "only support {}, but the type of return value is {}.".format(
                "str or None", type(op_bank_path)))
    return op_bank_path


def reset_repository() -> None:
    """
    Reload custom repository

    """
    lib = cdll.LoadLibrary(get_lib_path())
    lib.ResetCustomRepository()


def global_func_get_tiling(input_str: list, tiling_type_num: int, op_bank_path: str) -> str:
    """
    get the tiling from bank_cache

    Parameters
    ----------
    input_str: list
        input shape
    tiling_type_num: int
        tiling query type
    op_bank_path: str
        The result op_bank_path.

    Returns
    -------
    tiling_encode: str
        return tiling
    """
    lib = cdll.LoadLibrary(get_lib_path())
    paras_encode = input_str.encode()
    op_bank_path_encode = op_bank_path.encode()
    lib.GetTiling.restype = ctypes.c_void_p
    tiling_bytes = lib.GetTiling(paras_encode, tiling_type_num, op_bank_path_encode)
    tiling = str(cast(tiling_bytes, ctypes.c_char_p).value, encoding='utf-8')
    lib.FreeCharArray.argtypes = [ctypes.c_void_p]
    lib.FreeCharArray(tiling_bytes)
    return tiling


def global_func_tiling_query(shape_encode_array: list, tiling_type_num: int, \
                             op_bank_path: str) -> list:
    """
    get the tiling from bank_cache

    Parameters
    ----------
    shape_encode_array: list
        encode of shape
    tiling_type_num: int
        tiling query type
    op_bank_path: str
        The result op_bank_path.

    Returns
    -------
    tiling_encode: list
        encode of tiling
    """
    lib = cdll.LoadLibrary(get_lib_path())
    shape_encode_temp = ctypes.c_uint32 * len(shape_encode_array)
    shape_encode_encode = shape_encode_temp()
    for i, value in enumerate(shape_encode_array):
        shape_encode_encode[i] = value
    op_bank_path_encode = op_bank_path.encode()
    lib.TilingQueryProcess.restype = ctypes.POINTER(ctypes.c_uint32)
    tiling_encode_bytes = lib.TilingQueryProcess(shape_encode_encode, tiling_type_num, op_bank_path_encode)
    encode_bytes_size = tiling_encode_bytes[0]
    tiling_encode = [tiling_encode_bytes[i + 1] for i in range(encode_bytes_size)]
    lib.FreeUint32Array.argtypes = [ctypes.POINTER(ctypes.c_uint32)]
    lib.FreeUint32Array(tiling_encode_bytes)
    return tiling_encode


def global_func_tiling_analyzer(shape_encode_array: list, tuning_flag: int) -> list:
    """
    Tiling Analyzer interface on the python side

    Parameters
    ----------
    shape_encode_array: list
        encode of shape
    tuning_flag: int
        the flag of tuning mode.

    Returns
    -------
    tiling_space: list
        return the space of tiling
    """
    lib = cdll.LoadLibrary(get_lib_path())
    shape_encode = (ctypes.c_uint32 * len(shape_encode_array))()
    for i, value in enumerate(shape_encode_array):
        shape_encode[i] = value
    lib.TilingAnalyzer.restype = ctypes.POINTER(ctypes.c_uint64)
    tiling_space_pointer = lib.TilingAnalyzer(shape_encode, tuning_flag)
    space_size = tiling_space_pointer[0]
    tiling_space = [tiling_space_pointer[i + 1] for i in range(space_size)]
    lib.FreeUint64Array.argtypes = [ctypes.POINTER(ctypes.c_uint64)]
    lib.FreeUint64Array(tiling_space_pointer)
    return tiling_space


def global_func_flexible_analyzer(shape_encode_array: list, tuning_flag: int) -> list:
    """
    flexible Analyzer interface on the python side

    Parameters
    ----------
    shape_encode_array: list
        encode of shape
    tuning_flag: int
        the flag of tuning mode.

    Returns
    -------
    tiling_space: list
        return the space of tiling
    """
    lib = cdll.LoadLibrary(get_lib_path())
    if isinstance(shape_encode_array, str):
        shape_encode = shape_encode_array.encode()
    else:
        shape_encode = json.dumps(shape_encode_array).encode()
    lib.FlexibleAnalyzer.restype = ctypes.POINTER(ctypes.c_uint64)
    tiling_space_pointer = lib.FlexibleAnalyzer(shape_encode, tuning_flag)
    space_size = tiling_space_pointer[0]
    tiling_space = [tiling_space_pointer[i + 1] for i in range(space_size)]
    lib.FreeUint64Array.argtypes = [ctypes.POINTER(ctypes.c_uint64)]
    lib.FreeUint64Array(tiling_space_pointer)
    return tiling_space


def global_func_convert_output_format(input_str: str, tiling_index_array_str: str) -> str:
    """
    get tiling dict from tiling index array

    Parameters
    ----------
    input_str: str
        input shape
    tiling_index_array_str: str
        the tiling index array string

    Returns
    -------
    ret: str
        the tiling dict
    """
    lib = cdll.LoadLibrary(get_lib_path())
    paras_encode = input_str.encode()
    tiling_index_array_str_encode = tiling_index_array_str.encode()
    lib.ConvertTilingDict.restype = ctypes.c_void_p
    ret_bytes = lib.ConvertTilingDict(paras_encode, tiling_index_array_str_encode)
    ret = str(cast(ret_bytes, ctypes.c_char_p).value, encoding='utf-8')
    lib.FreeCharArray.argtypes = [ctypes.c_void_p]
    lib.FreeCharArray(ret_bytes)
    return ret


def check_params(info_dict: dict) -> None:
    """
    check the tiling from module

    Parameters
    ----------
    info_dict: dict
        the params of operator

    Returns
    -------

    """
    # check the info_dict
    if not isinstance(info_dict, dict):
        raise_err_message_cube(
            "info_dict should be dict, but the input type is {}.".format(type(info_dict)))

    # check the op_type info
    if "op_type" not in info_dict.keys():
        raise_err_message_cube(
            "the keyword {} is missing in input params.".format("op_type"))


def configure_model_type(info_dict: dict, tiling_type: str) -> None:
    """
    config the model type

    Parameters
    ----------
    info_dict: dict
        the params of operator
    tiling_type: str
        get tiling mode

    Returns
    -------
    """
    info_dict["model_type"] = MODEL_TYPE[0] # xgboost


def process_repo_hit_info(tilings: dict, dynamic_shape_flag: bool,
                          kernel_name: (str, None) = None, hit_repo_flag: bool = False) ->  dict:
    """
    record repository hit info
    :param tilings: dict of one tiling or a list of multiple tiling dicts
    :return tilings: tilings after discarding repo hit info
    """
    if tilings is None:
        return tilings
    # TILINGINSTANCE.set_tiling scenario or empty list(returned in dynamic scenario) will not hit repo
    ret = False
    if isinstance(tilings, list):
        for tmp_tiling in tilings:
            # tilings in dynamic scenario contains no repo hit info
            if dynamic_shape_flag:
                tmp_tiling["tiling"].pop("hit_repo", None)
                ret = None
            else:
                # tilings from cost model
                tmp_tiling.pop("hit_repo", False)
    elif "hit_repo" in tilings:
        ret = tilings.pop("hit_repo", False)
    else:
        ret = hit_repo_flag

    tmp_func = lambda x: "invalid" if x is None else x
    context = op_context.get_context()
    if context:
        op_name = context.get_addition("op_name")
        info_str = f"[auto_tiling]op_name:[{tmp_func(op_name)}] kernel_name:[{tmp_func(kernel_name)}]"
        if ret is None:
            AUTOTILINGLOG.debug(f"{info_str} contains no repository hit info,\
            check if it's in dynamic scenario")
        else:
            context.add_build_json_result("KBHit", ret)
            AUTOTILINGLOG.info(f"{info_str} hit bank result:{ret}")
    else:
        AUTOTILINGLOG.info(f"[auto_tiling]kernel_name:[{tmp_func(kernel_name)}] cannot get op context")

    return tilings


def is_support_fixpipe_flatform(op_type: str) -> bool:
    """
    Check if support fixpipe.
    """
    fixpipe_platform_flag = intrinsic_check_support("Intrinsic_fix_pipe_unit_list")
    is_support_op_flag = op_type in SUPPORT_NEW_TILING_OP
    return fixpipe_platform_flag and is_support_op_flag


def dict_value_none_deepcopy(input_args: dict) -> dict:
    """
    deepcopy cannot handle value = None

    Need to use json str do deepcopy

    Parameters
    ----------
    dict: with None value

    Returns
    -------
    dict: input dict
    """
    params_in = deepcopy(json.dumps(input_args))
    return json.loads(params_in)