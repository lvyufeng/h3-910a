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
CCE configuration constants
"""
from __future__ import absolute_import as _abs

import threading

from tbe import tvm


# Example: pylint: disable=invalid-name, useless-object-inheritance, too-few-public-methods
# Switch in tik/api/tik_build.py
class OUTPUT_PATH_CLASS(object):
    """
    cce output path
    """
    output_path = "kernel_meta"


# Example: pylint: disable=invalid-name, useless-object-inheritance, too-few-public-methods
class GM_NAME_MAP_CLASS(object):
    """
    get gm_name map
    """
    gm_name_map = {}


JSON_KEY_MIX_NAME = "mix_name"
JSON_KEY_MIX_AIC_NAME = "mix_aic_name"
JSON_KEY_MIX_AIV_NAME = "mix_aiv_name"
MIX_AIC_SUFFIX = "_mix_aic"
MIX_AIV_SUFFIX = "_mix_aiv"
KERNEL_META_PATH = "kernel_meta"
JSON_KEY_PARAMETERS = "parameters"
JSON_KEY_DETERMINISTIC = "deterministic"
JSON_KEY_OP_PARA_SIZE = "opParaSize"
JSON_KEY_WORKSPACE = "workspace"
JSON_KEY_KERNEL_LIST = "kernelList"
JSON_KEY_BLOCK_DIM = "blockDim"
JSON_KEY_SIZE = "size"
JSON_KEY_TASK_RATION = "taskRation"
JSON_KEY_CORE_TYPE = "coreType"
JSON_KEY_INTERCORE_SYNC_FLAG = "intercoreSync"
JSON_KEY_JSON_LIST = "jsonList"
JSON_KEY_JSON_FILE_NAME = "jsonFileName"
JSON_KEY_KERNEL_NAME = "kernelName"
JSON_KEY_BIN_FILE_NAME = "binFileName"
# OSError: [Errno 17] File exists
FILE_EXISTS_ERR = 17

# save tbe gm tensor whether is workspace in to a list
TBE_WORKSPACE_SIZE_LIST = threading.local()
TBE_WORKSPACE_SIZE_LIST.local_list = []

# save tik gm tensor whether is workspace in to a list
TIK_WORKSPACE_SIZE_LIST = threading.local()
TIK_WORKSPACE_SIZE_LIST.local_list = []

# Example: pylint: disable=invalid-name
# save tik gm tensor whether is atomic add in to a list
TIK_ATOMIC_ADD_LIST = threading.local()
TIK_ATOMIC_ADD_LIST.local_list = []
jump_expand_flag = False

# Example: pylint: disable=invalid-name
# def the buffer var
scope_cbuf = "local.L1"
scope_fb = "local.FB"
scope_fb0 = "local.FB0"
scope_fb1 = "local.FB1"
scope_fb2 = "local.FB2"
scope_fb3 = "local.FB3"
scope_bt = "local.BT"
scope_ubuf = "local.UB"
scope_ca = "local.L0A"
scope_cb = "local.L0B"
scope_cc = "local.L0C"
scope_reg = "local.REG"
scope_vreg = "local.VREG"
scope_preg = "local.PREG"
scope_areg = "local.AREG"
scope_ureg = "local.UREG"
scope_wreg = "local.WREG"
scope_aicpu = "local_aicpu"
scope_gm = "global"
scope_cbuf_fusion = "local.L1_Fusion"
scope_smask = "local.SMASK"

dma_copy = "dma_copy"
dma_copy_global = "global"

# def the cce thread axis for sync
CCE_AXIS = tvm.thread_axis("cce")
CCE_MASK_AXIS = tvm.thread_axis("cce_mask")
CCE_GLOBAL_AXIS = tvm.thread_axis("cce_global")

MASK_VAR = tvm.var("MASK")
RSVD_CNT = tvm.var("RSVD_CNT")

# def the cce vector intrinsic params
VECTOR_INST_BLOCK_WIDTH = 256
VECTOR_INST_BLOCK_NUM = 8
VECTOR_INST_MAX_REPEAT_TIMES = 255

# def the gemm const
WGT_WIDTH = 16
INP_WIDTH = 16
OUT_WIDTH = 16
BLOCK_IN = 16
BLOCK_OUT = 16
BLOCK_REDUCE = 16

# def the gemm int8/uint8 reduce const
BLOCK_REDUCE_INT8 = 32
# def the gemv/gevm vector const
BLOCK_VECTOR = 1

INP_ELEM_BYTES = (BLOCK_IN * BLOCK_REDUCE * INP_WIDTH // 8)
WGT_ELEM_BYTES = (BLOCK_OUT * BLOCK_REDUCE * WGT_WIDTH // 8)
OUT_ELEM_BYTES = (BLOCK_IN * BLOCK_OUT * OUT_WIDTH // 8)
GLB_ELEM_BYTES = (16 * OUT_WIDTH // 8)

# def the mad pattern
GEMM_MODE = 0
GEVM_MODE = 1
CONV_MODE = 2

C0_SIZE = 16
ELEMENTS_VECTOR_OP_FP16 = 128

DEFAULT_MUL_VALUE = 1
DEFAULT_ADD_VALUE = 0

CUBE_MKN = {"int4": {'mac': [16, 64, 16]},
            "int8": {'mac': [16, 32, 16]},
            "uint8": {'mac': [16, 32, 16]},
            "int16": {'mac': [16, 16, 16]},
            "int32": {'mac': [16, 16, 16]},
            "float16": {'mac': [16, 16, 16]},
            "float32": {'mac': [16, 8, 16]},
            "custom[bfloat]16": {'mac': [16, 16, 16]}}

# def limitation of repeart times for a single copy instrin and align value
VECTOR_COPY_NBURST_LIMIT = 4096
VECTOR_SINGLE_BLOCK_WIDTH_FP16 = 16


# Example: pylint: disable=invalid-name, useless-object-inheritance, too-few-public-methods
class conv_buffer_ex(object):
    """
    conv buffer
    """
    offsetPad = None


# represent 5 soc, currently contains in tik
ASCEND_310 = "Ascend310"
ASCEND_910 = "Ascend910"
ASCEND_910H = "Ascend910B"
ASCEND_910M = "Ascend910A"
ASCEND_910P = "Ascend910ProA"
HI3796CV300ES = "Hi3796CV300ES"
HI3796CV300CS = "Hi3796CV300CS"
SD3403 = "SD3403"
ASCEND_610 = "Ascend610"
ASCEND_310P = "Ascend310P"
BS9SX1A = "BS9SX1A"
ASCEND_610B = "Ascend610B"
ASCEND_910B = "Ascend910B"
ASCEND_910_93 = "Ascend910_93"
ASCEND_310B = "Ascend310B"
ASCEND_SD = "SD3403"
_AIC_ENGINE = "AiCore"
_VEC_ENGINE = "VectorCore"

AIC_310P = ASCEND_310P + _AIC_ENGINE
VEC_310P = ASCEND_310P + _VEC_ENGINE
AIC_610 = ASCEND_610 + _AIC_ENGINE
VEC_610 = ASCEND_610 + _VEC_ENGINE
AIC_BS9SX1A = BS9SX1A + _AIC_ENGINE
VEC_BS9SX1A = BS9SX1A + _VEC_ENGINE
AIC_610B = ASCEND_610B + _AIC_ENGINE
VEC_610B = ASCEND_610B + _VEC_ENGINE
AIC_310B = ASCEND_310B + _AIC_ENGINE
VEC_310B = ASCEND_310B + _VEC_ENGINE
HI3796CV300ESAIC = HI3796CV300ES + _AIC_ENGINE
HI3796CV300CSAIC = HI3796CV300CS + _AIC_ENGINE
SD3403AIC = SD3403 + _AIC_ENGINE
ASCEND_310AIC = ASCEND_310 + _AIC_ENGINE
ASCEND_910AIC = ASCEND_910 + _AIC_ENGINE
ASCEND_910BAIC = ASCEND_910B + _AIC_ENGINE
ASCEND_910BVEC = ASCEND_910B + _VEC_ENGINE
ASCEND_910_93AIC = ASCEND_910_93 + _AIC_ENGINE
ASCEND_910_93VEC = ASCEND_910_93 + _VEC_ENGINE
ASCEND_SD_AIC = "SD3403" + _AIC_ENGINE

ONLY_TIK_API_MAP = {
    """
    contains api that not is instr.
    """
    "vector_dup_v1": {
        ASCEND_310AIC: ["uint16", "int16", "float16", "uint32", "int32", "float32"],
        ASCEND_910AIC: ["uint16", "int16", "float16", "uint32", "int32", "float32"],
        ASCEND_910BAIC: ["uint16", "int16", "float16", "uint32", "int32", "float32"],
        ASCEND_910BVEC: ["uint16", "int16", "float16", "uint32", "int32", "float32"],
        ASCEND_910_93AIC: ["uint16", "int16", "float16", "uint32", "int32", "float32"],
        ASCEND_910_93VEC: ["uint16", "int16", "float16", "uint32", "int32", "float32"],
        HI3796CV300ESAIC: ["uint16", "int16", "float16", "uint32", "int32", "float32"],
        HI3796CV300CSAIC: ["uint16", "int16", "float16", "uint32", "int32",
                           "float32"],
        ASCEND_SD_AIC: ["uint16", "int16", "float16", "uint32", "int32", "float32"],
        AIC_310P: ["uint16", "int16", "float16", "uint32", "int32", "float32"],
        AIC_610: ["uint16", "int16", "float16", "uint32", "int32", "float32"],
        VEC_610: ["uint16", "int16", "float16", "uint32", "int32", "float32"],
        VEC_310P: ["uint16", "int16", "float16", "uint32", "int32", "float32"],
        AIC_BS9SX1A: ["uint16", "int16", "float16", "uint32", "int32", "float32"],
        AIC_610B: ["uint16", "int16", "float16", "uint32", "int32", "float32"]
    },
    "vector_dup": {
        ASCEND_310AIC: ["uint16", "int16", "float16", "uint32", "int32", "float32"],
        ASCEND_910AIC: ["uint16", "int16", "float16", "uint32", "int32", "float32"],
        ASCEND_910BAIC: ["uint16", "int16", "float16", "uint32", "int32", "float32"],
        ASCEND_910BVEC: ["uint16", "int16", "float16", "uint32", "int32", "float32"],
        ASCEND_910_93AIC: ["uint16", "int16", "float16", "uint32", "int32", "float32"],
        ASCEND_910_93VEC: ["uint16", "int16", "float16", "uint32", "int32", "float32"],
        HI3796CV300ESAIC: ["uint16", "int16", "float16", "uint32", "int32", "float32"],
        HI3796CV300CSAIC: ["uint16", "int16", "float16", "uint32", "int32",
                           "float32"],
        ASCEND_SD_AIC: ["uint16", "int16", "float16", "uint32", "int32", "float32"],
        AIC_310P: ["uint16", "int16", "float16", "uint32", "int32", "float32"],
        AIC_610: ["uint16", "int16", "float16", "uint32", "int32", "float32"],
        VEC_310P: ["uint16", "int16", "float16", "uint32", "int32", "float32"],
        VEC_610: ["uint16", "int16", "float16", "uint32", "int32", "float32"],
        AIC_BS9SX1A: ["uint16", "int16", "float16", "uint32", "int32", "float32"],
        AIC_610B: ["uint16", "int16", "float16", "uint32", "int32", "float32"],
        VEC_310B: ["uint16", "int16", "float16", "uint32", "int32", "float32"]
    },
    "vec_trans_scatter": {
        ASCEND_310AIC: ["int8", "uint8", "uint16", "int16", "float16"],
        ASCEND_910AIC: ["int8", "uint8", "uint16", "int16", "float16"],
        ASCEND_910BAIC: ["int8", "uint8", "uint16", "int16", "float16"],
        ASCEND_910BVEC: ["int8", "uint8", "uint16", "int16", "float16"],
        ASCEND_910_93AIC: ["int8", "uint8", "uint16", "int16", "float16"],
        ASCEND_910_93VEC: ["int8", "uint8", "uint16", "int16", "float16"],
        HI3796CV300ESAIC: ["int8", "uint8", "uint16", "int16", "float16",
                           "uint32", "int32", "float32"],
        HI3796CV300CSAIC: ["int8", "uint8", "uint16", "int16", "float16",
                           "uint32", "int32", "float32"],
        ASCEND_SD_AIC: ["int8", "uint8", "uint16", "int16", "float16",
                        "uint32", "int32", "float32"],
        AIC_310P: ["int8", "uint8", "uint16", "int16", "float16", "uint32",
                  "int32", "float32"],
        AIC_610: ["int8", "uint8", "uint16", "int16", "float16", "uint32",
                  "int32", "float32"],
        VEC_310P: ["int8", "uint8", "uint16", "int16", "float16", "uint32",
                  "int32", "float32"],
        VEC_610: ["int8", "uint8", "uint16", "int16", "float16", "uint32",
                  "int32", "float32"],
        AIC_BS9SX1A: ["int8", "uint8", "uint16", "int16", "float16", "uint32",
                      "int32", "float32"],
        AIC_610B: ["int8", "uint8", "uint16", "int16", "float16", "uint32",
                  "int32", "float32"]
    },
    "data_move_v1": {
        ASCEND_310AIC: [],
        ASCEND_910AIC: [],
        ASCEND_910BAIC: [],
        ASCEND_910BVEC: [],
        ASCEND_910_93AIC: [],
        ASCEND_910_93VEC: [],
        HI3796CV300ESAIC: [],
        HI3796CV300CSAIC: [],
        ASCEND_SD_AIC: [],
        AIC_310P: [],
        VEC_310P: [],
        AIC_610: [],
        VEC_610: [],
        AIC_BS9SX1A: [],
        AIC_610B: []
    },
    "vnchwtrans": {
        ASCEND_310AIC: ["uint16", "int16", "float16"],
        ASCEND_910AIC: ["uint16", "int16", "float16"],
        ASCEND_910BAIC: ["uint16", "int16", "float16"],
        ASCEND_910BVEC: ["uint16", "int16", "float16"],
        ASCEND_910_93AIC: ["uint16", "int16", "float16"],
        ASCEND_910_93VEC: ["uint16", "int16", "float16"],
        HI3796CV300ESAIC: ["uint16", "int16", "float16"],
        HI3796CV300CSAIC: ["uint16", "int16", "float16"],
        ASCEND_SD_AIC: ["uint16", "int16", "float16"],
        AIC_310P: ["uint16", "int16", "float16"],
        AIC_610: ["uint16", "int16", "float16"],
        VEC_310P: ["uint16", "int16", "float16"],
        VEC_610: ["uint16", "int16", "float16"],
        AIC_BS9SX1A: ["uint16", "int16", "float16"],
        AIC_610B: ["uint16", "int16", "float16"]
    },
    "scalar_conv": {
        ASCEND_910AIC: ['s322f32', 'f322s32r', 'f322s32a', 'f322s32f', 'f322s32c',
                        'f322s32z', 'f162f32', 'f322f16', 'f322f16o'],
        ASCEND_910BAIC: ['s322f32', 'f322s32r', 'f322s32a', 'f322s32f', 'f322s32c',
                        'f322s32z', 'f162f32', 'f322f16', 'f322f16o'],
        ASCEND_910BVEC: ['s322f32', 'f322s32r', 'f322s32a', 'f322s32f', 'f322s32c',
                        'f322s32z', 'f162f32', 'f322f16', 'f322f16o'],
        ASCEND_910_93AIC: ['s322f32', 'f322s32r', 'f322s32a', 'f322s32f', 'f322s32c',
                         'f322s32z', 'f162f32', 'f322f16', 'f322f16o'],
        ASCEND_910_93VEC: ['s322f32', 'f322s32r', 'f322s32a', 'f322s32f', 'f322s32c',
                         'f322s32z', 'f162f32', 'f322f16', 'f322f16o'],
        HI3796CV300ESAIC: ['s322f32', 'f322s32r', 'f322s32a', 'f322s32f', 'f322s32c',
                           'f322s32z', 'f162f32', 'f322f16', 'f322f16o'],
        HI3796CV300CSAIC: ['s322f32', 'f322s32r', 'f322s32a', 'f322s32f',
                           'f322s32c',
                           'f322s32z', 'f162f32', 'f322f16', 'f322f16o'],
        ASCEND_SD_AIC: ['s322f32', 'f322s32r', 'f322s32a', 'f322s32f',
                        'f322s32c', 'f322s32z', 'f162f32', 'f322f16', 'f322f16o'],
        AIC_310P: ['s322f32', 'f322s32r', 'f322s32a', 'f322s32f', 'f322s32c',
                  'f322s32z', 'f162f32', 'f322f16', 'f322f16o'],
        AIC_610: ['s322f32', 'f322s32r', 'f322s32a', 'f322s32f', 'f322s32c',
                  'f322s32z', 'f162f32', 'f322f16', 'f322f16o'],
        VEC_310P: ['s322f32', 'f322s32r', 'f322s32a', 'f322s32f', 'f322s32c',
                  'f322s32z', 'f162f32', 'f322f16', 'f322f16o'],
        VEC_610: ['s322f32', 'f322s32r', 'f322s32a', 'f322s32f', 'f322s32c',
                  'f322s32z', 'f162f32', 'f322f16', 'f322f16o'],
        VEC_BS9SX1A: ['s322f32', 'f322s32r', 'f322s32a', 'f322s32f', 'f322s32c',
                      'f322s32z', 'f162f32', 'f322f16', 'f322f16o'],
        AIC_BS9SX1A: ['s322f32', 'f322s32r', 'f322s32a', 'f322s32f', 'f322s32c',
                      'f322s32z', 'f162f32', 'f322f16', 'f322f16o'],
        VEC_610B: ['s322f32', 'f322s32r', 'f322s32a', 'f322s32f', 'f322s32c',
                  'f322s32z', 'f162f32', 'f322f16', 'f322f16o'],
        AIC_610B: ['s322f32', 'f322s32r', 'f322s32a', 'f322s32f', 'f322s32c',
                  'f322s32z', 'f162f32', 'f322f16', 'f322f16o']
    },
    "mov_rpn_cor_ir_to_scalar": {
        ASCEND_310AIC: [],
        ASCEND_910AIC: [],
        ASCEND_910BAIC: [],
        ASCEND_910BVEC: [],
        ASCEND_910_93AIC: [],
        ASCEND_910_93VEC: [],
        HI3796CV300ESAIC: [],
        HI3796CV300CSAIC: [],
        ASCEND_SD_AIC: [],
        AIC_310P: [],
        AIC_610: [],
        VEC_310P: [],
        VEC_610: [],
        AIC_BS9SX1A: [],
        AIC_610B: []
    },
    "load2dv1": {
        ASCEND_310AIC: ["int8", "uint8", "uint16", "int16", "float16"],
        ASCEND_910AIC: ["int8", "uint8", "uint16", "int16", "float16"],
        ASCEND_910BAIC: ["int8", "uint8", "uint16", "int16", "float16"],
        ASCEND_910BVEC: ["int8", "uint8", "uint16", "int16", "float16"],
        ASCEND_910_93AIC: ["int8", "uint8", "uint16", "int16", "float16"],
        ASCEND_910_93VEC: ["int8", "uint8", "uint16", "int16", "float16"],
        HI3796CV300ESAIC: ["uint4", "int4", "int8", "uint8", "uint16",
                           "int16", "float16"],
        HI3796CV300CSAIC: ["uint4", "int4", "int8", "uint8", "uint16",
                           "int16", "float16"],
        ASCEND_SD_AIC: ["uint4", "int4", "int8", "uint8", "uint16",
                        "int16", "float16"],
        AIC_310P: ["uint4", "int4", "int8", "uint8", "uint16", "int16",
                  "float16"],
        AIC_610: ["uint4", "int4", "int8", "uint8", "uint16", "int16",
                  "float16"],
        VEC_310P: ["uint4", "int4", "int8", "uint8", "uint16", "int16",
                  "float16"],
        VEC_610: ["uint4", "int4", "int8", "uint8", "uint16", "int16",
                  "float16"],
        AIC_BS9SX1A: ["uint4", "int4", "int8", "uint8", "uint16", "int16",
                      "float16"],
        AIC_610B: ["uint4", "int4", "int8", "uint8", "uint16", "int16",
                  "float16"]
    },
    "load2dv2": {
        HI3796CV300ESAIC: ["uint4", "int4", "int8", "uint8", "uint16",
                           "int16", "float16"],
        HI3796CV300CSAIC: ["uint4", "int4", "int8", "uint8", "uint16",
                           "int16", "float16"],
        ASCEND_SD_AIC: ["uint4", "int4", "int8", "uint8", "uint16",
                        "int16", "float16"],
        AIC_310P: ["uint4", "int4", "int8", "uint8", "uint16", "int16",
                   "float16"],
        AIC_610: ["uint4", "int4", "int8", "uint8", "uint16", "int16",
                  "float16"],
        VEC_310P: ["uint4", "int4", "int8", "uint8", "uint16", "int16",
                  "float16"],
        VEC_610: ["uint4", "int4", "int8", "uint8", "uint16", "int16",
                  "float16"],
        AIC_BS9SX1A: ["uint4", "int4", "int8", "uint8", "uint16", "int16",
                      "float16"],
        AIC_610B: ["uint4", "int4", "int8", "uint8", "uint16", "int16",
                  "float16"]
    },
    "assign": {
        ASCEND_310AIC: [],
        ASCEND_910AIC: [],
        ASCEND_910BAIC: [],
        ASCEND_910BVEC: [],
        ASCEND_910_93AIC: [],
        ASCEND_910_93VEC: [],
        HI3796CV300ESAIC: [],
        HI3796CV300CSAIC: [],
        ASCEND_SD_AIC: [],
        AIC_310P: [],
        AIC_610: [],
        VEC_310P: [],
        VEC_610: [],
        AIC_BS9SX1A: [],
        AIC_610B: []
    },
    "set_l0_set_value": {
        ASCEND_310AIC: [],
        ASCEND_910AIC: [],
        ASCEND_910BAIC: [],
        ASCEND_910BVEC: [],
        ASCEND_910_93AIC: [],
        ASCEND_910_93VEC: [],
        HI3796CV300ESAIC: [],
        HI3796CV300CSAIC: [],
        ASCEND_SD_AIC: [],
        AIC_310P: [],
        AIC_610: [],
        VEC_310P: [],
        VEC_610: [],
        AIC_BS9SX1A: [],
        AIC_610B: []
    },
    "load3dv1": {
        ASCEND_310AIC: [],
        ASCEND_910AIC: [],
        HI3796CV300ESAIC: [],
        HI3796CV300CSAIC: [],
        ASCEND_SD_AIC: [],
        AIC_310P: [],
        AIC_610: [],
        AIC_BS9SX1A: [],
        AIC_610B: []
    },
    "col2img": {
        ASCEND_910AIC: ["float16", "float32"],
        AIC_310P: ["float16", "float32"],
        AIC_610: ["float16", "float32"],
        AIC_BS9SX1A: ["float16", "float32"],
        AIC_610B: ["float16", "float32"]
    },
    "mmad_broadcast": {
        ASCEND_310AIC: ['f16f16', 'f32f32', 's32s32', 'f32f16'],
        ASCEND_910AIC: ['f16f16', 'f32f32', 's32s32', 'f32f16'],
        ASCEND_910BAIC: ['f16f16', 'f32f32', 's32s32', 'f32f16'],
        ASCEND_910BVEC: ['f16f16', 'f32f32', 's32s32', 'f32f16'],
        ASCEND_910_93AIC: ['f16f16', 'f32f32', 's32s32', 'f32f16'],
        ASCEND_910_93VEC: ['f16f16', 'f32f32', 's32s32', 'f32f16'],
        HI3796CV300ESAIC: ['f32f32', 's32s32'],
        HI3796CV300CSAIC: ['f32f32', 's32s32'],
        ASCEND_SD_AIC: ['f32f32', 's32s32'],
        AIC_310P: ['f16f16', 'f32f32', 's32s32', 'f32f16'],
        AIC_610: ['f16f16', 'f32f32', 's32s32', 'f32f16'],
        AIC_BS9SX1A: ['f16f16', 'f32f32', 's32s32', 'f32f16'],
        AIC_610B: ['f16f16', 'f32f32', 's32s32', 'f32f16']
    },
    "tensor_padding_with_matrix": {
        ASCEND_310AIC: [],
        ASCEND_910AIC: [],
        ASCEND_910BAIC: [],
        ASCEND_910BVEC: [],
        ASCEND_910_93AIC: [],
        ASCEND_910_93VEC: [],
        HI3796CV300ESAIC: [],
        HI3796CV300CSAIC: [],
        ASCEND_SD_AIC: [],
        AIC_310P: [],
        AIC_610: [],
        AIC_BS9SX1A: [],
        AIC_610B: []
    },
    "data_move": {
        ASCEND_310AIC: [],
        ASCEND_910AIC: [],
        ASCEND_910BAIC: [],
        ASCEND_910BVEC: [],
        ASCEND_910_93AIC: [],
        ASCEND_910_93VEC: [],
        HI3796CV300ESAIC: [],
        HI3796CV300CSAIC: [],
        ASCEND_SD_AIC: [],
        AIC_310P: [],
        AIC_610: [],
        VEC_310P: [],
        VEC_610: [],
        AIC_BS9SX1A: [],
        AIC_610B: []
    },
    "data_move_quant": {
        ASCEND_310AIC: [],
        ASCEND_910AIC: [],
        ASCEND_910BAIC: [],
        ASCEND_910BVEC: [],
        ASCEND_910_93AIC: [],
        ASCEND_910_93VEC: [],
        HI3796CV300ESAIC: [],
        HI3796CV300CSAIC: [],
        ASCEND_SD_AIC: [],
        AIC_310P: [],
        AIC_610: [],
        VEC_310P: [],
        VEC_610: [],
        AIC_BS9SX1A: [],
        AIC_610B: []
    },
    "tensor_mov": {
        ASCEND_310AIC: [],
        ASCEND_910AIC: [],
        ASCEND_910BAIC: [],
        ASCEND_910BVEC: [],
        ASCEND_910_93AIC: [],
        ASCEND_910_93VEC: [],
        HI3796CV300ESAIC: [],
        HI3796CV300CSAIC: [],
        ASCEND_SD_AIC: [],
        AIC_310P: [],
        AIC_610: [],
        VEC_310P: [],
        VEC_610: [],
        AIC_BS9SX1A: [],
        AIC_610B: []
    },
    "load3dv2": {
        HI3796CV300ESAIC: ["int8", "uint8", "float16"],
        HI3796CV300CSAIC: ["int8", "uint8", "float16"],
        ASCEND_SD_AIC: ["int8", "uint8", "float16"],
        AIC_310P: ["int8", "uint8", "float16"],
        AIC_610: ["int8", "uint8", "float16"],
        VEC_310P: ["int8", "uint8", "float16"],
        VEC_610: ["int8", "uint8", "float16"],
        AIC_BS9SX1A: ["int8", "uint8", "float16"],
        AIC_610B: ["int8", "uint8", "float16"]
    },
    "load_smask": {
        ASCEND_310AIC: [],
        ASCEND_910AIC: [],
        ASCEND_910BAIC: [],
        ASCEND_910BVEC: [],
        ASCEND_910_93AIC: [],
        ASCEND_910_93VEC: [],
        HI3796CV300ESAIC: [],
        HI3796CV300CSAIC: [],
        ASCEND_SD_AIC: [],
        AIC_310P: [],
        AIC_610: [],
        VEC_310P: [],
        VEC_610: [],
        AIC_BS9SX1A: [],
        AIC_610B: []
    },
    "load_image": {
        ASCEND_310AIC: [],
        ASCEND_910AIC: [],
        ASCEND_910BAIC: [],
        ASCEND_910BVEC: [],
        ASCEND_910_93AIC: [],
        ASCEND_910_93VEC: [],
        HI3796CV300ESAIC: [],
        AIC_310P: [],
        AIC_610: [],
        AIC_BS9SX1A: [],
        AIC_610B: []
    },
    "mov_vmrgsort4_sr_to_scalar": {
        ASCEND_310AIC: [],
        ASCEND_910AIC: [],
        ASCEND_910BAIC: [],
        ASCEND_910BVEC: [],
        ASCEND_910_93AIC: [],
        ASCEND_910_93VEC: [],
        HI3796CV300ESAIC: [],
        HI3796CV300CSAIC: [],
        ASCEND_SD_AIC: [],
        AIC_310P: [],
        AIC_610: [],
        VEC_310P: [],
        VEC_610: [],
        AIC_BS9SX1A: [],
        AIC_610B: []
    },
    "set_rpn_cor_ir": {
        ASCEND_310AIC: [],
        ASCEND_910AIC: [],
        ASCEND_910BAIC: [],
        ASCEND_910BVEC: [],
        ASCEND_910_93AIC: [],
        ASCEND_910_93VEC: [],
        HI3796CV300ESAIC: [],
        HI3796CV300CSAIC: [],
        ASCEND_SD_AIC: [],
        AIC_310P: [],
        AIC_610: [],
        VEC_310P: [],
        VEC_610: [],
        AIC_BS9SX1A: [],
        AIC_610B: []
    },
    "set_rpn_offset": {
        ASCEND_310AIC: [],
        ASCEND_910AIC: [],
        ASCEND_910BAIC: [],
        ASCEND_910BVEC: [],
        ASCEND_910_93AIC: [],
        ASCEND_910_93VEC: [],
        HI3796CV300ESAIC: [],
        HI3796CV300CSAIC: [],
        ASCEND_SD_AIC: [],
        AIC_310P: [],
        AIC_610: [],
        VEC_310P: [],
        VEC_610: [],
        AIC_BS9SX1A: [],
        AIC_610B: []
    },
    "mov_atomic_add_to_scalar": {
        ASCEND_910AIC: [],
        ASCEND_910BAIC: [],
        ASCEND_910BVEC: [],
        ASCEND_910_93AIC: [],
        ASCEND_910_93VEC: [],
        AIC_310P: [],
        AIC_610: [],
        VEC_310P: [],
        VEC_610: [],
        HI3796CV300ESAIC: [],
        HI3796CV300CSAIC: [],
        ASCEND_SD_AIC: [],
        AIC_BS9SX1A: [],
        AIC_610B: []
    },
    "mov_small_channel_to_scalar": {
        ASCEND_910AIC: [],
        ASCEND_910BAIC: [],
        ASCEND_910BVEC: [],
        ASCEND_910_93AIC: [],
        ASCEND_910_93VEC: [],
        AIC_310P: [],
        AIC_610: [],
        VEC_310P: [],
        VEC_610: [],
        HI3796CV300ESAIC: [],
        HI3796CV300CSAIC: [],
        ASCEND_SD_AIC: [],
        AIC_BS9SX1A: [],
        AIC_610B: []
    },
    "mov_system_cache_to_scalar": {
        ASCEND_910AIC: [],
        ASCEND_910BAIC: [],
        ASCEND_910BVEC: [],
        ASCEND_910_93AIC: [],
        ASCEND_910_93VEC: [],
        AIC_310P: [],
        AIC_610: [],
        VEC_310P: [],
        VEC_610: [],
        HI3796CV300ESAIC: [],
        HI3796CV300CSAIC: [],
        ASCEND_SD_AIC: [],
        AIC_BS9SX1A: [],
        AIC_610B: []
    },
    "mov_fp2int_mode_to_scalar": {
        ASCEND_910AIC: [],
        ASCEND_910BAIC: [],
        ASCEND_910BVEC: [],
        ASCEND_910_93AIC: [],
        ASCEND_910_93VEC: [],
        AIC_310P: [],
        AIC_610: [],
        VEC_310P: [],
        VEC_610: [],
        HI3796CV300ESAIC: [],
        HI3796CV300CSAIC: [],
        ASCEND_SD_AIC: [],
        AIC_BS9SX1A: [],
        AIC_610B: []
    },
    "set_atomic_add": {
        ASCEND_910AIC: [],
        ASCEND_910BAIC: [],
        ASCEND_910BVEC: [],
        ASCEND_910_93AIC: [],
        ASCEND_910_93VEC: [],
        AIC_310P: [],
        AIC_610: [],
        VEC_310P: [],
        VEC_610: [],
        ASCEND_SD_AIC: [],
        VEC_BS9SX1A: [],
        AIC_BS9SX1A: [],
        AIC_610B: []
    },
    "set_small_channel": {
        ASCEND_910AIC: [],
        ASCEND_910BAIC: [],
        ASCEND_910BVEC: [],
        ASCEND_910_93AIC: [],
        ASCEND_910_93VEC: [],
        AIC_310P: [],
        AIC_610: [],
        VEC_310P: [],
        VEC_610: [],
        HI3796CV300ESAIC: [],
        HI3796CV300CSAIC: [],
        ASCEND_SD_AIC: [],
        AIC_BS9SX1A: [],
        AIC_610B: []
    },
    "set_system_cache": {
        ASCEND_910AIC: [],
        ASCEND_910BAIC: [],
        ASCEND_910BVEC: [],
        ASCEND_910_93AIC: [],
        ASCEND_910_93VEC: [],
        AIC_310P: [],
        AIC_610: [],
        VEC_310P: [],
        VEC_610: [],
        HI3796CV300ESAIC: [],
        HI3796CV300CSAIC: [],
        ASCEND_SD_AIC: [],
        AIC_BS9SX1A: [],
        AIC_610B: []
    },
    "set_fp2int_mode": {
        ASCEND_910AIC: [],
        ASCEND_910BAIC: [],
        ASCEND_910BVEC: [],
        ASCEND_910_93AIC: [],
        ASCEND_910_93VEC: [],
        AIC_310P: [],
        AIC_610: [],
        VEC_310P: [],
        VEC_610: [],
        HI3796CV300ESAIC: [],
        HI3796CV300CSAIC: [],
        ASCEND_SD_AIC: [],
        AIC_BS9SX1A: [],
        AIC_610B: []
    },
    "instr_preload": {
        ASCEND_910AIC: [],
        ASCEND_910BAIC: [],
        ASCEND_910BVEC: [],
        ASCEND_910_93AIC: [],
        ASCEND_910_93VEC: [],
        HI3796CV300ESAIC: [],
        HI3796CV300CSAIC: [],
        ASCEND_SD_AIC: [],
        AIC_310P: [],
        AIC_610: [],
        VEC_310P: [],
        VEC_610: [],
        AIC_BS9SX1A: [],
        AIC_610B: []
    },
    "get_overflow_status": {
        ASCEND_910AIC: [],
        ASCEND_910BAIC: [],
        ASCEND_910BVEC: [],
        ASCEND_910_93AIC: [],
        ASCEND_910_93VEC: [],
    },
    "set_overflow_status": {
        ASCEND_910AIC: [],
        ASCEND_910BAIC: [],
        ASCEND_910BVEC: [],
        ASCEND_910_93AIC: [],
        ASCEND_910_93VEC: [],
    },
    "mov_cmpmask_to_tensor": {
        ASCEND_310AIC: [],
        ASCEND_910AIC: [],
        ASCEND_910BAIC: [],
        ASCEND_910BVEC: [],
        ASCEND_910_93AIC: [],
        ASCEND_910_93VEC: [],
        HI3796CV300ESAIC: [],
        HI3796CV300CSAIC: [],
        ASCEND_SD_AIC: [],
        AIC_310P: [],
        AIC_610: [],
        VEC_310P: [],
        VEC_610: [],
        AIC_BS9SX1A: [],
        AIC_610B: []
    },
    "mov_tensor_to_cmpmask": {
        ASCEND_310AIC: [],
        ASCEND_910AIC: [],
        ASCEND_910BAIC: [],
        ASCEND_910BVEC: [],
        ASCEND_910_93AIC: [],
        ASCEND_910_93VEC: [],
        HI3796CV300ESAIC: [],
        HI3796CV300CSAIC: [],
        ASCEND_SD_AIC: [],
        AIC_310P: [],
        AIC_610: [],
        VEC_310P: [],
        VEC_610: [],
        AIC_BS9SX1A: [],
        AIC_610B: []
    },
    "scatter_vmulva": {
        ASCEND_310AIC: [],
        ASCEND_910AIC: [],
        ASCEND_910BAIC: [],
        ASCEND_910BVEC: [],
        ASCEND_910_93AIC: [],
        ASCEND_910_93VEC: []
    },
    "scatter_vaddva": {
        ASCEND_310AIC: [],
        ASCEND_910AIC: [],
        ASCEND_910BAIC: [],
        ASCEND_910BVEC: [],
        ASCEND_910_93AIC: [],
        ASCEND_910_93VEC: []
    },
    "vec_rsqrt_high_preci": {
        ASCEND_310AIC: ["float16", "float32"],
        ASCEND_910AIC: ["float16", "float32"],
        ASCEND_910BAIC: ["float16", "float32"],
        ASCEND_910BVEC: ["float16", "float32"],
        ASCEND_910_93AIC: ["float16", "float32"],
        ASCEND_910_93VEC: ["float16", "float32"],
        AIC_310P: ["float16", "float32"],
        AIC_610: ["float16", "float32"],
        VEC_310P: ["float16", "float32"],
        VEC_610: ["float16", "float32"],
        AIC_BS9SX1A: ["float16", "float32"],
        AIC_610B: ["float16", "float32"]
    },
    "vec_ln_high_preci": {
        ASCEND_310AIC: ["float16"]
    },
    "vec_expm1_high_preci": {
        ASCEND_310AIC: ["float16"],
        ASCEND_910AIC: ["float16"],
        ASCEND_910BAIC: ["float16"],
        ASCEND_910BVEC: ["float16"],
        ASCEND_910_93AIC: ["float16"],
        ASCEND_910_93VEC: ["float16"],
        AIC_310P: ["float16"],
        AIC_610: ["float16"],
        VEC_310P: ["float16"],
        VEC_610: ["float16"],
        AIC_BS9SX1A: ["float16"],
        AIC_610B: ["float16"]
    },
    "vec_rec_high_preci": {
        ASCEND_310AIC: ["float16", "float32"],
        ASCEND_910AIC: ["float16", "float32"],
        ASCEND_910BAIC: ["float16", "float32"],
        ASCEND_910BVEC: ["float16", "float32"],
        ASCEND_910_93AIC: ["float16", "float32"],
        ASCEND_910_93VEC: ["float16", "float32"],
        AIC_310P: ["float16", "float32"],
        AIC_610: ["float16", "float32"],
        VEC_310P: ["float16", "float32"],
        VEC_610: ["float16", "float32"],
        AIC_BS9SX1A: ["float16", "float32"],
        AIC_610B: ["float16", "float32"]
    },
    "conv2d": {
        ASCEND_310AIC: ["s8s8s32", "u8s8s32", "f16f16f32"],
        ASCEND_910AIC: ["s8s8s32", "u8s8s32", "f16f16f32"],
        ASCEND_910BAIC: ["s8s8s32", "u8s8s32", "f16f16f32"],
        ASCEND_910BVEC: ["s8s8s32", "u8s8s32", "f16f16f32"],
        ASCEND_910_93AIC: ["s8s8s32", "u8s8s32", "f16f16f32"],
        ASCEND_910_93VEC: ["s8s8s32", "u8s8s32", "f16f16f32"],
        HI3796CV300CSAIC: ["s8s8s32", "u8s8s32"],
        ASCEND_SD_AIC: ["s8s8s32", "u8s8s32"],
        AIC_310P: ["s8s8s32", "u8s8s32", "f16f16f32"],
        AIC_610: ["s8s8s32", "u8s8s32", "f16f16f32"],
        AIC_BS9SX1A: ["s8s8s32", "u8s8s32", "f16f16f32"],
        AIC_610B: ["s8s8s32", "u8s8s32", "f16f16f32"]
    },
    "fixpipe": {
        ASCEND_310AIC: ["f32f16", "f32f32", "s32s32", "s32f16"],
        ASCEND_910AIC: ["f32f16", "f32f32", "s32s32", "s32f16"],
        ASCEND_910BAIC: ["f32f16", "f32f32", "s32s32", "s32f16"],
        ASCEND_910BVEC: ["f32f16", "f32f32", "s32s32", "s32f16"],
        ASCEND_910_93AIC: ["f32f16", "f32f32", "s32s32", "s32f16"],
        ASCEND_910_93VEC: ["f32f16", "f32f32", "s32s32", "s32f16"],
        HI3796CV300CSAIC: ["s32s32", "s32f16"],
        ASCEND_SD_AIC: ["s32s32", "s32f16"],
        AIC_310P: ["f32f16", "f32f32", "s32s32", "s32f16"],
        AIC_610: ["f32f16", "f32f32", "s32s32", "s32f16"],
        AIC_BS9SX1A: ["f32f16", "f32f32", "s32s32", "s32f16"],
        AIC_610B: ["f32f16", "f32f32", "s32s32", "s32f16"]
    },
    "matmul": {
        ASCEND_310AIC: ["s8s8s32", "u8s8s32", "f16f16f32"],
        ASCEND_910AIC: ["s8s8s32", "u8s8s32", "f16f16f32"],
        ASCEND_910BAIC: ["s8s8s32", "u8s8s32", "f16f16f32"],
        ASCEND_910BVEC: ["s8s8s32", "u8s8s32", "f16f16f32"],
        ASCEND_910_93AIC: ["s8s8s32", "u8s8s32", "f16f16f32"],
        ASCEND_910_93VEC: ["s8s8s32", "u8s8s32", "f16f16f32"],
        HI3796CV300CSAIC: ["s8s8s32", "u8s8s32"],
        ASCEND_SD_AIC: ["s8s8s32", "u8s8s32"],
        AIC_310P: ["s8s8s32", "u8s8s32", "f16f16f32"],
        AIC_610: ["s8s8s32", "u8s8s32", "f16f16f32"],
        AIC_BS9SX1A: ["s8s8s32", "u8s8s32", "f16f16f32"],
        AIC_610B: ["s8s8s32", "u8s8s32", "f16f16f32"]
    },
}

TRANS_TIK_API_TO_INSTR_MAP = {
    """
    use to save some api that is instr
    """
    "vcmpv_lt_v1": "vcmpv_lt",
    "vcmpv_gt_v1": "vcmpv_gt",
    "vcmpv_ge_v1": "vcmpv_ge",
    "vcmpv_eq_v1": "vcmpv_eq",
    "vcmpv_ne_v1": "vcmpv_ne",
    "vcmpv_le_v1": "vcmpv_le",
    "v_cpadd": "vcpadd",
    "scalar_sqrt": "sqrt",
    "scalar_abs": "abs",
    "scalar_countbit0": "bcnt0",
    "scalar_countbit1": "bcnt1",
    "scalar_countleading0": "clz",
    "scalar_max": "max",
    "scalar_min": "min",
    "v_add": "vadd",
    "v_sub": "vsub",
    "v_max": "vmax",
    "v_min": "vmin",
    "v_mul": "vmul",
    "v_and": "vand",
    "v_or": "vor",
    "v_relu": "vrelu",
    "v_abs": "vabs",
    "v_not": "vnot",
    "v_axpy": "vaxpy",
    "v_adds": "vadds",
    "vmuls": "vmuls",
    "v_sel": "vsel",
    "v_conv": "vconv",
    "vcmp_lt": "vcmp",
    "vcmp_gt": "vcmp",
    "vcmp_ge": "vcmp",
    "vcmp_eq": "vcmp",
    "vcmp_ne": "vcmp",
    "vcmp_le": "vcmp",
    "broadcast_ub_to_l0c": "broadcast_ub_to_cc",
    "vmrgch": "vmergech",
    "vrpsort16": "vbitsort",
    "scatter_vcmp_lt": "scatter_vcmp",
    "scatter_vcmp_gt": "scatter_vcmp",
    "scatter_vcmp_ge": "scatter_vcmp",
    "scatter_vcmp_eq": "scatter_vcmp",
    "scatter_vcmp_ne": "scatter_vcmp",
    "scatter_vcmp_le": "scatter_vcmp",
    "vreducemax": "vcmax",
    "vreducemin": "vcmin",
    "vreduceadd": "vcadd"
}

# split strategy
guard_with_if = "guard_with_if"
shift_inwards = "shift_inwards"
round_up = "round_up"
