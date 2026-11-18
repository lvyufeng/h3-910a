#!/usr/bin/env python
# -*- coding:utf-8 -*-
"""
Copyright (C) 2019. Huawei Technologies Co., Ltd. All rights reserved.
FILE:     tik_params.py
DESC:     provide params
CREATED:  2019-04-18 18:53:42
MODIFIED: 2020-12-7 19:17:00
"""
import numpy as np
from tbe import tvm
from tbe.tik.tik_lib.tik_soc_manager import TikSocManager
from tbe.tik.common.tik_get_soc_name import get_soc_name
from tbe.tik.common.tik_get_soc_name import get_soc_core_type

# for lru_cache parameter maxsize
MIN_LRU_CACHE_SIZE = 128
MAX_LRU_CACHE_SIZE = 1024


MAX_INT32_VALUE = 2**31 - 1
MIN_INT32_VALUE = -2**31
MAX_INT64_VALUE = 2**63 - 1
MIN_INT64_VALUE = -2**63

# def the cce thread axis for sync
CCE_GLOBAL_AXIS = tvm.thread_axis("cce_global")
CCE_MASK_AXIS = tvm.thread_axis("cce_mask")

CMPMASK_VAR = tvm.var("CMPMASK")
FMATRIX_VAR = tvm.var("FMATRIX")
FCOL2IMG_VAR = tvm.var("FCOL2IMG")
MASK_VAR = tvm.var("MASK")
DEQ_VAR = tvm.var("DEQ")
RPN_COR_IR = tvm.var("RPN_COR_IR")
RPN_OFFSET = tvm.var("RPN_OFFSET")
VMS4_SR = tvm.var("VMS4_SR")
CTRL_SPR = tvm.var("CTRL_SPR")
PADDING = tvm.var("PADDING")
RSVD_CNT = tvm.var("RSVD_CNT")
MAX_MIN_CNT = tvm.var("MAX_MIN_CNT")
L0_SET_VALUE = tvm.var("L0_SET_VALUE")
VA_REG = [tvm.var("VA0"), tvm.var("VA1"), tvm.var("VA2"), tvm.var("VA3"),
          tvm.var("VA4"), tvm.var("VA5"), tvm.var("VA6"), tvm.var("VA7")]

# version string of tik1.5
TIK_VERSION_1_5 = "1.5"

PRINT_SCOPE_MAP = {
    "local.L1": "scope_cbuf",
    "local.FB": "scope_fb",
    "local.UB": "scope_ubuf",
    "local.L0A": "scope_ca",
    "local.L0B": "scope_cb",
    "local.L0C": "scope_cc",
    "global": "scope_gm"
}

# get gm_name map
GM_NAME_MAP_CLASS = {}

AI_CORE_INDICATE = "aicore arch: "

PYTHON_VERSION3 = 3
PYTHON_VERSION_IDX = 0
GROUP_COPROC_SCOPE_VALUE = 1

# default 0 for strides
DEFAULT_STRIDE = 0

MAX_BLK_STRIDE_DOUBLE_BYTE = 65535
MAX_BLK_STRIDE_SINGLE_BYTE = 255
MAX_REP_STRIDE_DOUBLE_BYTE = 65535
MAX_REP_STRIDE_SINGLE_BYTE = 255
MAX_REP_STRIDE_12_BITS = 4095
MIN_REPEAT_TIMES = 1
MAX_REPEAT_TIMES = 255
MAX_REPEAT_TIMES_DOUBLE_BYTE = 65535
MAX_MATRIX = 4095
MIN_MATRIX = 0
MAX_N_SMALL_CHANNEL = 5

MIN_CSC_MATRIX = -32768
MAX_CSC_MATRIX = 32767
MAX_CSC_IN_BIAS = 255
MAX_CSC_OUT_BIAS = 255


# strides idx
SRC_BLK_STRIDE_IDX = 0
DST_BLK_STRIDE_IDX = 1

PIPE_S = 1
PIPE_V = 2
PIPE_M = 3
PIPE_MTE1 = 4
PIPE_MTE2 = 5
PIPE_MTE3 = 6
PIPE_MTE4 = 7
PIPE_MTE5 = 8
PIPE_V2 = 9
PIPE_ALL = 10
PIPE_FIX = 11

ONE_BYTE_BIT_LEN = 8
ONE_REP_BYTE_SIZE = 256
ONE_REP_BYTE_SIZE_FOR_NANO = 128
ONE_BLK_SIZE = 32
ONE_BLK_SIZE_FOR_NANO = 16
FRAC_LEN = 16

BLK_NUM_PER_REP = 8

KB_SIZE = 1024

PAD_LENGTH = 4
MAX_PADDING = 255
MAX_TENSOR_WIDTH = 32767
MAX_TENSOR_HEIGHT = 32767
MIN_TENSOR_WIDTH = 1
MIN_TENSOR_HEIGHT = 1
MAX_FETCH_POS = 255
MAX_START_POINT = 32767
MIN_START_POINT = -255
MAX_STRIDE = 63
MIN_STRIDE = 1
MAX_FILTER_WIDTH = 255
MIN_FILTER_WIDTH = 1
MAX_FILTER_HEIGHT = 255
MIN_FILTER_HEIGHT = 1
MAX_DILATION = 255
MIN_DILATION = 1

MAX_VA_ADDR_NUM = 8
VA0_INDEX = 0
VA2_INDEX = 2
VA3_INDEX = 3
VA4_INDEX = 4
ONE_VA_ADDR_NUM = 4
VA_ADDR_BIT_LEN = 16
VA_ADDR_BYTE_SIZE = 32
MAX_ADDR = 2**16 - 1
MAX_ADDR_HEX = 0xffff
MAX_COEFF = 2**5 - 1

MIN_NBURST = 1
MAX_NBURST_DOUBLE_BYTE = 4095
MAX_NBURST_SINGLE_BYTE = 255
MIN_BURST_LEN = 1
MAX_BURST_LEN_DOUBLE_BYTE = 65535
MAX_BURST_LEN_SINGLE_BYTE = 255
MAX_START_INDEX = 65535
MAX_SID = 15
MAX_DST_GAP_DOUBLE_BYTE = 65535
MAX_DST_GAP_SINGLE_BYTE = 255
MAX_SRC_GAP = 255
MIN_BURST_REPEAT = 1
MAX_BURST_REPEAT = 255

HAS_PARAM_CONCAT = 1
NEED_PARAM_CONCAT = 3

SCALAR_SUPPORT_DYTPE = ["uint8", "int8", "uint16", "int16", "float16", "bfloat16",
                        "uint32", "int32", "float32", "uint64", "int64"]
TENSOR_SUPPORT_DYTPE = ["uint8", "int8", "uint16", "int16", "float16", "bfloat16",
                        "uint32", "int32", "float32", "uint64", "int64", "bool"]

MAX_INPUT_OUTPUT_NUM = 192

# v4dtrans
MIN_M_LEN = 1
MAX_M_LEN = 4095
C1_C04 = 4
COUT_B16 = 16
MIN_C1_VALUE = 1
MAX_C1_VALUE = 256
BYTE_PER_C0 = 32
MIN_KH_W_CHANNELS = 1
MAX_KH_W_CHANNELS = 255
MIN_CHANNELS = 1
MAX_CHANNELS = 4096

# vpadding
MAX_PAD_MODE = 3

# vreduce
MIN_SRC1_PATTERN = 1
MAX_SRC1_PATTERN = 6
MAX_SRC1_PATTERN_V2 = 7

# load3dv2
MIN_CHANNEL_SIZE = 1
MAX_CHANNEL_SIZE = 65535
MIN_EXTENSION = 1
MAX_EXTENSION = 65535
MAX_START_PT = 65535
ELE_PER_FRACTAL_EDGE = 16

# load image
MAX_CLIP_NUMBER = 63

# vcmax, vcmin
MAXMIN_CNT_INDEX_LEN_1 = 1
MAXMIN_CNT_INDEX_LEN_3 = 3
FOUR_BYTE_VALUE = 0xffffffff
TWO_BYTE_VALUE = 0xffff
CNT_SHIFT_POS = 32
INDEX_SHIFT_POS = 48

# vconv
VDEQ_BLK_STRIDE = 0
VDEQ_REP_STRIDE = 0


# offset and segment list for concat param
FMATRIX_OFFSET_LIST = [0, 16, 32, 40, 48, 56]
FMATRIX_SEGMENT_LIST = [16, 16, 8, 8, 8, 8]
REG_FCOL2IMG_OFFSET_LIST = [0, 16, 32, 40, 48, 56]
REG_FCOL2IMG_SEGMENT_LIST = [16, 16, 8, 8, 8, 8]
PADDING_ONE_BYTE_OFFSET_LIST = [0, 8]
PADDING_ONE_BYTE_SEGMENT_LIST = [8, 8]
PADDING_TWO_BYTE_OFFSET_LIST = [0]
PADDING_TWO_BYTE_SEGMENT_LIST = [16]
LOAD3DV2_REG_XM_OFFSET_LIST = [0, 16, 32, 48]
LOAD3DV2_REG_XM_SEGMENT_LIST = [16, 16, 16, 16]
LOAD3DV2_REG_XT_OFFSET_LIST = [0, 6, 12, 20, 28, 36, 44, 48]
LOAD3DV2_REG_XT_SEGMENT_LIST = [6, 6, 8, 8, 8, 8, 1, 16]
VSCATTER_VGATHER_XT_SEGMENT_LIST = [32, 8, 8, 2, 8]
VSCATTER_VGATHER_XT_OFFSET_LIST = [0, 32, 40, 54, 56]
TENSOR_PADDING_OFFSET_LIST = [0]
TENSOR_PADDING_SEGMENT_LIST = [8]
OBJECT_SPECIAL_OFFSET_LIST = [56]
OBJECT_SPECIAL_SEGMENT_LIST = [8]
VEXTRACT_OFFSET_LIST = [16, 56]
VEXTRACT_SEGMENT_LIST = [3, 8]
VMRGSORT4_OFFSET_LIST = [0, 8, 20, 32, 44, 59, 60]
VMRGSORT4_SEGMENT_LIST = [8, 12, 12, 12, 12, 1, 4]
VMRGSORT4V2_OFFSET_LIST = [0, 8, 12]
VMRGSORT4V2_SEGMENT_LIST = [8, 4, 1]
VMRGSORT4V2_SRC1_OFFSET_LIST = [0, 16, 32, 48]
VMRGSORT4V2_SRC1_SEGMENT_LIST = [16, 16, 16, 16]
RPN_COR_OFFSET_LIST = [16, 32, 56]
RPN_COR_SEGMENT_LIST = [16, 16, 8]
VECTOR_PAIR_OFFSET_LIST = [0, 16, 32, 56]
VECTOR_PAIR_SEGMENT_LIST = [16, 16, 16, 8]
SIX_STRIDE_OFFSET_LIST = [0, 8, 16, 24, 32, 40, 56]
SIX_STRIDE_SEGMENT_LIST = [8, 8, 8, 8, 8, 8, 8]
FOUR_STRIDE_OFFSET_LIST = [0, 16, 32, 40, 56]
FOUR_STRIDE_SEGMENT_LIST = [16, 16, 8, 8, 8]
THREE_STRIDE_OFFSET_LIST = [0, 16, 32, 56]
THREE_STRIDE_SEGMENT_LIST = [16, 16, 16, 8]
TWO_STRIDE_OFFSET_LIST = [0, 16, 56]
TWO_STRIDE_SEGMENT_LIST = [16, 16, 8]
VREDUCE_OFFSET_LIST = [8, 16, 32, 40, 54, 56]
VREDUCE_SEGMENT_LIST = [8, 8, 8, 8, 2, 8]
V4DTRANS_OFFSET_LIST = [0, 12, 63]
V4DTRANS_SEGMENT_LIST = [12, 12, 1]
VPADDING_OFFSET_LIST = [0, 16, 32, 40, 48, 50, 54, 56]
VPADDING_SEGMENT_LIST = [16, 16, 8, 8, 2, 1, 2, 8]
VSEL_OFFSET_LIST = [0, 8, 16, 24, 32, 40, 48, 56]
VSEL_SEGMENT_LIST = [8, 8, 8, 8, 8, 8, 2, 8]
VEC_SCALAR_OFFSET_LIST = [56, 0, 16, 32, 40, 54]
VEC_SCALAR_SEGMENT_LIST = [8, 16, 16, 8, 8, 2]
VDP_OFFSET_LIST = [0, 16, 32, 48, 61, 62, 63]
PNT_COE_OFFSET_LIST = [0, 32]
PNT_COE_SEGMENT_LIST = [16, 16]

DEQSCALE_SHIFT_POS = 48
SCALE_SHIFT_POS = 47
SCALE_ADDR_BIT_POS = 15
DEQSCALE_46BIT_SHIFT_POS = 46
DEQSCALE_46BIT_MASK = 0x0000400000000000

MAX_C1_INDEX = 4095
MAX_JUMP_OFFSET = 127
MIN_JUMP_OFFSET = 1

DST_TYPE_LEN = 16
DST_TYPE_LEN_MAX = 32
MAX_REPEAT_MODE = 1
STRIDES_LEN = 2
PADMODE_NO_PADDING = 0
MAX_PADMODE = 5
PAD_MASK = 0xff
PADDING_SHIFT_POS = 8
PADDING_LEFT_IDX = 0
PADDING_RIGHT_IDX = 1
PADDING_TOP_IDX = 2
PADDING_BOT_IDX = 3

DEFAULT_VALUE_INDEX = 0
LEN_SHAPE_ONE = 1

MAX_MODE_NUMBER_VEXTRACT_V100 = 3
MIN_MODE_NUMBER = 0
MAX_MODE_NUMBER = 5
ELEMENTS_MULTIPLE = 16
ELEMENTS_MULTIPLE_EIGHT = 16*8
SRC_LIST_LEN = 4
MAX_ELEMENTS_LEN = 4095
MAX_ELEMENTS_LEN_V2 = 65535
VALID_BIT_TUPLE = (15, 7, 3)
VALID_BIT_TUPLE_V200 = (15, 7, 3, 1)
VMS4_SR_ARRAY_LEN = 4
VMS4_REG_BIT_ALL_ONE = 2**13 - 1
VMS4_REGION_LIST0_POS = 39
VMS4_REGION_LIST1_POS = 26
VMS4_REGION_LIST2_POS = 13
VMS4V2_REG_BIT_ALL_ONE = 2**16 - 1
VMS4V2_REGION_LIST_POS = 16
SCALAR_LIST_LEN = 4
MAX_NUMBER = 65535
MIN_STRIDE_UNIT = 0
MAX_STRIDE_UNIT = 3

SCALAR_EXTENTS = (1,)

MASK_LEN_CONTINOUS_MODE = 1
MASK_LEN_FULL_MODE = 2
MASK_LEN_B8_MODE = 4
MASK_LEN_64 = 64
MASK_LEN_128 = 128
MASK_LOW_SHIFT = 32
MASK_VALUE_ZERO = 0
SCALAR_MASK_VALUE_IDX = 0
MASK_VALUE_64 = 64
MASK_VALUE_128 = 128
MASK_HIGH_IDX = 0
MASK_LOW_IDX = 1
MAX_MASK_LOW_VALUE = 2**32 - 1
MIN_MASK = 1
MAX_MASK = 128
MAX_MASK_32 = 32
MAX_MASK_64 = 64
MAX_MASK_256 = 256
MIN_MASK_HALF = 0
MAX_MASK_HALF = 2**64 - 1
MAX_MASK_HALF_INT64 = 2**32-1
MAX_COUNTER_MASK = 2**32 - 1

OVERFLOW_WARNING_SHIFT_POS = 5
MAX_ATOMIC_ADD_MODE = 3
MAX_SMALL_CHANNEL_MODE = 2
MAX_FP2INT_MODE = 2
MAX_FP2FP_MODE = 2
SMALL_CHANNEL_ENABLE_SHIFT_POS = 63
FP2INT_SHIFT_POS = 59
FP2FP_SHIFT_POS = 48
MAX_SYSTEM_CACHE_MODE = 4
SYSTEM_CACHE_MODE_SHIFT_POS = 57
MAX_TWO_BITS_VALUE = 0b11
MAX_ONE_BIT_VALUE = 0b1
ATOMIC_ADD_MASK = 0xcfffffffffffffff
SMALL_CHANNEL_MASK = 0x7fffffffffffffff
SYSTEM_CACHE_MASK = 0xf9ffffffffffffff
FP2INT_MASK = 0xf7ffffffffffffff
FP2FP_MASK = 0xfffeffffffffffff
MASK_MODE_MASK = 0xfeffffffffffffff
MASK_COUNTER_MODE_ENABLE_SHIFT_POS = 56

BLK_32_LIST = [0xff, 0xff00, 0xff0000, 0xff000000, 0xff00000000,
               0xff0000000000, 0xff000000000000, 0xff00000000000000]
BLK_16_LIST = [0xffff, 0xffff0000, 0xffff00000000, 0xffff000000000000]

INSTR_UNIT = 32
CACHE_LINE_SIZE = 128

INDEX_IN_STOP = 9223372036854775807
INDEX_IN_START = 0

MIN_INDEX = 0
MAX_INDEX = 64
CONST_MASK_VALUE = 0x8000000000000000
MAX_LOW_MASK_LEN = 64

SIX_STRIDE = 6
FOUR_STRIDE = 4
THREE_STRIDE = 3
TWO_STRIDE = 2

MIN_RET = 0
MAX_RET = 2**64 - 1
BIT_LEN_32 = 32
BIT_LEN_8 = 8
BIT_LEN_16 = 16
BIT_LEN_64 = 64

MAX_STORE_MODE = 2
VSEL_PARALLEL_BIT = 2048
VSEL_BLK_PARALLEL_BIT = 256
MAX_XREG_ALLOCATED = 32
MAX_VAREG_ALLOCATED = 8
MAX_VREG_ALLOCATED = 32
MAX_PREG_ALLOCATED = 8
MAX_AREG_ALLOCATED = 8
MAX_UREG_ALLOCATED = 4
MIN_SREG_ALLOCATED = 6
MAX_SREG_ALLOCATED = 59
MAX_WREG_ALLOCATED = 4
MAX_IREG_ALLOCATED = 1

REGISTER_GROUP_NUM = 3
VREG_START_INDEX = 0
PREG_START_INDEX = REGISTER_GROUP_NUM * MAX_VREG_ALLOCATED
WREG_START_INDEX = PREG_START_INDEX + MAX_PREG_ALLOCATED*REGISTER_GROUP_NUM
UREG_START_INDEX = WREG_START_INDEX + MAX_VAREG_ALLOCATED
AREG_START_INDEX = UREG_START_INDEX + MAX_WREG_ALLOCATED
IREG_START_INDEX = AREG_START_INDEX + MAX_UREG_ALLOCATED
VAREG_START_INDEX = IREG_START_INDEX + MAX_IREG_ALLOCATED


INC_MODE = 0
DEC_MODE = 1

MAX_VSEL_MODE = 2
VSEL_MODE_TENSOR_SCALAR = 1
VSEL_MODE_DOUBLE_TENSOR_ONE_IT = 0
VSEL_MODE_DOUBLE_TENSOR_MANY_IT = 2

VSEL_TENSOR_TO_TENSOR_CMP_MODE = 2
VSEL_TENSOR_TO_SCALAR_CMP_MODE = 1

CRMODE_DEQSCALE_DEQ16 = 0b1011
CRMODE_DEQSCALE_VDEQ16 = 0b1010
CONV_F322F16_IS_RELU = 0b0010
CONV_F322F16_NO_RELU = 0b0001
CONV_S322F16_VECTOR_QUANT = 0b0111
CONV_RELU_VECTOR_QUANT = 7
CONV_RELU_QUANT = 3
BYTE_SIZE = 32
SPR_CONFIG_BIT_LEN = 15
CONV_S322F16_QUANT = 0b0011
CONV_F162F32_NO_RELU = 0b0100
NO_CONV_IS_RELU = 0b0101
CONV_L0C16_DEQ = 0b0110
CONV_S322B8_DEQ = 0b1000
VALUE_BI_1001 = 0b1001
VALUE_BI_1010 = 0b1010
VALUE_BI_1011 = 0b1011
VALUE_BI_1100 = 0b1100
VALUE_BI_1101 = 0b1101

ALIGN_TENSOR = 512
ALIGN_SRC = 32
ALIGN_SRC_EVEN = 2
ALIGN_DST = 1024
ALIGNED_ADDR = 32
ALIGNED_ADDR_NANO = 16


REPEAT_MODE_SHIFT_BIT = 52

MMAD_MATRIX_K_POS = 12
MMAD_MATRIX_N_POS = 24
MMAD_EN_WINOGRAD_A_POS = 58
MMAD_EN_WINOGRAD_B_POS = 59
MMAD_EN_WEIGHT_OFFSET_POS = 60
MMAD_EN_SSPARSE_POS = 61
MMAD_L0C_BIT_POS = 63

# object_detect
ADDR_BIT_LEN = 16
EXHAUSTED_SUSPENSION_POS = 59
EXHAUSTED_SUSPENSION_POS_V2 = 12
SRC_LIST_BIT_POS = 60
SRC_LIST_BIT_POS_V2 = 8
LENGTH_BIT_LEN = 12
LENGTH_BIAS = 8

# simd
SRC_BLOCK_STRIDE_SHIFT_POS = 16
STRIDE_UNIT_SHIFT_POS = 54
REPEAT_SHIFT_POS = 56
DST_REPEAT_STRIDE_SHIFT_POS = 32
SRC_REPEAT_STRIDE_SHIFT_POS = 40

#for data_move_multi_nd2nz
ND2NZ_ND_NUM_SHIFT_POS = 4
ND2NZ_ND_N_SHIFT_POS = 16
ND2NZ_ND_D_SHIFT_POS = 32
ND2NZ_SRC_ND_STRIDE_SHIFT_POS = 48
ND2NZ_DST_C0_STRIDE_SHIFT_POS = 16
ND2NZ_DST_N_STRIDE_SHIFT_POS = 32
ND2NZ_DST_ND_STRIDE_SHIFT_POS = 48

# tik_proposal
REPEAT_UNIT_ELEMENTS = 16

UINT64_BIT = 64

# buffer scope
L1_BUFFER = "L1_Buffer"
UB_BUFFER = "Unified_Buffer"
L0A_BUFFER = "L0A_Buffer"
L0B_BUFFER = "L0B_Buffer"
LOC_BUFFER = "L0C_Buffer"
SMASK_BUFFER = "SMASK_Buffer"
FB0_BUFFER = "FB0_Buffer"
BT_BUFFER = "BT_Buffer"

# reg scope
VECTOR_REG = "Vector_Reg"
ADDRESS_REG = "Address_Reg"
ADDRESS_REG_SIZE = 4  # unit is Byte
PREDICATE_REG = "Predicate_Reg"
ALIGNMENT_REG = "Alignment_Reg"
ALIGNMENT_REG_SIZE = 32  # unit is Byte, algnment register real size is 257bits, 32 is 257 // 8
WIDE_REG = "Wide_Reg"

# new SCOPE_CBUF_OUT, actual scope_cc
scope_cbuf_out = "local.L0C"

VNCHWCONV_LIST_LEN = 16
MAX_C_SIZE = 1

#winograd
MIN_L1_H_W_C = 1
MAX_L1_H_W = 65535
MAX_L1_C = 4095
MAX_DST_GAP_WINO = 63
MAX_K_WINO = 4095
MAX_COL_INDIC = 3
WINO_FM_XM_OFFSET_LIST = [0, 16, 32, 48, 54, 56, 59]
WINO_FM_XM_SEGMENT_LIST = [16, 16, 12, 6, 2, 3, 3]
WINO_FM_XT_OFFSET_LIST = [8, 20, 32, 48]
WINO_FM_XT_SEGMENT_LIST = [12, 12, 16, 16]
MAX_REP_DIR = 1
WINO_WGT_OFFSET_LIST = [8, 16, 32, 40, 52, 54, 55, 56]
WINO_WGT_SEGMENT_LIST = [8, 16, 8, 7, 2, 1, 1, 8]

SHIFT_BIT_POS_2 = 2
SHIFT_BIT_POS_3 = 3
SHIFT_BIT_POS_7 = 7
SHIFT_BIT_POS_8 = 8
SHIFT_BIT_POS_11 = 11
SHIFT_BIT_POS_12 = 12
SHIFT_BIT_POS_13 = 13
SHIFT_BIT_POS_16 = 16
SHIFT_BIT_POS_20 = 20
SHIFT_BIT_POS_24 = 24
SHIFT_BIT_POS_32 = 32
SHIFT_BIT_POS_47 = 47
SHIFT_BIT_POS_48 = 48
SHIFT_BIT_POS_52 = 52
SHIFT_BIT_POS_54 = 54
SHIFT_BIT_POS_55 = 55
SHIFT_BIT_POS_56 = 56
SHIFT_BIT_POS_59 = 59
SHIFT_BIT_POS_62 = 62
SHIFT_BIT_POS_63 = 63

# depthwise_conv

BYTE_PER_FRACTAL = 512
LOAD_SMASK_OFFSET_LIST = [11, 0, 7]
LOAD_SMASK_SEGMENT_LIST = [1, 7, 4]
# DATA MAX MIN
INT8_MIN = -128
INT8_MAX = 127
UINT_MIN = 0
UINT8_MAX = 255
INT16_MIN = -32768
INT16_MAX = 32767
UINT16_MAX = 65535

# BIT SEG
BIT_0 = 2
BIT16 = 16
BIT_16 = 655536

# MOV PAD
# padding counts
MIN_PADDING_COUNT = 0
MAX_PADDING_COUNT = 5
# AIV_BURST lenght
MAX_BURST_LENGHT_AIV_B8 = 262145
MAX_BURST_LENGHT_AIV_B16 = 131073
MAX_BURST_LENGHT_AIV_B32 = 65537
# AIVsrc/dst gap
MAX_GAP_AIV_UB = 8192
MAX_GAP_AIV_OUT = 262144

# AIC_BURST lenght
MAX_BURST_LENGHT_AIC_B8 = 393217
MAX_BURST_LENGHT_AIC_B16 = 196603
MAX_BURST_LENGHT_AIC_B32 = 98305
# AIC src/dst gap
MAX_GAP_AIC_UB = 12288
MAX_GAP_AIC_OUT = 393216

INSPECT_RANGE = 7
MIN_START_LINE_NO = 1
CUR_FRAME_IDX = 0

REAL_SCALE_MIN = 1.0
REAL_SCALE_MAX = 256.0

# vtranspose command data unit 512B, data type B16
PER_TRANSPOSE_DATA_SIZE = 256
MIN_VNCHWTRANS_STRIDE = 0
MAX_VNCHWTRANS_STRIDE = 4095
MAX_VNCHWTRANS_REPEAT_TIMES = 4095

VTRANSPOSE_REQUIRED_ELEMENT = 256

# for vec_reduce_max vec_reduce_min vec_reduce_add
MAX_VREDUCE_REPEAT_TIMES = 4095
VREDUCE_MIN_REPEAT_TIMES = 1
VREDUCE_DEFAULT_DST_REP_STRIDE = 1
VREDUCE_DEFAULT_DST_BLK_STRIDE = 1
VREDUCE_DEFAULT_SRC_BLK_STRIDE = 1
VREDUCE_DEFAULT_SRC_REP_STRIDE = 8

# vreduce per rep output count
VREDUCE_PER_REP_OUTPUT = 2

VREDUCE_DST_ALIGN = {
    "vcadd": 1,
    "vcgadd": 16,
    "vcpadd": 32,
    "vcmax": 4,
    "vcmin": 4,
    "vcgmax": 16,
    "vcgmin": 16
}

MIN_ONTHEFLY_MODE = 1
MAX_ONTHEFLY_MODE = 2

CONST_FIVE_THREE = 1.6666667
CONST_ONE_THREE = 0.3333333
CONST_NEG_FOUR_THREE = -1.3333333
# Log value
LOG_FOUR_THREE = 0.28768207
# const value
CONST_NEG_ONE = -1.0000000
CONST_ONE = 1.0000000
CONST_HALF = 0.5000000
CONST_THREE_FOUR = 0.7500000
CONST_ONE_FIVE = 0.2000000
CONST_NEG_ONE_FOUR = -0.2500000
CONST_NEG_HALF = -0.5000000
CONST_ONE_NINE = 0.1111111
CONST_NEG_ONE_EIGHT = -0.1250000
CONST_ONE_SEVEN = 0.1428571
CONST_NEG_ONE_SIX = -0.1666667


def gen_api_check_statement(dtype, intrinsic):
    """
    use to display which chip for api check support
    Parameters
    ----------
    dtype: instr's dtype
    intrinsic: which dtype

    Returns
    -------
    no return
    """
    soc_name = get_soc_name()
    core_type = get_soc_core_type()
    return "current chip: " + soc_name + \
           " " + core_type + \
           " doesn't support %s for Instruction %s" % (dtype, intrinsic)


MAX_PRINT_ARG_LENGTH = 6
MAGIC_WORD_FOR_CORE_HEAD = 0x6844
MAGIC_WORD_FOR_PRINT_HEAD = 0x6845
MAGIC_TL_TAG = 0x111
PRINT_CORE_HEAD_BYTES = 64
EACH_PRINT_HEAD_BYTES = 32
MAX_PRINT_LENGTH = 256
PRINT_TL_BYTES = 32
MAX_PRINT_WORKSPACE_SIZE = 32768
MIN_SINGLE_PRINT_LENGTH = 64
MAX_SINGLE_PRINT_LENGTH = 32768*1024*1024
PRINT_WORKSPACE_DTYPE = "uint8"
DEFAULT_PRINT_UB_BYTES = 288
MAX_PRINT_FORMAT_LENGTH = 128
MAX_FORMAT_NUM = 128
MAX_FLOAT16_VALUE = 65504.0
MIN_FLOAT16_VALUE = -65504.0
FAKE_TENSOR_NAME = "__fake_tensor"

DTYPE_IMM_MIN = {
    "uint8": 0,
    "int8": -2**7,
    "uint16": 0,
    "int16": -2**15,
    "uint32": 0,
    "int32": -2**31,
    "float16": -65504.0,
    "float32": -3.4e+38,
    "float64": -1.8e+308,
    "bfloat16": -3.4e+38
}

DTYPE_IMM_MAX = {
    "uint8": 2**8 - 1,
    "int8": 2**7 - 1,
    "uint16": 2**16 - 1,
    "int16": 2**15 - 1,
    "uint32": 2**32 - 1,
    "int32": 2**31 - 1,
    "int64": 2**63 - 1,
    "uint64": 2**64 - 1,
    "float16": 65504.0,
    "float32": 3.4e+38,
    "float64": -1.8e+308,
    "bfloat16": 3.4e+38
}

DTYPE_REL_TOL = {
    "float16": 1e-5,
    "float32": 1e-9,
    "float64": 1e-16
}

ROUND_MODE_MAP = {
    "": None,
    "none": None,
    "round": "ROUND_R",
    "floor": "ROUND_F",
    "ceil": "ROUND_C",
    "ceiling": "ROUND_C",
    "away-zero": "ROUND_A",
    "to-zero": "ROUND_Z",
    "odd": "ROUND_O"
}

VTRC_ROUND_MODE_MAP = {
    "round": "ROUND_R",
    "floor": "ROUND_F",
    "ceil": "ROUND_C",
    "away-zero": "ROUND_A",
    "to-zero": "ROUND_Z",
}

PAT_MODE_INSTR_CODE = {
    "PAT_ALL": 0,
    "PAT_VL1": 1,
    "PAT_VL2": 2,
    "PAT_VL3": 3,
    "PAT_VL4": 4,
    "PAT_VL8": 5,
    "PAT_VL16": 6,
    "PAT_VL32": 7,
    "PAT_VL64": 8,
    "PAT_VL128": 9,
    "PAT_M3": 10,
    "PAT_M4": 11,
    "PAT_H": 12,
    "PAT_Q": 13,
    "PAT_ALLF": 15,
}

PAT_STR_MODE = {
    "ALL": "PAT_ALL",
    "VL1": "PAT_VL1",
    "VL2": "PAT_VL2",
    "VL3": "PAT_VL3",
    "VL4": "PAT_VL4",
    "VL8": "PAT_VL8",
    "VL16": "PAT_VL16",
    "VL32": "PAT_VL32",
    "VL64": "PAT_VL64",
    "VL128": "PAT_VL128",
    "M3": "PAT_M3",
    "M4": "PAT_M4",
    "HALF": "PAT_H",
    "QUARTER": "PAT_Q",
    "ALLF": "PAT_ALLF"
}

PAT_MODE_MASK = {
    0: "ALLF",
    1: "VL1",
    2: "VL2",
    3: "VL3",
    4: "VL4",
    8: "VL8",
    16: "VL16",
    32: "VL32",
    64: "VL64",
    128: "VL128",
}

PAT_MODE_MASK_B32 = {
    0: "ALLF",
    1: "VL1",
    2: "VL2",
    3: "VL3",
    4: "VL4",
    8: "VL8",
    16: "VL16",
    32: "VL32",
    64: "VL64"
}

VSHR_TYPE_MAP = {
    'uint8': 'int8',
    'uint16': 'int16',
    'int16': 'int16',
    'uint32': 'int32',
    'int32': 'int32'
}

VRND_TYPE_MAP = {
    'int16': 'uint16',
    'int32': 'uint32'
}

VSHR_WIDE_TYPE_MAP = {
    'int8': 'uint8',
    'uint8': 'uint8',
    'uint16': 'uint16',
    'int16': 'uint16',
    'uint32': 'uint32',
    'int32': 'uint32'
}

WMUL_TYPE_MAP = {
    'int8': ['int8'],
    'uint8': ['uint8', 'int8'],
    'uint16': ['uint16', 'int16'],
    'int16': ['int16'],
}

WMUL_TYPE_MAP_WITH_PART_MODE = {
    'int8': ['int16'],
    'uint8': ['int16'],
    'uint16': ['uint32', 'int32'],
    'int16': ['uint32', 'int32'],
}

VL_T_MAP = {
    'bool': 256,
    'uint8': 256,
    'int8': 256,
    'uint16': 128,
    'int16': 128,
    'float16': 128,
    'uint32': 64,
    'int32': 64,
    'float32': 64
}

# winograd_v2
MIN_EXTENSION_WINO_V2 = 0
MAX_EXTENSION_WINO_V2 = 4095
MIN_CHANNEL_SIZE_WINO = 0
MAX_CHANNEL_SIZE_WINO = 65535
WINO_V2_FM_XM_OFFSET_LIST = [0, 16, 32, 56, 59]
WINO_V2_FM_XM_SEGMENT_LIST = [16, 16, 12, 3, 3]
WINO_V2_FM_XT_OFFSET_LIST = [8, 20, 32, 44]
WINO_V2_FM_XT_SEGMENT_LIST = [12, 12, 12, 12]
WINO_V2_WGT_OFFSET_LIST = [0, 16, 32, 48]
WINO_V2_WGT_SEGMENT_LIST = [16, 16, 16, 1]
SHIFT_BIT_POS_44 = 44

# for multi-core multi-process response msg
RUN_CMD_FINISHED = "Run command over!"

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
    "vec_reduce_max": "vcmax",
    "vec_reduce_min": "vcmin",
    "vec_reduce_add": "vcadd"
}

typeMap = {
    "uint8": np.uint8,
    "int8": np.int8,
    "uint16": np.uint16,
    "int16": np.int16,
    "float16": np.float16,
    "bfloat16": np.float32,
    "uint32": np.uint32,
    "int32": np.int32,
    "float32": np.float32,
    "uint64": np.uint64,
    "int64": np.int64

}

FLOAT_TYPES = (float, np.floating)

VECTOR_PRINTF_DTYPE_MAP = {
    "bool": "uint32",
    "int24": "int32",
    "int48": "int64",
}

ATOMIC_ADD_MODE_SHIFT_POS = 60
ATOMIC_ADD_MODE_SHIFT_POS_910B = 6

# for_range thread_num Boundary Value
MAX_BUFFER_NUM = 4


def get_atomic_add_mod_start_pos():
    """
    get_atomic_add_mod_start_pos
    """
    atomic_add_mod_start_pos = ATOMIC_ADD_MODE_SHIFT_POS
    if TikSocManager.is_910b_soc() or TikSocManager.is_310b_610l_soc():
        atomic_add_mod_start_pos = ATOMIC_ADD_MODE_SHIFT_POS_910B

    return atomic_add_mod_start_pos


def get_atomic_add_mod_end_pos():
    """
    get_atomic_add_mod_end_pos
    """
    atomic_add_mod_end_pos = ATOMIC_ADD_MODE_SHIFT_POS + 2
    if TikSocManager.is_910b_soc() or TikSocManager.is_310b_610l_soc():
        atomic_add_mod_end_pos = ATOMIC_ADD_MODE_SHIFT_POS_910B + 3

    return atomic_add_mod_end_pos
