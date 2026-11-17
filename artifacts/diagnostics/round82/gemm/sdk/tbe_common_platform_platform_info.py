#!/usr/bin/env python
# -*- coding:utf-8 -*-
"""
Copyright (C) 2019. Huawei Technologies Co., Ltd. All rights reserved.
CREATED:  2021-01-25 18:53:42
MODIFIED: 2021-01-25 18:53:42
"""
import functools

from tbe.tvm import get_global_func

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
# def the gemm int4/uint4 reduce const
BLOCK_REDUCE_INT4 = 64
# def the gemm float32 reduce const
BLOCK_REDUCE_FP32 = 8
# def the gemv/gevm vector const
BLOCK_VECTOR = 1

INP_ELEM_BYTES = (BLOCK_IN * BLOCK_REDUCE * INP_WIDTH // 8)
WGT_ELEM_BYTES = (BLOCK_OUT * BLOCK_REDUCE * WGT_WIDTH // 8)
OUT_ELEM_BYTES = (BLOCK_IN * BLOCK_OUT * OUT_WIDTH // 8)
GLB_ELEM_BYTES = (16 * OUT_WIDTH // 8)

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
            "bfloat16": {'mac': [16, 16, 16]}}

# def limitation of repeart times for a single copy instrin and align value
VECTOR_COPY_NBURST_LIMIT = 4096
VECTOR_SINGLE_BLOCK_WIDTH_FP16 = 16

# represent 5 soc, currently contains in tik
ASCEND_310 = "Ascend310"
ASCEND_310B = "Ascend310B"
AS31XM1 = "AS31XM1"
ASCEND_031 = "Ascend031"
ASCEND_035 = "Ascend035"
ASCEND_035A = "Ascend035A"
ASCEND_035B = "Ascend035B"
ASCEND_910 = "Ascend910"
ASCEND_910H = "Ascend910B"
ASCEND_910B = "Ascend910B"
ASCEND_910_93 = "Ascend910_93"
ASCEND_910_96 = "Ascend910_96"
ASCEND_9602 = "Ascend9602"
ASCEND_950 = "Ascend950"
ASCEND_350 = "Ascend350"
ASCEND_910M = "Ascend910A"
ASCEND_910P = "Ascend910ProA"
MC62 = "MC62"
HI3796CV300ES = "Hi3796CV300ES"
HI3796CV300CS = "Hi3796CV300CS"
SD3403 = "SD3403"
ASCEND_610 = "Ascend610"
ASCEND_610LITE = "Ascend610Lite"
BS9SX2A = "BS9SX2A"
MC61AM21A = "MC61AM21A"
ASCEND_310P = "Ascend310P"
BS9SX1A = "BS9SX1A"
ASCEND_610B = "Ascend610B"
ASCEND_SD = "SD3403"
_AIC_ENGINE = "AiCore"
_VEC_ENGINE = "VectorCore"
Kirin9020 = "Kirin9020"
Kirinx90 = "KirinX90"
Kirin9030 = "Kirin9030"

AIC_310P = ASCEND_310P + _AIC_ENGINE
VEC_310P = ASCEND_310P + _VEC_ENGINE
AIC_610 = ASCEND_610 + _AIC_ENGINE
VEC_610 = ASCEND_610 + _VEC_ENGINE
VEC_BS9SX1A = BS9SX1A + _VEC_ENGINE
AIC_BS9SX1A = BS9SX1A + _AIC_ENGINE
AIC_610B = ASCEND_610B + _AIC_ENGINE
VEC_610B = ASCEND_610B + _VEC_ENGINE
AIC_310B = ASCEND_310B + _AIC_ENGINE
VEC_310B = ASCEND_310B + _VEC_ENGINE
AIC_AS31XM1 = AS31XM1 + _AIC_ENGINE
VEC_AS31XM1 = AS31XM1 + _VEC_ENGINE
HI3796CV300ESAIC = HI3796CV300ES + _AIC_ENGINE
HI3796CV300CSAIC = HI3796CV300CS + _AIC_ENGINE
SD3403AIC = SD3403 + _AIC_ENGINE
ASCEND_910BAIC = ASCEND_910B + _AIC_ENGINE
ASCEND_910BVEC = ASCEND_910B + _VEC_ENGINE
ASCEND_910_93AIC = ASCEND_910_93 + _AIC_ENGINE
ASCEND_910_93VEC = ASCEND_910_93 + _VEC_ENGINE
ASCEND_SD_AIC = "SD3403" + _AIC_ENGINE

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
scope_ubuf_fusion = "local.UB_Fusion"
scope_smask = "local.SMASK"
scope_fb = "local.FB"
scope_fb0 = "local.FB0"
scope_fb1 = "local.FB1"
scope_fb2 = "local.FB2"
scope_fb3 = "local.FB3"
scope_bt = "local.BT"

dma_copy = "dma_copy"
dma_copy_global = "global"

SOC_VERSION = "SOC_VERSION"
FULL_SOC_VERSION = "FULL_SOC_VERSION"
SHORT_SOC_VERSION = "SHORT_SOC_VERSION"
AICORE_TYPE = "AICORE_TYPE"
CORE_NUM = "CORE_NUM"
CUBE_CORE_CNT = "CUBE_CORE_CNT"
VECTOR_CORE_CNT = "VECTOR_CORE_CNT"
MAX_CORE_NUM = "MAX_CORE_NUM"
UB_SIZE = "UB_SIZE"
L2_SIZE = "L2_SIZE"
L1_SIZE = "L1_SIZE"
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
COMPILER_ARCH = "Compiler_arch"
FB_SIZE = "FB_SIZE"
FB0_SIZE = "FB0_SIZE"
FB1_SIZE = "FB1_SIZE"
FB2_SIZE = "FB2_SIZE"
FB3_SIZE = "FB3_SIZE"
BT_SIZE = "BT_SIZE"
CORE_TYPE_LIST = "CORE_TYPE_LIST"
L0A_LAYOUT_IS_zN = "L0A_LAYOUT_IS_zN"
UB_BLOCK_SIZE = "ubblock_size"


def get_soc_spec(key):
    """
    call global func to get soc spec.

    Parameters
    ----------
    key

    Returns
    -------
    value
    """

    func = get_global_func("cce.get_soc_spec")
    value = func(key)
    if value == "":
        raise RuntimeError("Unsupported Key Value of get_soc_spec(): %s" % key)

    str2int_list = (CORE_NUM, UB_SIZE, L2_SIZE, L1_SIZE, L0A_SIZE, L0B_SIZE, L0C_SIZE, SMASK_SIZE, VREG_SIZE, WREG_SIZE,
                    WIDE_REG_WIDTH, AREG_SIZE, PREG_SIZE, UREG_SIZE, CUBE_VECTOR_SPLIT, FB_SIZE, FB0_SIZE, FB1_SIZE,
                    FB2_SIZE, FB3_SIZE, BT_SIZE, UB_BLOCK_SIZE)

    str2bool_list = (L0A_LAYOUT_IS_zN,)

    if key in str2int_list:
        try:
            value = int(value)
        except Exception:
            raise RuntimeError("return value %s is not 'int' type" % value)
    elif key in (CUBE_SIZE, UNZIP):
        value_str_list = value.split(",")
        value_int_list = []
        for i in value_str_list:
            try:
                value_int_list.append(int(i))
            except Exception:
                raise RuntimeError("return value %s is not 'int' type" % value)
        value = value_int_list
    elif key == CORE_TYPE_LIST:
        value_str_list = value.split(",")
        value = value_str_list
    elif key in str2bool_list:
        try:
            # "0" stands for False, "1" stands for True
            int_value = int(value)
            assert(0 <= int_value <= 1)
            value = int_value == 1
        except Exception as casting_error:
            raise RuntimeError("return value %s cannot be interpret "
                               "as 'bool' type" % value) from casting_error
    return value


# Example: pylint: disable=unused-argument
def set_current_compile_soc_info(soc_version, core_type="AiCore", aicore_num=None, l1_fusion=None):
    """
    set version info

    Parameters
    ----------
    soc_version : str
    -    "Ascend310"/"Ascend910"/"Ascend310P1~P4"/"Ascend610" ...
    core_type : str
    -    "AiCore" or "VectorCore"
    aicore_num: int
    -    example: 32
    l1_fusion: bool
    -    example: True/False

    Returns
    -------
    errmsg : str
    -    error message, 'success' for OK.
    """
    if core_type in (None, ""):
        core_type = "AiCore"
    if aicore_num in [None, "0", 0, ""]:
        aicore_num = ""
    else:
        aicore_num = str(aicore_num)
    if l1_fusion is None:
        l1_fusion = ""
    l1_fusion = str(l1_fusion).lower()
    func = get_global_func("cce.product_init")
    value = func(soc_version, core_type, aicore_num, l1_fusion)
    if value != "success":
        raise RuntimeError("set_current_compile_soc_info() return error.")

    return value


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
    from tbe.dsl.compute.util import dsl_check_support as _dsl_check_support
    from tbe.tik.common.common_util import tik_api_check_support as _deal_tik_api

    if not isinstance(intrinsic, str):
        raise RuntimeError("intrinsic type should be 'str', it is [%s]"
                           % type(intrinsic))
    if not isinstance(dtype, str):
        raise RuntimeError("dtype type should be 'str', it is [%s]"
                           % type(dtype))

    if intrinsic.startswith("tik."):
        return _deal_tik_api(intrinsic, dtype)
    if intrinsic.startswith("te.lang.cce.") or intrinsic.startswith("tbe.dsl."):
        return _dsl_check_support(intrinsic, dtype)

    return False


def get_align_factor(dtype):
    """
    get_align_factor
    """
    # base on the diff data type, get the align_factor
    if dtype in ('int8', 'uint8'):
        align_factor = 32
        dtype_bytes = 1
    elif dtype in ('float16', 'int16', 'uint16'):
        align_factor = 16
        dtype_bytes = 2
    else:
        align_factor = 8
        dtype_bytes = 4
    return align_factor, dtype_bytes


def get_bit_len(dtype):
    """
    calculate bits of dtype of TVM
    Parameters
    ----------
    dtype : string
    -    dtype of TVM

    Returns
    -------
    ret : int
    -    bit length of dtype.
    """

    # logic use vmul vadd vsub vcmp vsel, only support float16, so
    # 1. int8 -> float16 2.  mul vadd vsub vcmp vsel 3. float16 ->int8
    # bit len use float16 length
    from tbe.dsl.instrinsic import cce_intrin
    return cce_intrin.get_bit_len(dtype)


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
    if not isinstance(intrinsic, str):
        raise RuntimeError("intrinsic type should be 'str', it is [%s]"
                           % type(intrinsic))
    if not intrinsic.startswith("Intrinsic_"):
        raise RuntimeError("intrinsic type should start with Intrinsic_")
    if not isinstance(dtype, str):
        raise RuntimeError("dtype type should be 'str', it is [%s]"
                           % type(dtype))
    func = get_global_func("cce.intrinsic_check_support")
    value = func(intrinsic, dtype)
    if value in (-1, ""):
        raise RuntimeError("Unsupported Key Value"
                           " of get_soc_spec(): %s" % intrinsic)
    if value == "True":
        return True
    return False


def register_api_check_support_func(comp):
    """
    the api_check_support register.

    Parameters
    ----------
    comp : str, should be 'TIK' or 'DSL' .
    """
    from .platform_info_ import TIK_API_CHECK_SUPPORT_FUNC_TYPE
    from .platform_info_ import DSL_API_CHECK_SUPPORT_FUNC_TYPE
    from .platform_info_ import api_check_support_func
    if comp not in (TIK_API_CHECK_SUPPORT_FUNC_TYPE, DSL_API_CHECK_SUPPORT_FUNC_TYPE):
        raise RuntimeError("comp should be 'TIK' or 'DSL' .")

    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            return func(*args, **kwargs)

        api_check_support_func[comp] = wrapper
        return wrapper

    return decorator


def getValue(key):
    """
    call global func to get product value

    Parameters
    ----------
    Returns
    -------
    value
    """
    value = None
    if "Buffer" in key:
        func = get_global_func("cce.product_conf_buffer")
        value = func(key)
        if value == -1:
            raise RuntimeError("Unsupported buffer name: %s" %
                               key.split("_Buffer")[0])
        return value

    if "Compiler" in key:
        func = get_global_func("cce.product_conf_compiler")
        value = func(key)
        if value == "":
            raise RuntimeError("Unsupported compiler param: %s" %
                               key.split("Compiler_")[1])
        return value

    if "Intrinsic" in key:
        func = get_global_func("cce.product_conf_intrinsic")
        value = func(key)
        if value == "":
            raise RuntimeError("Unsupported intrinsic: %s" %
                               key.split("Intrinsic_")[1])
        return value

    if "Sid" in key:
        func = get_global_func("cce.product_conf_sid")
        value = func(key)
        if value == "":
            raise RuntimeError("Unsupported sid param: %s" %
                               key.split("Sid_")[1])
        return value

    if "Device" in key:
        func = get_global_func("cce.product_conf_device")
        value = func(key)
        if value == -1:
            raise RuntimeError("Unsupported device param: %s" %
                               key.split("Device_")[1])
        return value

    return value


def get_cube_mkn(dtype):
    soc_version = get_soc_spec("SHORT_SOC_VERSION")
    if (soc_version == "Ascend035" or soc_version == "Ascend035A" or soc_version == "Ascend035B"):
        if dtype == "uint8" or dtype == "int8":
            bytes = 1
        elif dtype == "uint16" or dtype == "int16" or dtype == "float16":
            bytes = 2
        M = int(get_soc_spec("cube_m_size"))
        K = int(get_soc_spec("cube_k_size"))
        N = int(get_soc_spec("cube_n_size"))
        return [M, K // bytes, N]
    return CUBE_MKN[dtype]['mac']


def set_core_num_by_core_type(core_type):
    """
    set core number by core type
    Parameters
    ----------
    core_type: core type

    Returns None
    -------

    """
    if not core_type:
        # default val is aic
        core_type = "0"
    func = get_global_func("cce.set_core_num_by_core_type")
    value = func(core_type)
    if not value:
        raise RuntimeError("set_core_num_by_core_type return error!")


def set_platform_info_res(device_id, res):
    """
    set platform info
    Parameters
    ----------
    device_id: device id
    res: map contains aic num, aiv num, hbm size, l2 size, memory bw

    Returns
    -------

    """
    if device_id not in ("", None) and res:
        func = get_global_func("cce.set_platform_info_res")
        value = func(int(device_id), res)
        if not value:
            raise RuntimeError("set_platform_info_res return error!")


def te_update_version(soc_version=None, core_type=None,
                      aicore_num=None, l1_fusion=None,
                      l2_mode=None, l2_fusion=None, kwargs=None):
    """
    set version info

    Parameters
    ----------
    soc_version : str
    -    "Ascend310"/"Ascend910"/"Ascend310P1~P4"/"Ascend610" ...
    core_type : str
    -    "AiCore" or "VectorCore"
    aicore_num: int
    -    example: 32
    l1_fusion: bool
    -    example: True/False

    Returns
    -------
    errmsg : str
    -    error message, 'success' for OK.
    """
    if soc_version is None:
        soc_version = ""
    if core_type in (None, ""):
        core_type = ""
    if aicore_num in [None, "0", 0, ""]:
        aicore_num = ""
    else:
        aicore_num = str(aicore_num)
    if l1_fusion is None:
        l1_fusion = ""
    else:
        l1_fusion = str(l1_fusion).lower()
    func = get_global_func("cce.te_update_version")
    value = func(soc_version, core_type, aicore_num, l1_fusion)

    from tbe.common.buildcfg import compatible_interface
    if l2_fusion in ("true", True):
        compatible_interface.set_L1_info("L2_fusion_enabled", True)
    elif l2_fusion in ("false", False):
        compatible_interface.set_L1_info("L2_fusion_enabled", False)

    import te.platform.cce_conf as cce_conf
    cce_conf.te_set_op_impl_mode(kwargs)
    cce_conf.set_mdl_bank_path(kwargs)
    cce_conf.set_op_bank_path(kwargs)

    if value != "success":
        raise RuntimeError("te_update_version() return error.")

    return value


def set_soc_spec(key):
    """
    set soc_version or core_type or aicore_num or l1_fusion

    Parameters
    ----------
    key

    Returns
    -------
    errmsg : str
    -    error message, 'success' for OK.
    """
    func = get_global_func("cce.set_soc_spec")
    if key is True:
        key = "true"
    elif key is False:
        key = "false"
    else:
        key = str(key)
    value = func(key)
    if value != "success":
        raise RuntimeError("set_soc_spec() return error.")

    return value


class CceProductParams:
    """
    define Cce Product Params class
    """

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
        if key is None:
            raise RuntimeError("key is miss!")
        value = getValue(key)

        return value

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

    cce_product = None
    build_options = {}


def multicore_sync_shape():
    """
    Multicore Sync Plan: workspace size is AICoreNum * (4 + 32) int64
    4 int64 is for set_block_sync()
    32 int64 is for wait_block_sync()
    """
    workspace_byte = int(get_soc_spec("MAX_CORE_NUM")) * (4 + 32)
    return workspace_byte


class KernelName():
    """
    store kernel name, to across python > c++ > python
    """
    kernel_name = ''

    @classmethod
    def set_kernel_name(cls, kernel_name):
        """
        set kernel name
        """
        cls.kernel_name = kernel_name

    @classmethod
    def get_kernel_name(cls):
        """
        return kernel name set before
        """
        return cls.kernel_name


def get_block_size():
    """
    get the block size of the currently set chip.
    """
    return get_soc_spec(UB_BLOCK_SIZE)
