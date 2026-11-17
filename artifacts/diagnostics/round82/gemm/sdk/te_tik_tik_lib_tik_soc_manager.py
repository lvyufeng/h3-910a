#!/usr/bin/env python
# -*- coding:utf-8 -*-
"""
Copyright (C) 2019. Huawei Technologies Co., Ltd. All rights reserved.
FILE:     tik_soc_manager_.py
DESC:     soc type and soc core type check file for tik
CREATED:  2021-10-20 19:02:50
MODIFIED: 2021-10-20 19:17:00
"""
from tbe.tik.common.tik_api_map import ASCEND_910
from tbe.tik.common.tik_api_map import ASCEND_310
from tbe.tik.common.tik_api_map import ASCEND_610
from tbe.tik.common.tik_api_map import ASCEND_610L
from tbe.tik.common.tik_api_map import BS9SX2A
from tbe.tik.common.tik_api_map import MC61AM21A
from tbe.tik.common.tik_api_map import ASCEND_310P
from tbe.tik.common.tik_api_map import HI3796CV300CS
from tbe.tik.common.tik_api_map import SD3403
from tbe.tik.common.tik_api_map import HI3796CV300ES
from tbe.tik.common.tik_api_map import BS9SX1A
from tbe.tik.common.tik_api_map import ASCEND_910B
from tbe.tik.common.tik_api_map import ASCEND_910_93
from tbe.tik.common.tik_api_map import ASCEND_950
from tbe.tik.common.tik_api_map import ASCEND_310B
from tbe.tik.common.tik_api_map import ASCEND_310M
from tbe.tik.common.tik_api_map import ASCEND_031
from tbe.tik.common.tik_api_map import ASCEND_035
from tbe.tik.common.tik_get_soc_name import get_soc_name
from tbe.tik.common.tik_get_soc_name import get_soc_core_type


class TikSocManager:
    """
    static class for check soc version and soc core type
    """
    @staticmethod
    def is_mini_soc():
        """
        use to check whether current soc is v100
        """
        soc_name = get_soc_name()
        return soc_name == ASCEND_310

    @staticmethod
    def is_cloud_core():
        """
        use to check whether is 910 soc
        """
        soc_name = get_soc_name()
        return soc_name == ASCEND_910

    @staticmethod
    def is_v100_soc():
        """
        use to check whether current soc is v100
        """
        soc_name = get_soc_name()
        return soc_name in (ASCEND_310, ASCEND_910)

    @staticmethod
    def is_v200_soc():
        """
        use to check whether current soc is v200
        """
        soc_name = get_soc_name()
        return soc_name in (ASCEND_610, ASCEND_310P, SD3403, HI3796CV300ES, HI3796CV300CS)

    @staticmethod
    def is_v210_soc():
        """
        use to check whether current soc is v210
        """
        soc_name = get_soc_name()
        return soc_name == BS9SX1A

    @staticmethod
    def is_v210_vec_soc():
        """
        use to check whether current soc is v210
        """
        soc_core_type = get_soc_core_type()
        return TikSocManager.is_v210_soc() and soc_core_type == "VectorCore"

    @staticmethod
    def is_950_soc():
        """
        use to check whether is v300 soc
        """
        soc_name = get_soc_name()
        return soc_name == ASCEND_950

    @staticmethod
    def is_vector_core():
        """
        use to check whether is vector core type
        """
        soc_core_type = get_soc_core_type()
        return soc_core_type == "VectorCore"

    @staticmethod
    def is_aicore_core():
        """
        use to check whether is AiCore core type
        """
        soc_core_type = get_soc_core_type()
        return soc_core_type == "AiCore"

    @staticmethod
    def is_910b_soc():
        """
        check is 910B soc
        :return: True or False
        """
        soc_name = get_soc_name()
        return soc_name in (ASCEND_910B, ASCEND_910_93)

    @staticmethod
    def is_v300_soc():
        """
        use to check whether is v300 soc
        """
        soc_name = get_soc_name()
        return TikSocManager.is_310b_soc() or soc_name == ASCEND_031

    @staticmethod
    def is_310b_soc():
        soc_name = get_soc_name()
        return soc_name in (ASCEND_310B, ASCEND_310M)

    @staticmethod
    def is_v300_vec_soc():
        """
        use to check whether is v300 vec soc
        """
        soc_core_type = get_soc_core_type()
        return TikSocManager.is_310b_soc() and soc_core_type == "VectorCore"

    @staticmethod
    def is_v300_aicore():
        """
        use to check whether is v300 vec soc
        """
        soc_core_type = get_soc_core_type()
        return TikSocManager.is_310b_soc() and soc_core_type == "AiCore"

    @staticmethod
    def is_hisi_sd_cs():
        """
        use to check whether is SD3403 and Hi3796CV300CS
        """
        soc_name = get_soc_name()
        return soc_name in (SD3403, HI3796CV300CS)

    @staticmethod
    def is_hisi_soc():
        """
        use to check whether is SD3403, Hi3796CV300CS or Hi3796CV300ES
        """
        soc_name = get_soc_name()
        return soc_name in (SD3403, HI3796CV300CS, HI3796CV300ES)

    @staticmethod
    def is_610l_soc():
        """
        use to check whether is 610Lite soc
        """
        soc_name = get_soc_name()
        return soc_name in (ASCEND_610L, BS9SX2A, MC61AM21A)

    @staticmethod
    def is_310b_610l_soc():
        """
        use to check whether is 310B or 610Lite soc
        """
        return TikSocManager.is_v300_soc() or TikSocManager.is_610l_soc() or TikSocManager().is_950_soc()

    @staticmethod
    def is_v300_610l_soc():
        """
        use to check whether is v300 or 610Lite soc
        """
        return TikSocManager.is_v300_soc() or TikSocManager.is_610l_soc()

    @staticmethod
    def is_nano_soc():
        """
        use to check whether is nano soc
        """
        soc_name = get_soc_name()
        return soc_name == ASCEND_035
