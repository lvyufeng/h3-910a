#!/usr/bin/env python
# -*- coding:utf-8 -*-
"""
Copyright (C) 2022. Huawei Technologies Co., Ltd. All rights reserved.
External interfaces of the cann knowledege bank manager.
"""
import sys

from tbe.common.platform.platform_info import get_soc_spec
from tbe.common.repository_manager.route import RouteServer
from tbe.common.repository_manager.route import public_interact
from tbe.common.repository_manager.route import public_interact_async
from tbe.common.repository_manager.route import public_interact_get
from tbe.common.repository_manager.route import validate_cannkb_entry
from tbe.common.repository_manager.utils.common import generate_unique
from tbe.common.repository_manager.utils.define import RepoMgrRetStatus


__all__ = ["cann_kb_init", "cann_kb_write", "cann_kb_search", "cann_kb_search_async", "cann_kb_search_get",
    "cann_kb_finalize", "cann_kb_delete", "cann_kb_sigdeal"]


def cann_kb_init(sys_config: dict, load_config: dict = {}, option: dict = {}) -> RepoMgrRetStatus:
    """initialize cann knowledge bank service

    Parameters:
    sys_config(dict): System Configuration For Software Such As soc_version and core_num
    load_config(dict): Additional Load Knowledge Bank Configuration Such As kb_path
    option(dict): Reserved Configuration Item

    Return:
    RepoMgrRetStatus
    """
    if not validate_cannkb_entry():
        return RepoMgrRetStatus.SUCCESS
    return RouteServer.initialize(**locals())


def cann_kb_finalize() -> None:
    """Finalize Cann Knowledge Bank Service

    Parameters: None
    Return: None
    """
    if not validate_cannkb_entry():
        return
    RouteServer.finalize()


def cann_kb_search(info_dict: str, search_config: dict, option: dict = {}) -> list:
    """Cann Knowledge Search

    Parameters:
    info_dict(str): Operator Info To Search Knowledge
    search_config(dict): Additional Configuration Items for Knowledge Search
    option(dict): Reserved Configuration Item

    Return: Knowledge List
    """
    if not validate_cannkb_entry():
        return []
    search_config["core_num"] = str(get_soc_spec("CORE_NUM"))
    return public_interact(sys._getframe().f_code.co_name, locals(), generate_unique())


def cann_kb_search_async(info_dict: str, search_config: dict, option: dict = {}) -> str:
    """Async Cann Knowledge Search
    Parameters:
    info_dict(str): Operator Info To Search Knowledge
    search_config(dict): Additional Configuration Items for Knowledge Search
    option(dict): Reserved Configuration Item
    Return: unique key to search knowledge item
    """
    if not validate_cannkb_entry():
        return ""
    search_config["core_num"] = str(get_soc_spec("CORE_NUM"))
    kw = locals()
    key = generate_unique()
    if not public_interact_async("cann_kb_search", kw, key):
        return ""
    return key


def cann_kb_search_get(key: str) -> list:
    """Get async Cann Knowledge Search result
    Parameters:
    info_dict(str): Operator Info To Search Knowledge
    search_config(dict): Additional Configuration Items for Knowledge Search
    option(dict): Reserved Configuration Item
    Return: Knowledge List
    """
    if not validate_cannkb_entry():
        return []
    return public_interact_get(key)


def cann_kb_write(info_dict: str,
    knowledge: str,
    write_config: dict,
    flush: bool = False,
    option: dict = {}) -> RepoMgrRetStatus:
    """Cann Knowledge write(include add and replace)

    Parameters:
    info_dict(str): Operator Info ToSearch Knowledge
    knowledege(str): Operator Knowledge, specially, dict is allowed;
    write_config(dict): Knowledge Write Config
    flush(bool): force flush knowledege to file
    option(dict): Reserved Configuration Item

    Return:
    RepoMgrRetStatus
    """
    if not validate_cannkb_entry():
        return RepoMgrRetStatus.SUCCESS
    write_config["core_num"] = str(get_soc_spec("CORE_NUM"))
    return public_interact(sys._getframe().f_code.co_name, locals(), generate_unique())


def cann_kb_delete(info_dict: str,
    delete_config: dict,
    flush: bool = False,
    option: dict = {}) -> RepoMgrRetStatus:
    """Cann Knowledge delete custom repo

    Parameters:
    info_dict(str): Operator Info ToSearch Knowledge
    knowledege(str): Operator Knowledge
    delete_config(dict): Knowledge delete Config
    flush(bool): force flush knowledege to file
    option(dict): Reserved Configuration Item

    Return:
    RepoMgrRetStatus
    """
    if not validate_cannkb_entry():
        return RepoMgrRetStatus.SUCCESS
    delete_config["core_num"] = str(get_soc_spec("CORE_NUM"))
    return public_interact(sys._getframe().f_code.co_name, locals(), generate_unique())


def cann_kb_sigdeal(signum: int) -> None:
    """cann knowledge bank service deal with signal

    Parameters:
    signum(int): signal number

    Return:
    None
    """
    if not validate_cannkb_entry():
        return
    RouteServer.sig_deal(signum)