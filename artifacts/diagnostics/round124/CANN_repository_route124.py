#!/usr/bin/env python
# -*- coding:utf-8 -*-
"""
Copyright (C) 2022. Huawei Technologies Co., Ltd. All rights reserved.

cann knowledge bank server
"""
import os
import time
import pickle
import signal
import getpass
import functools
import threading
import multiprocessing
import logging
from multiprocessing.util import Finalize

from tbe.common.context import op_context
from tbe.common.platform.platform_info import get_soc_spec
from tbe.common.repository_manager.utils.define import RepoMgrRetStatus
from tbe.common.repository_manager.utils.file_sys_util import FileSysUtil
from tbe.common.repository_manager.utils.common import restore_main_info
from tbe.common.repository_manager.utils.common import config_main_info
from tbe.common.repository_manager.utils.common import timer
from tbe.common.repository_manager.utils.common import generate_unique
from tbe.common.repository_manager.utils.common import daemon_process
from tbe.common.repository_manager.utils.common import pid_exists
from tbe.common.repository_manager.utils.common import get_msg_file_dir
from tbe.common.repository_manager.utils.common import get_mac_addr
from tbe.common.repository_manager.utils.multiprocess_util import GlobalMgr
from tbe.common.repository_manager.utils.multiprocess_util import MessageManager
from tbe.common.repository_manager.utils.multiprocess_util import client_get_msg_obj
from tbe.common.repository_manager.utils.repository_manager_log import LOG_INSTANCE
from tbe.common.repository_manager.cann_kb_manager.knowledge_bank_manager import CannKbManager


TASK_QUEUE = "task_queue"
RES_DICT = "res_dict"
RES_LOCK = "res_lock"
SUB_PROCESS_STATE = "sub_process_state"
MSG_FILE_NAME = "cann_kb_" + getpass.getuser() + get_mac_addr()
QRY_INTERVAL = 0.00002
EXIT_PRIORITY = 100
CANN_KB_PROCESS_NUM = 8
CANN_KB_TYPE_TOTAL = 8
SUB_PROCESS_STATE_FLAG = False


def validate_cannkb_entry() -> bool:
    env_str = os.getenv("CANN_KNOWLEDGE_BANK_PROCESS_NUM", "8")
    if env_str.isdigit() and int(env_str) == 0:
        LOG_INSTANCE.warn(f"CANN_KNOWLEDGE_BANK_PROCESS_NUM is {env_str}, cannkb will not be enabled.")
        return False
    return True


def reentry_lock(func):
    """RouteServer reentry lock wrapper"""
    @functools.wraps(func)
    def wrapper(cls, *args, **kwargs):
        with getattr(cls, "route_lock"):
            return func(cls, *args, **kwargs)
    return wrapper


def subprocess_troubleshoot(func):
    """Capture all exceptions, output error logs, and re-throw exceptions."""
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        try:
            func(*args, **kwargs)
        except (EOFError, KeyboardInterrupt) as e:
            LOG_INSTANCE.error("Subprocess[%s] raised error[%s]" % (func.__name__, str(e)))
            logging.getLogger("").error(
                f"[ERROR] TBE Subprocess[{func.__name__}] raised error[{str(e)}], main process disappeared!")
            os._exit(1)
        except Exception as exp:
            LOG_INSTANCE.error("Subprocess[%s] raised error[%s]" % (func.__name__, str(exp)))
            raise exp
    return wrapper


class RouteServer:
    """Task Route Server Class

    Description:
    The knowledge base management module requires multi-process query.
    Therefore, the knowledge base service process and
    service process need to serialize multi-process communication objects to transfer handles.
    This class implements the functions of the server.
    """
    # message manager object
    message_mgr = None
    # route server
    route_lock = threading.Lock()
    route_status = threading.Event()
    # daemon process
    daemon_process = None
    daemon_running = None
    # task distribute process list
    bank_process_list = []
    # server and distribute process communicate
    task_queue = None
    result_dict = None
    result_lock = None
    sub_proc_state_list = None
    # Facade object of the knowledge base management module
    cann_kb_mgr = None
    # Global resource management thread object
    global_mgr = None
    # pid
    pid = None

    @classmethod
    @reentry_lock
    def initialize(cls, sys_config: dict, load_config: dict = {}, option: dict = {}) -> RepoMgrRetStatus:
        """initialize RouteServer

        Parameters:
        sys_config(dict): System Configuration For Software Such As soc_version and core_num
        load_config(dict): Additional Load Knowledge Bank Configuration Such As kb_path
        option(dict): Reserved Configuration Item

        Return:
        RepoMgrRetStatus
        """
        if cls.message_mgr is not None:
            LOG_INSTANCE.warn("RouteServer.initialize Repeatedly Invoked")
            return RepoMgrRetStatus.DOUBLEINIT
        msg_file_root_dir = get_msg_file_dir()
        if FileSysUtil.create_folder_recursively(
            msg_file_root_dir, FileSysUtil.DIR_MODE_750) != RepoMgrRetStatus.SUCCESS:
            return RepoMgrRetStatus.FAIL

        deamon_status = multiprocessing.current_process().daemon
        multiprocessing.current_process().daemon = False
        LOG_INSTANCE.info("RouteServer starts initializing.")

        # Initialize the knowledge base management facade object.
        cls.global_mgr = GlobalMgr(daemon=True)
        cls.global_mgr.start()

        # Initialize the knowledge base management facade object.
        try:
            cls.cann_kb_mgr = CannKbManager({
                "task_q": cls.global_mgr.task_q,
                "res_d": cls.global_mgr.res_d,
                "resource_l": cls.global_mgr.resource_l,
                "dict_list": [cls.global_mgr.dict() for _ in range(CANN_KB_TYPE_TOTAL)],
                "task_lock": cls.global_mgr.lock()
                }, sys_config, load_config, option)
        except RuntimeError as e:
            cls.global_mgr.finalize()
            cls.global_mgr = None
            raise e
        finally:
            pass

        cls.task_queue = cls.global_mgr.queue()
        cls.result_dict = cls.global_mgr.dict()
        cls.result_lock = cls.global_mgr.lock()
        cls.daemon_running = cls.global_mgr.event()
        cls.sub_proc_state_list = cls.global_mgr.list()

        resource_proxy = {
            TASK_QUEUE: cls.task_queue,
            RES_DICT: cls.result_dict,
            RES_LOCK: cls.result_lock,
            SUB_PROCESS_STATE: cls.sub_proc_state_list
        }

        file_name = MSG_FILE_NAME + str(os.getpid())
        cls.message_mgr = MessageManager(os.path.join(msg_file_root_dir, file_name))
        if cls.message_mgr.msg_mgr_dump(resource_proxy, file_name) != RepoMgrRetStatus.SUCCESS:
            return RepoMgrRetStatus.FAIL

        main_mod, main_path = config_main_info()
        ctx = multiprocessing.get_context("forkserver")
        ctx.set_forkserver_preload([])

        # create sub-processes for bank process distribution task.
        cann_kb_process_num = get_cann_kb_proc_num_env()
        for _ in range(cann_kb_process_num):
            process_item = ctx.Process(target=cls.task_distribute, args=(resource_proxy, cls.cann_kb_mgr,), daemon=True)
            process_item.start()
            cls.bank_process_list.append(process_item)

        cls.daemon_process = ctx.Process(target=daemon_process,
            args=(cls.global_mgr.mgr._process.pid, os.getpid(), cls.daemon_running,),
            daemon=True)
        cls.daemon_process.start()
        restore_main_info(main_mod, main_path)

        Finalize(None, RouteServer.finalize, exitpriority=EXIT_PRIORITY)
        multiprocessing.current_process().daemon = deamon_status
        cls.pid = os.getpid()
        cls.route_status.set()
        LOG_INSTANCE.info("RouteServer initialize success.")
        return RepoMgrRetStatus.SUCCESS

    @classmethod
    @reentry_lock
    def finalize(cls) -> None:
        """finalize RouteServer"""
        if not cls.route_status.is_set() or cls.message_mgr is None or cls.pid != os.getpid():
            LOG_INSTANCE.warn("RouteServer does not initialize!")
            return

        # Do not receive new processing tasks.
        for item in cls.bank_process_list:
            item.terminate()
            item = None
        cls.bank_process_list = []

        # Close the knowledge base management portal object.
        cls.cann_kb_mgr.finalize()
        cls.cann_kb_mgr = None

        # Close the daemon_process object.
        cls.daemon_process.terminate()
        cls.daemon_process = None

        cls.message_mgr.release_resource()
        cls.message_mgr = None

        # Disabling the Global Resource Management Thread
        cls.global_mgr.finalize()
        cls.global_mgr = None

        cls.task_queue = None
        cls.result_lock = None
        cls.result_dict = None
        cls.sub_proc_state_list = None
        cls.daemon_running = None
        cls.route_status.clear()
        LOG_INSTANCE.info("RouteServer finalize success")

    @classmethod
    def sig_deal(cls, sig_num: int, frame: object = None) -> None:
        if RouteServer.message_mgr is not None and cls.pid == os.getpid():
            for item in cls.bank_process_list:
                item.terminate()
                item = None
            cls.bank_process_list = []
            RouteServer.daemon_process.terminate()
            RouteServer.message_mgr.release_resource()
            RouteServer.message_mgr = None
            RouteServer.global_mgr.finalize()
            RouteServer.route_status.clear()
        if sig_num == signal.SIGINT or sig_num == signal.SIGHUP:
            os._exit(1)

    @staticmethod
    @subprocess_troubleshoot
    def task_distribute(resource_proxy: dict, cann_kb_mgr: object) -> None:
        """task distribute func

        Description:
        get task from cls.task_queue and distribute task to knowledge manager module.

        Parameters:
        resource_proxy: {
            task_queue(queue): Task distribution queue
            result_dict(dict): Storage of detailed task results
            result_lock(dict): Multi-process lock ensures multi-process security for result modification
        }
        cann_kb_mgr(CannKbManager): Knowledge Base Management Module Object (Method Concurrency Security)
        task_distribute_running(Event): set running event to parent process

        Return: None
        """
        resource_proxy[SUB_PROCESS_STATE].append(True)
        pid = str(os.getpid())
        LOG_INSTANCE.event("task_distribute process[%s] running!" % pid)
        while True:
            key, func_name, detail = resource_proxy[TASK_QUEUE].get()
            LOG_INSTANCE.debug("task_key: %s" % str(key))
            LOG_INSTANCE.debug("task_func_name: %s" % str(func_name))
            LOG_INSTANCE.debug("task_detail: %s" % str(detail))
            if func_name not in ("cann_kb_search", "cann_kb_write", "cann_kb_delete"):
                LOG_INSTANCE.error("Unsupported method: %s" % func_name)
                break
            func = getattr(cann_kb_mgr, func_name)
            # result only support [str, str ...] and RepoMgrRetStatus
            res  = None
            try:
                res = func(**detail)
            except Exception as exp:
                LOG_INSTANCE.error("[%s] raise error[%s]" % (func.__name__, str(exp)))
                if func_name == "cann_kb_search":
                    res = []
                else:
                    res = RepoMgrRetStatus.FAIL
            finally:
                pass
            with resource_proxy[RES_LOCK]:
                resource_proxy[RES_DICT][key] = res
        LOG_INSTANCE.event("task_distribute process[%s] end!" % pid)

def get_cann_kb_proc_num_env() -> int:
    """get_cann_kb_proc_num_env
    Description:
    Get the env: CANN_KNOWLEDGE_BANK_PROCESS_NUM
    Return: int
    """
    proc_num = CANN_KB_PROCESS_NUM
    env_str = os.getenv("CANN_KNOWLEDGE_BANK_PROCESS_NUM")
    if not env_str:
        env_str = os.getenv("MAX_COMPILE_CORE_NUMBER", "8")
    if env_str.isdigit() and int(env_str) > 0 and int(env_str) <= CANN_KB_PROCESS_NUM:
        proc_num = int(env_str)
    LOG_INSTANCE.event("get_cann_kb_proc_num_env proc_num = [%d]." % proc_num)
    return proc_num


def get_msg_obj() -> tuple:
    """get_msg_obj

    Description:
    Get msg obj created in main process in sub-process

    Parameters:
    Retrun: tuple(status, msg_obj)
    """
    master_pid = op_context.get_context().get_addition("master_pid") if op_context.get_context() else None
    if not master_pid and not RouteServer.route_status.is_set():
        RouteServer.initialize(
            {"soc_version": get_soc_spec("FULL_SOC_VERSION"), "core_num": get_soc_spec("CORE_NUM")}, {}, {})
    file_name = MSG_FILE_NAME + str(master_pid if master_pid else os.getpid())
    msg_file_path = os.path.join(get_msg_file_dir(), file_name)

    return client_get_msg_obj(msg_file_path, file_name)


@timer
def public_interact(func: str, kwargs: dict, key: str) -> object:
    """public interact interface

    Description:
    Public interfaces for communication and interaction with sockets

    Parameters:
    func(str): Name of the method invoked in the subprocess.
    kwargs(dict): Function Input Parameters.
    key(str): Task Unique

    Retrun: Return value of the function corresponding to the facade mode
    """
    global SUB_PROCESS_STATE_FLAG
    while not SUB_PROCESS_STATE_FLAG:
        status, msg_mgr = get_msg_obj()
        if status != RepoMgrRetStatus.SUCCESS:
            return []
        SUB_PROCESS_STATE_FLAG = any(msg_mgr[SUB_PROCESS_STATE])
        time.sleep(QRY_INTERVAL)

    status, msg_mgr = get_msg_obj()
    if status != RepoMgrRetStatus.SUCCESS:
        return RepoMgrRetStatus.FAIL if func in ("cann_kb_write", "cann_kb_delete") else []

    msg_mgr[TASK_QUEUE].put((key, func, kwargs))
    while msg_mgr[RES_DICT].get(key, None) is None:
        time.sleep(QRY_INTERVAL)

    with msg_mgr[RES_LOCK]:
        return msg_mgr[RES_DICT].pop(key)


def public_interact_async(func: str, kwargs: dict, key: str) -> bool:
    """public interact interface async quest

    Description:
    Public interfaces for communication and interaction with sockets

    Parameters:
    func(str): Name of the method invoked in the subprocess.
    kwargs(dict): Function Input Parameters.
    key(str): Task Unique

    Retrun: add request to queue status
    """
    global SUB_PROCESS_STATE_FLAG
    while not SUB_PROCESS_STATE_FLAG:
        status, msg_mgr = get_msg_obj()
        if status != RepoMgrRetStatus.SUCCESS:
            return False
        SUB_PROCESS_STATE_FLAG = any(msg_mgr[SUB_PROCESS_STATE])
        time.sleep(QRY_INTERVAL)

    status, msg_mgr = get_msg_obj()
    if status != RepoMgrRetStatus.SUCCESS:
        return False
    msg_mgr[TASK_QUEUE].put((key, func, kwargs))
    return True


def public_interact_get(key: str) -> object:
    """public interact interface get

    Description:
    Public interfaces for communication and interaction with sockets

    Parameters:
    func(str): Name of the method invoked in the subprocess.
    kwargs(dict): Function Input Parameters.
    key(str): Task Unique

    Retrun: result of the async request
    """
    status, msg_mgr = get_msg_obj()
    if status != RepoMgrRetStatus.SUCCESS:
        return False, []

    if msg_mgr[RES_DICT].get(key, None) is None:
        return False, []

    with msg_mgr[RES_LOCK]:
        return True, msg_mgr[RES_DICT].pop(key)
