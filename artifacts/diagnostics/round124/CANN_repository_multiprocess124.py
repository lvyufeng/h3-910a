#!/usr/bin/env python
# -*- coding:utf-8 -*-
"""
Copyright (C) 2022. Huawei Technologies Co., Ltd. All rights reserved.

multiprocess resource manager
"""
import os
import sys
import hmac
import time
import queue
import pickle
import hashlib
import threading
import multiprocessing as mp

from tbe.common.repository_manager.utils.define import RepoMgrRetStatus
from tbe.common.repository_manager.utils.file_sys_util import FileSysUtil
from tbe.common.repository_manager.utils.file_sys_util import LockedOpen
from tbe.common.repository_manager.utils.repository_manager_log import LOG_INSTANCE


TIMEOUT = 0.1
MARK_DIGEST_SIZE = 5
HEXDIGEST_RATE = 2
END = "end"


def get_resource(task_q: object, res_d: object, lock: object, content: tuple) -> object:
    """get resource such as mp.magager().lock list or dict..."""
    key, func, args, kwargs = content
    task_q.put((key, func, args, kwargs))
    while key not in res_d:
        pass
    lock.acquire()
    res = res_d[key]
    del res_d[key]
    lock.release()
    return res


class GlobalMgr(threading.Thread):
    """Global Mulitprocess Data Communication Management"""
    def __init__(self, *args, **kwargs):
        """initialize GlobalMgr"""
        super().__init__(*args, **kwargs)
        try:
            if sys.version_info >= (3, 14):
                ctx = mp.get_context("fork")
                self.mgr = ctx.Manager()
            else:
                self.mgr = mp.Manager()
        except EOFError as e:
            LOG_INSTANCE.error("Python built-in object multiprocessing.Manager instantiation failed.")
            raise RuntimeError({
                "errCode": "EC0009",
                "err_detail": "Python built-in object multiprocessing.Manager instantiation failed"
            }) from e
        self.task_q = self.mgr.Queue()
        self.res_d = self.mgr.dict()
        self.resource_l = self.mgr.Lock()

    def finalize(self):
        """finalize GlobalMgr"""
        self.task_q.put(END)
        self.join()
        self.mgr.shutdown()

    def run(self):
        """GlobalMgr func server"""
        while True:
            item = self.task_q.get()
            if item == END:
                break
            key, func, args, kwargs = item
            self.resource_l.acquire()
            self.res_d[key] = getattr(self, func)(*args, **kwargs)
            self.resource_l.release()

    def queue(self, *args, **kwargs):
        """return a multiprocessing.Manager.Queue
        if return None because not initialize
        """
        return self.mgr.Queue(*args, **kwargs)

    def dict(self, *args, **kwargs):
        """return a multiprocessing.Manager.dict
        if return None because not initialize
        """
        return self.mgr.dict(*args, **kwargs)

    def event(self, *args, **kwargs):
        """return a multiprocessing.Manager.Event
        if return None because not initialize
        """
        return self.mgr.Event(*args, **kwargs)

    def value(self, *args, **kwargs):
        """return a multiprocessing.Manager.Event
        if return None because not initialize
        """
        return self.mgr.Value(*args, **kwargs)

    def lock(self, *args, **kwargs):
        """return a multiprocessing.Manager.Lock
        if return None because not initialize
        """
        return self.mgr.Lock(*args, **kwargs)

    def list(self, *args, **kwargs):
        """return a multiprocessing.Manager.list
        if return None because not initialize
        """
        return self.mgr.list(*args, **kwargs)


class MessageManager:
    file = None

    def __init__(self, file_path: str) -> None:
        """MessageManager __init__"""
        self.file = file_path

    def __del__(self) -> None:
        """MessageManager __del__"""
        self.release_resource()

    def msg_mgr_dump(self, obj: object, hmac_key: str) -> RepoMgrRetStatus:
        """MessageManager msg_mgr_dump"""
        LOG_INSTANCE.debug("MessageManager[%s] try to write file!" % self.file)
        res = RepoMgrRetStatus.SUCCESS
        try:
            obj_bytes = pickle.dumps(obj)
        except pickle.PickleError as e:
            LOG_INSTANCE.error("MessageManager.msg_mgr_dump : %s" % str(e))
            res = RepoMgrRetStatus.MSG_DUMP_FAIL
        finally:
            pass
        if res != RepoMgrRetStatus.SUCCESS:
            return res
        secrets = hmac.new(hmac_key.encode(), obj_bytes, hashlib.md5)
        secrets_size = str(secrets.digest_size * HEXDIGEST_RATE).zfill(MARK_DIGEST_SIZE).encode()
        secrets_hexdigest = secrets.hexdigest().encode()
        try:
            with os.fdopen(os.open(self.file, FileSysUtil.FILE_FLAG, FileSysUtil.FILE_MODE_640), "wb") as fp:
                fp.write(secrets_size + secrets_hexdigest + obj_bytes)
        except OSError as e:
            LOG_INSTANCE.error("msg_mgr_dump raise error[%s]" % str(e))
            file_dir = self.file.split(".lock")[0]
            raise RuntimeError({"errCode": "EC0013", "file": file_dir, "reason": str(e)})
        finally:
            pass
        LOG_INSTANCE.debug("MessageManager[%s] create success!" % self.file)
        return res

    def release_resource(self) -> None:
        """MessageManager release_resource"""
        LOG_INSTANCE.debug("MessageManager[%s] release_resource entry!" % self.file)
        if self.file is None:
            return
        if os.path.exists(self.file):
            os.remove(self.file)
            LOG_INSTANCE.debug("MessageManager[%s] remove file success!" % self.file)
        LOG_INSTANCE.debug("MessageManager[%s] release_resource success!" % self.file)
        self.file = None


def client_get_msg_obj(file_path: str, hmac_key: str) -> object:
    """get msg object from file"""
    with LockedOpen(file_path, mode="ab+") as fp:
        fp.seek(0)
        secrets_size = int(fp.read(MARK_DIGEST_SIZE).decode())
        secrets_hexdigest = fp.read(secrets_size).decode()
        obj_bytes = fp.read()
        secrets = hmac.new(hmac_key.encode(), obj_bytes, hashlib.md5)
        if secrets.digest_size * HEXDIGEST_RATE == secrets_size and secrets_hexdigest == secrets.hexdigest():
            res = RepoMgrRetStatus.SUCCESS
            try:
                obj = pickle.loads(obj_bytes)
            except pickle.PickleError as e:
                LOG_INSTANCE.error("client_get_msg_obj : %s" % str(e))
                res = RepoMgrRetStatus.FAIL
                obj = {}
            finally:
                pass
            return res, obj
        LOG_INSTANCE.error("client_get_msg_obj[%s] get msg object failed!" % file_path)
    return RepoMgrRetStatus.FAIL, {}

