#!/usr/bin/env python3
"""R188: is the FA core budget a hardware limit or an arbitrary cap?

fa910a.py sets instance.core_num = min(instance.core_num, core_budget) where core_budget comes
from a driver query (infoType 101). R159's log printed core_budget=30. If 30 is the hardware AI
Core count then the FA kernel already uses every core and the core budget is not a bottleneck,
which rules out the cheap "just use more cores" path and means only a better kernel or tiling
can help.

Read-only: queries the driver, allocates nothing of significance.
"""
from __future__ import annotations
import ctypes
import json
import sys
from pathlib import Path

OUT = Path('/tmp/h3_core188/corecount188.json')
# fa910a.py queries ACL's aclrtGetDeviceInfo, not the HAL directly
LIB = '/home/mseco/Ascend/cann-9.2.0/lib64/libascendcl.so'


def main():
    import torch
    import torch_npu  # noqa: F401
    torch.npu.set_device(0)

    result = {'device': 0, 'queries': {}}

    # replicate _query_runtime_aicore_count exactly
    try:
        hal = ctypes.CDLL(LIB)
        q = hal.aclrtGetDeviceInfo
        q.argtypes = [ctypes.c_uint32, ctypes.c_int, ctypes.POINTER(ctypes.c_int64)]
        q.restype = ctypes.c_int
    except OSError as exc:
        result['error'] = f'cannot load {LIB}: {exc}'
        OUT.parent.mkdir(parents=True, exist_ok=True)
        OUT.write_text(json.dumps(result, indent=2) + '\n')
        print(json.dumps(result, indent=1))
        return

    # 101 is what fa910a.py uses; sweep a few neighbours so we can label them
    for info_type in (101, 100, 102, 109, 110):
        v = ctypes.c_int64(0)
        rc = q(0, info_type, ctypes.byref(v))
        result['queries'][str(info_type)] = {'rc': rc, 'value': int(v.value)}

    # what does the kernel default its core_num to, before the min()
    try:
        sys.path.insert(0, '/tmp/h3_perf12601/native_sources')
        import fa910a
        budget = fa910a._runtime_core_budget()
        result['fa910a_core_budget'] = budget
    except Exception as exc:  # noqa: BLE001
        result['fa910a_core_budget_error'] = repr(exc)[:300]

    # the ACL platform view of the same hardware
    try:
        import acl
        acl.init()
        acl.rt.set_device(0)
        for name, fn in (('aicore_num', 'get_aicore_num'),):
            f = getattr(acl, fn, None)
            if f is None:
                continue
            try:
                result[name] = int(f())
            except Exception as exc:  # noqa: BLE001
                result[name + '_error'] = repr(exc)[:200]
        acl.rt.reset_device(0)
        acl.finalize()
    except Exception as exc:  # noqa: BLE001
        result['acl_error'] = repr(exc)[:300]

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result, indent=1))


if __name__ == '__main__':
    main()
