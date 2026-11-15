#!/usr/bin/env python3
"""Read-only compare ACL runtime vs DCMI per-device AI Core budgets."""
import ctypes as C
import json
import torch
import torch_npu
from h3_core_audit_probe import DCMI, load_backend

torch.npu.set_device(0)
lib = load_backend()._lib()
f = lib.aclrtGetDeviceInfo
f.argtypes = [C.c_uint32, C.c_int, C.POINTER(C.c_int64)]
f.restype = C.c_int
out = []
for device in range(8):
    cores = C.c_int64(-1)
    rc = f(device, 101, C.byref(cores))  # ACL_DEV_ATTR_AICORE_CORE_NUM from installed acl_rt.h
    d = DCMI(device)
    # Use the typed DCMI getter without model or NPU compute allocations.
    from h3_core_audit_probe import ChipInfo
    chip = ChipInfo()
    drc = d.lib.dcmi_get_device_chip_info(device, 0, C.byref(chip))
    freq = d.sample()
    out.append({"device": device, "acl_rc": rc, "acl_aicore_num": cores.value,
                "dcmi_rc": drc, "dcmi_aicore_num": chip.aicore_cnt if drc == 0 else None,
                "aicore_current_mhz": freq["aicore_current_mhz"],
                "aicore_max_mhz": freq["aicore_max_mhz"]})
print(json.dumps(out, indent=2), flush=True)
