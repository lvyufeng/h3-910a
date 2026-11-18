#!/usr/bin/env python3
"""Refresh manifest96.json file SHAs after any source edit (stdlib only)."""
import hashlib
import importlib.util
import json
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
DELIVERED = ["pv_high_low96.py", "accurate_vision_attention_pv96.py",
             "test_pv_high_low96.py", "test_accurate_vision_attention_pv96.py",
             "test_emission96.py", "README.md", "refresh_manifest96.py"]


def sha(name):
    return hashlib.sha256((HERE / name).read_bytes()).hexdigest()


def main():
    manifest_path = HERE / "manifest96.json"
    manifest = json.loads(manifest_path.read_text())
    spec = importlib.util.spec_from_file_location("pv_high_low96",
                                                  HERE / "pv_high_low96.py")
    pv = importlib.util.module_from_spec(spec)
    sys.modules["pv_high_low96"] = pv
    spec.loader.exec_module(pv)
    manifest["files"] = {name: {"sha256": sha(name),
                                "bytes": (HERE / name).stat().st_size}
                         for name in DELIVERED}
    manifest["kernel"]["name_pattern"] = "pv_hi_lo96_h{H}_m{M}_k{K}_n{N}_mb128_nb128_kb128_c30"
    manifest["kernel"]["kernel_arg_order"] = list(pv.KERNEL_ARG_ORDER)
    manifest["tests"]["modules"] = ["test_pv_high_low96",
                                    "test_accurate_vision_attention_pv96",
                                    "test_emission96"]
    manifest["tests"]["command"] = ("python3 -m unittest test_pv_high_low96 "
                                    "test_accurate_vision_attention_pv96 test_emission96")
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    for name in DELIVERED:
        print(name, manifest["files"][name]["sha256"])


if __name__ == "__main__":
    main()
