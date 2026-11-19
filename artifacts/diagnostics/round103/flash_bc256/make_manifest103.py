#!/usr/bin/env python3
"""Write manifest103.json: source pin, plan/API identity, tests and budget evidence.

Stdlib only, no CANN/TIK/torch/NPU. It records what the source *says*; it does not
compile, execute or measure anything.
"""
from __future__ import annotations

import hashlib
import importlib.util
import inspect
import json
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
SOURCE = HERE / "flash_vision103.py"
START = HERE.parent.parent / "round101" / "flash_vision_mask_fixed101.py"
OUT = HERE / "manifest103.json"


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load():
    spec = importlib.util.spec_from_file_location("flash_vision103_manifest", SOURCE)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main():
    flash = load()
    plan = flash.plan_kernel()
    smoke = flash.plan_kernel(*flash.SMOKE_GEOMETRY)
    buffer = plan["buffers"]
    signature = inspect.signature(flash.build_kernel)
    manifest = {
        "round": 103,
        "experiment": "MB=32 x BC=256 single-launch streaming vision flash attention",
        "status": "SOURCE ONLY",
        "source_only": True,
        "author_never_executed_it": True,
        "compiled": False,
        "executed": False,
        "accuracy_claim": None,
        "speed_claim": None,
        "numerical_tolerance": None,
        "default_enabled": False,
        "full_encoder_precision_accepted": False,
        "whole_goal_complete": False,
        "purpose": ("halve the per-row Float row-broadcast and cumulative O/L update "
                    "count versus Round101 MB=64 x BC=128, at the explicit cost of "
                    "roughly doubling K/V reload traffic; the net effect is NOT assumed"),
        "files": {},
        "starting_source": {
            "path": "../round101/flash_vision_mask_fixed101.py",
            "sha256": sha256(START) if START.is_file() else None,
            "expected_sha256":
                "ef21123261574bf31aa03e8d0ba8666dfc0f5fbb54a0c1d4694cce4c0749e69d",
            "immutable_during_root_execution": True,
            "note": ("Round103 was authored from this exact file; the executed Round101 "
                     "kernel is unchanged by this experiment"),
        },
        "api": {
            "module": "flash_vision103",
            "build_kernel_signature": str(signature),
            "parameter_order": list(signature.parameters),
            "plan_kernel": str(inspect.signature(flash.plan_kernel)),
            "metadata_fields": ["obj_path", "obj_sha256", "kernel_symbol", "block_dim",
                                "GM_argument_count", "kernel_arg_order", "logical",
                                "grid", "dma", "buffers", "vector", "math",
                                "host_protocol", "assumptions", "capability"],
            "GM_argument_count": 5,
            "kernel_arg_order": list(flash.KERNEL_ARG_ORDER),
            "compile_soc": flash.PINNED_COMPILE_SOC,
            "kernel_name_for_real_geometry": flash.kernel_name_for(plan),
            "build_outputs": ["<out_dir>/<kernel>/<kernel>.o",
                              "<out_dir>/<kernel>/<kernel>.json",
                              "<out_dir>/<kernel>/spec103.json"],
        },
        "geometry": {
            "real": {key: plan["logical"][key] for key in
                     ("heads", "m", "n", "n_pad", "d", "padded_head_dim", "mb", "bc",
                      "kb1", "tail_keys", "phantom_keys", "phantom_groups",
                      "real_groups_last", "score_groups", "scale")},
            "grid": plan["grid"],
            "smoke": {"geometry": list(flash.SMOKE_GEOMETRY),
                      "logical": {key: smoke["logical"][key] for key in
                                  ("m", "n", "n_pad", "tail_keys", "phantom_keys")},
                      "grid": smoke["grid"]},
            "query_tiles": 126,
            "kv_tiles": 16,
            "tasks": 2016,
            "launched_cores": 30,
            "physical_cores": 32,
            "dynamic_steps": 68,
        },
        "memory": {
            "l1_total_bytes": buffer["l1_total_bytes"],
            "l0c_total_bytes": buffer["l0c_total_bytes"],
            "ub_total_bytes": buffer["ub_total_bytes"],
            "ub_named_total_bytes": buffer["ub_named_total_bytes"],
            "ub_shift_extra_bytes": buffer["ub_shift_extra_bytes"],
            "ub_safe_bound_bytes": flash.PINNED_UB_SAFE_BYTES,
            "ub_derived_detail": buffer["ub_named"],
            "ub_informational_aliases_not_summed":
                buffer["informational_ub_aliases_not_summed"],
            "cube_operand_bytes": buffer["cube_operand_bytes"],
            "l0c_capacity_bytes": 262144,
            "l1_capacity_bytes": 1024 * 1024,
            "every_ub_allocation_is_derived_from_plan": True,
            "no_fake_capacity_caps": True,
        },
        "math": plan["math"],
        "dma": plan["dma"],
        "vector": plan["vector"],
        "host_protocol": plan["host_protocol"],
        "assumptions": plan["assumptions"],
        "unverified_on_real_cann": [
            "masked data_move with burst=256 32 B units (8192 B)",
            "Cube m=32,k=80,n=256 (QK) and m=32,k=256,n=80 (P@V) operand residency",
            "32-lane Float vector_dup tail mask (128 B) for an MB=32 row",
            "two 16-row uint16 vec_or + native Half vec_trans shift broadcast",
            "16-group explicit vec_max / vec_add folds",
            "a 32 KiB single tensor_mov L0C->UB burst",
        ],
        "not_run_here": True,
    }
    tests = HERE / "test_flash_vision103.py"
    if tests.is_file():
        manifest["files"]["test_flash_vision103.py"] = {
            "sha256": sha256(tests),
            "tests": 64,
            "structural_only": True,
            "imports_cann_or_torch": False,
        }
    manifest["files"]["flash_vision103.py"] = {
        "sha256": sha256(SOURCE),
        "lines": len(SOURCE.read_text().splitlines()),
    }
    manifest["files"]["make_manifest103.py"] = {
        "sha256": sha256(Path(__file__)),
    }
    for name in ("dryrun_build103.py", "dryrun103.json", "README.md"):
        path = HERE / name
        if path.is_file():
            manifest["files"][name] = {"sha256": sha256(path)}
    dryrun = HERE / "dryrun103.json"
    if dryrun.is_file():
        report = json.loads(dryrun.read_text())
        manifest["fake_sdk_dry_run"] = {
            "file": "dryrun103.json",
            "fake_sdk": report["fake_sdk"],
            "compiled": report["compiled"],
            "executed": report["executed"],
            "proves": ("build_kernel spec assembly, object-SHA self-consistency, "
                       "cache hit, block_dim guard and the exact metadata keys the "
                       "root engine and Runtime5GM read"),
            "does_not_prove": ("that the emitted program compiles or runs; the .o is "
                              "a placeholder blob, not a binary"),
            "block_dim": report["real_geometry"]["block_dim"],
            "kernel_symbol": report["real_geometry"]["kernel_symbol"],
            "runtime5gm_keys_present": report["real_geometry"]["runtime5gm_keys_present"],
            "all_required_metadata_keys_present":
                report["real_geometry"]["all_required_metadata_keys_present"],
            "cache_hit": report["cache_hit"]["same_obj_sha256"],
            "block_dim_guard_refused": report["block_dim_guard"]["refused"],
            "plan_refusals": report["plan_refusals"],
            "smoke_block_dim": report["smoke_geometry"]["block_dim"],
        }
    if OUT.is_file():
        raise SystemExit(f"refusing to overwrite {OUT}")
    OUT.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"manifest": str(OUT), "source_sha256": manifest["files"]
                      ["flash_vision103.py"]["sha256"]}, indent=2))


if __name__ == "__main__":
    main()
