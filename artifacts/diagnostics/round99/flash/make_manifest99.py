#!/usr/bin/env python3
"""Regenerate manifest99.json from the delivered files (stdlib only).

Run from this directory: ``python3 make_manifest99.py``. It records bytes/SHA256
for every delivered file plus the plan-derived geometry, budgets and the list of
SDK assumptions the root CANN smoke must verify. It never compiles anything.
"""
import hashlib
import importlib.util
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
DELIVERED = ("flash_vision99.py", "test_flash_vision99.py", "README.md")


def file_receipt(path):
    payload = path.read_bytes()
    return {"bytes": len(payload), "sha256": hashlib.sha256(payload).hexdigest()}


def load_source():
    spec = importlib.util.spec_from_file_location("flash_vision99_manifest",
                                                  HERE / "flash_vision99.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main():
    flash = load_source()
    plan = flash.plan_kernel()
    smoke = flash.plan_kernel(*flash.SMOKE_GEOMETRY)
    manifest = {
        "round": 99,
        "scope": ("single-launch streaming vision flash attention: one QK FP32 matmul, "
                  "Half-only row shift estimate, FP32 online shift/denominator/output, "
                  "two-Half probability P@V in two FP32 L0C, one final Half cast"),
        "source_only": True,
        "executed": False,
        "compiled_now": False,
        "not_a_speed_claim": True,
        "not_h3_end_to_end_result": True,
        "full_encoder_precision_accepted": False,
        "whole_goal_complete": False,
        "numerical_tolerance": None,
        "default_enabled": False,
        "files": {name: file_receipt(HERE / name) for name in DELIVERED},
        "api": {
            "builder": "flash_vision99.build_kernel(batch, m, n, d, out_dir=None)",
            "plan": "flash_vision99.plan_kernel(batch, m, n, d, mb=64, bc=64, cores_max=30)",
            "protocol": "flash_vision99.check_host_protocol(max_abs_scaled_qk=None, "
                        "q_absmax=None, k_absmax=None, d=72)",
            "capability": "flash_vision99.capability_report(plan=None)",
            "constants": ["HEADS", "SEQ", "HEAD_DIM", "PAD_DIM", "MB", "BC",
                          "SHIFT_BOUND", "SMOKE_GEOMETRY", "KERNEL_ARG_ORDER"],
        },
        "geometry": {
            "real": {"heads": plan["logical"]["heads"], "m": plan["logical"]["m"],
                     "n": plan["logical"]["n"], "d": plan["logical"]["d"],
                     "padded_head_dim": plan["logical"]["padded_head_dim"],
                     "mb": plan["logical"]["mb"], "bc": plan["logical"]["bc"],
                     "tasks": plan["grid"]["tasks"], "cores": plan["grid"]["cores"],
                     "steps": plan["grid"]["steps"],
                     "kv_tiles": plan["grid"]["kv_tiles"]},
            "smoke": {"geometry": list(flash.SMOKE_GEOMETRY),
                      "tasks": smoke["grid"]["tasks"], "cores": smoke["grid"]["cores"],
                      "kv_tiles": smoke["grid"]["kv_tiles"]},
            "scale": plan["logical"]["scale"],
            "scale_is_original_head_dim": True,
            "padded_width_is_not_the_scale": True,
            "unmasked": True,
            "causal": False,
        },
        "gm_abi": {
            "argument_order": list(flash.KERNEL_ARG_ORDER),
            "count": 5,
            "inputs": ["q_gm", "k_gm", "v_gm", "aux_gm"],
            "outputs": ["out_gm"],
            "shapes": plan["gm_shapes"],
            "head_stride_elems": plan["head_stride_elems"],
            "bytes": plan["gm_bytes"],
            "aux_is_unused_float_placeholder": True,
        },
        "dma": plan["dma"],
        "buffers": {k: v for k, v in plan["buffers"].items() if k != "ub_named"},
        "vector": plan["vector"],
        "math": plan["math"],
        "host_protocol": plan["host_protocol"],
        "assumptions": plan["assumptions"],
        "root_smoke_must_check": [
            "the two real geometries (H16 M=N4032 D72) and the H2/M=N128 smoke both compile",
            "h_reduce_max/h_reduce_sum with axis=1 return (MB,) rows on a (MB,16) tile",
            "reshape((n1,MB,16)) is a metadata view and the buffer reuse order is respected",
            "masked vector_dup (repeat_times=n1, blk_stride=1, rep_stride=2*MB) writes the "
            "16-value row slot of every NZ group and the last row ends exactly at the tile end",
            "in-place h_exp/h_sub/h_mul/h_add with dst==src0 behave as the FA source's rows do",
            "the fused output matches the frozen real QKV CPU Half MATH reference and is "
            "compared against the executed Round97 leaf before any whole-vision claim",
        ],
        "evidence_read_only": {
            "sdk_sheet": "../actual_sdk_highlevel99.json",
            "float_vcmax_unsupported": True,
            "half_vcmax_supported": True,
            "float_vcadd_vmax_vexp_vdiv_supported": True,
            "h_broadcast_absent": True,
        },
        "non_claims": [
            "no FP32 linear weight, teacher tensor or BF16 gold cast",
            "no cached pixel/patch/hidden/DeepStack/conditioning value",
            "no Half cumulative output or Half denominator",
            "no clamp and no post-hoc numerical tolerance",
            "no measured speedup or accuracy result from the author",
        ],
    }
    (HERE / "manifest99.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    print("wrote manifest99.json")
    for name, receipt in manifest["files"].items():
        print(f"  {name}: {receipt['bytes']} B {receipt['sha256']}")


if __name__ == "__main__":
    main()
