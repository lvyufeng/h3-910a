#!/usr/bin/env python3
"""Write manifest101.json: the auditable identity of this source-only delivery.

Stdlib only. This does not compile, launch or measure anything: it records the
source/test hashes, the plan digest, the geometry and budget constants, the build
API, the pinned upstream Round99 source, and the explicit list of CANN behaviours
the root smoke must verify.
"""
import hashlib
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
UPSTREAM = HERE.parent.parent / "round99" / "flash" / "flash_vision_shift_temp_fixed99.py"
UPSTREAM_SHA = "d45cac797d25c8f5b0534e6cf12fb6fcad6f531ee33557debb35bc59cdcf6013"


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    import flash_vision101 as flash

    plan = flash.plan_kernel()
    smoke = flash.plan_kernel(*flash.SMOKE_GEOMETRY)
    buf = plan["buffers"]
    upstream_sha = sha256(UPSTREAM) if UPSTREAM.is_file() else None
    manifest = {
        "round": 101,
        "scope": "flash_bc128 source-only builder (MB=64 x BC=128 streaming attention)",
        "authority": "source only: not compiled, not launched, no accuracy or speed claim",
        "executed": False,
        "compiled_now": False,
        "not_a_speed_claim": True,
        "not_h3_end_to_end_result": True,
        "full_encoder_precision_accepted": False,
        "whole_goal_complete": False,
        "numerical_tolerance": None,
        "files": {
            "flash_vision101.py": {"sha256": sha256(HERE / "flash_vision101.py"),
                                   "role": "builder: plan + emitter + build_kernel"},
            "test_flash_vision101.py": {"sha256": sha256(HERE / "test_flash_vision101.py"),
                                        "role": "stdlib-only contract tests + recording TIK stub"},
            "make_manifest101.py": {"sha256": sha256(HERE / "make_manifest101.py"),
                                    "role": "this generator"},
            "README.md": {"sha256": sha256(HERE / "README.md"),
                          "role": "design/memory/masking notes and non-claims"},
            # NOTE: manifest101.json cannot record its own hash; the generator
            # prints it on stdout when the file is written.
        },
        "upstream_start_source": {
            "path": str(UPSTREAM.relative_to(HERE.parent.parent.parent.parent)),
            "expected_sha256": UPSTREAM_SHA,
            "actual_sha256": upstream_sha,
            "unchanged_on_disk": upstream_sha == UPSTREAM_SHA,
            "note": ("immutable executed Round99 kernel this builder starts from; it was "
                     "not modified by this round"),
        },
        "build_api": {
            "entry": "build_kernel(batch, m, n, d, out_dir=None, cores_max=30)",
            "cli": "--plan H M N D | --capability | --build H M N D --out-dir DIR",
            "kernel_name": flash.kernel_name_for(plan),
            "kernel_name_rule": "flash_vision101_h{H}_m{M}_n{N}_np{NPAD}_d{D}_mb{MB}_bc{BC}_c{cores}_kv{kvtiles}",
            "GM_argument_count": 5,
            "kernel_arg_order": list(flash.KERNEL_ARG_ORDER),
            "build_cce": {"inputs": ["q_gm_float16", "k_gm_float16", "v_gm_float16",
                                     "aux_gm_float32"],
                          "outputs": ["out_gm_float16"]},
            "metadata_fields": ["kernel_name", "kernel_symbol", "block_dim", "core_type",
                                "obj_path", "obj_sha256", "json_path", "source_sha256",
                                "logical", "grid", "dma", "buffers", "vector", "math",
                                "host_protocol", "assumptions", "capability"],
            "explicit_4_inputs_1_output": True,
        },
        "geometry": {
            "real": "H=16, M=N=4032 (48x84), D=72 padded to 80 = 5 NZ blocks",
            "query_tile_mb": 64, "kv_block_bc": 128,
            "query_tiles": 63, "padded_key_length": 4096, "kv_tiles": 32,
            "tail_real_keys_in_last_kv_tile": 64, "phantom_keys_in_last_kv_tile": 64,
            "phantom_groups": 4, "score_groups": 8, "pv_groups": 5,
            "tasks": 1008, "cores": 30, "steps": 34,
            "unmasked": True, "causal": False,
            "scale": plan["logical"]["scale"],
            "scale_is_original_head_dim_not_padded": True,
            "smoke": {"geometry": list(flash.SMOKE_GEOMETRY),
                      "n_pad": smoke["logical"]["n_pad"],
                      "kv_tiles": smoke["grid"]["kv_tiles"],
                      "phantom_keys": smoke["logical"]["phantom_keys"],
                      "cores": smoke["grid"]["cores"],
                      "note": "last KV tile is half real, so the smoke covers padding"},
        },
        "memory": {
            "ub_total_bytes": buf["ub_total_bytes"],
            "ub_named_total_bytes": buf["ub_named_total_bytes"],
            "ub_shift_extra_bytes": buf["ub_shift_extra_bytes"],
            "ub_safe_bound_bytes": flash.PINNED_UB_SAFE_BYTES,
            "ub_headroom_bytes": flash.PINNED_UB_SAFE_BYTES - buf["ub_total_bytes"],
            "l1_total_bytes": buf["l1_total_bytes"],
            "l0c_total_bytes": buf["l0c_total_bytes"],
            "l0c_breakdown": {"score": buf["l0c_score_bytes"],
                              "pv_hi": buf["l0c_pv_bytes"], "pv_lo": buf["l0c_pv_bytes"]},
            "removed_round99_buffer": "separate Float low-residual score tile (16384 B)",
            "doubled_buffers": ["score Float (also exp, also low residual)",
                                "P_hi Half", "P_hi Float (also row broadcast)",
                                "P_lo Half"],
            "bytes_delta_vs_round99": {
                "round99_ub_total": 151424 + 6656,
                "doubling_adds": 4 * 16384,
                "removed_low_residual_float": -16384,
                "round101_ub_total": buf["ub_total_bytes"],
            },
        },
        "math": plan["math"],
        "host_protocol": plan["host_protocol"],
        "capability_required": [f"{i}:{d}" for i, d in flash.REQUIRED_INTRINSICS],
        "capability_optional_never_required": [f"{i}:{d}" for i, d
                                               in flash.OPTIONAL_INTRINSICS],
        "unverified_cann_behaviours_for_the_root_smoke": list(flash.ASSUMPTIONS),
        "must_be_verified_by_root": [
            "compile of the 32 KiB single L0C->UB tensor_mov burst",
            "compile of the dynamic if_scope (no else) containing only Float vector_dup",
            "compile of 8-group explicit vec_max Half fold and vec_add Float fold",
            "compile of the 1024-element Float holding-tile h_cast plus 8 NZ copies",
            "compile of the 8192-element whole-tile h_sub/h_exp/h_cast chain",
            "exact real-input comparison against the Round78 CPU Half MATH reference on "
            "the frozen Round88 QKV (3fb5d524...)",
            "smoke H=2 M=128 N=192 first, then the real 4032 geometry",
        ],
        "non_claims": [
            "no accuracy result", "no speed result", "no compiler result",
            "no tolerance", "no default-enable", "no Float linear weight teacher",
            "no BF16 gold cast", "no clamps", "no cached conditioning",
            "no promotion of the Round99 bit pattern: the KV blocking and the Float "
            "reduction association changed, so low bits may differ",
        ],
    }
    out = HERE / "manifest101.json"
    out.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"written": str(out),
                      "sha256": sha256(out),
                      "upstream_unchanged": manifest["upstream_start_source"]
                      ["unchanged_on_disk"],
                      "ub_total_bytes": buf["ub_total_bytes"]}, indent=2))


if __name__ == "__main__":
    main()