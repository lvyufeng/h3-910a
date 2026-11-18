#!/usr/bin/env python3
"""Regenerate manifest97.json from the delivered files (stdlib only).

Run from this directory: ``python3 make_manifest97.py``. It records bytes/SHA256 for
every delivered file plus the capability, API-evidence and non-claim receipts. It
does not compile, launch or measure anything.
"""
import hashlib
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
DELIVERED = ("softmax_pack97.py", "test_softmax_pack97.py", "README.md")

SDK_DIR = HERE.parent.parent / "round83" / "bias" / "sdk"

API_EVIDENCE = [
    {"need": "vec_muls/vec_adds accept a TIK Scalar",
     "evidence": "common_params_check_.py:193 _check_scalar accepts (int,float,Scalar,Expr)"},
    {"need": "vector_dup accepts a Scalar src",
     "evidence": "common_params_check_.py:303 same tuple"},
    {"need": "vec_add/vec_max/vec_mul signatures",
     "evidence": "tik_vector_api.py:53/113/173"},
    {"need": "vec_exp / vec_abs", "evidence": "tik_vector_api.py:315/341"},
    {"need": "vec_rec(mask,dst,src,repeat,dst_str,src_str)",
     "evidence": "tik_vector_api.py:414"},
    {"need": "vec_adds / vec_muls with scalar",
     "evidence": "tik_vector_api.py:599/631"},
    {"need": "vcmax/vcadd intrinsics exist",
     "evidence": "tik_params.py:908-910; VREDUCE_DST_ALIGN vcmax=4 at tik_params.py:619-626"},
    {"need": "vector repeat limit 255",
     "evidence": "tik_params.py:77 MAX_REPEAT_TIMES"},
    {"need": "vreduce repeat limit 4095",
     "evidence": "tik_params.py:607 MAX_VREDUCE_REPEAT_TIMES"},
    {"need": "vec_reduce_add call shape",
     "evidence": "executed Round61/62 AdaLN norm: (mask,dst,src,work,repeat,src_rep_stride)"},
    {"need": "h_cast both directions",
     "evidence": "executed Round90 erf: Half->Float widen and one Float->Half narrow"},
    {"need": "FP32 4096 tile = 64 repeats x rep_stride 8",
     "evidence": "executed Round90 erf_gelu_fixed90.py FP32_MASK=64, FP32_REP_STRIDE=8"},
    {"need": "offset UB slice as a vector dst",
     "evidence": "project FA patch vector_dup(..., scores[block,0,0], ...)"},
    {"need": "t.Scalar('float32',name=...).set_as(t[idx])",
     "evidence": "executed Round61 adaln_vector61.py"},
    {"need": "t.else_scope()", "evidence": "FA patch / Round57 sources"},
    {"need": "vec_reduce_max call shape",
     "evidence": ("ASSUMPTION, no executed precedent in this project: taken as the same "
                  "vreduce signature as the verified vec_reduce_add, one call over all 64 "
                  "repeats leaving the row result in dst[0]. Root build/probe is the "
                  "authority; _emit_kernel refuses if the method is absent.")},
]


def file_receipt(path: Path):
    payload = path.read_bytes()
    return {"bytes": len(payload), "sha256": hashlib.sha256(payload).hexdigest()}


def main():
    import importlib.util
    spec = importlib.util.spec_from_file_location("sp97", HERE / "softmax_pack97.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    plans = {str(m): module.plan_kernel(m=m) for m in module.M_CHOICES}
    manifest = {
        "round": 97,
        "scope": ("Float32 NZ score row -> true-4032 unmasked Float stable softmax -> "
                  "Half hi + Half lo NZ probability packing, one launch, five GM pointers"),
        "executed": False,
        "compiled_now": False,
        "not_a_speed_claim": True,
        "not_h3_end_to_end_result": True,
        "full_encoder_precision_accepted": False,
        "whole_goal_complete": False,
        "numerical_tolerance": None,
        "author_computer_execution": "none: no CANN/TBE build, no NPU, no torch, no model, no SSH",
        "files": {name: file_receipt(HERE / name) for name in DELIVERED},
        "api": module.build_kernel.__doc__.splitlines()[0] if module.build_kernel.__doc__ else None,
        "public_api": {
            "build_kernel": "build_kernel(batch=16, m=512|448, n=4096, valid_keys=4032, out_dir=None)",
            "plan_kernel": "plan_kernel(batch, m, n, valid_keys, cores)",
            "helpers": ["flatten_task", "row_offsets", "is_padded_row",
                        "row_element_coverage", "score_scale", "capability_report"],
        },
        "geometry": {m: {"logical": plan["logical"], "padded": plan["padded"],
                         "grid": plan["grid"], "head_stride_elems": plan["head_stride_elems"],
                         "gm_shapes": plan["gm_shapes"], "gm_bytes": plan["gm_bytes"],
                         "dma": plan["dma"], "vector": plan["vector"],
                         "buffers": plan["buffers"]}
                     for m, plan in plans.items()},
        "wire": {"GM_argument_count": 5,
                 "kernel_arg_order": plans["512"]["kernel_arg_order"],
                 "BuildCCE": "inputs=[scores,aux0,aux1], outputs=[hi,lo]"},
        "masking": {"valid_keys": module.VALID_KEYS, "tail_keys": module.TAIL_KEYS,
                    "negative_max": module.NEG_MAX,
                    "tail_masked_before_max_and_exp": True,
                    "tail_zeroed_before_denominator": True,
                    "padded_query_rows_forced_zero": True,
                    "padded_rows_uniform_softmax_forbidden": True,
                    "causal": False, "unmasked": True},
        "arithmetic": {"softmax_dtype": "float32",
                       "hi": "Half(p)", "lo": "Half((p - float(hi)) * 1024)",
                       "lo_rescale_inverse": module.LO_SCALE,
                       "score_scale": module.score_scale(),
                       "score_scale_basis": "head_dim 72 ** -0.5 (original; not padded 128)",
                       "reciprocal": "vec_rec + %d Newton (denominator only)" % module.NEWTON_STEPS,
                       "bit_identity_with_torch_softmax_claimed": False},
        "capability_gate": {
            "compile_soc_must_be": module.PINNED_COMPILE_SOC,
            "core_num_must_be": module.CORE_NUM,
            "ub_safe_bytes": module.PINNED_UB_SAFE_BYTES,
            "intrinsics": list(module.REDUCE_INTRINSICS),
            "dtype_spellings": list(module.DTYPE_SPELLINGS),
            "on_unsupported": "raise SoftmaxPackError; no non-Float/A2 substitute",
        },
        "api_evidence": API_EVIDENCE,
        "api_evidence_root": str(SDK_DIR),
        "no_float_linear_weight": True,
        "no_teacher_or_bf16_gold_cast": True,
        "caches_no_pixel_patch_hidden_deepstack_or_conditioning": True,
        "aux_gm_operands_are_unused_placeholders": True,
        "layout_equals_pv_a_nz": True,
        "frozen_inputs_untouched": ["Round88 QKV fixture 3fb5...", "CPU Half MATH reference"],
        "tests": {"runner": "python3 -m unittest test_softmax_pack97",
                  "count": 29, "stdlib_only": True, "compiles_or_runs_kernel": False},
    }
    (HERE / "manifest97.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    print("wrote manifest97.json")
    for name, receipt in manifest["files"].items():
        print(" %-26s %7d B  %s" % (name, receipt["bytes"], receipt["sha256"]))


if __name__ == "__main__":
    main()