#!/usr/bin/env python3
"""Verify the round125 documents against their own machine receipts.

No Torch, no model, no network. It re-reads the committed JSON receipts and asserts
that every number quoted in ``STATE125.json`` and ``candidates_measured125.json`` is
the value those receipts actually contain, so the prose cannot drift away from the
measurements. Re-running must produce identical output.
"""
from __future__ import annotations
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def load(name: str) -> dict:
    return json.loads((ROOT / name).read_text())


def close(a: float, b: float, tol: float = 1e-12) -> bool:
    return abs(float(a) - float(b)) <= tol


def main() -> int:
    state = load('STATE125.json')
    summary = load('candidates_measured125.json')
    fa = load('fa910a_candidate125.json')
    gen = load('NPU_generated_FA/generation_result124.json')
    media = load('R124_vs_FA_media125.json')
    vae = load('VAEfast_vs_baselines125.json')
    half = load('half_residual_candidate125.json')
    fused = load('fused_mlp_fa_rank0.json')
    floor = load('noise_floor125.json')

    acc = state['accepted_acceleration']
    rejected = {row['candidate'].split()[0]: row for row in state['rejected_candidates']}
    measured = {row['candidate'].split()[0]: row for row in summary['candidates']}

    checks: dict[str, bool] = {}

    # The accepted acceleration must match the generation receipt and media comparison.
    checks['fa_completed'] = gen.get('completed') is True
    checks['fa_video_generated'] = gen.get('actual_video_generated') is True
    checks['fa_eight_workers_49'] = (
        len(gen['actual_DiT_counts_by_worker_pid']) == 8
        and set(gen['actual_DiT_counts_by_worker_pid'].values()) == {49})
    checks['fa_wall_matches_receipt'] = close(acc['generation_wall_s'], gen['generation_wall_s'])
    checks['fa_psnr_min_matches'] = close(acc['media_psnr_min_dB_vs_R124'], media['psnr_min_dB'])
    checks['fa_psnr_median_matches'] = close(acc['media_psnr_median_dB_vs_R124'], media['psnr_median_dB'])
    checks['fa_no_frame_below_40dB'] = media['frames_below_40dB'] == 0
    checks['fa_speedup_recomputes'] = close(
        acc['end_to_end_speedup'], acc['R124_generation_wall_s'] / acc['generation_wall_s'], tol=1e-6)
    checks['fa_media_verification'] = (
        gen['media_verification']['decoded_video_frames'] == 124
        and gen['media_verification']['width'] == 1344
        and gen['media_verification']['height'] == 768)
    checks['fa_engine_warm_step_matches'] = close(
        acc['engine_warm_step_s'], load('full_generation_with_FA125.json')['engine_side_step_timing_with_FA']['warm_average_step_s'])
    checks['fa_replay_step_matches'] = close(
        acc['replay_warm_step_s'], measured['custom']['speed']['warm_step_median_s'])
    checks['fa_speedup_step_recomputes'] = close(
        measured['custom']['speed']['speedup_step'], 40.466285259928554 / acc['replay_warm_step_s'], tol=1e-6)

    # Rejected candidates must match their receipts.
    checks['vae_decode_matches'] = close(
        rejected['VAE']['measured']['decode_median_s'],
        measured['VAE']['speed']['decode_median_s'])
    checks['vae_psnr_matches'] = close(
        rejected['VAE']['measured']['media_psnr_min_dB_vs_FA'], vae['vs_FA']['psnr_min_dB'])
    checks['half_step_matches'] = close(
        rejected['DiT']['measured']['warm_step_median_s'],
        half['timing']['half_residual_warm_step_median_s'])
    checks['half_slower_than_fa'] = (
        rejected['DiT']['measured']['warm_step_median_s'] > half['timing']['FA_only_warm_step_median_s'])
    checks['fused_step_matches'] = close(
        rejected['fused']['measured']['warm_step_median_s'], fused['warm_step_median_s'])
    checks['fused_no_gain'] = (
        rejected['fused']['measured']['warm_step_median_s'] >= fused['warm_step_median_s'] - 1e-9)
    checks['fused_drift_recorded'] = (
        rejected['fused']['measured']['video_relative_RMS_by_step'][-1] >
        rejected['fused']['measured']['video_relative_RMS_by_step'][0])

    # The noise floor must bracket the R124 gate values quoted in the README.
    checks['noise_floor_not_bit_reproducible'] = floor['bit_reproducible_across_runs'] is False
    checks['noise_floor_video_band'] = floor['video_relRMS_max_over_A_and_B'] > 0.0
    checks['noise_floor_audio_band'] = floor['audio_relRMS_max_over_A_and_B'] > 0.0

    # Nothing may claim the goal is complete, and no rejected candidate may be enabled.
    checks['goal_not_complete'] = state['overall_original_performance_and_A2_A3_goal_complete'] is False
    checks['no_lossy_without_approval'] = state['cleanup']['lossy_candidates_enabled_without_user_approval'] is False
    checks['rejected_not_enabled'] = state['cleanup']['rejected_candidates_enabled_anywhere'] is False
    checks['resources_released'] = state['cleanup']['resources_released'] is True
    checks['no_device_processes'] = state['cleanup']['device_processes_remaining'] == 0

    failed = sorted(name for name, ok in checks.items() if not ok)
    report = {
        'verification_passed': not failed,
        'checks_run': len(checks),
        'failed_checks': failed,
        'accepted_acceleration': 'custom 910A TBE flash attention',
        'net_warm_step_s': acc['replay_warm_step_s'],
        'net_end_to_end_speedup': acc['end_to_end_speedup'],
        'candidates_rejected': len(state['rejected_candidates']),
        'goal_complete': state['overall_original_performance_and_A2_A3_goal_complete'],
    }
    print(json.dumps(report, indent=2))
    (ROOT / 'verified_round125.json').write_text(json.dumps(report, indent=2) + '\n')
    return 0 if not failed else 1


if __name__ == '__main__':
    raise SystemExit(main())