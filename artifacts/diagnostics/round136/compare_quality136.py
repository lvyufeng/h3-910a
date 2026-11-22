#!/usr/bin/env python3
"""Host-only R136 postprocessing after all model processes have exited.

No model, checkpoint, encoder, VAE or NPU computation is performed here. Torch
is imported lazily ONLY to deserialize checksum-verified, CPU-saved final
latents and own-live encoder conditioning with weights_only=True. No teacher
features or model weights are loaded. Native frames/audio remain on host.
The report is scalar JSON, not an adoption decision or a numerical tolerance.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import re
import stat
import sys

import numpy as np

from quality_metrics136 import (
    MAX_CHUNK_WORDS,
    compare_audio_float32,
    compare_float_arrays,
    compare_video_uint8,
    sha256_file,
)

ROOT = Path("/tmp/h3_quality13601")
SESSION_NAME = "engine_quality136a"
LABELS = ("r01_original_false", "r02_candidate_true", "r03_restored_false")
FRAME_SHAPE = (124, 768, 1344, 3)
AUDIO_SHAPE = (2, 165600)
CHECKS_PER_REQUEST = 49 * 155 * 8
TOTAL_CHECKS = 3 * CHECKS_PER_REQUEST
MAX_LATENT_FILE_BYTES = 512 * 1024 ** 2
SHA256_PATTERN = re.compile(r"[0-9a-f]{64}\Z")
LATENT_PATTERN = re.compile(r"final_latents_pid[0-9]+127\.pt\Z")
ENCODER_PATTERN = re.compile(r"live_encoder_pid[0-9]+127\.pt\Z")


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _exact_int(value, expected: int, name: str) -> None:
    _require(type(value) is int and value == expected, f"Invalid {name}")


def validate_quality_receipts(process: dict, lifecycle: dict) -> list[dict]:
    """Pure scalar receipt gates; callable by CPU tests without any file/Torch IO."""
    _require(isinstance(process, dict) and isinstance(lifecycle, dict), "Receipt objects must be dictionaries")
    _exact_int(process.get("exit_code"), 0, "model process exit_code")
    _require(process.get("all_private_children_exited") is True, "Model/private child teardown is not verified")
    _require(not process.get("completion_unknown"), "Model completion is unknown")
    _require(not process.get("cleanup_failure"), "Model cleanup failed")
    _require(not process.get("cohort_monitor_errors"), "Model cohort monitor reported an error")
    _require(not process.get("private_children_after_parent_exit"), "Private children were still present")
    _require(lifecycle.get("completed") is True, "Generation lifecycle is incomplete")
    _require(not lifecycle.get("failure"), "Generation lifecycle failed")
    _exact_int(lifecycle.get("strict_checks_allranks"), TOTAL_CHECKS, "total strict checks")
    requests = lifecycle.get("requests")
    _require(isinstance(requests, list) and len(requests) == 3, "Exactly three complete request receipts are required")
    for index, (row, label) in enumerate(zip(requests, LABELS), start=1):
        _require(isinstance(row, dict), "Invalid request receipt")
        _exact_int(row.get("index"), index, "request index/order")
        _require(row.get("label") == label, "Unexpected original/candidate/restored label")
        _require(row.get("completed") is True, "A request did not complete")
        _exact_int(row.get("actual_DiT_evaluations"), 49, "actual DiT evaluations")
        _exact_int(row.get("strict_checks_allranks"), CHECKS_PER_REQUEST, "request strict checks")
        for key in ("rawframes_sha256", "rawaudio_sha256", "final_latents_sha256", "encoder_hidden_file_sha256"):
            value = row.get(key)
            _require(isinstance(value, str) and SHA256_PATTERN.fullmatch(value) is not None, f"Missing/invalid whole-file {key}")
    return requests


def _directory(path: Path, expected: Path) -> Path:
    _require(not path.is_symlink(), "Refuse symlinked private directory")
    _require(path.is_dir() and path.resolve() == expected.resolve(), "Unexpected private directory")
    return path.resolve()


def _regular_file(path: Path, parent: Path) -> Path:
    _require(not path.is_symlink(), "Refuse symlinked evidence file")
    _require(path.parent.resolve() == parent.resolve(), "Evidence escaped its exact private directory")
    info = path.stat()
    _require(stat.S_ISREG(info.st_mode), "Evidence must be a regular file")
    _require(path.resolve().parent == parent.resolve(), "Resolved evidence escaped its directory")
    return path.resolve()


def _read_json(path: Path, parent: Path) -> dict:
    path = _regular_file(path, parent)
    _require(path.stat().st_size <= 32 * 1024 ** 2, "Scalar receipt exceeds the bounded JSON budget")
    value = json.loads(path.read_text(encoding="utf-8"))
    _require(isinstance(value, dict), "Expected a scalar JSON object receipt")
    return value


def _file_signature(path: Path) -> tuple:
    info = path.stat()
    return (info.st_dev, info.st_ino, info.st_size, info.st_mtime_ns, info.st_ctime_ns)


def _verify_source_files(root: Path, process: dict) -> dict:
    sources = process.get("source_sha256")
    _require(isinstance(sources, dict) and bool(sources), "Executed source checksums are missing")
    verified = {}
    for original_path, digest in sources.items():
        _require(isinstance(original_path, str), "Invalid source path")
        p = Path(original_path)
        if p.is_absolute():
            _require(p.parent.resolve() == root, "Source checksum path is outside the experiment root")
        else:
            _require(len(p.parts) == 1, "Relative source names must be basenames")
        _require(p.suffix == ".py" and p.name not in verified, "Invalid/duplicate executed source entry")
        _require(isinstance(digest, str) and SHA256_PATTERN.fullmatch(digest) is not None, "Invalid executed source SHA")
        source = _regular_file(root / p.name, root)
        _require(sha256_file(source) == digest, "Executed source changed before postprocessing")
        verified[p.name] = digest
    return verified


def _load_latents_cpu(path: Path) -> tuple[dict, object]:
    # This function is never used by the metric library or its NumPy tests.
    # The caller has already checked completed model teardown and file SHA.
    import torch

    _require(path.stat().st_size <= MAX_LATENT_FILE_BYTES, "Latent artifact exceeds the CPU-only load budget")
    payload = torch.load(path, map_location="cpu", weights_only=True)
    _require(isinstance(payload, dict) and set(payload) == {"video_latent", "audio_latent", "actual_DiT_evaluations"}, "Not an original final-latent artifact")
    _exact_int(payload["actual_DiT_evaluations"], 49, "latent DiT evaluations")
    arrays = {}
    for name in ("video_latent", "audio_latent"):
        tensor = payload[name]
        _require(isinstance(tensor, torch.Tensor) and tensor.device.type == "cpu", "Latents must deserialize as CPU tensors only")
        _require(tensor.dtype in (torch.float16, torch.float32), "Do not cast or replace the original latent dtype")
        arrays[name] = tensor.detach().numpy()
    return arrays, payload


def _load_encoder_hidden_cpu(path: Path) -> tuple[np.ndarray, object]:
    """Deserialize only the existing own-live Half conditioning, never a teacher."""
    import torch

    _require(path.stat().st_size <= MAX_LATENT_FILE_BYTES, "Own-live conditioning exceeds the CPU-only load budget")
    payload = torch.load(path, map_location="cpu", weights_only=True)
    _require(isinstance(payload, dict) and set(payload) == {"hidden_states"}, "Not an own-live encoder artifact")
    tensor = payload["hidden_states"]
    _require(isinstance(tensor, torch.Tensor) and tensor.device.type == "cpu" and tensor.dtype == torch.float16, "Own-live conditioning must remain CPU Half")
    return tensor.detach().numpy(), payload


def _write_new_json(output: Path, report: dict, root: Path) -> None:
    """Exclusive atomic publication; an existing report is never overwritten."""
    intended = root / "quality_comparison136.json"
    _require(output.resolve() == intended and output.parent.resolve() == root, "Unexpected report destination")
    text = json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n"
    temporary = root / f".quality_comparison136.{os.getpid()}.pending"
    _require(temporary.resolve() == root / f".quality_comparison136.{os.getpid()}.pending", "Unexpected private temporary file")
    created = False
    try:
        descriptor = os.open(temporary, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
        created = True
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            stream.write(text)
            stream.flush()
            os.fsync(stream.fileno())
        _require(output.resolve() == intended, "Resolved output changed")
        os.link(temporary, output)  # Fails on an existing destination; no overwrite.
    finally:
        if created:
            # Check the exact resolved target before deleting only our temporary.
            expected = root / f".quality_comparison136.{os.getpid()}.pending"
            _require(temporary.resolve() == expected and temporary.parent.resolve() == root, "Unexpected temporary cleanup target")
            temporary.unlink()


def run_quality_comparison(root: Path = ROOT, *, chunk_words: int = MAX_CHUNK_WORDS) -> dict:
    """Run scalar host postprocessing only after finished/verified model execution."""
    root = Path(root)
    _require(type(chunk_words) is int and 1 <= chunk_words <= MAX_CHUNK_WORDS, "Chunk size must be a bounded integer before any evidence/Torch loading")
    _require(root.resolve() == ROOT.resolve(), "Only the exact R136 owned experiment root is supported")
    root = _directory(root, ROOT)
    session = _directory(root / SESSION_NAME, root / SESSION_NAME)
    output = root / "quality_comparison136.json"
    _require(not output.exists() and not output.is_symlink(), "Preserve the existing quality report; do not overwrite it")
    process = _read_json(root / "engine_quality136a_process.json", root)
    lifecycle = _read_json(session / "lifecycle_result136.json", session)
    requests = validate_quality_receipts(process, lifecycle)
    sources = _verify_source_files(root, process)
    snapshots = {}
    artifacts = []
    signatures = {}
    owners = []
    maps = []
    try:
        # Verify ALL three raw-file/latent hashes before importing Torch or
        # opening any arrays. Hash scope is complete .npy/.pt file bytes.
        verified_paths = []
        for row in requests:
            directory = _directory(session / row["label"], session / row["label"])
            frame_path = _regular_file(directory / "nativeframes136.npy", directory)
            audio_path = _regular_file(directory / "audio136.npy", directory)
            matches = sorted(directory.glob("final_latents_pid*127.pt"))
            _require(len(matches) == 1 and LATENT_PATTERN.fullmatch(matches[0].name) is not None, "Exactly one native final-latent file is required per request")
            latent_path = _regular_file(matches[0], directory)
            encoder_matches = sorted(directory.glob("live_encoder_pid*127.pt"))
            _require(len(encoder_matches) == 1 and ENCODER_PATTERN.fullmatch(encoder_matches[0].name) is not None, "Exactly one own-live encoder artifact is required per request")
            encoder_path = _regular_file(encoder_matches[0], directory)
            _require(latent_path.stat().st_size <= MAX_LATENT_FILE_BYTES and encoder_path.stat().st_size <= MAX_LATENT_FILE_BYTES, "Refuse oversized/non-native CPU artifacts")
            paths = (("rawframes_sha256", frame_path), ("rawaudio_sha256", audio_path), ("final_latents_sha256", latent_path), ("encoder_hidden_file_sha256", encoder_path))
            for key, path in paths:
                signature = _file_signature(path)
                _require(sha256_file(path) == row[key], "Native media/latent whole-file SHA differs from the completed lifecycle receipt")
                _require(_file_signature(path) == signature, "Artifact changed while it was being hashed")
                signatures[path] = signature
            verified_paths.append((row, frame_path, audio_path, latent_path, encoder_path))
            artifacts.append({"index": row["index"], "label": row["label"], "files": {key: {"path": str(path), "sha256": row[key], "bytes": path.stat().st_size} for key, path in paths}})
        for row, frame_path, audio_path, latent_path, encoder_path in verified_paths:
            frames = np.load(frame_path, mmap_mode="r", allow_pickle=False)
            maps.append(frames)
            audio = np.load(audio_path, mmap_mode="r", allow_pickle=False)
            maps.append(audio)
            _require(isinstance(frames, np.memmap) and frames.shape == FRAME_SHAPE and frames.dtype == np.uint8, "Native RGB frame shape/dtype must not be altered")
            _require(isinstance(audio, np.memmap) and audio.shape == AUDIO_SHAPE and audio.dtype == np.float32, "Native stereo audio shape/dtype must not be altered")
            latents, owner = _load_latents_cpu(latent_path)
            owners.append(owner)  # Keep the NumPy views' original CPU storages alive.
            conditioning, encoder_owner = _load_encoder_hidden_cpu(encoder_path)
            owners.append(encoder_owner)
            snapshots[row["label"]] = {"frames": frames, "audio": audio, "encoder_hidden": conditioning, **latents}
        combinations = (
            ("candidate_vs_original", LABELS[1], LABELS[0]),
            ("restored_vs_original", LABELS[2], LABELS[0]),
            ("candidate_vs_restored", LABELS[1], LABELS[2]),
        )
        comparisons = []
        for pair_id, actual_label, reference_label in combinations:
            actual, reference = snapshots[actual_label], snapshots[reference_label]
            comparisons.append({
                "pair_id": pair_id,
                "actual_request_label": actual_label,
                "reference_request_label": reference_label,
                "comparison_direction": "actual_minus_reference",
                "original_restored_noise_is_not_a_candidate_tolerance": True,
                "own_live_encoder_conditioning": compare_float_arrays(actual["encoder_hidden"], reference["encoder_hidden"], chunk_words=chunk_words),
                "video_latent": compare_float_arrays(actual["video_latent"], reference["video_latent"], chunk_words=chunk_words),
                "audio_latent": compare_float_arrays(actual["audio_latent"], reference["audio_latent"], chunk_words=chunk_words),
                "native_decoded_video": compare_video_uint8(actual["frames"], reference["frames"], chunk_words=chunk_words),
                "native_decoded_audio": compare_audio_float32(actual["audio"], reference["audio"], chunk_words=chunk_words),
            })
        for path, signature in signatures.items():
            _require(_file_signature(path) == signature, "Native artifact changed during postprocessing")
        report = {
            "completed": True,
            "scope": "Host-only unchanged final latents/native RGB uint8/stereo FP32 after 49 actual evaluations",
            "comparison_scope": "DefaultFalse original/restored vs temporaryTrue plus final-AdaLN candidate COMBINATION; not isolated AdaLN-only attribution",
            "full49_candidate_repeatability_tested": False,
            "own_live_encoder_conditioning_not_teacher_features": True,
            "global_deterministic_flag_can_affect_encoder_and_VAE_too": True,
            "generation_requests": 3,
            "strict_checks_allranks": TOTAL_CHECKS,
            "model_process_pid": process.get("pid"),
            "model_process_exit_code": 0,
            "model_private_children_exited_before_postprocessing": True,
            "executed_source_SHA_verified": sources,
            "artifact_checksum_scope": "Complete saved .npy/.pt file bytes, including headers",
            "host_only_artifacts": artifacts,
            "numpy_version": np.__version__,
            "chunk_words": chunk_words,
            "metric_arithmetic": "FP64 chunk error/moment reductions; original arrays/dtypes preserved",
            "arrays_normalized_resampled_aligned_clipped_or_gain_fitted": False,
            "encoded_media_PSNR_measured": False,
            "raw_decoded_RGB_PSNR_is_not_an_automatic_tolerance": True,
            "numeric_baseline_adopted": False,
            "new_numeric_tolerance": False,
            "quality_pass_or_acceptance_decided": False,
            "weights_or_activation_archives_transferred_to_client": False,
            "comparisons": comparisons,
        }
        _write_new_json(output, report, root)
        return report
    finally:
        for array in maps:
            if getattr(array, "_mmap", None) is not None:
                array._mmap.close()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--chunk-words", type=int, default=MAX_CHUNK_WORDS)
    args = parser.parse_args(argv)
    try:
        report = run_quality_comparison(args.root, chunk_words=args.chunk_words)
        summary = {
            "completed": True,
            "report": str(args.root / "quality_comparison136.json"),
            "numeric_baseline_adopted": False,
            "quality_pass_or_acceptance_decided": False,
            "comparisons": [{
                "pair_id": pair["pair_id"],
                "own_live_encoder_relative_RMS_percent": pair["own_live_encoder_conditioning"]["relative_RMS_percent"],
                "video_latent_relative_RMS_percent": pair["video_latent"]["relative_RMS_percent"],
                "audio_latent_relative_RMS_percent": pair["audio_latent"]["relative_RMS_percent"],
                "native_RGB_PSNR_db": pair["native_decoded_video"]["psnr_db"],
                "native_RGB_perfect": pair["native_decoded_video"]["psnr_perfect"],
                "native_audio_SNR_db": pair["native_decoded_audio"]["snr_db"],
                "native_audio_SNR_status": pair["native_decoded_audio"]["snr_status"],
            } for pair in report["comparisons"]],
        }
        print(json.dumps(summary, sort_keys=True, allow_nan=False))
        return 0
    except Exception as error:
        print(json.dumps({"completed": False, "error": {"type": type(error).__name__, "message": str(error)}, "model_rerun_requested": False}, allow_nan=False), file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
