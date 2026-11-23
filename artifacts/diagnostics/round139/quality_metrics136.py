"""Host-only R136 quality metrics on unchanged NumPy arrays.

These are measurements, not acceptance tests or numerical tolerances. The
original/restored difference is not an allowance for a candidate. Iteration is
in logical C order with bounded buffers; only metric arithmetic uses FP64.
No resampling, alignment search, gain fitting, normalization or clipping occurs.
"""
from __future__ import annotations

import hashlib
import math
from numbers import Integral
from pathlib import Path
from typing import Iterator

import numpy as np

MAX_CHUNK_WORDS = 262144
MAX_COORDINATES = 16
MAX_WORST_FRAMES = 12


def _bounded_integer(value: int, name: str, minimum: int, maximum: int) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral):
        raise TypeError(f"{name} must be an integer")
    if not minimum <= int(value) <= maximum:
        raise ValueError(f"{name} must be in [{minimum}, {maximum}]")
    return int(value)


def _arrays(actual: np.ndarray, reference: np.ndarray) -> None:
    if not isinstance(actual, np.ndarray) or not isinstance(reference, np.ndarray):
        raise TypeError("Pass NumPy arrays, not tensors or implicitly converted data")
    if actual.shape != reference.shape:
        raise ValueError("Actual/reference shapes must match exactly")
    if actual.dtype != reference.dtype:
        raise ValueError("Actual/reference dtypes must match exactly")
    if actual.size == 0:
        raise ValueError("Empty arrays do not establish a quality comparison")


def _float_arrays(actual: np.ndarray, reference: np.ndarray) -> None:
    _arrays(actual, reference)
    if actual.dtype.kind != "f" or actual.dtype.itemsize not in (2, 4):
        raise TypeError("Only original FP16 or FP32 arrays are supported")


def _paired_chunks(
    actual: np.ndarray, reference: np.ndarray, chunk_words: int
) -> Iterator[tuple[np.ndarray, np.ndarray]]:
    # Buffered external-loop iteration also handles strided and negative-stride
    # views without reshape()/contiguous() copying the entire activation.
    with np.nditer(
        [actual, reference],
        flags=["external_loop", "buffered", "zerosize_ok"],
        op_flags=[["readonly"], ["readonly"]],
        order="C",
        buffersize=chunk_words,
    ) as iterator:
        for a, b in iterator:
            if a.size > chunk_words or b.size != a.size:
                raise RuntimeError("NumPy iterator exceeded the bounded chunk contract")
            yield a, b


class _Sum:
    """Compensated accumulation of already-FP64 chunk reductions."""

    def __init__(self) -> None:
        self.total = 0.0
        self.correction = 0.0

    def add(self, value: float) -> None:
        adjusted = float(value) - self.correction
        result = self.total + adjusted
        self.correction = (result - self.total) - adjusted
        self.total = result


class _Errors:
    def __init__(self) -> None:
        self.count = 0
        self.absolute = _Sum()
        self.squared = _Sum()
        self.reference_squared = _Sum()
        self.max_abs = 0.0

    def add(self, actual: np.ndarray, reference: np.ndarray) -> None:
        # Cast BEFORE subtraction/squaring, particularly for uint8 and Half.
        a = np.asarray(actual, dtype=np.float64)
        b = np.asarray(reference, dtype=np.float64)
        delta = a - b
        absolute = np.abs(delta)
        self.count += int(delta.size)
        self.absolute.add(float(np.sum(absolute, dtype=np.float64)))
        self.squared.add(float(np.sum(delta * delta, dtype=np.float64)))
        self.reference_squared.add(float(np.sum(b * b, dtype=np.float64)))
        self.max_abs = max(self.max_abs, float(np.max(absolute)))

    def basic(self) -> dict:
        mse = self.squared.total / self.count
        return {
            "max_abs_difference": self.max_abs,
            "mean_absolute_error": self.absolute.total / self.count,
            "mean_squared_error": mse,
            "root_mean_square_error": math.sqrt(mse),
        }


def _coordinates(mask: np.ndarray, offset: int, shape: tuple, remaining: int) -> list:
    if remaining == 0:
        return []
    locations = np.flatnonzero(mask)[:remaining]
    return [
        [int(c) for c in np.unravel_index(offset + int(i), shape, order="C")]
        for i in locations
    ]


def compare_float_arrays(
    actual: np.ndarray,
    reference: np.ndarray,
    *,
    chunk_words: int = MAX_CHUNK_WORDS,
    max_coordinates: int = MAX_COORDINATES,
) -> dict:
    """Compare complete FP16/FP32 logical words and FP64 error statistics.

    Direction is actual minus reference. The relative RMS ratio is the L2
    difference divided by max(reference L2, 1e-30); percent is 100 * ratio.
    Bit equality and numeric equality differ for positive/negative zero.
    """
    _float_arrays(actual, reference)
    chunk_words = _bounded_integer(chunk_words, "chunk_words", 1, MAX_CHUNK_WORDS)
    max_coordinates = _bounded_integer(
        max_coordinates, "max_coordinates", 0, MAX_COORDINATES
    )
    word_dtype = np.dtype(f"u{actual.dtype.itemsize}").newbyteorder(actual.dtype.byteorder)
    errors = _Errors()
    changed = 0
    offset = 0
    coords: list = []
    for a, b in _paired_chunks(actual, reference, chunk_words):
        if not bool(np.isfinite(a).all()) or not bool(np.isfinite(b).all()):
            raise ValueError("Nonfinite actual/reference values are not quality evidence")
        mask = a.view(word_dtype) != b.view(word_dtype)
        changed += int(np.count_nonzero(mask))
        coords.extend(_coordinates(mask, offset, actual.shape, max_coordinates - len(coords)))
        errors.add(a, b)
        offset += int(a.size)
    if offset != actual.size:
        raise RuntimeError("The whole logical word range was not traversed")
    difference_l2 = math.sqrt(errors.squared.total)
    reference_l2 = math.sqrt(errors.reference_squared.total)
    ratio = difference_l2 / max(reference_l2, 1e-30)
    result = {
        "shape": list(actual.shape),
        "dtype": str(actual.dtype),
        "word_bits": 8 * actual.dtype.itemsize,
        "total_words": int(actual.size),
        "different_words": changed,
        "all_words_equal": changed == 0,
        "numerically_perfect": errors.squared.total == 0.0,
        "first_changed_coordinates": coords,
        "squared_L2_difference": errors.squared.total,
        "squared_L2_reference": errors.reference_squared.total,
        "difference_L2_norm": difference_l2,
        "reference_L2_norm": reference_l2,
        "reference_root_mean_square": reference_l2 / math.sqrt(errors.count),
        "reference_norm_clamped_at_1e_minus30": reference_l2 < 1e-30,
        "relative_RMS_ratio": ratio,
        "relative_RMS_percent": 100.0 * ratio,
        "complete_logical_word_range_compared": True,
        "finite_inputs": True,
        "chunk_words": chunk_words,
        "comparison_direction": "actual_minus_reference",
        "no_alignment_gain_normalization_or_clipping": True,
    }
    result.update(errors.basic())
    return result


def _psnr(mse: float) -> dict:
    if mse == 0.0:
        return {"psnr_db": None, "psnr_perfect": True, "psnr_status": "positive_infinity_perfect"}
    return {
        "psnr_db": 10.0 * (math.log10(255.0 ** 2) - math.log10(mse)),
        "psnr_perfect": False,
        "psnr_status": "finite",
    }


def compare_video_uint8(
    actual: np.ndarray,
    reference: np.ndarray,
    *,
    chunk_words: int = MAX_CHUNK_WORDS,
    max_coordinates: int = MAX_COORDINATES,
    worst_frames: int = MAX_WORST_FRAMES,
) -> dict:
    """Compare native RGB uint8 [frames, height, width, 3] without alteration.

    PSNR uses data_range=255 as the unit definition, not as array normalization.
    Changed channel values and changed RGB pixels are counted separately.
    Arithmetic mean per-frame PSNR is infinite if ANY frame is perfect; a
    separately named finite-only mean is also provided, never substituted for it.
    """
    _arrays(actual, reference)
    if actual.dtype != np.dtype(np.uint8) or actual.ndim != 4 or actual.shape[-1] != 3:
        raise TypeError("Video must be unchanged uint8 RGB [T, H, W, 3]")
    chunk_words = _bounded_integer(chunk_words, "chunk_words", 1, MAX_CHUNK_WORDS)
    max_coordinates = _bounded_integer(
        max_coordinates, "max_coordinates", 0, MAX_COORDINATES
    )
    worst_frames = _bounded_integer(worst_frames, "worst_frames", 0, MAX_WORST_FRAMES)
    total_errors = _Errors()
    frames: list = []
    coords: list = []
    changed_channels = 0
    changed_pixels = 0
    frame_words = int(actual[0].size)
    for index in range(actual.shape[0]):
        errors = _Errors()
        channel_count = 0
        pixel_count = 0
        offset = index * frame_words
        visited = 0
        carry = np.empty(0, dtype=np.bool_)
        for a, b in _paired_chunks(actual[index], reference[index], chunk_words):
            mask = a != b
            channel_count += int(np.count_nonzero(mask))
            coords.extend(_coordinates(mask, offset + visited, actual.shape, max_coordinates - len(coords)))
            # At most two channel flags cross a chunk boundary. Chunk sizes need
            # not be multiples of three, and no whole-frame pixel mask is made.
            grouped = np.concatenate((carry, mask)) if carry.size else mask
            complete = (grouped.size // 3) * 3
            if complete:
                pixel_count += int(np.count_nonzero(np.any(grouped[:complete].reshape(-1, 3), axis=1)))
            carry = grouped[complete:].copy()
            errors.add(a, b)
            visited += int(a.size)
        if visited != frame_words or carry.size:
            raise RuntimeError("Incomplete RGB logical-channel traversal")
        total_errors.count += errors.count
        total_errors.absolute.add(errors.absolute.total)
        total_errors.squared.add(errors.squared.total)
        total_errors.reference_squared.add(errors.reference_squared.total)
        total_errors.max_abs = max(total_errors.max_abs, errors.max_abs)
        changed_channels += channel_count
        changed_pixels += pixel_count
        frame = {
            "frame_index": int(index),
            "different_channel_values": channel_count,
            "different_RGB_pixels": pixel_count,
        }
        frame.update(errors.basic())
        frame.update(_psnr(frame["mean_squared_error"]))
        frames.append(frame)
    finite_frames = [f for f in frames if not f["psnr_perfect"]]
    finite_values = [f["psnr_db"] for f in finite_frames]
    perfect_count = len(frames) - len(finite_frames)
    finite_mean = math.fsum(finite_values) / len(finite_values) if finite_values else None
    worst = sorted(finite_frames, key=lambda f: (f["psnr_db"], f["frame_index"]))[:worst_frames]
    result = {
        "shape": list(actual.shape),
        "dtype": "uint8",
        "total_channel_values": int(actual.size),
        "total_RGB_pixels": int(actual.size // 3),
        "different_channel_values": changed_channels,
        "different_RGB_pixels": changed_pixels,
        "all_channel_values_equal": changed_channels == 0,
        "first_changed_coordinates": coords,
        "data_range_for_PSNR_only": 255,
        "per_frame": frames,
        "perfect_frame_count": perfect_count,
        "mean_per_frame_psnr_db": None if perfect_count else finite_mean,
        "mean_per_frame_psnr_infinite": bool(perfect_count),
        "finite_only_frame_psnr_mean_db": finite_mean,
        "finite_frame_count": len(finite_frames),
        "worst_frame_indices": [f["frame_index"] for f in worst],
        "complete_logical_channel_and_RGB_pixel_range_compared": True,
        "chunk_words": chunk_words,
        "comparison_direction": "actual_minus_reference",
        "no_alignment_gain_normalization_or_clipping": True,
    }
    result.update(total_errors.basic())
    result.update(_psnr(result["mean_squared_error"]))
    return result


class _BivariateMoments:
    """Bounded Chan-merge centered moments for Pearson correlation."""

    def __init__(self) -> None:
        self.count = 0
        self.mean_actual = 0.0
        self.mean_reference = 0.0
        self.m2_actual = 0.0
        self.m2_reference = 0.0
        self.cross = 0.0

    def add(self, actual: np.ndarray, reference: np.ndarray) -> None:
        a = np.asarray(actual, dtype=np.float64)
        b = np.asarray(reference, dtype=np.float64)
        n = int(a.size)
        mean_a = float(np.mean(a, dtype=np.float64))
        mean_b = float(np.mean(b, dtype=np.float64))
        da = a - mean_a
        db = b - mean_b
        m2a = float(np.sum(da * da, dtype=np.float64))
        m2b = float(np.sum(db * db, dtype=np.float64))
        cross = float(np.sum(da * db, dtype=np.float64))
        if self.count == 0:
            self.count = n
            self.mean_actual, self.mean_reference = mean_a, mean_b
            self.m2_actual, self.m2_reference, self.cross = m2a, m2b, cross
            return
        combined = self.count + n
        delta_a = mean_a - self.mean_actual
        delta_b = mean_b - self.mean_reference
        factor = self.count * n / combined
        self.m2_actual += m2a + delta_a * delta_a * factor
        self.m2_reference += m2b + delta_b * delta_b * factor
        self.cross += cross + delta_a * delta_b * factor
        self.mean_actual += delta_a * n / combined
        self.mean_reference += delta_b * n / combined
        self.count = combined

    def result(self) -> dict:
        if self.m2_actual == 0.0 or self.m2_reference == 0.0:
            reason = "constant_actual_and_reference" if self.m2_actual == self.m2_reference == 0.0 else (
                "constant_actual" if self.m2_actual == 0.0 else "constant_reference"
            )
            correlation = None
        else:
            reason = None
            correlation = self.cross / math.sqrt(self.m2_actual * self.m2_reference)
        return {
            "pearson_correlation": correlation,
            "correlation_defined": correlation is not None,
            "correlation_undefined_reason": reason,
            "actual_mean": self.mean_actual,
            "reference_mean": self.mean_reference,
            "correlation_method": "FP64_centered_Chan_merge_no_alignment_or_gain_fit",
        }


def _audio_signal_metrics(
    actual: np.ndarray, reference: np.ndarray, errors: dict, chunk_words: int
) -> dict:
    difference_squared = errors["squared_L2_difference"]
    reference_squared = errors["squared_L2_reference"]
    perfect = difference_squared == 0.0
    reference_zero = reference_squared == 0.0
    if perfect:
        snr, status = None, "positive_infinity_perfect_numeric"
    elif reference_zero:
        snr, status = None, "negative_infinity_zero_reference"
    else:
        snr = 10.0 * (math.log10(reference_squared) - math.log10(difference_squared))
        status = "finite"
    moments = _BivariateMoments()
    for a, b in _paired_chunks(actual, reference, chunk_words):
        moments.add(a, b)
    result = {
        "snr_db": snr,
        "snr_perfect": perfect,
        "snr_reference_zero": reference_zero,
        "snr_status": status,
    }
    result.update(moments.result())
    return result


def compare_audio_float32(
    actual: np.ndarray,
    reference: np.ndarray,
    *,
    chunk_words: int = MAX_CHUNK_WORDS,
    max_coordinates: int = MAX_COORDINATES,
) -> dict:
    """Compare unchanged stereo FP32 [2, samples], globally and per channel."""
    _arrays(actual, reference)
    if actual.dtype != np.dtype(np.float32) or actual.ndim != 2 or actual.shape[0] != 2:
        raise TypeError("Audio must remain stereo FP32 [2, samples]")
    chunk_words = _bounded_integer(chunk_words, "chunk_words", 1, MAX_CHUNK_WORDS)
    result = compare_float_arrays(
        actual, reference, chunk_words=chunk_words, max_coordinates=max_coordinates
    )
    result.update(_audio_signal_metrics(actual, reference, result, chunk_words))
    channels = []
    for channel in range(2):
        item = compare_float_arrays(
            actual[channel], reference[channel], chunk_words=chunk_words,
            max_coordinates=max_coordinates,
        )
        item.update(_audio_signal_metrics(actual[channel], reference[channel], item, chunk_words))
        item["channel_index"] = channel
        channels.append(item)
    result["channels"] = channels
    return result


def sha256_file(path: str | Path, *, chunk_bytes: int = 1048576) -> str:
    """Hash complete file bytes, including .npy/.pt headers; no tensor loading."""
    chunk_bytes = _bounded_integer(chunk_bytes, "chunk_bytes", 1, 8 * 1048576)
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        while True:
            block = stream.read(chunk_bytes)
            if not block:
                break
            digest.update(block)
    return digest.hexdigest()
