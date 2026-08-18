"""Detection, reporting, and pitch-preserving audio alignment."""

from __future__ import annotations

import shutil
import subprocess
import tempfile
from dataclasses import asdict, dataclass
from itertools import pairwise
from pathlib import Path
from typing import Any

import numpy as np
import soundfile as sf

from rift_svc.audio_tools import parse_time, read_float_audio


@dataclass(frozen=True)
class Anchor:
    source: float
    target: float


def read_mono(path: Path) -> tuple[np.ndarray, int]:
    audio, sample_rate = read_float_audio(path)
    return audio.mean(axis=1, dtype=np.float32), sample_rate


def log_rms_envelope(
    audio: np.ndarray, window_frames: int, hop_frames: int
) -> np.ndarray:
    if window_frames <= 0 or hop_frames <= 0:
        raise ValueError("analysis window and hop must be positive")
    if len(audio) < window_frames:
        raise ValueError("audio is shorter than the analysis window")
    squared = np.square(audio, dtype=np.float64)
    cumulative = np.concatenate(([0.0], np.cumsum(squared, dtype=np.float64)))
    energy = (cumulative[window_frames:] - cumulative[:-window_frames]) / window_frames
    return np.log(np.sqrt(energy[::hop_frames] + 1e-12) + 1e-8)


def normalized_correlation(left: np.ndarray, right: np.ndarray) -> float:
    left = left - left.mean()
    right = right - right.mean()
    denominator = np.linalg.norm(left) * np.linalg.norm(right)
    return float(np.dot(left, right) / denominator) if denominator else 0.0


def lag_score(
    reference: np.ndarray,
    source: np.ndarray,
    lag: int,
    reference_first: int,
    reference_last: int,
) -> float:
    """Score source[index + lag] against reference[index]."""
    first = max(reference_first, -lag)
    last = min(reference_last, len(source) - lag)
    if last - first < 8:
        return float("-inf")
    return normalized_correlation(reference[first:last], source[first + lag : last + lag])


def best_lag(
    reference: np.ndarray,
    source: np.ndarray,
    first_lag: int,
    last_lag: int,
    reference_first: int,
    reference_last: int,
) -> tuple[int, float]:
    scores = [
        (lag_score(reference, source, lag, reference_first, reference_last), lag)
        for lag in range(first_lag, last_lag + 1)
    ]
    score, lag = max(scores)
    if not np.isfinite(score):
        raise ValueError("no overlapping audio remained for alignment detection")
    return lag, score


def affine_boundary_anchors(
    source_duration: float,
    target_duration: float,
    intercept: float,
    ratio: float,
) -> list[Anchor]:
    """Clip source = intercept + ratio * target to both file timelines."""
    if ratio <= 0:
        raise ValueError("detected alignment ratio is not positive")
    target_first = max(0.0, -intercept / ratio)
    target_last = min(target_duration, (source_duration - intercept) / ratio)
    if target_last <= target_first:
        raise ValueError("detected source and reference timelines do not overlap")
    return [
        Anchor(max(0.0, intercept + ratio * target_first), target_first),
        Anchor(min(source_duration, intercept + ratio * target_last), target_last),
    ]


def _weighted_affine_fit(
    target_centers: np.ndarray,
    source_centers: np.ndarray,
    correlations: np.ndarray,
    min_correlation: float,
) -> tuple[float, float]:
    if min_correlation >= 1.0:
        weights = np.ones_like(correlations)
    else:
        weights = np.clip(
            (correlations - min_correlation) / (1.0 - min_correlation), 0.05, 1.0
        )
    ratio, intercept = np.polyfit(
        target_centers, source_centers, 1, w=weights
    )
    return float(ratio), float(intercept)


def detect_alignment(
    source_path: Path,
    reference_path: Path,
    *,
    search_seconds: float,
    local_window_seconds: float,
    local_search_seconds: float,
    min_correlation: float,
    max_fit_residual_ms: float,
    analysis_window_ms: float,
    analysis_hop_ms: float,
    edge_seconds: float,
) -> tuple[list[Anchor], dict[str, Any]]:
    source, source_rate = read_mono(source_path)
    reference, reference_rate = read_mono(reference_path)
    if source_rate != reference_rate:
        raise ValueError(
            f"sample rates differ: source={source_rate}, reference={reference_rate}"
        )
    sample_rate = source_rate
    window_frames = round(analysis_window_ms * sample_rate / 1000.0)
    hop_frames = round(analysis_hop_ms * sample_rate / 1000.0)
    source_env = log_rms_envelope(source, window_frames, hop_frames)
    reference_env = log_rms_envelope(reference, window_frames, hop_frames)
    envelope_rate = sample_rate / hop_frames
    edge = round(edge_seconds * envelope_rate)
    if min(len(source_env), len(reference_env)) <= 2 * edge + 8:
        raise ValueError("audio is too short for the requested edge exclusion")

    maximum_lag = round(search_seconds * envelope_rate)
    global_lag, global_score = best_lag(
        reference_env,
        source_env,
        -maximum_lag,
        maximum_lag,
        edge,
        len(reference_env) - edge,
    )
    if global_score < min_correlation:
        raise ValueError(
            f"global alignment correlation {global_score:.3f} is below "
            f"the required {min_correlation:.3f}"
        )

    source_duration = len(source) / sample_rate
    target_duration = len(reference) / sample_rate
    global_lag_seconds = global_lag / envelope_rate
    usable_target_first = max(edge_seconds, -global_lag_seconds + edge_seconds)
    usable_target_last = min(
        target_duration - edge_seconds,
        source_duration - global_lag_seconds - edge_seconds,
    )
    if usable_target_last - usable_target_first < local_window_seconds:
        raise ValueError("aligned overlap is shorter than --local-window-seconds")

    local_radius = round(local_search_seconds * envelope_rate)
    local_window_frames = round(local_window_seconds * envelope_rate)
    local_results: list[dict[str, Any]] = []
    first_frame = round(usable_target_first * envelope_rate)
    last_frame = round(usable_target_last * envelope_rate)
    for window_first in range(first_frame, last_frame, local_window_frames):
        window_last = min(window_first + local_window_frames, last_frame)
        if window_last - window_first < local_window_frames // 2:
            continue
        lag, score = best_lag(
            reference_env,
            source_env,
            global_lag - local_radius,
            global_lag + local_radius,
            window_first,
            window_last,
        )
        target_center = (window_first + window_last) / 2.0 / envelope_rate
        source_center = target_center + lag / envelope_rate
        local_results.append(
            {
                "target_start": window_first / envelope_rate,
                "target_end": window_last / envelope_rate,
                "target_center": target_center,
                "source_center": source_center,
                "source_minus_target_ms": lag / envelope_rate * 1000.0,
                "correlation": score,
                "passed_correlation": score >= min_correlation,
                "used_in_final_fit": False,
            }
        )

    accepted_indices = [
        index
        for index, result in enumerate(local_results)
        if result["passed_correlation"]
    ]
    if len(accepted_indices) < 2:
        raise ValueError("fewer than two local windows passed the correlation threshold")
    target_centers = np.asarray(
        [local_results[index]["target_center"] for index in accepted_indices]
    )
    source_centers = np.asarray(
        [local_results[index]["source_center"] for index in accepted_indices]
    )
    correlations = np.asarray(
        [local_results[index]["correlation"] for index in accepted_indices]
    )
    initial_ratio, initial_intercept = _weighted_affine_fit(
        target_centers, source_centers, correlations, min_correlation
    )
    initial_residual = source_centers - (
        initial_intercept + initial_ratio * target_centers
    )
    residual_median = float(np.median(initial_residual))
    mad = float(np.median(np.abs(initial_residual - residual_median)))
    outlier_limit_seconds = max(
        2.0 * analysis_hop_ms / 1000.0,
        3.0 * 1.4826 * mad,
    )
    inliers = np.abs(initial_residual - residual_median) <= outlier_limit_seconds
    final_indices = [
        accepted_indices[index] for index, included in enumerate(inliers) if included
    ]
    if len(final_indices) >= 2:
        ratio, intercept = _weighted_affine_fit(
            target_centers[inliers],
            source_centers[inliers],
            correlations[inliers],
            min_correlation,
        )
    else:
        ratio, intercept = initial_ratio, initial_intercept
        final_indices = accepted_indices
        inliers = np.ones(len(accepted_indices), dtype=np.bool_)
    for index in final_indices:
        local_results[index]["used_in_final_fit"] = True

    fitted_target = target_centers[inliers]
    fitted_source = source_centers[inliers]
    residual_ms = (fitted_source - (intercept + ratio * fitted_target)) * 1000.0
    residual_rms_ms = float(np.sqrt(np.mean(np.square(residual_ms))))
    residual_max_abs_ms = float(np.max(np.abs(residual_ms)))
    candidate_anchors = affine_boundary_anchors(
        source_duration, target_duration, intercept, ratio
    )
    unsafe_reasons = []
    if len(final_indices) < 3:
        unsafe_reasons.append("fewer than three local windows support the final fit")
    if residual_max_abs_ms > max_fit_residual_ms:
        unsafe_reasons.append(
            f"fit residual {residual_max_abs_ms:.3f} ms exceeds "
            f"{max_fit_residual_ms:g} ms"
        )
    safe_to_apply = not unsafe_reasons
    suggested_anchors = candidate_anchors if safe_to_apply else []
    report: dict[str, Any] = {
        "schema": 2,
        "method": "weighted log-RMS cross-correlation with robust affine drift fit",
        "source": str(source_path.resolve()),
        "reference": str(reference_path.resolve()),
        "sample_rate": sample_rate,
        "source_duration": source_duration,
        "reference_duration": target_duration,
        "parameters": {
            "search_seconds": search_seconds,
            "local_window_seconds": local_window_seconds,
            "local_search_seconds": local_search_seconds,
            "min_correlation": min_correlation,
            "max_fit_residual_ms": max_fit_residual_ms,
            "analysis_window_ms": analysis_window_ms,
            "analysis_hop_ms": analysis_hop_ms,
            "edge_seconds": edge_seconds,
        },
        "global_source_minus_target_ms": global_lag_seconds * 1000.0,
        "global_correlation": global_score,
        "local_windows": local_results,
        "fit": {
            "source_seconds_at_target_zero": intercept,
            "source_seconds_per_target_second": ratio,
            "target_duration_change_percent": (1.0 / ratio - 1.0) * 100.0,
            "residual_rms_ms": residual_rms_ms,
            "residual_max_abs_ms": residual_max_abs_ms,
            "outlier_limit_ms": outlier_limit_seconds * 1000.0,
            "correlation_pass_count": len(accepted_indices),
            "final_window_count": len(final_indices),
        },
        "safe_to_apply": safe_to_apply,
        "unsafe_reasons": unsafe_reasons,
        "candidate_anchors": [asdict(anchor) for anchor in candidate_anchors],
        "suggested_anchors": [asdict(anchor) for anchor in suggested_anchors],
        "note": "inspect local confidence and audition after applying suggested anchors",
    }
    return suggested_anchors, report


def read_anchors(path: Path) -> list[Anchor]:
    anchors: list[Anchor] = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        text = line.split("#", 1)[0].strip()
        if not text:
            continue
        fields = text.split()
        if len(fields) != 2:
            raise ValueError(f"{path}:{line_number}: expected SOURCE_TIME TARGET_TIME")
        try:
            anchor = Anchor(parse_time(fields[0]), parse_time(fields[1]))
        except ValueError as exc:
            raise ValueError(f"{path}:{line_number}: {exc}") from exc
        if anchors and (
            anchor.source <= anchors[-1].source or anchor.target <= anchors[-1].target
        ):
            raise ValueError(
                f"{path}:{line_number}: source and target times must both increase"
            )
        anchors.append(anchor)
    if len(anchors) < 2:
        raise ValueError(f"{path}: at least two anchors are required")
    return anchors


def audio_info(path: Path) -> sf.SoundFile:
    try:
        return sf.info(path)
    except RuntimeError as exc:
        raise ValueError(f"cannot read audio metadata from {path}: {exc}") from exc


def canonicalize_anchors(
    anchors: list[Anchor], sample_rate: int, source_frames: int, target_frames: int
) -> list[Anchor]:
    """Snap textual timestamps to the only positions an audio file can represent."""
    if len(anchors) < 2:
        raise ValueError("at least two anchors are required")
    canonical: list[Anchor] = []
    for anchor in anchors:
        source_frame = round(anchor.source * sample_rate)
        target_frame = round(anchor.target * sample_rate)
        if source_frame < 0 or source_frame > source_frames:
            raise ValueError("source anchors exceed the source duration")
        if target_frame < 0 or target_frame > target_frames:
            raise ValueError("target anchors exceed the reference duration")
        snapped = Anchor(source_frame / sample_rate, target_frame / sample_rate)
        if canonical and (
            snapped.source <= canonical[-1].source
            or snapped.target <= canonical[-1].target
        ):
            raise ValueError(
                "source and target anchors must remain increasing after sample rounding"
            )
        canonical.append(snapped)
    return canonical


def alignment_report(source: Path, reference: Path, anchors: list[Anchor]) -> dict[str, Any]:
    source_info = audio_info(source)
    reference_info = audio_info(reference)
    if source_info.samplerate != reference_info.samplerate:
        raise ValueError(
            f"sample rates differ: source={source_info.samplerate}, "
            f"reference={reference_info.samplerate}"
        )
    anchors = canonicalize_anchors(
        anchors,
        source_info.samplerate,
        source_info.frames,
        reference_info.frames,
    )
    source_duration = source_info.frames / source_info.samplerate
    target_duration = reference_info.frames / reference_info.samplerate

    segments = []
    for left, right in pairwise(anchors):
        source_seconds = right.source - left.source
        target_seconds = right.target - left.target
        tempo = source_seconds / target_seconds
        segments.append(
            {
                "source_start": left.source,
                "source_end": right.source,
                "target_start": left.target,
                "target_end": right.target,
                "source_seconds": source_seconds,
                "target_seconds": target_seconds,
                "tempo": tempo,
                "stretch_percent": (target_seconds / source_seconds - 1.0) * 100.0,
                "drift_change_ms": (
                    (right.target - right.source) - (left.target - left.source)
                )
                * 1000.0,
            }
        )
    offsets_ms = [(anchor.target - anchor.source) * 1000.0 for anchor in anchors]
    return {
        "schema": 1,
        "source": str(source.resolve()),
        "reference": str(reference.resolve()),
        "sample_rate": source_info.samplerate,
        "source_frames": source_info.frames,
        "reference_frames": reference_info.frames,
        "source_duration": source_duration,
        "reference_duration": target_duration,
        "anchors": [asdict(anchor) for anchor in anchors],
        "anchor_offsets_ms": offsets_ms,
        "offset_range_ms": max(offsets_ms) - min(offsets_ms),
        "segments": segments,
        "uncovered_source_head_seconds": anchors[0].source,
        "uncovered_source_tail_seconds": source_duration - anchors[-1].source,
        "silent_target_head_seconds": anchors[0].target,
        "silent_target_tail_seconds": target_duration - anchors[-1].target,
    }


def atempo_chain(tempo: float) -> str:
    if tempo <= 0 or not np.isfinite(tempo):
        raise ValueError(f"invalid tempo ratio: {tempo}")
    factors = []
    remaining = tempo
    while remaining > 2.0:
        factors.append(2.0)
        remaining /= 2.0
    while remaining < 0.5:
        factors.append(0.5)
        remaining /= 0.5
    factors.append(remaining)
    return ",".join(f"atempo={factor:.12g}" for factor in factors)


def render_segment(
    source: Path,
    source_start: float,
    source_end: float,
    source_duration: float,
    output_frames: int,
    sample_rate: int,
    output: Path,
) -> None:
    """Render an exact segment using real guard audio, never post-tempo zero fill."""
    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg is None:
        raise RuntimeError("ffmpeg is required but was not found on PATH")
    source_seconds = source_end - source_start
    target_seconds = output_frames / sample_rate
    tempo = source_seconds / target_seconds
    guard_source_seconds = 0.1 * tempo
    extended_start = max(0.0, source_start - guard_source_seconds)
    extended_end = min(source_duration, source_end + guard_source_seconds)
    skip_frames = round((source_start - extended_start) / tempo * sample_rate)
    last_frame = skip_frames + output_frames
    subprocess.run(
        [
            ffmpeg,
            "-v",
            "error",
            "-xerror",
            "-nostdin",
            "-i",
            str(source),
            "-af",
            (
                f"atrim=start={extended_start:.12f}:end={extended_end:.12f},"
                f"asetpts=PTS-STARTPTS,apad=pad_dur=0.25,"
                f"{atempo_chain(tempo)},"
                f"atrim=start_sample={skip_frames}:end_sample={last_frame},"
                "asetpts=PTS-STARTPTS"
            ),
            "-ar",
            str(sample_rate),
            "-c:a",
            "pcm_f32le",
            "-y",
            str(output),
        ],
        check=True,
    )


def apply_alignment(
    source: Path,
    reference: Path,
    anchors: list[Anchor],
    crossfade_ms: float,
    max_stretch_percent: float,
) -> tuple[np.ndarray, dict[str, Any]]:
    report = alignment_report(source, reference, anchors)
    anchors = [Anchor(**anchor) for anchor in report["anchors"]]
    excessive = [
        segment
        for segment in report["segments"]
        if abs(segment["stretch_percent"]) > max_stretch_percent
    ]
    if excessive:
        worst = max(excessive, key=lambda item: abs(item["stretch_percent"]))
        raise ValueError(
            f"segment {worst['source_start']:.3f}-{worst['source_end']:.3f}s "
            f"requires {worst['stretch_percent']:+.4f}% stretch; "
            f"limit is ±{max_stretch_percent:g}%"
        )

    sample_rate = report["sample_rate"]
    source_info = audio_info(source)
    target_frames = report["reference_frames"]
    channels = source_info.channels
    output = np.zeros((target_frames, channels), dtype=np.float32)
    weights = np.zeros(target_frames, dtype=np.float32)
    fade_frames = round(crossfade_ms * sample_rate / 1000.0)
    if fade_frames and any(
        round((right.target - left.target) * sample_rate) <= fade_frames
        for left, right in pairwise(anchors)
    ):
        raise ValueError("crossfade is not shorter than every target segment")

    with tempfile.TemporaryDirectory(prefix="rift-align-") as temporary_name:
        temporary = Path(temporary_name)
        for index, (left, right) in enumerate(pairwise(anchors)):
            source_ratio = (right.source - left.source) / (right.target - left.target)
            pre_frames = fade_frames // 2 if index else 0
            post_frames = fade_frames - fade_frames // 2 if index < len(anchors) - 2 else 0
            target_first = round(left.target * sample_rate) - pre_frames
            target_last = round(right.target * sample_rate) + post_frames
            source_first = left.source - pre_frames / sample_rate * source_ratio
            source_last = right.source + post_frames / sample_rate * source_ratio
            if target_first < 0 or target_last > target_frames:
                raise ValueError("crossfade extends beyond the reference timeline")
            if source_first < 0 or source_last > report["source_duration"]:
                raise ValueError("crossfade extends beyond the source audio")
            segment_path = temporary / f"segment-{index:03d}.wav"
            render_segment(
                source,
                source_first,
                source_last,
                report["source_duration"],
                target_last - target_first,
                sample_rate,
                segment_path,
            )
            segment, segment_rate = read_float_audio(segment_path)
            if segment_rate != sample_rate or segment.shape != (
                target_last - target_first,
                channels,
            ):
                raise RuntimeError(f"unexpected rendered segment shape: {segment.shape}")
            blend = np.ones(len(segment), dtype=np.float32)
            if pre_frames:
                phase = np.linspace(0.0, np.pi, fade_frames, dtype=np.float32)
                blend[:fade_frames] = 0.5 - 0.5 * np.cos(phase)
            if post_frames:
                phase = np.linspace(0.0, np.pi, fade_frames, dtype=np.float32)
                blend[-fade_frames:] = 0.5 + 0.5 * np.cos(phase)
            output[target_first:target_last] += segment * blend[:, np.newaxis]
            weights[target_first:target_last] += blend

    covered = weights > 0
    output[covered] /= weights[covered, np.newaxis]
    report["crossfade_ms"] = crossfade_ms
    report["max_stretch_percent"] = max_stretch_percent
    return output, report
