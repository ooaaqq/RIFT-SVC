#!/usr/bin/env python3
"""Apply localized dynamic control to breath-heavy upper-mid/high vocal energy."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from scipy.ndimage import uniform_filter1d
from scipy.signal import butter, sosfiltfilt

from rift_svc.audio_tools import (
    concise_cli,
    ensure_new_paths,
    read_float_audio,
    write_float_wav_new,
)


def parse_region(value: str) -> tuple[float, float]:
    try:
        start_text, end_text = value.replace(",", ":").split(":", 1)
        start = float(start_text)
        end = float(end_text)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            f"invalid region {value!r}; expected START:END"
        ) from exc
    if start < 0.0 or end <= start:
        raise argparse.ArgumentTypeError(
            f"invalid region {value!r}; END must be greater than START"
        )
    return start, end


def read_regions(path: Path | None, regions: list[tuple[float, float]]) -> list[tuple[float, float]]:
    if path is not None:
        for line_number, line in enumerate(path.read_text().splitlines(), 1):
            text = line.split("#", 1)[0].strip()
            if not text:
                continue
            try:
                regions.append(parse_region(text))
            except argparse.ArgumentTypeError as exc:
                raise ValueError(f"{path}:{line_number}: {exc}") from exc
    if not regions:
        raise ValueError("at least one breath-control region is required")
    return sorted(regions)


def asymmetric_smooth(
    values: np.ndarray, sample_rate: int, attack_ms: float, release_ms: float
) -> np.ndarray:
    """Smooth a reduction curve with faster gain reduction and slower release."""
    attack = 1.0 - np.exp(-1.0 / max(1.0, attack_ms * sample_rate / 1000.0))
    release = 1.0 - np.exp(-1.0 / max(1.0, release_ms * sample_rate / 1000.0))
    output = np.empty_like(values)
    output[0] = values[0]
    for index in range(1, len(values)):
        coefficient = attack if values[index] < output[index - 1] else release
        output[index] = output[index - 1] + coefficient * (
            values[index] - output[index - 1]
        )
    return output


def localized_reduction_mask(
    length: int,
    sample_rate: int,
    regions: list[tuple[float, float]],
    fade_ms: float,
) -> np.ndarray:
    mask = np.zeros(length, dtype=np.float32)
    fade = max(0, round(fade_ms * sample_rate / 1000.0))
    merged: list[tuple[float, float]] = []
    for start, end in sorted(regions):
        if merged and start <= merged[-1][1]:
            merged[-1] = (merged[-1][0], max(merged[-1][1], end))
        else:
            merged.append((start, end))
    for start, end in merged:
        first = max(0, round(start * sample_rate))
        last = min(length, round(end * sample_rate))
        if last <= first:
            continue
        region_mask = np.ones(last - first, dtype=np.float32)
        left = min(fade, last - first)
        right = min(fade, last - first)
        if left:
            region_mask[:left] = np.minimum(
                region_mask[:left],
                np.linspace(0.0, 1.0, left, endpoint=False, dtype=np.float32),
            )
        if right:
            region_mask[-right:] = np.minimum(
                region_mask[-right:],
                np.linspace(1.0, 0.0, right, endpoint=False, dtype=np.float32),
            )
        mask[first:last] = np.maximum(mask[first:last], region_mask)
    return mask


def process_channel(
    channel: np.ndarray,
    sample_rate: int,
    regions: list[tuple[float, float]],
    highpass_hz: float,
    lowpass_hz: float,
    max_reduction_db: float,
    threshold_percentile: float,
    ratio: float,
    attack_ms: float,
    release_ms: float,
    fade_ms: float,
) -> tuple[np.ndarray, list[dict[str, float]]]:
    if len(channel) < 32:
        return channel.copy(), []
    highpass = butter(4, highpass_hz, btype="highpass", fs=sample_rate, output="sos")
    lowpass = butter(4, lowpass_hz, btype="lowpass", fs=sample_rate, output="sos")
    high_band = sosfiltfilt(lowpass, sosfiltfilt(highpass, channel))
    power = np.square(high_band, dtype=np.float64)
    window = max(8, round(0.020 * sample_rate))
    smoothed_power = uniform_filter1d(power, size=window, mode="nearest")
    envelope = np.sqrt(np.maximum(smoothed_power, 0.0))
    envelope_db = 20.0 * np.log10(envelope + 1e-12)
    reduction_db = np.zeros(len(channel), dtype=np.float64)
    reports: list[dict[str, float]] = []

    for start, end in regions:
        first = max(0, round(start * sample_rate))
        last = min(len(channel), round(end * sample_rate))
        if last <= first:
            continue
        threshold = float(
            np.percentile(envelope_db[first:last], threshold_percentile)
        )
        target = -np.clip(
            (envelope_db - threshold) * (ratio - 1.0) / ratio,
            0.0,
            max_reduction_db,
        )
        target[:first] = 0.0
        target[last:] = 0.0
        smoothed = asymmetric_smooth(
            target, sample_rate, attack_ms=attack_ms, release_ms=release_ms
        )
        region_mask = localized_reduction_mask(
            len(channel), sample_rate, [(start, end)], fade_ms
        )
        applied = smoothed * region_mask
        reduction_db = np.minimum(reduction_db, applied)
        reports.append(
            {
                "start": start,
                "end": end,
                "threshold_db": threshold,
                "max_reduction_db": float(-np.min(applied[first:last])),
                "mean_reduction_db": float(-np.mean(applied[first:last])),
            }
        )

    gain = np.power(10.0, reduction_db / 20.0).astype(np.float32)
    processed_high = high_band.astype(np.float32) * gain
    output = channel.astype(np.float32) + processed_high - high_band.astype(np.float32)
    return output, reports


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Dynamically attenuate localized breath-heavy high-frequency energy."
    )
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--regions-file", type=Path)
    parser.add_argument("--region", action="append", type=parse_region, default=[])
    parser.add_argument("--sample-rate", type=int, default=44100)
    parser.add_argument("--highpass-hz", type=float, required=True)
    parser.add_argument("--lowpass-hz", type=float, required=True)
    parser.add_argument("--max-reduction-db", type=float, required=True)
    parser.add_argument("--threshold-percentile", type=float, required=True)
    parser.add_argument("--ratio", type=float, required=True)
    parser.add_argument("--attack-ms", type=float, required=True)
    parser.add_argument("--release-ms", type=float, required=True)
    parser.add_argument("--fade-ms", type=float, required=True)
    args = parser.parse_args()

    if args.sample_rate <= 0 or args.highpass_hz <= 0:
        parser.error("sample rate and highpass frequency must be positive")
    if not 0.0 < args.highpass_hz < args.lowpass_hz < args.sample_rate / 2:
        parser.error("frequencies must satisfy 0 < highpass < lowpass < Nyquist")
    if args.max_reduction_db < 0.0 or args.ratio < 1.0:
        parser.error("max reduction must be non-negative and ratio must be at least 1")
    if args.attack_ms <= 0.0 or args.release_ms <= 0.0 or args.fade_ms < 0.0:
        parser.error("attack/release must be positive and fade cannot be negative")
    if not 0.0 <= args.threshold_percentile <= 100.0:
        parser.error("threshold percentile must be between 0 and 100")
    regions = read_regions(args.regions_file, list(args.region))

    ensure_new_paths([args.output])

    audio, sample_rate = read_float_audio(args.input)
    if sample_rate != args.sample_rate:
        raise ValueError(
            f"{args.input} has {sample_rate} Hz; expected {args.sample_rate} Hz"
        )
    if audio.shape[1] != 1:
        raise ValueError(
            "breath control expects a mono lead vocal; "
            f"got {audio.shape[1]} channels"
        )
    duration = audio.shape[0] / sample_rate
    outside = [(start, end) for start, end in regions if end > duration]
    if outside:
        values = ", ".join(f"{start:g}:{end:g}" for start, end in outside)
        raise ValueError(
            f"breath-control region exceeds the {duration:.6f}s input: {values}"
        )

    processed = np.empty_like(audio)
    processed[:, 0], reports = process_channel(
        audio[:, 0],
        sample_rate,
        regions,
        args.highpass_hz,
        args.lowpass_hz,
        args.max_reduction_db,
        args.threshold_percentile,
        args.ratio,
        args.attack_ms,
        args.release_ms,
        args.fade_ms,
    )

    write_float_wav_new(args.output, processed, sample_rate)
    print(f"input:  {args.input}")
    print(f"output: {args.output}")
    print(
        f"band: {args.highpass_hz:g}-{args.lowpass_hz:g} Hz; "
        f"max reduction: {args.max_reduction_db:g} dB; "
        f"ratio: {args.ratio:g}"
    )
    for report in reports:
        print(
            f"region {report['start']:.3f}-{report['end']:.3f}s: "
            f"threshold={report['threshold_db']:.1f} dB, "
            f"max={report['max_reduction_db']:.1f} dB, "
            f"mean={report['mean_reduction_db']:.1f} dB"
        )


if __name__ == "__main__":
    concise_cli(main, "apply_breath_control")
