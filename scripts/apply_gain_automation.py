#!/usr/bin/env python3
"""Apply human-authored, time-range gain automation to an audio file."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from rift_svc.audio_tools import (
    concise_cli,
    parse_time,
    read_float_audio,
    write_float_wav_new,
)


@dataclass(frozen=True)
class GainRegion:
    start: float
    end: float
    gain_db: float


def read_gain_regions(path: Path) -> list[GainRegion]:
    regions: list[GainRegion] = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        text = line.split("#", 1)[0].strip()
        if not text:
            continue
        fields = text.split()
        if len(fields) != 3:
            raise ValueError(
                f"{path}:{line_number}: expected START END GAIN_DB"
            )
        try:
            start = parse_time(fields[0])
            end = parse_time(fields[1])
            gain_db = float(fields[2])
        except ValueError as exc:
            raise ValueError(f"{path}:{line_number}: {exc}") from exc
        if end <= start:
            raise ValueError(f"{path}:{line_number}: END must be greater than START")
        if not np.isfinite(gain_db):
            raise ValueError(f"{path}:{line_number}: gain must be finite")
        regions.append(GainRegion(start, end, gain_db))
    if not regions:
        raise ValueError(f"no gain regions found in {path}")
    return regions


def gain_curve_db(
    length: int,
    sample_rate: int,
    regions: list[GainRegion],
    fade_ms: float,
) -> np.ndarray:
    """Build an additive dB curve; overlapping regions intentionally add."""
    curve = np.zeros(length, dtype=np.float64)
    fade_frames = round(fade_ms * sample_rate / 1000.0)
    for region in regions:
        first = round(region.start * sample_rate)
        last = round(region.end * sample_rate)
        if first < 0 or last > length or last <= first:
            raise ValueError(
                f"gain region {region.start:g}-{region.end:g}s exceeds the audio"
            )
        region_length = last - first
        ramp = min(fade_frames, region_length // 2)
        shape = np.ones(region_length, dtype=np.float64)
        if ramp:
            phase = np.linspace(0.0, np.pi, ramp, endpoint=False)
            shape[:ramp] = 0.5 - 0.5 * np.cos(phase)
            shape[-ramp:] = shape[:ramp][::-1]
        curve[first:last] += region.gain_db * shape
    return curve


def apply_gain(
    audio: np.ndarray,
    sample_rate: int,
    regions: list[GainRegion],
    fade_ms: float,
) -> tuple[np.ndarray, np.ndarray]:
    curve = gain_curve_db(len(audio), sample_rate, regions, fade_ms)
    gain = np.power(10.0, curve / 20.0).astype(np.float32)
    return audio.astype(np.float32) * gain[:, np.newaxis], curve


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Apply START END GAIN_DB regions from a human-authored file."
    )
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--automation-file", type=Path, required=True)
    parser.add_argument("--fade-ms", type=float, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    if args.fade_ms < 0:
        parser.error("--fade-ms cannot be negative")
    regions = read_gain_regions(args.automation_file)
    audio, sample_rate = read_float_audio(args.input)
    processed, curve = apply_gain(audio, sample_rate, regions, args.fade_ms)
    write_float_wav_new(args.output, processed, sample_rate)

    print(f"input:  {args.input}")
    print(f"output: {args.output}")
    print(f"regions: {len(regions)}; transition: {args.fade_ms:g} ms")
    print(f"automation range: {np.min(curve):+.2f} to {np.max(curve):+.2f} dB")


if __name__ == "__main__":
    concise_cli(main, "apply_gain_automation")
