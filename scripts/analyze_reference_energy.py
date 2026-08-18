#!/usr/bin/env python3
"""Read-only broad-energy analysis for an original-song reference."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from rift_svc.audio_tools import (
    concise_cli,
    read_float_audio,
    write_json_new,
)


def windowed_energy_curve(
    audio: np.ndarray,
    sample_rate: int,
    window_seconds: float,
    hop_seconds: float,
) -> tuple[np.ndarray, np.ndarray]:
    window = round(window_seconds * sample_rate)
    hop = round(hop_seconds * sample_rate)
    if window <= 0 or hop <= 0:
        raise ValueError("analysis window and hop must be positive")
    if len(audio) < window:
        raise ValueError("audio is shorter than the analysis window")
    power = np.mean(np.square(audio, dtype=np.float64), axis=1)
    cumulative = np.concatenate(([0.0], np.cumsum(power, dtype=np.float64)))
    starts = np.arange(0, len(audio) - window + 1, hop, dtype=np.int64)
    mean_power = (cumulative[starts + window] - cumulative[starts]) / window
    dbfs = 10.0 * np.log10(mean_power + 1e-15)
    centers = (starts + window / 2.0) / sample_rate
    return centers, dbfs


def section_summaries(
    audio: np.ndarray,
    sample_rate: int,
    section_seconds: float,
    baseline_dbfs: float,
) -> list[dict[str, float]]:
    section_frames = round(section_seconds * sample_rate)
    summaries = []
    for first in range(0, len(audio), section_frames):
        last = min(len(audio), first + section_frames)
        power = float(np.mean(np.square(audio[first:last], dtype=np.float64)))
        dbfs = 10.0 * np.log10(power + 1e-15)
        summaries.append(
            {
                "start": first / sample_rate,
                "end": last / sample_rate,
                "rms_dbfs": dbfs,
                "relative_db": dbfs - baseline_dbfs,
            }
        )
    return summaries


def highlight_ranges(
    times: np.ndarray,
    relative_db: np.ndarray,
    hop_seconds: float,
    threshold_db: float,
) -> dict[str, list[dict[str, float | str]]]:
    result: dict[str, list[dict[str, float | str]]] = {"higher": [], "lower": []}
    labels = np.zeros(len(relative_db), dtype=np.int8)
    labels[relative_db >= threshold_db] = 1
    labels[relative_db <= -threshold_db] = -1
    for value, name in ((1, "higher"), (-1, "lower")):
        first: int | None = None
        for index in range(len(labels) + 1):
            active = index < len(labels) and labels[index] == value
            if active and first is None:
                first = index
            if not active and first is not None:
                last = index - 1
                result[name].append(
                    {
                        "start": max(0.0, float(times[first] - hop_seconds / 2.0)),
                        "end": float(times[last] + hop_seconds / 2.0),
                        "mean_relative_db": float(np.mean(relative_db[first:index])),
                        "classification": name,
                    }
                )
                first = None
    return result


def analyze_reference(
    path: Path,
    *,
    window_seconds: float,
    hop_seconds: float,
    section_seconds: float,
    silence_floor_dbfs: float,
    highlight_db: float,
) -> dict:
    audio, sample_rate = read_float_audio(path)
    times, dbfs = windowed_energy_curve(
        audio, sample_rate, window_seconds, hop_seconds
    )
    active = dbfs > silence_floor_dbfs
    if not np.any(active):
        raise ValueError("no analysis windows are above the silence floor")
    baseline = float(np.median(dbfs[active]))
    relative = dbfs - baseline
    return {
        "schema": 1,
        "method": "multichannel RMS energy with a slow sliding window",
        "input": str(path.resolve()),
        "sample_rate": sample_rate,
        "channels": audio.shape[1],
        "frames": len(audio),
        "duration": len(audio) / sample_rate,
        "parameters": {
            "window_seconds": window_seconds,
            "hop_seconds": hop_seconds,
            "section_seconds": section_seconds,
            "silence_floor_dbfs": silence_floor_dbfs,
            "highlight_db": highlight_db,
        },
        "active_median_rms_dbfs": baseline,
        "active_percentiles_dbfs": {
            str(percentile): float(np.percentile(dbfs[active], percentile))
            for percentile in (10, 25, 50, 75, 90)
        },
        "curve": [
            {
                "time": float(time),
                "rms_dbfs": float(level),
                "relative_db": float(delta),
                "active": bool(is_active),
            }
            for time, level, delta, is_active in zip(
                times, dbfs, relative, active, strict=True
            )
        ],
        "sections": section_summaries(
            audio, sample_rate, section_seconds, baseline
        ),
        "highlights": highlight_ranges(
            times, relative, hop_seconds, highlight_db
        ),
        "note": "read-only evidence; do not copy this curve directly as vocal gain",
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Analyze broad original-song energy without rendering audio."
    )
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--window-seconds", type=float, default=3.0)
    parser.add_argument("--hop-seconds", type=float, default=0.5)
    parser.add_argument("--section-seconds", type=float, default=10.0)
    parser.add_argument("--silence-floor-dbfs", type=float, default=-60.0)
    parser.add_argument("--highlight-db", type=float, default=1.5)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()

    if min(args.window_seconds, args.hop_seconds, args.section_seconds) <= 0:
        parser.error("window, hop, and section durations must be positive")
    if args.highlight_db <= 0:
        parser.error("--highlight-db must be positive")
    report = analyze_reference(
        args.input,
        window_seconds=args.window_seconds,
        hop_seconds=args.hop_seconds,
        section_seconds=args.section_seconds,
        silence_floor_dbfs=args.silence_floor_dbfs,
        highlight_db=args.highlight_db,
    )
    write_json_new(args.report, report)
    print(f"report: {args.report}")
    print(
        f"active median: {report['active_median_rms_dbfs']:.2f} dBFS; "
        f"higher/lower ranges: {len(report['highlights']['higher'])}/"
        f"{len(report['highlights']['lower'])}"
    )
    for section in report["sections"]:
        print(
            f"{section['start']:7.2f}-{section['end']:7.2f}s  "
            f"{section['relative_db']:+6.2f} dB relative"
        )


if __name__ == "__main__":
    concise_cli(main, "analyze_reference_energy")
