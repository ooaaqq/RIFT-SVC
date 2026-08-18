from __future__ import annotations

import shutil
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

from rift_svc.alignment import (
    Anchor,
    alignment_report,
    apply_alignment,
    detect_alignment,
    read_anchors,
)


def test_report_exposes_offset_drift_and_segment_stretch(tmp_path: Path) -> None:
    sample_rate = 8000
    source = tmp_path / "source.wav"
    reference = tmp_path / "reference.wav"
    sf.write(source, np.zeros((16000, 1), dtype=np.float32), sample_rate)
    sf.write(reference, np.zeros((16200, 2), dtype=np.float32), sample_rate)
    anchors_file = tmp_path / "anchors.txt"
    anchors_file.write_text("0 0.010\n1.000 1.020\n2.000 2.025\n")

    anchors = read_anchors(anchors_file)
    report = alignment_report(source, reference, anchors)

    assert anchors == [Anchor(0.0, 0.01), Anchor(1.0, 1.02), Anchor(2.0, 2.025)]
    assert report["anchor_offsets_ms"] == pytest.approx([10.0, 20.0, 25.0])
    assert report["offset_range_ms"] == pytest.approx(15.0)
    assert report["segments"][0]["stretch_percent"] == pytest.approx(1.0)
    assert report["segments"][1]["stretch_percent"] == pytest.approx(0.5)


def test_report_accepts_six_decimal_duration_at_sample_boundary(
    tmp_path: Path,
) -> None:
    sample_rate = 44100
    frames = 100
    source = tmp_path / "source.wav"
    reference = tmp_path / "reference.wav"
    sf.write(source, np.zeros((frames, 1), dtype=np.float32), sample_rate)
    sf.write(reference, np.zeros((frames, 1), dtype=np.float32), sample_rate)
    duration = frames / sample_rate
    anchors_file = tmp_path / "anchors.txt"
    anchors_file.write_text(f"0 0\n{duration:.9f} {duration:.6f}\n", encoding="utf-8")

    report = alignment_report(source, reference, read_anchors(anchors_file))

    assert report["anchors"][-1]["target"] == duration


@pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="FFmpeg is required")
def test_piecewise_render_matches_reference_without_silent_padding(
    tmp_path: Path,
) -> None:
    sample_rate = 8000
    source = tmp_path / "source.wav"
    reference = tmp_path / "reference.wav"
    time = np.arange(16000, dtype=np.float32) / sample_rate
    audio = (0.1 * np.sin(2 * np.pi * 220 * time))[:, np.newaxis]
    sf.write(source, audio, sample_rate, subtype="FLOAT")
    sf.write(reference, np.zeros((16080, 2), dtype=np.float32), sample_rate)

    rendered, report = apply_alignment(
        source,
        reference,
        [Anchor(0.0, 0.0), Anchor(1.0, 1.005), Anchor(2.0, 2.01)],
        crossfade_ms=10,
        max_stretch_percent=1,
    )

    assert rendered.shape == (16080, 1)
    assert np.isfinite(rendered).all()
    assert np.sqrt(np.mean(np.square(rendered[8000:8160]))) > 0.05
    assert np.sqrt(np.mean(np.square(rendered[-160:]))) > 0.05
    assert report["segments"][0]["stretch_percent"] == pytest.approx(0.5)


def test_rejects_bad_anchors_and_excessive_stretch(tmp_path: Path) -> None:
    anchors_file = tmp_path / "bad-anchors.txt"
    anchors_file.write_text("0 0\n1 1\n0.5 2\n")
    with pytest.raises(ValueError, match="must both increase"):
        read_anchors(anchors_file)

    sample_rate = 8000
    source = tmp_path / "source.wav"
    reference = tmp_path / "reference.wav"
    sf.write(source, np.zeros((16000, 1), dtype=np.float32), sample_rate)
    sf.write(reference, np.zeros((17600, 1), dtype=np.float32), sample_rate)
    with pytest.raises(ValueError, match="requires .10.0000% stretch"):
        apply_alignment(
            source,
            reference,
            [Anchor(0.0, 0.0), Anchor(2.0, 2.2)],
            crossfade_ms=10,
            max_stretch_percent=2,
        )


def test_detector_finds_offset_and_linear_drift(tmp_path: Path) -> None:
    sample_rate = 1000
    duration = 60.0
    time = np.arange(round(duration * sample_rate)) / sample_rate
    rng = np.random.default_rng(7)
    control_times = np.arange(0.0, duration + 0.2, 0.2)
    envelope = np.interp(
        time, control_times, rng.uniform(0.05, 0.8, len(control_times))
    )
    reference_audio = envelope * np.sin(2.0 * np.pi * 83.0 * time)
    intercept = 0.120
    ratio = 1.0005
    source_time = (
        np.arange(round((intercept + ratio * duration) * sample_rate)) / sample_rate
    )
    source_audio = np.interp(
        (source_time - intercept) / ratio,
        time,
        reference_audio,
        left=0.0,
        right=0.0,
    )
    reference = tmp_path / "reference.wav"
    source = tmp_path / "source.wav"
    sf.write(reference, reference_audio, sample_rate, subtype="FLOAT")
    sf.write(source, source_audio, sample_rate, subtype="FLOAT")

    anchors, report = detect_alignment(
        source,
        reference,
        search_seconds=1.0,
        local_window_seconds=10.0,
        local_search_seconds=0.1,
        min_correlation=0.7,
        max_fit_residual_ms=15.0,
        analysis_window_ms=20.0,
        analysis_hop_ms=5.0,
        edge_seconds=0.5,
    )

    assert report["global_correlation"] > 0.9
    assert report["safe_to_apply"]
    assert report["fit"]["source_seconds_at_target_zero"] == pytest.approx(
        intercept, abs=0.015
    )
    assert report["fit"]["source_seconds_per_target_second"] == pytest.approx(
        ratio, abs=0.0004
    )
    assert anchors[0].source == pytest.approx(intercept, abs=0.015)
    assert anchors[0].target == 0.0
    assert anchors[-1].target == pytest.approx(duration, abs=0.02)


def test_detector_with_only_two_windows_is_report_only(tmp_path: Path) -> None:
    sample_rate = 1000
    time = np.arange(25 * sample_rate) / sample_rate
    rng = np.random.default_rng(9)
    envelope = np.interp(
        time,
        np.arange(0.0, 25.2, 0.2),
        rng.uniform(0.05, 0.8, 126),
    )
    audio = envelope * np.sin(2 * np.pi * 71 * time)
    source = tmp_path / "source.wav"
    reference = tmp_path / "reference.wav"
    sf.write(source, audio, sample_rate, subtype="FLOAT")
    sf.write(reference, audio, sample_rate, subtype="FLOAT")

    anchors, report = detect_alignment(
        source,
        reference,
        search_seconds=0.5,
        local_window_seconds=10.0,
        local_search_seconds=0.1,
        min_correlation=0.7,
        max_fit_residual_ms=15.0,
        analysis_window_ms=20.0,
        analysis_hop_ms=5.0,
        edge_seconds=1.0,
    )

    assert not report["safe_to_apply"]
    assert anchors == []
    assert "fewer than three" in report["unsafe_reasons"][0]
