from __future__ import annotations

import shutil
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

from rift_svc.audio_tools import parse_time
from rift_svc.mixctl import check_delivery
from scripts.analyze_reference_energy import analyze_reference
from scripts.apply_breath_control import localized_reduction_mask, process_channel
from scripts.apply_gain_automation import (
    GainRegion,
    apply_gain,
    read_gain_regions,
)
from scripts.render_m7_returns import (
    PRESET_FILES,
    read_ir,
    render_return,
    resolve_ir_pair,
)


def test_m7_everyday_presets_and_explicit_pair(tmp_path: Path) -> None:
    left, right = resolve_ir_pair("small-vox-room", tmp_path, None, None)
    explicit_left = tmp_path / "Custom L.wav"
    explicit_right = tmp_path / "Custom R.wav"

    assert set(PRESET_FILES) == {"small-vox-room", "studio-b-close"}
    assert left.name == "3 Rooms 17 Small Vox Room, 44K L.wav"
    assert right.name == "3 Rooms 17 Small Vox Room, 44K R.wav"
    assert resolve_ir_pair(
        "studio-b-close", tmp_path, explicit_left, explicit_right
    ) == (explicit_left, explicit_right)
    with pytest.raises(ValueError):
        resolve_ir_pair("small-vox-room", tmp_path, explicit_left, None)


def test_m7_return_matches_input_samples_and_refuses_overwrite(tmp_path: Path) -> None:
    sample_rate = 1000
    audio = np.ones((1000, 1), dtype=np.float32) * 0.1
    ir_left = np.zeros((20, 2), dtype=np.float32)
    ir_right = np.zeros((20, 2), dtype=np.float32)
    ir_left[0] = 1.0
    ir_right[0] = 1.0
    output = tmp_path / "m7-main-wet.wav"
    full_output = tmp_path / "m7-main-wet-full.wav"

    report = render_return(
        "main", audio, ir_left, ir_right, -18.0, output, sample_rate, False
    )
    rendered, rendered_rate = sf.read(output, always_2d=True)
    full_report = render_return(
        "main", audio, ir_left, ir_right, -18.0, full_output, sample_rate, True
    )
    rendered_full, _ = sf.read(full_output, always_2d=True)

    assert rendered_rate == sample_rate
    assert rendered.shape == (audio.shape[0], 2)
    assert report["tail_policy"] == "match-input"
    assert report["full_convolution_seconds"] > report["output_seconds"]
    assert full_report["tail_policy"] == "full"
    assert rendered_full.shape[0] > rendered.shape[0]
    assert np.allclose(rendered_full[: len(rendered)], rendered, atol=1e-7)
    with pytest.raises(FileExistsError):
        render_return(
            "main", audio, ir_left, ir_right, -18.0, output, sample_rate, False
        )


def test_m7_full_ir_is_not_modified_and_float_peak_is_not_clamped(
    tmp_path: Path,
) -> None:
    sample_rate = 1000
    source_ir = np.linspace(-1.0, 1.0, 40, dtype=np.float32).reshape(20, 2)
    ir_path = tmp_path / "IR.wav"
    sf.write(ir_path, source_ir, sample_rate, subtype="FLOAT")

    loaded = read_ir(ir_path, sample_rate, 1.0, 0.0)
    assert np.array_equal(loaded, source_ir)

    audio = np.ones((1000, 1), dtype=np.float32) * 0.1
    impulse = np.zeros((2, 2), dtype=np.float32)
    impulse[0] = 1.0
    output = tmp_path / "hot-float-wet.wav"
    report = render_return(
        "main", audio, impulse, impulse, 30.0, output, sample_rate, False
    )
    rendered, _ = sf.read(output, always_2d=True)

    assert report["wet_peak"] > 1.0
    assert np.max(np.abs(rendered)) > 1.0


def test_breath_control_is_unchanged_outside_selected_region() -> None:
    sample_rate = 44100
    time = np.arange(sample_rate, dtype=np.float32) / sample_rate
    audio = (0.1 * np.sin(2 * np.pi * 7000 * time)).astype(np.float32)

    processed, reports = process_channel(
        audio,
        sample_rate,
        [(0.25, 0.5)],
        4500.0,
        14000.0,
        4.0,
        55.0,
        4.0,
        10.0,
        140.0,
        45.0,
    )

    assert reports
    assert np.allclose(processed[:8000], audio[:8000], atol=1e-6)
    assert np.allclose(processed[30000:], audio[30000:], atol=1e-6)


def test_breath_region_fades_do_not_dip_where_regions_overlap() -> None:
    mask = localized_reduction_mask(
        1000,
        1000,
        [(0.10, 0.30), (0.25, 0.45)],
        50.0,
    )

    assert mask[250] > 0.9
    assert mask[275] > 0.9


def test_gain_automation_parses_human_times_and_adds_overlapping_regions(
    tmp_path: Path,
) -> None:
    automation = tmp_path / "vocal-gain.txt"
    automation.write_text(
        "# START END GAIN_DB\n00:00.200 00:00.600 +6\n00:00.400 00:00.800 -3\n",
        encoding="utf-8",
    )
    regions = read_gain_regions(automation)
    audio = np.ones((1000, 2), dtype=np.float32)
    processed, curve = apply_gain(audio, 1000, regions, fade_ms=0)

    assert parse_time("01:02.5") == 62.5
    assert regions == [GainRegion(0.2, 0.6, 6.0), GainRegion(0.4, 0.8, -3.0)]
    assert np.allclose(processed[:200], audio[:200])
    assert np.allclose(curve[250], 6.0)
    assert np.allclose(curve[500], 3.0)
    assert np.allclose(curve[700], -3.0)
    assert np.allclose(processed[900:], audio[900:])


def test_gain_automation_uses_smooth_boundaries_and_rejects_overrun() -> None:
    audio = np.ones((1000, 1), dtype=np.float32)
    processed, curve = apply_gain(audio, 1000, [GainRegion(0.2, 0.8, 6.0)], fade_ms=100)

    assert curve[199] == 0.0
    assert curve[200] == 0.0
    assert 0.0 < curve[250] < 6.0
    assert curve[350] == pytest.approx(6.0)
    assert np.allclose(processed[:200], audio[:200])
    assert np.allclose(processed[800:], audio[800:])
    with pytest.raises(ValueError):
        apply_gain(audio, 1000, [GainRegion(0.9, 1.1, 1.0)], fade_ms=10)


def test_reference_energy_analysis_reports_broad_sections_without_rendering(
    tmp_path: Path,
) -> None:
    sample_rate = 1000
    time = np.arange(20 * sample_rate) / sample_rate
    audio = np.sin(2 * np.pi * 50 * time).astype(np.float32)
    audio[: 10 * sample_rate] *= 0.1
    audio[10 * sample_rate :] *= 0.2
    source = tmp_path / "original.wav"
    sf.write(source, np.column_stack((audio, audio)), sample_rate, subtype="FLOAT")

    report = analyze_reference(
        source,
        window_seconds=1.0,
        hop_seconds=1.0,
        section_seconds=10.0,
        silence_floor_dbfs=-60.0,
        highlight_db=1.0,
    )

    assert report["frames"] == 20 * sample_rate
    assert len(report["sections"]) == 2
    assert report["sections"][1]["rms_dbfs"] - report["sections"][0][
        "rms_dbfs"
    ] == pytest.approx(6.0206, abs=0.01)
    assert report["highlights"]["higher"]
    assert report["highlights"]["lower"]


@pytest.mark.skipif(
    shutil.which("ffmpeg") is None or shutil.which("ffprobe") is None,
    reason="FFmpeg tools are required",
)
def test_delivery_check_fully_decodes_measures_and_applies_limits(
    tmp_path: Path,
) -> None:
    sample_rate = 48000
    time = np.arange(sample_rate, dtype=np.float32) / sample_rate
    mono = 0.1 * np.sin(2 * np.pi * 440 * time)
    stereo = np.column_stack((mono, mono)).astype(np.float32)
    source = tmp_path / "tone.wav"
    sf.write(source, stereo, sample_rate, subtype="PCM_24")

    accepted = check_delivery(
        [source],
        sample_rate=48000,
        channels=2,
        max_true_peak=-0.5,
        reference=source,
    )
    rejected = check_delivery([source], sample_rate=44100, channels=2)

    assert accepted["ok"]
    assert accepted["results"][0]["loudness"]["integrated_lufs"] < 0
    assert accepted["results"][0]["sha256"]
    assert accepted["results"][0]["reference_duration_delta_ms"] == 0
    assert accepted["results"][0]["reference_duration_delta_samples"] == 0
    assert not rejected["ok"]
    assert "sample rate" in rejected["failures"][0]["error"]
