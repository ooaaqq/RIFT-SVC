from __future__ import annotations

import json
import subprocess
from pathlib import Path
from unittest.mock import patch

import pytest

from scripts.kaggle_common import (
    exclusive_kaggle_channel,
    frame_count_at_rate,
    frame_delta,
    publish_directory_atomic,
    safe_audio_name,
    wait_for_dataset,
    wait_for_kernel,
)


def test_kaggle_channel_lock_rejects_a_second_local_launcher() -> None:
    with (
        exclusive_kaggle_channel("owner/dataset|owner/kernel"),
        pytest.raises(RuntimeError, match="already in use"),
        exclusive_kaggle_channel("owner/dataset|owner/kernel"),
    ):
        pass


def test_atomic_directory_publish_is_complete_or_absent(tmp_path: Path) -> None:
    first = tmp_path / "first.wav"
    second = tmp_path / "second.wav"
    first.write_bytes(b"first")
    second.write_bytes(b"second")
    destination = tmp_path / "Run"

    with (
        patch(
            "scripts.kaggle_common.shutil.copy2",
            side_effect=[None, OSError("copy failed")],
        ),
        pytest.raises(OSError, match="copy failed"),
    ):
        publish_directory_atomic(
            destination,
            [(first, Path("first.wav")), (second, Path("second.wav"))],
        )

    assert not destination.exists()
    assert not list(tmp_path.glob(".Run.publishing-*"))

    published = publish_directory_atomic(
        destination,
        [(first, Path("first.wav")), (second, Path("nested/second.wav"))],
    )
    assert published == [destination / "first.wav", destination / "nested/second.wav"]
    assert (destination / "first.wav").read_bytes() == b"first"
    assert (destination / "nested/second.wav").read_bytes() == b"second"


def test_safe_audio_name_removes_kaggle_unsafe_characters(tmp_path) -> None:
    assert safe_audio_name(tmp_path / "vocal [mvsep.com].WAV") == "vocal_mvsep.com.wav"


def test_frame_count_uses_exact_time_base_and_resampling() -> None:
    input_probe = {
        "sample_rate": "48000",
        "duration_ts": "48000",
        "time_base": "1/48000",
        "duration": "1.0",
    }
    output_probe = {
        "sample_rate": "44100",
        "duration_ts": "44101",
        "time_base": "1/44100",
        "duration": "1.000023",
    }

    assert frame_count_at_rate(input_probe, 44100) == 44100
    assert frame_delta(input_probe, output_probe) == 1


def test_wait_for_dataset_checks_current_files() -> None:
    response = subprocess.CompletedProcess(
        args=[],
        returncode=0,
        stdout=json.dumps([{"name": "job.json"}, {"name": "input.wav"}]),
        stderr="",
    )

    with patch("scripts.kaggle_common.run", return_value=response) as run:
        wait_for_dataset("owner/dataset", {"job.json", "input.wav"}, timeout=1)

    run.assert_called_once_with(
        [
            "kaggle",
            "datasets",
            "files",
            "owner/dataset",
            "--format",
            "json",
        ],
        capture=True,
        check=False,
    )


def test_wait_for_kernel_retries_transient_status_failure() -> None:
    with (
        patch(
            "scripts.kaggle_common.kernel_status",
            side_effect=[subprocess.CalledProcessError(1, ["kaggle"]), "COMPLETE"],
        ),
        patch("scripts.kaggle_common.time.sleep") as sleep,
    ):
        wait_for_kernel("owner/kernel", poll_interval=5, timeout=60)

    sleep.assert_called_once_with(5)
