from __future__ import annotations

import json
import subprocess
from unittest.mock import patch

from scripts.kaggle_common import safe_audio_name, wait_for_dataset, wait_for_kernel


def test_safe_audio_name_removes_kaggle_unsafe_characters(tmp_path) -> None:
    assert safe_audio_name(tmp_path / "vocal [mvsep.com].WAV") == "vocal_mvsep.com.wav"


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
