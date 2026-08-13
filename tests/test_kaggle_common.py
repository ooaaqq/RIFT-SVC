from __future__ import annotations

import json
import subprocess
from unittest.mock import patch

from scripts.kaggle_common import wait_for_dataset


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
