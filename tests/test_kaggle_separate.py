from __future__ import annotations

import argparse
import ast
import hashlib
from pathlib import Path
from unittest.mock import patch

import pytest

from scripts.kaggle_separate import DEFAULT_ACCELERATOR, build_job, render_kernel
from scripts.separation_profiles import MSST_COMMIT, MSST_REPO, PROFILES


def test_cloud_kernel_profiles_match_local_cli() -> None:
    rendered = render_kernel()
    assignments = {}
    for node in ast.parse(rendered).body:
        if isinstance(node, ast.Assign) and len(node.targets) == 1:
            target = node.targets[0]
            if isinstance(target, ast.Name) and target.id in {
                "PROFILES",
                "MSST_REPO",
                "MSST_COMMIT",
            }:
                assignments[target.id] = ast.literal_eval(node.value)

    assert "__SEPARATION_PROFILES__" not in rendered
    assert assignments["PROFILES"] == PROFILES
    assert assignments["MSST_REPO"] == MSST_REPO
    assert assignments["MSST_COMMIT"] == MSST_COMMIT


def test_cloud_jobs_use_kaggle_t4() -> None:
    assert DEFAULT_ACCELERATOR == "NvidiaTeslaT4"


@pytest.mark.parametrize("profile", sorted(PROFILES))
@patch(
    "scripts.kaggle_separate.probe_audio",
    return_value={"sample_rate": "44100", "channels": 2, "duration": "1.0"},
)
def test_build_job_supports_every_profile(
    _probe_audio, profile: str, tmp_path: Path
) -> None:
    input_path = tmp_path / "mix.wav"
    input_path.write_bytes(b"audio")
    args = argparse.Namespace(model=profile)

    job = build_job(args, input_path)

    assert job["profile"] == profile
    assert profile in job["job_id"]
    assert job["profile_config"] == PROFILES[profile]
    assert job["input_sha256"] == hashlib.sha256(b"audio").hexdigest()
