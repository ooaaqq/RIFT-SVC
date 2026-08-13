from __future__ import annotations

import argparse
import hashlib
import importlib.util
from pathlib import Path
from unittest.mock import patch

import pytest

from scripts.kaggle_separate import DEFAULT_ACCELERATOR, build_job
from scripts.separation_profiles import PROFILES


def test_cloud_kernel_profiles_match_local_cli() -> None:
    kernel_path = Path(__file__).parents[1] / "kaggle/separation/kernel.py"
    spec = importlib.util.spec_from_file_location("separation_kernel", kernel_path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    assert module.PROFILES == PROFILES


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
