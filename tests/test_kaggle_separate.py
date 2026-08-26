from __future__ import annotations

import argparse
import ast
import hashlib
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest
import soundfile as sf

from kaggle.separation.kernel import max_fft_ensemble, mixture_residual
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


def test_background_profile_pins_requested_models_and_max_spec() -> None:
    profile = PROFILES["big-beta7-bs-roformer-mag-max-spec"]

    assert profile["ensemble_algorithm"] == "max_fft"
    assert [model["model_repo"] for model in profile["models"]] == [
        "pcunwa/Mel-Band-Roformer-big",
        "anvuew/BS_RoFormer_mag",
    ]
    assert profile["models"][0]["checkpoint"] == "big_beta7.ckpt"
    assert profile["models"][1]["checkpoint"] == "bs_roformer_mag_anvuew.ckpt"


def test_max_fft_vocal_and_residual_remain_additive(tmp_path: Path) -> None:
    sample_rate = 8000
    frames = sample_rate
    time = np.arange(frames, dtype=np.float32) / sample_rate
    first = np.sin(2 * np.pi * 220 * time).astype(np.float32)[:, None] * 0.2
    second = np.sin(2 * np.pi * 880 * time).astype(np.float32)[:, None] * 0.1
    mixture = first + second
    first_path = tmp_path / "first.wav"
    second_path = tmp_path / "second.wav"
    mixture_path = tmp_path / "mixture.wav"
    vocal_path = tmp_path / "vocal.wav"
    residual_path = tmp_path / "instrumental.wav"
    for path, audio in (
        (first_path, first),
        (second_path, second),
        (mixture_path, mixture),
    ):
        sf.write(path, audio, sample_rate, subtype="FLOAT")

    assert max_fft_ensemble([first_path, second_path], vocal_path) == (
        sample_rate,
        1,
        frames,
    )
    assert mixture_residual(mixture_path, vocal_path, residual_path) == (
        sample_rate,
        1,
        frames,
    )

    vocal, _ = sf.read(vocal_path, dtype="float32", always_2d=True)
    residual, _ = sf.read(residual_path, dtype="float32", always_2d=True)
    np.testing.assert_allclose(vocal + residual, mixture, atol=1e-6)


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
