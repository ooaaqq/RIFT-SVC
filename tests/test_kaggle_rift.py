from __future__ import annotations

import argparse
import hashlib
from pathlib import Path
from unittest.mock import patch

from scripts.kaggle_rift import DEFAULT_ACCELERATOR, build_job, name_value


def arguments() -> argparse.Namespace:
    return argparse.Namespace(
        repo_url="https://github.com/ooaaqq/RIFT-SVC.git",
        repo_ref="master",
        model_repo="ooaaqq/rift-svc-luzao-25k",
        model_filename="rift25k.ckpt",
        model_sha256="abc123",
        model_revision=None,
        modules_revision=None,
        speaker="target",
        key_shift=0,
        steps=32,
        ds=0.2,
        spk=0.8,
        cfg_rescale=0.7,
        robust_f0=0,
        seed=7,
    )


def test_name_value_is_filename_safe() -> None:
    assert name_value(-1.5) == "m1p5"


def test_cloud_jobs_use_kaggle_t4() -> None:
    assert DEFAULT_ACCELERATOR == "NvidiaTeslaT4"


@patch(
    "scripts.kaggle_rift.probe_audio",
    return_value={"sample_rate": "44100", "channels": 1, "duration": "1.0"},
)
def test_build_job_records_hash_parameters_and_dynamic_name(
    _probe_audio, tmp_path: Path
) -> None:
    input_path = tmp_path / "dry vocal.wav"
    input_path.write_bytes(b"audio")

    job = build_job(arguments(), input_path)

    assert job["input_sha256"] == hashlib.sha256(b"audio").hexdigest()
    assert job["params"]["infer_steps"] == 32
    assert job["params"]["robust_f0"] == 0
    assert job["input_audio"]["sample_rate"] == "44100"
    assert job["output_name"] == (
        "dry vocal__rift25k-spk-target-k0-steps32-ds0p2-spk0p8-cfg0p7-rf0-seed7.wav"
    )
