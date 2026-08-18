from __future__ import annotations

import argparse
import hashlib
import importlib.util
from pathlib import Path
from unittest.mock import patch

import pytest

from scripts.kaggle_rift import (
    DEFAULT_ACCELERATOR,
    DEFAULT_MODEL_REVISION,
    DEFAULT_MODULES_REVISION,
    DEFAULT_REPO_REF,
    build_job,
    build_run_name,
    name_value,
    render_kernel,
)


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


def test_cloud_runtime_defaults_are_immutable_revisions() -> None:
    assert len(DEFAULT_REPO_REF) == 40
    assert len(DEFAULT_MODEL_REVISION) == 40
    assert len(DEFAULT_MODULES_REVISION) == 40
    assert DEFAULT_REPO_REF != "master"


def test_run_name_contains_shared_batch_parameters() -> None:
    assert build_run_name(arguments()) == (
        "RIFT25K-k0-s32-ds0p2-spk0p8-cfg0p7-rf0-seed7"
    )


@patch(
    "scripts.kaggle_rift.probe_audio",
    return_value={"sample_rate": "44100", "channels": 1, "duration": "1.0"},
)
def test_build_job_records_multiple_inputs_and_human_output_names(
    _probe_audio, tmp_path: Path
) -> None:
    first = tmp_path / "歌名-歌手-bs124-vocal-l1.wav"
    second = tmp_path / "歌名-歌手-bs124-vocal-l1-anvuew-karaoke-lead.wav"
    first.write_bytes(b"first")
    second.write_bytes(b"second")

    job = build_job(arguments(), [first, second])

    assert job["schema_version"] == 2
    assert job["run_name"] == "RIFT25K-k0-s32-ds0p2-spk0p8-cfg0p7-rf0-seed7"
    assert [item["input_name"] for item in job["items"]] == [
        "input-0001.wav",
        "input-0002.wav",
    ]
    assert [item["transport_output_name"] for item in job["items"]] == [
        "output-0001.wav",
        "output-0002.wav",
    ]
    assert job["items"][0]["input_sha256"] == hashlib.sha256(b"first").hexdigest()
    assert job["items"][0]["output_name"] == (
        "歌名-歌手-bs124-vocal-l1-rift25k.wav"
    )
    assert job["items"][1]["output_name"] == (
        "歌名-歌手-bs124-vocal-l1-anvuew-karaoke-lead-rift25k.wav"
    )
    assert job["params"]["infer_steps"] == 32


@patch(
    "scripts.kaggle_rift.probe_audio",
    return_value={"sample_rate": "44100", "channels": 1, "duration": "1.0"},
)
def test_batch_refuses_duplicate_human_output_names(
    _probe_audio, tmp_path: Path
) -> None:
    first_dir = tmp_path / "one"
    second_dir = tmp_path / "two"
    first_dir.mkdir()
    second_dir.mkdir()
    first = first_dir / "same.wav"
    second = second_dir / "same.wav"
    first.write_bytes(b"first")
    second.write_bytes(b"second")

    with pytest.raises(ValueError, match="duplicate batch output name"):
        build_job(arguments(), [first, second])


def test_kernel_uses_up_to_two_dynamic_gpu_workers() -> None:
    kernel_path = Path(__file__).parents[1] / "kaggle/rift/kernel.py"
    spec = importlib.util.spec_from_file_location("rift_kernel", kernel_path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    assert module.worker_count(gpu_count=2, item_count=8) == 2
    assert module.worker_count(gpu_count=2, item_count=1) == 1
    assert module.worker_count(gpu_count=1, item_count=8) == 1


def test_rendered_kernel_embeds_the_worker_for_kaggle_single_file_upload() -> None:
    rendered = render_kernel()

    assert 'BATCH_WORKER_SOURCE = "__BATCH_WORKER_SOURCE__"' not in rendered
    assert "Run a dynamic multi-file RIFT queue" in rendered
    assert "def gpu_worker(" in rendered
