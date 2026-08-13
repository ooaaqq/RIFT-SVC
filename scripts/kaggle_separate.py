#!/usr/bin/env python3
"""Run one supported source-separation model through a Kaggle GPU kernel."""

from __future__ import annotations

import argparse
import json
import shutil
import tempfile
from datetime import UTC, datetime
from pathlib import Path

from scripts.kaggle_common import (
    AUDIO_EXTENSIONS,
    current_datasets,
    probe_audio,
    run,
    sha256,
    wait_for_dataset,
    wait_for_kernel,
)
from scripts.separation_profiles import PROFILES

REPO_ROOT = Path(__file__).resolve().parents[1]
KERNEL_SOURCE = REPO_ROOT / "kaggle/separation/kernel.py"
DEFAULT_OWNER = "eeviriyi"
DEFAULT_DATASET_SLUG = "rift-separation-cli-input"
DEFAULT_KERNEL_SLUG = "rift-separation-cli"
DEFAULT_ACCELERATOR = "NvidiaTeslaT4"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Upload audio, run a pinned separation model on Kaggle, and download both Float WAV stems."
    )
    parser.add_argument("input", type=Path)
    parser.add_argument("--model", choices=sorted(PROFILES), required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/separation"))
    parser.add_argument("--owner", default=DEFAULT_OWNER)
    parser.add_argument("--dataset-slug", default=DEFAULT_DATASET_SLUG)
    parser.add_argument("--kernel-slug", default=DEFAULT_KERNEL_SLUG)
    parser.add_argument("--poll-interval", type=int, default=30)
    parser.add_argument("--timeout", type=int, default=6 * 60 * 60)
    parser.add_argument(
        "--keep-dataset-versions",
        action="store_true",
        help="retain older versions of the designated transient input dataset",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="prepare and print the job without uploading or running it",
    )
    args = parser.parse_args()
    if args.poll_interval < 5:
        parser.error("--poll-interval must be at least 5 seconds")
    if args.timeout < 60:
        parser.error("--timeout must be at least 60 seconds")
    return args


def build_job(args: argparse.Namespace, input_path: Path) -> dict:
    now = datetime.now(UTC)
    return {
        "schema_version": 1,
        "job_id": (
            f"{now.strftime('%Y%m%dT%H%M%SZ')}-{args.model}-"
            f"{sha256(input_path)[:8]}"
        ),
        "submitted_at": now.isoformat(),
        "input_name": input_path.name,
        "input_sha256": sha256(input_path),
        "input_audio": probe_audio(input_path),
        "profile": args.model,
        "profile_config": PROFILES[args.model],
    }


def validate_outputs(job: dict, download_dir: Path) -> tuple[list[Path], Path]:
    manifests = sorted(download_dir.rglob("manifest.json"))
    if len(manifests) != 1:
        raise RuntimeError(f"expected one manifest, found {manifests}")
    manifest = json.loads(manifests[0].read_text(encoding="utf-8"))
    if manifest.get("job_id") != job["job_id"]:
        raise RuntimeError("downloaded output belongs to a different Kaggle job")

    expected = {item["name"]: item for item in manifest.get("outputs", [])}
    outputs = sorted(
        path for path in download_dir.rglob("*.wav") if path.name in expected
    )
    if len(outputs) != 2 or len(expected) != 2:
        raise RuntimeError(f"expected two output stems, found {outputs}")
    input_duration = float(job["input_audio"]["duration"])
    for path in outputs:
        if sha256(path) != expected[path.name]["sha256"]:
            raise RuntimeError(f"output SHA-256 mismatch: {path}")
        probe = probe_audio(path)
        if abs(float(probe["duration"]) - input_duration) > 0.1:
            raise RuntimeError(f"output duration differs from input: {path}")
    return outputs, manifests[0]


def main() -> None:
    args = parse_args()
    input_path = args.input.expanduser().resolve()
    if not input_path.is_file():
        raise SystemExit(f"input does not exist: {input_path}")
    if input_path.suffix.lower() not in AUDIO_EXTENSIONS:
        raise SystemExit(f"unsupported audio extension: {input_path.suffix}")
    if shutil.which("kaggle") is None or shutil.which("ffprobe") is None:
        raise SystemExit(
            "kaggle and ffprobe must be available; enter the Flake environment"
        )

    job = build_job(args, input_path)
    dataset_ref = f"{args.owner}/{args.dataset_slug}"
    kernel_ref = f"{args.owner}/{args.kernel_slug}"
    print(json.dumps(job, ensure_ascii=False, indent=2), flush=True)
    if args.dry_run:
        print("Dry run: no Kaggle dataset or kernel was changed.")
        return

    with tempfile.TemporaryDirectory(prefix="separation-kaggle-") as temporary:
        root = Path(temporary)
        dataset_dir = root / "dataset"
        kernel_dir = root / "kernel"
        download_dir = root / "download"
        dataset_dir.mkdir()
        kernel_dir.mkdir()
        download_dir.mkdir()

        shutil.copy2(input_path, dataset_dir / input_path.name)
        (dataset_dir / "job.json").write_text(
            json.dumps(job, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        (dataset_dir / "dataset-metadata.json").write_text(
            json.dumps(
                {
                    "title": "RIFT Separation CLI Input",
                    "id": dataset_ref,
                    "licenses": [{"name": "other"}],
                    "subtitle": "Transient private input for source separation",
                    "description": "Managed by scripts/kaggle_separate.py; contains only the current job.",
                },
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )
        if dataset_ref in current_datasets():
            dataset_command = [
                "kaggle",
                "datasets",
                "version",
                "-p",
                str(dataset_dir),
                "-m",
                f"{job['job_id']} {args.model}",
            ]
            if not args.keep_dataset_versions:
                dataset_command.append("--delete-old-versions")
            run(dataset_command)
        else:
            run(["kaggle", "datasets", "create", "-p", str(dataset_dir)])
        wait_for_dataset(
            dataset_ref,
            {input_path.name, "job.json"},
            min(args.timeout, 30 * 60),
        )

        shutil.copy2(KERNEL_SOURCE, kernel_dir / "kernel.py")
        (kernel_dir / "kernel-metadata.json").write_text(
            json.dumps(
                {
                    "id": kernel_ref,
                    "title": "RIFT Separation CLI",
                    "code_file": "kernel.py",
                    "language": "python",
                    "kernel_type": "script",
                    "is_private": "true",
                    "enable_gpu": "true",
                    "enable_tpu": "false",
                    "machine_shape": DEFAULT_ACCELERATOR,
                    "enable_internet": "true",
                    "dataset_sources": [dataset_ref],
                    "competition_sources": [],
                    "kernel_sources": [],
                    "model_sources": [],
                },
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )
        run(
            [
                "kaggle",
                "kernels",
                "push",
                "-p",
                str(kernel_dir),
                "--accelerator",
                DEFAULT_ACCELERATOR,
            ]
        )
        wait_for_kernel(kernel_ref, args.poll_interval, args.timeout)
        run(["kaggle", "kernels", "output", kernel_ref, "-p", str(download_dir), "-o"])

        outputs, manifest = validate_outputs(job, download_dir)
        destination = args.output_dir.expanduser().resolve() / job["job_id"]
        destination.mkdir(parents=True, exist_ok=False)
        for output in outputs:
            shutil.copy2(output, destination / output.name)
        shutil.copy2(manifest, destination / "manifest.json")
        print(f"Downloaded {len(outputs)} stems to: {destination}")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        raise SystemExit("Interrupted; the Kaggle job may still be running.") from None
