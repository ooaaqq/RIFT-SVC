#!/usr/bin/env python3
"""Submit one RIFT-SVC inference job to Kaggle and download its output."""

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
    safe_audio_name,
    sha256,
    wait_for_dataset,
    wait_for_kernel,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
KERNEL_SOURCE = REPO_ROOT / "kaggle/rift/kernel.py"
DEFAULT_OWNER = "eeviriyi"
DEFAULT_DATASET_SLUG = "rift-svc-cli-input"
DEFAULT_KERNEL_SLUG = "rift-svc-cli-inference"
DEFAULT_ACCELERATOR = "NvidiaTeslaT4"
DEFAULT_REPO_URL = "https://github.com/ooaaqq/RIFT-SVC.git"
DEFAULT_MODEL_REPO = "ooaaqq/rift-svc-luzao-25k"
DEFAULT_MODEL_SHA256 = (
    "3db77a14098d87359dd69156973e2e315c642da8df17de1686831c91faed0c86"
)


def name_value(value: object) -> str:
    return str(value).replace("-", "m").replace(".", "p")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Upload audio, run the private Kaggle RIFT kernel, and download Float WAV output."
    )
    parser.add_argument("input", type=Path)
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/kaggle"))
    parser.add_argument("--owner", default=DEFAULT_OWNER)
    parser.add_argument("--dataset-slug", default=DEFAULT_DATASET_SLUG)
    parser.add_argument("--kernel-slug", default=DEFAULT_KERNEL_SLUG)
    parser.add_argument("--repo-url", default=DEFAULT_REPO_URL)
    parser.add_argument("--repo-ref", default="master")
    parser.add_argument("--model-repo", default=DEFAULT_MODEL_REPO)
    parser.add_argument("--model-filename", default="rift25k.ckpt")
    parser.add_argument("--model-sha256", default=DEFAULT_MODEL_SHA256)
    parser.add_argument("--model-revision")
    parser.add_argument("--modules-revision")
    parser.add_argument("--speaker", default="target")
    parser.add_argument("--key-shift", type=int, default=0)
    parser.add_argument("--steps", type=int, default=32)
    parser.add_argument("--ds", type=float, default=0.2)
    parser.add_argument("--spk", type=float, default=0.8)
    parser.add_argument("--cfg-rescale", type=float, default=0.7)
    parser.add_argument("--robust-f0", type=int, choices=(0, 1, 2), default=0)
    parser.add_argument("--seed", type=int, default=7)
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
    if args.steps < 2:
        parser.error("--steps must be at least 2")
    if args.poll_interval < 5:
        parser.error("--poll-interval must be at least 5 seconds")
    if args.timeout < 60:
        parser.error("--timeout must be at least 60 seconds")
    return args


def build_job(args: argparse.Namespace, input_path: Path) -> dict:
    suffix = (
        f"rift25k-spk-{args.speaker}-k{name_value(args.key_shift)}"
        f"-steps{args.steps}-ds{name_value(args.ds)}-spk{name_value(args.spk)}"
        f"-cfg{name_value(args.cfg_rescale)}-rf{args.robust_f0}-seed{args.seed}"
    )
    safe_name = safe_audio_name(input_path)
    output_name = f"{Path(safe_name).stem}__{suffix}.wav"
    now = datetime.now(UTC)
    input_probe = probe_audio(input_path)
    return {
        "schema_version": 1,
        "job_id": f"{now.strftime('%Y%m%dT%H%M%SZ')}-{sha256(input_path)[:8]}",
        "submitted_at": now.isoformat(),
        "input_name": safe_name,
        "source_input_name": input_path.name,
        "input_sha256": sha256(input_path),
        "input_audio": input_probe,
        "output_name": output_name,
        "repo_url": args.repo_url,
        "repo_ref": args.repo_ref,
        "model_repo": args.model_repo,
        "model_filename": args.model_filename,
        "model_sha256": args.model_sha256.lower(),
        "model_revision": args.model_revision,
        "modules_revision": args.modules_revision,
        "params": {
            "speaker": args.speaker,
            "key_shift": args.key_shift,
            "infer_steps": args.steps,
            "ds_cfg_strength": args.ds,
            "spk_cfg_strength": args.spk,
            "cfg_rescale": args.cfg_rescale,
            "robust_f0": args.robust_f0,
            "seed": args.seed,
        },
    }


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

    with tempfile.TemporaryDirectory(prefix="rift-kaggle-") as temporary:
        root = Path(temporary)
        dataset_dir = root / "dataset"
        kernel_dir = root / "kernel"
        download_dir = root / "download"
        dataset_dir.mkdir()
        kernel_dir.mkdir()
        download_dir.mkdir()

        shutil.copy2(input_path, dataset_dir / job["input_name"])
        (dataset_dir / "job.json").write_text(
            json.dumps(job, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        (dataset_dir / "dataset-metadata.json").write_text(
            json.dumps(
                {
                    "title": "RIFT SVC CLI Input",
                    "id": dataset_ref,
                    "licenses": [{"name": "other"}],
                    "subtitle": "Transient private input for automated RIFT inference",
                    "description": "Managed by scripts/kaggle_rift.py; contains only the current inference job.",
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
                job["job_id"],
            ]
            if not args.keep_dataset_versions:
                dataset_command.append("--delete-old-versions")
            run(dataset_command)
        else:
            run(["kaggle", "datasets", "create", "-p", str(dataset_dir)])
        wait_for_dataset(
            dataset_ref,
            {job["input_name"], "job.json"},
            min(args.timeout, 30 * 60),
        )

        shutil.copy2(KERNEL_SOURCE, kernel_dir / "kernel.py")
        (kernel_dir / "kernel-metadata.json").write_text(
            json.dumps(
                {
                    "id": kernel_ref,
                    "title": "RIFT SVC CLI Inference",
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

        manifests = sorted(download_dir.rglob("manifest.json"))
        outputs = sorted(download_dir.rglob(job["output_name"]))
        if len(manifests) != 1 or len(outputs) != 1:
            raise RuntimeError(
                f"expected one manifest and output; found manifests={manifests}, outputs={outputs}"
            )
        manifest = json.loads(manifests[0].read_text(encoding="utf-8"))
        if manifest.get("job_id") != job["job_id"]:
            raise RuntimeError("downloaded output belongs to a different Kaggle job")
        if sha256(outputs[0]) != manifest.get("output_sha256"):
            raise RuntimeError("downloaded output SHA-256 mismatch")
        probe = probe_audio(outputs[0])
        if int(probe["sample_rate"]) != 44100 or int(probe["channels"]) != 1:
            raise RuntimeError(f"unexpected downloaded audio: {probe}")
        input_duration = float(job["input_audio"]["duration"])
        output_duration = float(probe["duration"])
        if abs(input_duration - output_duration) > 0.1:
            raise RuntimeError(
                "downloaded audio duration differs from input: "
                f"input={input_duration:.3f}s output={output_duration:.3f}s"
            )

        destination = args.output_dir.expanduser().resolve() / job["job_id"]
        destination.mkdir(parents=True, exist_ok=False)
        final_audio = destination / outputs[0].name
        final_manifest = destination / "manifest.json"
        shutil.copy2(outputs[0], final_audio)
        shutil.copy2(manifests[0], final_manifest)
        print(f"Downloaded audio: {final_audio}")
        print(f"Run manifest: {final_manifest}")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        raise SystemExit("Interrupted; the Kaggle job may still be running.") from None
