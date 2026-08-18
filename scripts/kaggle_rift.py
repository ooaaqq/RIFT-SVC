#!/usr/bin/env python3
"""Submit one multi-file RIFT-SVC batch to a dual-T4 Kaggle kernel."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import tempfile
from datetime import UTC, datetime
from pathlib import Path

from scripts.kaggle_common import (
    AUDIO_EXTENSIONS,
    current_datasets,
    exclusive_kaggle_channel,
    frame_delta,
    probe_audio,
    publish_directory_atomic,
    run,
    sha256,
    wait_for_dataset,
    wait_for_kernel,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
KERNEL_SOURCE = REPO_ROOT / "kaggle/rift/kernel.py"
KERNEL_WORKER_SOURCE = REPO_ROOT / "kaggle/rift/batch_worker.py"
WORKER_MARKER = 'BATCH_WORKER_SOURCE = "__BATCH_WORKER_SOURCE__"'
DEFAULT_OWNER = "eeviriyi"
DEFAULT_DATASET_SLUG = "rift-svc-cli-input"
DEFAULT_KERNEL_SLUG = "rift-svc-cli-inference"
DEFAULT_ACCELERATOR = "NvidiaTeslaT4"
DEFAULT_REPO_URL = "https://github.com/ooaaqq/RIFT-SVC.git"
DEFAULT_REPO_REF = "4536e77f9d05759e3ea6054d72848a2dbd97b4d3"
DEFAULT_MODEL_REPO = "ooaaqq/rift-svc-luzao-25k"
DEFAULT_MODEL_REVISION = "020cacf61bc073b33a5e0ea1c20a909ec96c8545"
DEFAULT_MODULES_REVISION = "03c1662ba24a76fa3a653c33bc983ce6422620b4"
DEFAULT_MODEL_SHA256 = (
    "3db77a14098d87359dd69156973e2e315c642da8df17de1686831c91faed0c86"
)


def name_value(value: object) -> str:
    return str(value).replace("-", "m").replace(".", "p")


def build_run_name(args: argparse.Namespace) -> str:
    return (
        f"RIFT25K-k{name_value(args.key_shift)}-s{args.steps}"
        f"-ds{name_value(args.ds)}-spk{name_value(args.spk)}"
        f"-cfg{name_value(args.cfg_rescale)}-rf{args.robust_f0}-seed{args.seed}"
    )


def render_kernel() -> str:
    kernel = KERNEL_SOURCE.read_text(encoding="utf-8")
    if kernel.count(WORKER_MARKER) != 1:
        raise RuntimeError("RIFT kernel worker marker is missing or ambiguous")
    worker = KERNEL_WORKER_SOURCE.read_text(encoding="utf-8")
    return kernel.replace(WORKER_MARKER, f"BATCH_WORKER_SOURCE = {worker!r}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Upload multiple vocals, load RIFT once per available T4, and download "
            "one validated Float WAV for every input."
        )
    )
    parser.add_argument("inputs", type=Path, nargs="+")
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/kaggle"))
    parser.add_argument("--owner", default=DEFAULT_OWNER)
    parser.add_argument("--dataset-slug", default=DEFAULT_DATASET_SLUG)
    parser.add_argument("--kernel-slug", default=DEFAULT_KERNEL_SLUG)
    parser.add_argument("--repo-url", default=DEFAULT_REPO_URL)
    parser.add_argument("--repo-ref", default=DEFAULT_REPO_REF)
    parser.add_argument("--model-repo", default=DEFAULT_MODEL_REPO)
    parser.add_argument("--model-filename", default="rift25k.ckpt")
    parser.add_argument("--model-sha256", default=DEFAULT_MODEL_SHA256)
    parser.add_argument("--model-revision", default=DEFAULT_MODEL_REVISION)
    parser.add_argument("--modules-revision", default=DEFAULT_MODULES_REVISION)
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
        help="prepare and print the batch without uploading or running it",
    )
    args = parser.parse_args()
    if args.steps < 2:
        parser.error("--steps must be at least 2")
    if args.poll_interval < 5:
        parser.error("--poll-interval must be at least 5 seconds")
    if args.timeout < 60:
        parser.error("--timeout must be at least 60 seconds")
    return args


def build_job(args: argparse.Namespace, input_paths: list[Path]) -> dict:
    if not input_paths:
        raise ValueError("at least one input is required")
    items = []
    output_names: set[str] = set()
    input_digests = []
    for index, input_path in enumerate(input_paths, start=1):
        input_path = input_path.expanduser().resolve()
        if not input_path.is_file():
            raise FileNotFoundError(input_path)
        if input_path.suffix.lower() not in AUDIO_EXTENSIONS:
            raise ValueError(f"unsupported audio extension: {input_path.suffix}")
        digest = sha256(input_path)
        output_name = f"{input_path.stem}-rift25k.wav"
        if output_name in output_names:
            raise ValueError(f"duplicate batch output name: {output_name}")
        output_names.add(output_name)
        input_digests.append(digest)
        items.append(
            {
                "index": index,
                "input_name": f"input-{index:04d}{input_path.suffix.lower()}",
                "source_input_name": input_path.name,
                "input_sha256": digest,
                "input_audio": probe_audio(input_path),
                "transport_output_name": f"output-{index:04d}.wav",
                "output_name": output_name,
            }
        )
    now = datetime.now(UTC)
    batch_digest = hashlib.sha256("\0".join(input_digests).encode()).hexdigest()[:8]
    return {
        "schema_version": 2,
        "job_id": f"{now.strftime('%Y%m%dT%H%M%SZ')}-batch-{batch_digest}",
        "submitted_at": now.isoformat(),
        "run_name": build_run_name(args),
        "items": items,
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


def validate_outputs(job: dict, download_dir: Path) -> list[tuple[Path, dict]]:
    manifests = sorted(download_dir.rglob("manifest.json"))
    if len(manifests) != 1:
        raise RuntimeError(f"expected one manifest, found {manifests}")
    manifest = json.loads(manifests[0].read_text(encoding="utf-8"))
    if manifest.get("job_id") != job["job_id"]:
        raise RuntimeError("downloaded output belongs to a different Kaggle batch")
    records = {
        item["transport_output_name"]: item for item in manifest.get("outputs", [])
    }
    if len(records) != len(job["items"]):
        raise RuntimeError(
            f"expected {len(job['items'])} output records, found {len(records)}"
        )
    validated = []
    for item in job["items"]:
        transport_name = item["transport_output_name"]
        record = records.get(transport_name)
        matches = sorted(download_dir.rglob(transport_name))
        if record is None or len(matches) != 1:
            raise RuntimeError(f"missing unique output for {transport_name}: {matches}")
        output = matches[0]
        if sha256(output) != record.get("sha256"):
            raise RuntimeError(f"downloaded output SHA-256 mismatch: {output}")
        probe = probe_audio(output)
        if int(probe["sample_rate"]) != 44100 or int(probe["channels"]) != 1:
            raise RuntimeError(f"unexpected downloaded audio: {probe}")
        delta = frame_delta(item["input_audio"], probe)
        if abs(delta) > 2:
            raise RuntimeError(
                "downloaded audio frame count differs from input after resampling: "
                f"delta={delta} frames at {probe['sample_rate']} Hz"
            )
        validated.append((output, item))
    return validated


def run_job(args: argparse.Namespace) -> None:
    input_paths = [path.expanduser().resolve() for path in args.inputs]
    if shutil.which("kaggle") is None or shutil.which("ffprobe") is None:
        raise SystemExit(
            "kaggle and ffprobe must be available; enter the Flake environment"
        )

    job = build_job(args, input_paths)
    dataset_ref = f"{args.owner}/{args.dataset_slug}"
    kernel_ref = f"{args.owner}/{args.kernel_slug}"
    print(json.dumps(job, ensure_ascii=False, indent=2), flush=True)
    if args.dry_run:
        print("Dry run: no Kaggle dataset or kernel was changed.")
        return

    destination = args.output_dir.expanduser().resolve() / job["run_name"]
    if destination.exists() or destination.is_symlink():
        raise FileExistsError(f"refusing to overwrite existing Run: {destination}")

    with tempfile.TemporaryDirectory(prefix="rift-kaggle-") as temporary:
        root = Path(temporary)
        dataset_dir = root / "dataset"
        kernel_dir = root / "kernel"
        download_dir = root / "download"
        dataset_dir.mkdir()
        kernel_dir.mkdir()
        download_dir.mkdir()

        for source, item in zip(input_paths, job["items"], strict=True):
            shutil.copy2(source, dataset_dir / item["input_name"])
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
                    "subtitle": "Transient private input for one RIFT batch",
                    "description": "Managed by scripts/kaggle_rift.py; contains only the current batch.",
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
        expected_files = {"job.json"} | {item["input_name"] for item in job["items"]}
        wait_for_dataset(dataset_ref, expected_files, min(args.timeout, 30 * 60))

        (kernel_dir / "kernel.py").write_text(render_kernel(), encoding="utf-8")
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

        validated = validate_outputs(job, download_dir)
        targets = publish_directory_atomic(
            destination,
            [(output, Path(item["output_name"])) for output, item in validated],
        )
        for target in targets:
            print(f"Downloaded audio: {target}")
        print(f"Validated {len(validated)} outputs from one batch into: {destination}")


def main() -> None:
    args = parse_args()
    if args.dry_run:
        run_job(args)
        return
    channel = f"{args.owner}/{args.dataset_slug}|{args.owner}/{args.kernel_slug}"
    with exclusive_kaggle_channel(channel):
        run_job(args)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        raise SystemExit(
            "Interrupted; the Kaggle batch may still be running."
        ) from None
